import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import glob
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import torch
import torch.nn.functional as F
import matplotlib.font_manager as fm
import platform
import functools

# 设置OpenMP环境变量以避免警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 设置中文字体
def set_chinese_font():
    system = platform.system()
    if system == 'Windows':
        # Windows系统
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':
        # macOS系统
        font_names = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
    else:
        # Linux系统
        font_names = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC',
                      'Noto Sans CJK TC']

    font_found = False
    for font_name in font_names:
        for font in fm.findSystemFonts():
            try:
                if font_name.lower() in os.path.basename(font).lower():
                    plt.rcParams['font.family'] = ['sans-serif']
                    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                    print(f"使用字体: {font_name}")
                    font_found = True
                    break
            except:
                continue
        if font_found:
            break

    if not font_found:
        # 使用英文标题替代
        print("未找到合适的中文字体，将使用英文标题")
        return False

    return True


# 在脚本开始时调用设置中文字体函数
use_chinese = set_chinese_font()


def load_data(data_dir):
    """加载从提取脚本保存的数据"""
    # 查找所有level_X目录
    level_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('level_')])
    if not level_dirs:
        print(f"错误: 在 {data_dir} 中未找到任何level_X目录")
        return None, None, None

    attention_list = []
    kurtosis_list = []
    kurtosis_weight_list = []

    print(f"找到 {len(level_dirs)} 个特征层级")

    for level_dir in level_dirs:
        level_path = os.path.join(data_dir, level_dir)
        attention_path = os.path.join(level_path, 'attention_weights.npy')
        kurtosis_path = os.path.join(level_path, 'kurtosis.npy')
        kurtosis_weight_path = os.path.join(level_path, 'kurtosis_weight.npy')

        if not os.path.exists(attention_path):
            print(f"警告: 未找到注意力权重文件 {attention_path}")
            continue

        # 加载数据
        attention = np.load(attention_path)
        attention_list.append(attention)

        if os.path.exists(kurtosis_path):
            kurtosis = np.load(kurtosis_path)
            kurtosis_list.append(kurtosis)

        if os.path.exists(kurtosis_weight_path):
            kurtosis_weight = np.load(kurtosis_weight_path)
            kurtosis_weight_list.append(kurtosis_weight)

        print(f"已加载 {level_dir} 的数据:")
        print(f"  - 注意力权重形状: {attention.shape}")
        if os.path.exists(kurtosis_path):
            print(f"  - 峰度值形状: {kurtosis_list[-1].shape}")
        if os.path.exists(kurtosis_weight_path):
            print(f"  - 峰度权重形状: {kurtosis_weight_list[-1].shape}")

    if not attention_list:
        print("错误: 未能加载任何数据")
        return None, None, None

    return attention_list, kurtosis_list if kurtosis_list else None, kurtosis_weight_list if kurtosis_weight_list else None


def resize_feature_map(feature_map, target_size):
    """将特征图调整到目标尺寸"""
    print(f"调整特征图: 原始形状 = {feature_map.shape}, 目标尺寸 = {target_size}")

    # 根据维度处理特征图
    if feature_map.ndim == 4:  # [N, C, H, W]
        if feature_map.shape[0] == 1 and feature_map.shape[1] == 1:
            # 单通道特征图，直接取内容
            feature_map = feature_map[0, 0]  # 变成 [H, W]
            print(f"  → 提取单通道特征图: {feature_map.shape}")
        else:
            # 多通道特征图，取第一个样本
            feature_map = feature_map[0]  # 变成 [C, H, W]
            # 如果是多通道，取平均
            if feature_map.shape[0] > 1:
                feature_map = np.mean(feature_map, axis=0)  # 变成 [H, W]
                print(f"  → 对通道取平均: {feature_map.shape}")
    elif feature_map.ndim > 4:  # 形如 [N, C, 4, H, W] 的峰度数据
        print(f"  → 处理高维特征图")
        if feature_map.shape[2] == 4:  # 典型的峰度数据格式 [N, C, 4, H, W]
            # 对4个方向取平均
            feature_map = np.mean(feature_map, axis=2)  # 变成 [N, C, H, W]
            print(f"  → 对方向取平均: {feature_map.shape}")
            # 然后提取单通道
            feature_map = feature_map[0, 0] if feature_map.shape[1] == 1 else np.mean(feature_map[0], axis=0)
            print(f"  → 最终形状: {feature_map.shape}")

    # 确保特征图是2D的
    if feature_map.ndim > 2:
        print(f"  ! 警告: 特征图维度大于2: {feature_map.shape}")
        feature_map = np.mean(feature_map, axis=0)
        print(f"  → 强制转换为2D: {feature_map.shape}")

    # 特征图可能是1D的(错误的处理)，转换为2D
    if feature_map.ndim == 1:
        side_length = int(np.sqrt(feature_map.size))
        feature_map = feature_map.reshape(side_length, side_length)
        print(f"  → 将1D转换为2D: {feature_map.shape}")

    # 调整特征图大小
    print(f"  → 调整前特征图范围: [{np.min(feature_map)}, {np.max(feature_map)}]")

    # 使用OpenCV直接调整大小(更可靠)
    feature_map_resized = cv2.resize(feature_map, (target_size[0], target_size[1]))
    print(f"  → 调整后特征图形状: {feature_map_resized.shape}")

    return feature_map_resized


def normalize_feature_map(feature_map, min_percentile=1, max_percentile=99):
    """归一化特征图，去除极值噪声点"""
    if feature_map.size == 0:
        print("警告: 特征图为空!")
        return np.zeros(feature_map.shape)

    flat_map = feature_map.flatten()
    min_val = np.percentile(flat_map, min_percentile)
    max_val = np.percentile(flat_map, max_percentile)

    # 裁剪到百分位数范围
    clipped = np.clip(feature_map, min_val, max_val)

    # 归一化到 [0, 1]
    normalized = (clipped - min_val) / (max_val - min_val + 1e-8)

    # 打印归一化前后的值域
    print(f"归一化: 原始范围=[{np.min(feature_map):.4f}, {np.max(feature_map):.4f}], " +
          f"裁剪范围=[{min_val:.4f}, {max_val:.4f}], " +
          f"归一化后范围=[{np.min(normalized):.4f}, {np.max(normalized):.4f}]")

    return normalized


def create_heatmap_overlay(img, attention_map, colormap='jet', alpha=0.5):
    """创建热力图叠加效果"""
    print("\n=== 创建热力图叠加 ===")
    print(f"图像形状: {img.shape}, 注意力图形状: {attention_map.shape}")

    # 确保图像是RGB格式
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print("将灰度图转换为RGB")
    elif img.shape[2] == 4:  # 带alpha通道的图像
        img = img[:, :, :3]
        print("从RGBA中去除alpha通道")

    # 调整注意力图尺寸以匹配原图
    target_size = (img.shape[1], img.shape[0])  # 宽, 高
    attention_resized = resize_feature_map(attention_map, target_size)

    # 归一化并应用颜色映射
    attention_normalized = normalize_feature_map(attention_resized)

    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(attention_normalized)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)

    # 确保热力图和图像尺寸完全一致
    if heatmap.shape != img.shape:
        print(f"调整热力图大小从 {heatmap.shape} 到 {img.shape}")
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 确保热力图和图像通道数一致
    if heatmap.shape[2] != img.shape[2]:
        print(f"热力图通道数 {heatmap.shape[2]} 与图像通道数 {img.shape[2]} 不匹配，进行调整")
        if heatmap.shape[2] == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif heatmap.shape[2] == 1 and img.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

    # 创建叠加效果，确保两个图像类型一致
    img_uint8 = img.astype(np.uint8)
    heatmap_uint8 = heatmap.astype(np.uint8)

    try:
        overlay = cv2.addWeighted(
            img_uint8,
            1 - alpha,
            heatmap_uint8,
            alpha,
            0
        )
        print("热力图叠加成功")
    except Exception as e:
        print(f"叠加过程中出错: {e}")
        print(f"img_uint8 类型: {type(img_uint8)}, 形状: {img_uint8.shape}, 数据类型: {img_uint8.dtype}")
        print(
            f"heatmap_uint8 类型: {type(heatmap_uint8)}, 形状: {heatmap_uint8.shape}, 数据类型: {heatmap_uint8.dtype}")
        # 在出错的情况下返回原图
        overlay = img_uint8

    return overlay, attention_normalized


def visualize_probability_comparison(prob, attention_weights, save_dir, layer_idx=0):
    """可视化概率分布在应用attention_weights前后的对比

    Args:
        prob (np.ndarray): 原始概率分布，形状为 [N, 4, reg_max+1, H, W]
        attention_weights (np.ndarray): 注意力权重，形状为 [N, 1, H, W]
        save_dir (str): 保存可视化结果的目录
        layer_idx (int): 要可视化的层索引
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 打印详细的调试信息
        print("\n=== 调试信息 ===")
        print(f"输入数据形状:")
        print(f"- 概率分布: {prob.shape}")
        print(f"- 注意力权重: {attention_weights.shape}")

        # 提取当前层的概率分布和注意力权重
        layer_prob = prob[layer_idx]  # [4, reg_max+1, H, W]
        layer_weights = attention_weights[layer_idx]  # [1, H, W]

        print(f"\n当前层数据形状:")
        print(f"- 层概率分布: {layer_prob.shape}")
        print(f"- 层注意力权重: {layer_weights.shape}")

        # 确保注意力权重是2D的
        if layer_weights.ndim == 1:
            print("警告: 注意力权重是1D的，尝试重塑为2D")
            side_length = int(np.sqrt(layer_weights.size))
            layer_weights = layer_weights.reshape(side_length, side_length)
        elif layer_weights.ndim == 3:
            print("警告: 注意力权重是3D的，提取第一个通道")
            layer_weights = layer_weights[0]

        print(f"\n处理后数据形状:")
        print(f"- 层注意力权重: {layer_weights.shape}")

        # 计算加权后的概率分布
        weighted_prob = layer_prob * (1 + layer_weights[None, None, :, :])  # 残差形式

        print(f"\n加权后数据形状:")
        print(f"- 加权概率分布: {weighted_prob.shape}")

        # 创建更大的对比图，包含原始分布和加权后的分布
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(4, 4, figure=fig)

        # 设置总标题
        fig.suptitle(f'Probability Distribution Analysis - Layer {layer_idx} (Shape: {layer_prob.shape})', fontsize=16)

        # 1. 原始概率分布 (左上角)
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            prob_2d = layer_prob[i].mean(axis=0)
            im = ax.imshow(prob_2d, cmap='hot')
            ax.set_title(f'Original Prob - Dir {i}')
            plt.colorbar(im, ax=ax)

        # 2. 注意力权重 (第二行)
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(layer_weights, cmap='hot')
            ax.set_title(f'Attention Weights')
            plt.colorbar(im, ax=ax)

        # 3. 加权后的概率分布 (第三行)
        for i in range(4):
            ax = fig.add_subplot(gs[2, i])
            weighted_prob_2d = weighted_prob[i].mean(axis=0)
            im = ax.imshow(weighted_prob_2d, cmap='hot')
            ax.set_title(f'Weighted Prob - Dir {i}')
            plt.colorbar(im, ax=ax)

        # 4. 差异图 (最后一行)
        for i in range(4):
            ax = fig.add_subplot(gs[3, i])
            orig_prob_2d = layer_prob[i].mean(axis=0)
            weighted_prob_2d = weighted_prob[i].mean(axis=0)
            diff = weighted_prob_2d - orig_prob_2d
            im = ax.imshow(diff, cmap='bwr')  # 使用蓝白红色图显示差异
            ax.set_title(f'Difference - Dir {i}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prob_comparison_layer_{layer_idx}.png'))
        plt.close()

        print(f"\n已保存概率分布对比图到: {os.path.join(save_dir, f'prob_comparison_layer_{layer_idx}.png')}")

        # 保存统计信息
        stats = {
            'original_mean': layer_prob.mean(),
            'original_std': layer_prob.std(),
            'weighted_mean': weighted_prob.mean(),
            'weighted_std': weighted_prob.std(),
            'attention_mean': layer_weights.mean(),
            'attention_std': layer_weights.std(),
            'shape': {
                'prob': layer_prob.shape,
                'attention': layer_weights.shape,
                'weighted_prob': weighted_prob.shape
            }
        }

        # 保存统计信息到文件
        with open(os.path.join(save_dir, f'stats_layer_{layer_idx}.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"已保存统计信息到: {os.path.join(save_dir, f'stats_layer_{layer_idx}.txt')}")

    except Exception as e:
        print(f"可视化概率分布对比时出错: {e}")
        import traceback
        traceback.print_exc()


def process_single_image(image_path, output_dir, attention_weights, kurtosis=None, data_dir=None, start_idx=0,
                         kurtosis_weight=None):
    """处理单个图像的注意力可视化"""
    try:
        # 读取并预处理图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None

        # 图像预处理
        img = original_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 确保只处理5个特征层
        num_levels = 5
        attention_weights = attention_weights[:num_levels]  # 只取前5层

        # 为每一层的注意力权重创建可视化
        for layer_idx, layer_attention in enumerate(attention_weights):
            # 直接使用layer_idx作为层号，不需要加start_idx
            layer_num = layer_idx
            print(f"\n处理第 {layer_num} 层...")

            # 获取当前层的峰度值（如果有）
            layer_mean_feat = None
            if kurtosis is not None and layer_idx < len(kurtosis):
                layer_mean_feat = kurtosis[layer_idx]

            # 获取当前层的峰度权重（如果有）
            layer_kurtosis_weight = None
            if kurtosis_weight is not None and layer_idx < len(kurtosis_weight):
                layer_kurtosis_weight = kurtosis_weight[layer_idx]
            elif data_dir:
                level_dir = os.path.join(data_dir, f'level_{layer_num}')
                kurtosis_weight_path = os.path.join(level_dir, 'kurtosis_weight.npy')
                if os.path.exists(kurtosis_weight_path):
                    try:
                        layer_kurtosis_weight = np.load(kurtosis_weight_path)
                    except Exception as e:
                        print(f"    警告: 无法加载峰度权重: {e}")

            # 创建带有colorbar空间的图像布局，增加图像大小以适应更大的字体
            fig = plt.figure(figsize=(32, 24))  # 增加整体高度以容纳更大字体
            gs = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], figure=fig)

            # 根据是否支持中文选择标题
            title_mean = '原始特征均值' if use_chinese else 'Original Mean Features'
            title_kurtosis = '峰度权重' if use_chinese else 'Kurtosis Weight'
            title_attention = '注意力权重' if use_chinese else 'Attention Weight'
            title_weighted = '加权后结果' if use_chinese else 'Weighted Result'
            title_cls_score = '分类分数热力图' if use_chinese else 'Cls Score Heatmap'

            # 增加字体大小设置
            TITLE_FONT_SIZE = 48  # 标题字体大小
            COLORBAR_LABEL_SIZE = 32  # 颜色条标签字体大小
            COLORBAR_TICK_SIZE = 32  # 颜色条刻度字体大小
            TITLE_PAD = 30  # 标题间距

            try:
                # 准备所有需要可视化的数据
                if layer_mean_feat is not None:
                    if layer_mean_feat.ndim == 5:  # [N, 4, 1, H, W]
                        mean_feat = np.mean(layer_mean_feat[0], axis=0)  # 对4个方向取平均
                    else:
                        mean_feat = layer_mean_feat[0, 0] if layer_mean_feat.ndim == 4 else layer_mean_feat

                    # 调整特征图大小以匹配原图
                    target_size = (img.shape[1], img.shape[0])  # 宽, 高
                    mean_resized = resize_feature_map(mean_feat, target_size)

                if layer_attention.ndim == 4 and layer_attention.shape[0] == 1 and layer_attention.shape[1] == 1:
                    attention_avg = layer_attention[0, 0]
                else:
                    attention_avg = np.mean(layer_attention,
                                            axis=(0, 1)) if layer_attention.ndim > 3 else layer_attention

                # 调整注意力权重到原图大小
                attention_resized = resize_feature_map(attention_avg, target_size)

                # 对原始特征和注意力权重分别归一化到[0,1]范围
                mean_norm = normalize_feature_map(mean_resized)
                attention_norm = normalize_feature_map(attention_resized)

                # 使用加法而不是乘法来计算加权结果
                weighted_result = mean_norm * (1 + attention_norm)
                # 确保结果在[0,1]范围内
                # weighted_result = weighted_result/2

                # 打印调试信息
                print("\n=== 加权计算调试信息 ===")
                print(f"归一化后的原始特征范围: [{np.min(mean_norm):.4f}, {np.max(mean_norm):.4f}]")
                print(f"归一化后的注意力权重范围: [{np.min(attention_norm):.4f}, {np.max(attention_norm):.4f}]")
                print(f"加权结果范围: [{np.min(weighted_result):.4f}, {np.max(weighted_result):.4f}]")

                # 统一使用jet颜色映射
                colormap = 'jet'
                alpha = 0.7

                # 1. 原始特征均值热力图
                ax1 = fig.add_subplot(gs[0, 0])
                overlay, _ = create_feature_overlay(img, mean_resized, alpha=alpha, colormap=colormap)
                ax1.imshow(overlay)
                ax1.set_title(title_mean, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax1.axis('off')

                # 2. 峰度权重热力图
                ax2 = fig.add_subplot(gs[0, 1])
                if layer_kurtosis_weight is not None:
                    overlay, _ = create_feature_overlay(img, layer_kurtosis_weight, alpha=alpha, colormap=colormap)
                    ax2.imshow(overlay)
                    ax2.set_title(title_kurtosis, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                else:
                    ax2.imshow(img)
                    ax2.set_title(title_kurtosis + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax2.axis('off')

                # 3. 注意力权重热力图
                ax3 = fig.add_subplot(gs[0, 2])
                overlay, _ = create_feature_overlay(img, attention_resized, alpha=alpha, colormap=colormap)
                ax3.imshow(overlay)
                ax3.set_title(title_attention, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax3.axis('off')

                # 4. 加权后的结果
                ax4 = fig.add_subplot(gs[1, 0])
                overlay, _ = create_feature_overlay(img, weighted_result, alpha=alpha, colormap=colormap)
                ax4.imshow(overlay)
                ax4.set_title(title_weighted, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax4.axis('off')
                
                # 5. 分类分数热力图 (新增)
                cls_score_path = os.path.join(data_dir, f'level_{layer_num}', 'cls_score.npy')
                print(f"\n尝试加载分类分数文件: {cls_score_path}")
                print(f"文件是否存在: {os.path.exists(cls_score_path)}")

                if os.path.exists(cls_score_path):
                    try:
                        cls_score = np.load(cls_score_path)  # [N, num_classes, H, W]
                        print(f"\n=== cls_score 信息 ===")
                        print(f"- 文件路径: {cls_score_path}")
                        print(f"- 数据形状: {cls_score.shape}")
                        print(f"- 数据类型: {cls_score.dtype}")
                        print(f"- 数值范围: [{np.min(cls_score)}, {np.max(cls_score)}]")
                        
                        # 获取最大类别分数
                        if cls_score.ndim == 4:  # [N, num_classes, H, W]
                            print("处理4D张量...")
                            cls_score_map = np.max(cls_score[0], axis=0)  # 取第一个样本，跨类别取最大值
                        elif cls_score.ndim == 3:  # [num_classes, H, W]
                            print("处理3D张量...")
                            cls_score_map = np.max(cls_score, axis=0)
                        else:
                            print("处理2D张量...")
                            cls_score_map = cls_score  # 如果已经是2D
                        
                        print(f"- cls_score_map 形状: {cls_score_map.shape}")
                        print(f"- cls_score_map 范围: [{np.min(cls_score_map)}, {np.max(cls_score_map)}]")
                        
                        # 调整到与原图相同大小
                        cls_score_resized = resize_feature_map(cls_score_map, target_size)
                        cls_score_norm = normalize_feature_map(cls_score_resized)
                        
                        # 创建热力图
                        ax5 = fig.add_subplot(gs[1, 1])
                        overlay, _ = create_feature_overlay(img, cls_score_norm, alpha=alpha, colormap=colormap)
                        ax5.imshow(overlay)
                        ax5.set_title(title_cls_score, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                        ax5.axis('off')
                        
                        print(f"- 最终热力图范围: [{np.min(cls_score_norm):.4f}, {np.max(cls_score_norm):.4f}]")
                    except Exception as e:
                        print(f"\n处理cls_score时出错: {e}")
                        print("详细错误信息:")
                        import traceback
                        traceback.print_exc()
                        # 如果出错，显示原始图像作为后备
                        ax5 = fig.add_subplot(gs[1, 1])
                        ax5.imshow(img)
                        ax5.set_title(title_cls_score + " (处理错误)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                        ax5.axis('off')
                else:
                    print(f"\n未找到cls_score文件: {cls_score_path}")
                    print(f"当前目录内容:")
                    level_dir = os.path.join(data_dir, f'level_{layer_num}')
                    if os.path.exists(level_dir):
                        print("\n".join(os.listdir(level_dir)))
                    else:
                        print(f"目录 {level_dir} 不存在")
                
                # 6. 原始图像
                ax6 = fig.add_subplot(gs[1, 2])
                ax6.imshow(img)
                ax6.set_title("原始图像" if use_chinese else "Original Image", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax6.axis('off')

                # 添加统一的颜色条，增加字体大小
                cax = fig.add_subplot(gs[:, 3])
                cmap = plt.get_cmap(colormap)
                norm = plt.Normalize(vmin=0, vmax=1)
                cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                cb.set_label('归一化值' if use_chinese else 'Normalized Value', 
                            fontsize=COLORBAR_LABEL_SIZE)  # 增大颜色条标签字体
                cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)  # 增大颜色条刻度字体

                # 调整子图间距，增加间距以适应更大的字体
                plt.tight_layout(pad=4.0)  # 增加间距

                # 保存图像，增加DPI以提高清晰度
                plt.savefig(os.path.join(output_dir, f'{layer_num}_heatmaps.png'),
                            dpi=300,  # 增加DPI以提高清晰度
                            bbox_inches='tight')
                plt.close(fig)

                print(f"已保存可视化结果: {layer_num}_heatmaps.png")

            except Exception as e:
                print(f"生成可视化时出错: {e}")
                import traceback
                traceback.print_exc()
                plt.close(fig)

        return True
    except Exception as e:
        print(f"处理单个图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_attention_on_images(image_dir, output_dir, attention_weights, kurtosis=None, kurtosis_weight=None,
                                  num_images=5, data_dir=None, layer_idx=None, single_image=None):
    """对一批图像可视化注意力权重"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取图像文件列表
    if single_image:
        image_files = [single_image]
    else:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 限制处理的图像数量
    if num_images and num_images > 0:
        image_files = image_files[:num_images]

    print(f"将处理 {len(image_files)} 张图像...")

    # 处理每个图像
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file) if not single_image else single_image
        filename = os.path.basename(image_path)
        output_image_dir = os.path.join(output_dir, os.path.splitext(filename)[0])

        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        print(f"处理图像 {i + 1}/{len(image_files)}: {filename}")

        # 只处理指定的层（如果提供了layer_idx）
        if layer_idx is not None:
            if 0 <= layer_idx < len(attention_weights):
                single_layer_weights = [attention_weights[layer_idx]]
                single_layer_kurtosis = [kurtosis[layer_idx]] if kurtosis is not None else None
                single_layer_kurtosis_weight = [kurtosis_weight[layer_idx]] if kurtosis_weight is not None else None

                process_single_image(
                    image_path, output_image_dir, single_layer_weights, single_layer_kurtosis,
                    data_dir=data_dir, start_idx=layer_idx, kurtosis_weight=single_layer_kurtosis_weight
                )
            else:
                print(f"错误: 层索引 {layer_idx} 超出范围 (0-{len(attention_weights) - 1})")
        else:
            # 处理所有层
            process_single_image(
                image_path, output_image_dir, attention_weights, kurtosis,
                data_dir=data_dir, kurtosis_weight=kurtosis_weight
            )


def main():
    parser = argparse.ArgumentParser(description='在原始图像上可视化注意力权重和峰度')
    parser.add_argument('--img-dir', help='测试图像目录')
    parser.add_argument('--single-image', help='单个图像文件路径，用于只处理一张图片')
    parser.add_argument('--data-dir', required=True, help='包含提取数据的目录')
    parser.add_argument('--save-dir', default='attention_visualization', help='可视化结果保存目录')
    parser.add_argument('--alpha', type=float, default=0.6, help='热力图透明度 (0-1)')

    args = parser.parse_args()

    # 检查参数
    if not args.img_dir and not args.single_image:
        print("错误: 必须提供--img-dir或--single-image参数")
        return

    # 加载数据
    attention, kurtosis, kurtosis_weight = load_data(args.data_dir)
    if attention is None:
        return

    # 可视化
    visualize_attention_on_images(args.img_dir, args.save_dir, attention, kurtosis, kurtosis_weight,
                                  single_image=args.single_image, data_dir=args.data_dir)

    print(f"可视化完成! 结果已保存到: {args.save_dir}")


def create_feature_overlay(original_img, feature_map, alpha=0.5, colormap='jet'):
    """将特征图映射回原始图像大小并创建叠加可视化

    Args:
        original_img (np.ndarray): 原始图像，形状为 [H, W, 3]
        feature_map (np.ndarray): 特征图，形状为 [H', W']
        alpha (float): 叠加透明度
        colormap (str): 使用的颜色映射

    Returns:
        np.ndarray: 叠加后的图像
    """
    # 确保特征图是2D的
    if feature_map.ndim > 2:
        feature_map = feature_map.squeeze()

    # 调整特征图大小以匹配原始图像
    target_size = (original_img.shape[1], original_img.shape[0])
    feature_resized = cv2.resize(feature_map, target_size)

    # 归一化特征图到[0,1]范围
    feature_norm = (feature_resized - feature_resized.min()) / (feature_resized.max() - feature_resized.min() + 1e-6)

    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(feature_norm)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)

    # 创建叠加效果
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

    return overlay, feature_norm


if __name__ == '__main__':
    main() 