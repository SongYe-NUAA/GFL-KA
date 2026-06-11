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


def load_data(data_dir, image_name):
    """加载从提取脚本保存的数据
    
    Args:
        data_dir (str): 数据根目录
        image_name (str): 图片名称（不含扩展名）
    """
    # 构建图片对应的数据目录路径
    image_data_dir = os.path.join(data_dir, image_name)
    if not os.path.exists(image_data_dir):
        print(f"错误: 在 {data_dir} 中未找到图片 {image_name} 对应的数据目录")
        return None, None, None, None

    # 查找所有level_X目录
    level_dirs = sorted([d for d in os.listdir(image_data_dir) if d.startswith('level_')])
    if not level_dirs:
        print(f"错误: 在 {image_data_dir} 中未找到任何level_X目录")
        return None, None, None, None

    attention_list = []
    kurtosis_list = []
    kurtosis_weight_list = []
    cls_score_list = []  # 新增分类分数列表

    print(f"找到 {len(level_dirs)} 个特征层级")

    for level_dir in level_dirs:
        level_path = os.path.join(image_data_dir, level_dir)
        attention_path = os.path.join(level_path, 'attention_weights.npy')
        kurtosis_path = os.path.join(level_path, 'kurtosis.npy')
        kurtosis_weight_path = os.path.join(level_path, 'kurtosis_weight.npy')
        cls_score_path = os.path.join(level_path, 'cls_score.npy')  # 分类分数文件路径

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
            
        # 加载分类分数
        if os.path.exists(cls_score_path):
            cls_score = np.load(cls_score_path)
            cls_score_list.append(cls_score)
        else:
            cls_score_list.append(None)

        print(f"已加载 {level_dir} 的数据:")
        print(f"  - 注意力权重形状: {attention.shape}")
        if os.path.exists(kurtosis_path):
            print(f"  - 峰度值形状: {kurtosis_list[-1].shape}")
        if os.path.exists(kurtosis_weight_path):
            print(f"  - 峰度权重形状: {kurtosis_weight_list[-1].shape}")
        if os.path.exists(cls_score_path):
            print(f"  - 分类分数形状: {cls_score_list[-1].shape}")

    if not attention_list:
        print("错误: 未能加载任何数据")
        return None, None, None, None

    return attention_list, kurtosis_list if kurtosis_list else None, kurtosis_weight_list if kurtosis_weight_list else None, cls_score_list if cls_score_list else None


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


def create_feature_overlay(original_img, feature_map, alpha=0.5, colormap='jet', center_zero=False, background_alpha=1.0):
    """将特征图映射回原始图像大小并创建叠加可视化

    Args:
        original_img (np.ndarray): 原始图像，形状为 [H, W, 3]
        feature_map (np.ndarray): 特征图，形状为 [H', W']
        alpha (float): 叠加透明度
        colormap (str): 使用的颜色映射
        center_zero (bool): 是否将0值居中(用于差异图)
        background_alpha (float): 背景图像的透明度，用于降低原图亮度

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
    if center_zero:
        # 对于差异图，使用对称色图，确保0在中间
        max_abs = np.max(np.abs(feature_resized)) + 1e-6
        feature_norm = (feature_resized / (2 * max_abs)) + 0.5  # 映射到[0,1]，0值对应0.5
    else:
        # 标准归一化，映射到[0,1]
        feature_norm = (feature_resized - feature_resized.min()) / (feature_resized.max() - feature_resized.min() + 1e-6)

    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(feature_norm)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 根据background_alpha调整原图亮度
    if background_alpha < 1.0:
        darkened_img = (original_img * background_alpha).astype(np.uint8)
    else:
        darkened_img = original_img

    # 创建叠加效果
    overlay = cv2.addWeighted(darkened_img, 1 - alpha, heatmap, alpha, 0)

    return overlay, feature_norm


def process_cls_score(cls_score, target_size):
    """处理分类分数数据以用于可视化"""
    if cls_score is None:
        return None
        
    print(f"\n=== cls_score 信息 ===")
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
    
    return cls_score_norm


def create_difference_map(orig_score, imp_score, threshold=0.1, positive_only=False):
    """创建并优化分类分数差异图
    
    Args:
        orig_score (np.ndarray): 原始模型分数图
        imp_score (np.ndarray): 改进模型分数图
        threshold (float): 差异显示阈值 (0-1)
        positive_only (bool): 是否只显示正向改进
        
    Returns:
        np.ndarray: 处理后的差异图
        dict: 差异统计信息
    """
    # 计算差异
    diff_map = imp_score - orig_score
    
    # 统计指标
    total_pixels = diff_map.size
    improved_pixels = np.sum(diff_map > threshold)  # 明显改进的像素数
    degraded_pixels = np.sum(diff_map < -threshold)  # 明显下降的像素数
    
    # 计算改进/下降百分比
    improved_percent = (improved_pixels / total_pixels) * 100
    degraded_percent = (degraded_pixels / total_pixels) * 100
    net_improvement = improved_percent - degraded_percent
    
    stats = {
        'total_pixels': total_pixels,
        'improved_pixels': improved_pixels,
        'degraded_pixels': degraded_pixels,
        'improved_percent': improved_percent,
        'degraded_percent': degraded_percent,
        'net_improvement': net_improvement
    }
    
    print(f"\n=== 差异统计 ===")
    print(f"- 总像素数: {total_pixels}")
    print(f"- 改进像素数: {improved_pixels} ({improved_percent:.2f}%)")
    print(f"- 下降像素数: {degraded_pixels} ({degraded_percent:.2f}%)")
    print(f"- 净改进: {net_improvement:.2f}%")
    
    # 如果只显示正向改进，则将负值部分置为0
    if positive_only:
        diff_map = np.maximum(diff_map, 0)
    
    # 应用阈值，将小差异处理为0
    diff_map[np.abs(diff_map) < threshold] = 0
    
    return diff_map, stats


def compare_models_visualization(image_path, output_dir, original_data_dir, improved_data_dir, layer_idx=None, alpha=0.7, 
                                diff_threshold=0.1, positive_only=True, show_decrease=False):
    """比较原始模型和改进模型的可视化效果
    
    Args:
        image_path (str): 输入图像路径
        output_dir (str): 输出目录
        original_data_dir (str): 原始模型数据目录
        improved_data_dir (str): 改进模型数据目录
        layer_idx (int): 要可视化的层索引
        alpha (float): 热力图透明度
        diff_threshold (float): 差异图阈值
        positive_only (bool): 是否只显示正向改进
        show_decrease (bool): 是否显示下降区域
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取并预处理图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None
            
        # 图像预处理
        img = original_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取图片名称（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 加载原始模型数据
        print("\n加载原始模型数据...")
        orig_attention, orig_kurtosis, orig_kurtosis_weight, orig_cls_score = load_data(original_data_dir, image_name)
        if orig_attention is None:
            print("无法加载原始模型数据")
            return False
            
        # 加载改进模型数据
        print("\n加载改进模型数据...")
        imp_attention, imp_kurtosis, imp_kurtosis_weight, imp_cls_score = load_data(improved_data_dir, image_name)
        if imp_attention is None:
            print("无法加载改进模型数据")
            return False
            
        # 确定要处理的层级
        num_levels = min(len(orig_attention), len(imp_attention))
        if num_levels == 0:
            print("错误: 没有找到可用的特征层级")
            return False
            
        # 如果指定了特定层级，只处理该层级，否则处理所有层级
        layer_indices = [layer_idx] if layer_idx is not None else range(num_levels)
        
        # 处理每个层级
        for layer_idx in layer_indices:
            if layer_idx >= num_levels:
                print(f"警告: 层索引 {layer_idx} 超出范围 (0-{num_levels-1})，跳过")
                continue
                
            print(f"\n=== 处理第 {layer_idx} 层 ===")
            
            # 获取当前层的数据
            orig_layer_attention = orig_attention[layer_idx]
            imp_layer_attention = imp_attention[layer_idx]
            
            # 获取当前层的峰度值
            orig_layer_kurtosis = None
            if orig_kurtosis is not None and layer_idx < len(orig_kurtosis):
                orig_layer_kurtosis = orig_kurtosis[layer_idx]
                
            # 获取原始模型分类分数
            orig_layer_cls_score = None
            if orig_cls_score is not None and layer_idx < len(orig_cls_score):
                orig_layer_cls_score = orig_cls_score[layer_idx]
                
            # 获取改进模型分类分数
            imp_layer_cls_score = None
            if imp_cls_score is not None and layer_idx < len(imp_cls_score):
                imp_layer_cls_score = imp_cls_score[layer_idx]
            
            # 创建可视化图像
            # 设置字体大小和标题
            TITLE_FONT_SIZE = 48
            COLORBAR_LABEL_SIZE = 32
            COLORBAR_TICK_SIZE = 32
            TITLE_PAD = 30
            colormap = 'jet'
            background_alpha = 1.0  # 定义背景图像的透明度，默认为1.0（不透明）
            
            # 选择分类分数差异图的颜色映射
            if show_decrease:
                colormap_diff = 'coolwarm'  # 红蓝对比色图：红色表示改进，蓝色表示下降
            else:
                colormap_diff = 'hot'  # 热力图：黄红色突出显示改进区域
            
            # 根据是否支持中文选择标题
            title_mean = '原始特征均值' if use_chinese else 'Original Mean Features'
            title_weighted = '加权后结果' if use_chinese else 'Weighted Result'
            title_orig_cls = '原模型分类分数' if use_chinese else 'Original Model Score'
            title_imp_cls = '改进模型分类分数' if use_chinese else 'Improved Model Score'
            
            if show_decrease:
                title_diff = '分类分数变化区域' if use_chinese else 'Score Changes'
            else:
                title_diff = '分类分数改进区域' if use_chinese else 'Score Improvement'
            
            title_image = '原始图像' if use_chinese else 'Original Image'
            
            # 创建带有colorbar空间的图像布局
            fig = plt.figure(figsize=(32, 24))
            gs = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], figure=fig)
            
            # 准备可视化所需数据
            target_size = (img.shape[1], img.shape[0])  # 宽, 高
            
            # 1. 原始特征均值
            if orig_layer_kurtosis is not None:
                if orig_layer_kurtosis.ndim == 5:  # [N, 4, 1, H, W]
                    mean_feat = np.mean(orig_layer_kurtosis[0], axis=0)  # 对4个方向取平均
                else:
                    mean_feat = orig_layer_kurtosis[0, 0] if orig_layer_kurtosis.ndim == 4 else orig_layer_kurtosis
                    
                mean_resized = resize_feature_map(mean_feat, target_size)
                mean_norm = normalize_feature_map(mean_resized)
                
                ax1 = fig.add_subplot(gs[0, 0])
                overlay, _ = create_feature_overlay(img, mean_resized, alpha=alpha, colormap=colormap)
                ax1.imshow(overlay)
                ax1.set_title(title_mean, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax1.axis('off')
            else:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(img)
                ax1.set_title(title_mean + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax1.axis('off')
            
            # 2. 加权后结果 (用改进模型的注意力权重)
            if orig_layer_kurtosis is not None and imp_layer_attention is not None:
                # 处理改进模型的注意力权重
                if imp_layer_attention.ndim == 4 and imp_layer_attention.shape[0] == 1 and imp_layer_attention.shape[1] == 1:
                    attention_avg = imp_layer_attention[0, 0]
                else:
                    attention_avg = np.mean(imp_layer_attention, axis=(0, 1)) if imp_layer_attention.ndim > 3 else imp_layer_attention
                    
                attention_resized = resize_feature_map(attention_avg, target_size)
                attention_norm = normalize_feature_map(attention_resized)
                
                # 使用加法而不是乘法来计算加权结果
                weighted_result = mean_norm * (1 + attention_norm)
                
                ax2 = fig.add_subplot(gs[0, 1])
                overlay, _ = create_feature_overlay(img, weighted_result, alpha=alpha, colormap=colormap)
                ax2.imshow(overlay)
                ax2.set_title(title_weighted, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax2.axis('off')
            else:
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(img)
                ax2.set_title(title_weighted + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                ax2.axis('off')

            # 3. 原模型分类分数热力图
            ax3 = fig.add_subplot(gs[0, 2])
            orig_cls_score_norm = None
            if orig_layer_cls_score is not None:
                orig_cls_score_norm = process_cls_score(orig_layer_cls_score, target_size)
                if orig_cls_score_norm is not None:
                    overlay, _ = create_feature_overlay(img, orig_cls_score_norm, alpha=alpha, colormap=colormap)
                    ax3.imshow(overlay)
                    ax3.set_title(title_orig_cls, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                else:
                    ax3.imshow(img)
                    ax3.set_title(title_orig_cls + " (处理错误)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            else:
                ax3.imshow(img)
                ax3.set_title(title_orig_cls + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            ax3.axis('off')
            
            # 4. 改进模型分类分数热力图
            ax4 = fig.add_subplot(gs[1, 0])
            imp_cls_score_norm = None
            if imp_layer_cls_score is not None:
                imp_cls_score_norm = process_cls_score(imp_layer_cls_score, target_size)
                if imp_cls_score_norm is not None:
                    overlay, _ = create_feature_overlay(img, imp_cls_score_norm, alpha=alpha, colormap=colormap)
                    ax4.imshow(overlay)
                    ax4.set_title(title_imp_cls, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                else:
                    ax4.imshow(img)
                    ax4.set_title(title_imp_cls + " (处理错误)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            else:
                ax4.imshow(img)
                ax4.set_title(title_imp_cls + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            ax4.axis('off')
            
            # 5. 分类分数改进区域图
            ax5 = fig.add_subplot(gs[1, 1])
            if orig_cls_score_norm is not None and imp_cls_score_norm is not None:
                # 创建优化的差异图
                diff_map, stats = create_difference_map(
                    orig_cls_score_norm, 
                    imp_cls_score_norm, 
                    threshold=diff_threshold,
                    positive_only=positive_only if not show_decrease else False  # 如果显示下降区域，则不使用positive_only
                )
                
                # 计算改进和下降百分比
                improved_percent = stats['improved_percent']
                degraded_percent = stats['degraded_percent'] if show_decrease else 0
                
                # 对于差异图，选择颜色映射
                if show_decrease:
                    # 使用红蓝对比色图，红色表示增强，蓝色表示减弱
                    colormap_diff = 'RdBu_r'  # 红蓝反转色图，确保红色表示增强
                else:
                    # 只显示增强区域时，使用红色热力图
                    colormap_diff = 'Reds'  # 红色热力图
                
                # 使用与其他热力图相同的渲染方式
                # 对于差异图，处理零值
                center_zero = show_decrease
                
                # 使用与其他热力图相同的方式创建叠加效果
                overlay, _ = create_feature_overlay(
                    img, 
                    diff_map, 
                    alpha=alpha, 
                    colormap=colormap_diff, 
                    center_zero=center_zero,
                    background_alpha=background_alpha
                )
                
                ax5.imshow(overlay)
                ax5.set_title(title_diff, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
                
            else:
                ax5.imshow(img)
                ax5.set_title(title_diff + " (无数据)", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            ax5.axis('off')
            
            # 6. 原始图像
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(img)
            ax6.set_title(title_image, fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
            ax6.axis('off')
            
            # 添加统一的颜色条
            cax = fig.add_subplot(gs[:, 3])
            cmap = plt.get_cmap(colormap)
            norm = plt.Normalize(vmin=0, vmax=1)
            cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.set_label('归一化值' if use_chinese else 'Normalized Value', fontsize=COLORBAR_LABEL_SIZE)
            cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
            
            # 调整子图间距
            plt.tight_layout(pad=4.0)
            
            # 获取文件名和保存路径
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f'compare_layer_{layer_idx}_{os.path.splitext(filename)[0]}.png')
            
            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"已保存比较结果: {output_path}")
        return True
        
    except Exception as e:
        print(f"比较模型可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='比较原始模型和改进模型的可视化效果')
    parser.add_argument('--img-dir', required=True, help='输入图像目录')
    parser.add_argument('--original-data-dir', required=True, help='原始模型数据目录')
    parser.add_argument('--improved-data-dir', required=True, help='改进模型数据目录')
    parser.add_argument('--save-dir', default='model_comparison', help='可视化结果保存目录')
    parser.add_argument('--layer-idx', type=int, help='要可视化的层索引，不指定则处理所有层级')
    parser.add_argument('--alpha', type=float, default=0.7, help='热力图透明度 (0-1)')
    parser.add_argument('--diff-threshold', type=float, default=0.1, help='差异显示阈值 (0-1)')
    parser.add_argument('--no-positive-only', action='store_false', dest='positive_only', help='显示所有变化区域（包括正向和负向）')
    parser.add_argument('--show-decrease', action='store_true', help='同时显示下降区域')
    parser.add_argument('--num-images', type=int, default=45, help='处理的图像数量 (当使用图像目录时)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 获取图像文件列表
    image_files = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    # 限制处理的图像数量
    if args.num_images and args.num_images > 0:
        image_files = image_files[:args.num_images]
        
    print(f"将处理 {len(image_files)} 张图像...")
    
    # 处理每个图像
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        image_name = os.path.splitext(filename)[0]
        print(f"\n处理图像 {i + 1}/{len(image_files)}: {filename}")
        
        # 创建该图像的输出目录
        image_output_dir = os.path.join(args.save_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # 比较模型可视化
        compare_models_visualization(
            image_path, 
            image_output_dir, 
            args.original_data_dir, 
            args.improved_data_dir,
            layer_idx=args.layer_idx,
            alpha=args.alpha,
            diff_threshold=args.diff_threshold,
            positive_only=args.positive_only,
            show_decrease=args.show_decrease
        )


if __name__ == '__main__':
    main() 