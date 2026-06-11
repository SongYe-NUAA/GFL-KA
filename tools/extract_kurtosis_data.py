import torch
import pickle
import os
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import MODELS
from mmengine.registry import init_default_scope
import functools
import sys
from mmcv.transforms import Compose
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
import torch.nn.functional as F
import functools
# 设置OpenMP环境变量以避免警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# === 改进的跟踪代码 ===
def trace_forward_single(func):
    """跟踪GFocalHead的forward_single方法，记录关键变量"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]  # GFocalHead 实例
        x = args[1]     # 输入特征图
        scale = args[2] # 尺度因子
        
        # 从全局变量获取保存路径
        global current_save_dir
        if not hasattr(wrapper, "current_save_dir"):
            wrapper.current_save_dir = "kurtosis_data"
        save_dir = wrapper.current_save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取当前特征层级的索引
        if not hasattr(wrapper, 'level_idx'):
            wrapper.level_idx = 0
        
        # 获取当前层的索引
        current_level = wrapper.level_idx % 5  # 假设有5个特征层级
        
        # 创建level目录
        level_dir = os.path.join(save_dir, f'level_{current_level}')
        os.makedirs(level_dir, exist_ok=True)
        
        print(f"\n=== 处理特征层 {current_level} ===")
        print(f"保存目录: {level_dir}")
        
        try:
            # 1. 首先保存输入特征
            np.save(os.path.join(level_dir, "input_feature.npy"), x.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存输入特征 x 到 {os.path.join(level_dir, 'input_feature.npy')}")
            
            # 2. 调用原始前向函数的前半部分来获取边界框预测
            reg_feat = x
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()
            
            # 3. 从bbox_pred计算概率分布
            N, C, H, W = bbox_pred.size()
            # 重塑并应用softmax得到概率分布
            prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            np.save(os.path.join(level_dir, "prob.npy"), prob.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存概率分布 prob 到 {os.path.join(level_dir, 'prob.npy')}")
            
            # 4. 获取概率最高的top-k个值与均值
            prob_topk, _ = prob.topk(self.reg_topk, dim=2)
            np.save(os.path.join(level_dir, "prob_topk.npy"), prob_topk.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存top-k概率 prob_topk")
            
            # 5. 计算均值特征
            mean_feat = prob.mean(dim=2, keepdim=True)  # [N, 4, 1, H, W]
            np.save(os.path.join(level_dir, "mean_feat.npy"), mean_feat.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存均值特征 mean_feat")
            
            # 6. 计算标准化的峰度
            mean_all = prob.mean(dim=2, keepdim=True)
            centered = prob - mean_all
            std = torch.std(prob, dim=2, keepdim=True) + 1e-6
            normalized = centered / std
            kurtosis = torch.mean(normalized ** 4, dim=2, keepdim=False)  # [N, 4, H, W]
            np.save(os.path.join(level_dir, "kurtosis.npy"), kurtosis.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存峰度 kurtosis")
            
            # 7. 计算kurtosis权重
            kurtosis_weight = torch.mean(kurtosis, dim=1, keepdim=True)  # [N, 1, H, W]
            if hasattr(self, 'adaptive_linear_map'):
                # 如果模型有这个方法
                kurtosis_weight = self.adaptive_linear_map(kurtosis_weight)
            np.save(os.path.join(level_dir, "kurtosis_weight.npy"), kurtosis_weight.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存峰度权重 kurtosis_weight")
            
            # 8. 计算注意力权重
            attention_weights = torch.sigmoid(kurtosis_weight * 3.0)  # 放大对比度
            np.save(os.path.join(level_dir, "attention_weights.npy"), attention_weights.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存注意力权重 attention_weights")
            
            # 9. 构建加权前的stat
            if hasattr(self, 'add_mean') and self.add_mean:
                # 如果使用了add_mean，则stat应该是prob_topk和mean_feat的拼接
                stat_before = torch.cat([prob_topk, mean_feat], dim=2)
            else:
                # 否则stat就是prob_topk
                stat_before = prob_topk
            
            np.save(os.path.join(level_dir, "stat_before_weighting.npy"), stat_before.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存加权前的stat")
            
            # 10. 计算加权后的stat（残差形式：stat * (1 + attention_weights)）
            attention_expanded = attention_weights
            if stat_before.dim() > attention_weights.dim():
                # 需要扩展attention维度以匹配stat
                attention_expanded = attention_weights.unsqueeze(2) if attention_weights.dim() == 4 else attention_weights
            
            stat_after = stat_before * (1 + attention_expanded)  # 残差形式
            np.save(os.path.join(level_dir, "stat_after_weighting.npy"), stat_after.detach().cpu().numpy())
            print(f"✓ Level {current_level}: 已保存加权后的stat")
            
            # 打印数据形状信息，方便调试
            print(f"\nLevel {current_level} 重要张量形状信息:")
            print(f"- 输入特征 x: {x.shape}")
            print(f"- 概率分布 prob: {prob.shape}")
            print(f"- 峰度 kurtosis: {kurtosis.shape}")
            print(f"- 峰度权重 kurtosis_weight: {kurtosis_weight.shape}")
            print(f"- 注意力权重 attention_weights: {attention_weights.shape}")
            
            # 保存形状信息到文件
            with open(os.path.join(level_dir, "shapes.txt"), "w") as f:
                f.write(f"Level {current_level} 张量形状信息:\n")
                f.write(f"输入特征 x: {x.shape}\n")
                f.write(f"概率分布 prob: {prob.shape}\n")
                f.write(f"峰度 kurtosis: {kurtosis.shape}\n")
                f.write(f"峰度权重 kurtosis_weight: {kurtosis_weight}\n")
                f.write(f"注意力权重 attention_weights: {attention_weights.shape}\n")
            
            # 调用原始前向函数，获取最终的分类和边界框预测
            cls_score, bbox_pred = func(*args, **kwargs)
            
            # 添加调试信息
            print(f"\n=== 保存cls_score信息 ===")
            print(f"特征层: {current_level}")
            print(f"cls_score 形状: {cls_score.shape}")
            print(f"cls_score 类型: {cls_score.dtype}")
            print(f"cls_score 设备: {cls_score.device}")
            print(f"保存路径: {os.path.join(level_dir, 'cls_score.npy')}")

            try:
                # 保存最终的分类得分和边界框预测
                cls_score_np = cls_score.detach().cpu().numpy()
                print(f"转换后的numpy数组形状: {cls_score_np.shape}")
                print(f"numpy数组范围: [{np.min(cls_score_np)}, {np.max(cls_score_np)}]")
                
                np.save(os.path.join(level_dir, "cls_score.npy"), cls_score_np)
                np.save(os.path.join(level_dir, "bbox_pred_final.npy"), bbox_pred.detach().cpu().numpy())
                print(f"✓ Level {current_level}: 已成功保存最终输出张量 cls_score 和 bbox_pred_final")
            except Exception as e:
                print(f"保存cls_score时出错: {e}")
                import traceback
                traceback.print_exc()
            
            # 更新level_idx
            wrapper.level_idx += 1
            
            return cls_score, bbox_pred
            
        except Exception as e:
            print(f"Level {current_level} 处理出错: {e}")
            import traceback
            traceback.print_exc()
            wrapper.level_idx += 1
            return func(*args, **kwargs)
    
    # 重置level_idx
    wrapper.level_idx = 0
    return wrapper

# 全局变量保存当前处理的目录
current_save_dir = "kurtosis_data"

# 应用修改到GFocalHead
from mmdet.models.dense_heads.gfocal_head import GFocalHead
# 保存原始forward_single方法
original_forward_single = GFocalHead.forward_single
# 应用装饰器
GFocalHead.forward_single = trace_forward_single(GFocalHead.forward_single)
print("已添加GFocalHead.forward_single跟踪功能")

def process_image(image_path, model, device='cuda', save_dir='kurtosis_data'):
    """处理单张图像并提取峰度信息"""
    global current_save_dir
    
    # 为每张图片创建独立的结果目录
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_dir = os.path.join(save_dir, img_name)
    os.makedirs(img_save_dir, exist_ok=True)
    
    # 更新当前保存目录
    current_save_dir = img_save_dir
    # 设置装饰器的保存目录
    GFocalHead.forward_single.current_save_dir = img_save_dir
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_shape = img.shape[:2]  # 保存原始尺寸
    
    # 创建数据样本
    data_sample = DetDataSample()
    
    # 图像预处理：调整大小至(1333, 800)，这是COCO数据集常用的大小
    target_size = (1333, 800)
    img_resized = cv2.resize(img, (target_size[0], target_size[1]))
    new_shape = img_resized.shape[:2]
    
    # 计算缩放因子 - 修改为更简单的格式
    scale_w = target_size[0] / orig_shape[1]
    scale_h = target_size[1] / orig_shape[0]
    scale_factor = np.array([scale_w, scale_h, scale_w, scale_h], dtype=np.float32)
    
    # 设置必要的元信息
    data_sample.set_metainfo({
        'img_id': 0, 
        'img_shape': new_shape,
        'ori_shape': orig_shape,
        'scale_factor': scale_factor,
        'pad_shape': new_shape,
        'batch_input_shape': new_shape  # 只包含空间维度 (H, W)
    })
    
    # 保存原始图像以便可视化
    images_dir = os.path.join(img_save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    img_save_path = os.path.join(images_dir, os.path.basename(image_path))
    cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✓ 已保存原始图像到 {img_save_path}")
    
    # 转换为张量
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(device)
    img_tensor = img_tensor / 255.0  # 归一化到[0,1]
    
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)
    
    # 模拟标准预处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    # 推理
    try:
        with torch.no_grad():
            model.eval()
            # 使用测试模式，让模型仅提取特征不进行后处理，避免尺寸不匹配错误
            # 这样我们仍然可以获取到kurtosis相关数据
            try:
                # 首先尝试完整推理
                results = model(img_tensor, [data_sample], mode='predict')
                print(f"图像 {os.path.basename(image_path)} 完整处理成功")
            except Exception as e:
                print(f"完整推理失败，但可能已捕获特征数据: {e}")
                # 只提取特征
                try:
                    # 尝试直接提取特征
                    _ = model.extract_feat(img_tensor)
                    print("特征提取成功")
                except Exception as feat_error:
                    print(f"特征提取也失败: {feat_error}")
            
        # 返回保存目录
        return img_save_dir  
    except Exception as e:
        print(f"处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_kurtosis_during_inference(config_file, checkpoint, img_dir=None, single_image=None, save_dir='kurtosis_results'):
    """
    在推理过程中提取峰度信息
    
    Args:
        config_file: 模型配置文件路径
        checkpoint: 模型检查点文件路径
        img_dir: 测试图像目录
        single_image: 单个图像文件路径
        save_dir: 保存结果的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载配置
    print(f"加载配置: {config_file}")
    cfg = Config.fromfile(config_file)
    
    # 设置工作环境
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # 构建模型
    print("构建模型...")
    model = MODELS.build(cfg.model)
    model.to(device)
    
    # 加载检查点权重
    if checkpoint:
        print(f"加载检查点: {checkpoint}")
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        
        # 检查检查点格式
        if 'state_dict' in checkpoint_dict:
            checkpoint_dict = checkpoint_dict['state_dict']
        
        # 兼容性处理
        if list(checkpoint_dict.keys())[0].startswith('module.'):
            checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}
        
        model.load_state_dict(checkpoint_dict, strict=False)
        print("模型权重加载完成")
    
    # 处理单个图片或图片目录
    img_files = []
    
    if single_image and os.path.isfile(single_image):
        # 单个图像模式
        img_files = [single_image]
        print(f"使用单个指定图像: {single_image}")
    elif img_dir:
        # 目录模式
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in img_extensions:
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))
            img_files.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))
        
        if not img_files:
            print(f"错误: 在 {img_dir} 目录中未找到图像文件")
            return
        
        print(f"找到 {len(img_files)} 个图像文件")
    else:
        print("错误: 必须提供img_dir或single_image参数")
        return
    
    # 处理图像
    processed_dirs = []
    for i, img_file in enumerate(img_files):
        print(f"\n处理图像 {i+1}/{len(img_files)}: {img_file}")
        result_dir = process_image(img_file, model, device, save_dir)
        if result_dir:
            processed_dirs.append(result_dir)
            
            # 为当前图片生成可视化
            try:
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                vis_cmd = f"python tools/visualize_attention_on_image.py --img-dir {os.path.dirname(img_file)} --data-dir {result_dir} --save-dir {result_dir}/visualization --single-image {img_file}"
                print(f"\n可以使用以下命令为这张图片生成可视化:\n{vis_cmd}")
            except Exception as e:
                print(f"生成可视化命令时出错: {e}")
    
    print(f"\n处理完成! 结果已保存到: {save_dir}")
    return processed_dirs

def compare_model_kurtosis(original_config, original_checkpoint, 
                           improved_config, improved_checkpoint,
                           img_dir=None, single_image=None, save_dir='comparison_results'):
    """
    比较原始模型和改进模型的峰度分布
    
    Args:
        original_config: 原始模型配置文件路径
        original_checkpoint: 原始模型检查点文件路径
        improved_config: 改进模型配置文件路径
        improved_checkpoint: 改进模型检查点文件路径
        img_dir: 测试图像目录
        single_image: 单个图像文件路径
        save_dir: 保存结果的目录
    """
    # 创建子目录
    original_dir = os.path.join(save_dir, 'original')
    improved_dir = os.path.join(save_dir, 'improved')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(improved_dir, exist_ok=True)
    
    # 提取原始模型的峰度信息
    print("\n====== 提取原始模型的峰度信息 ======")
    original_results = extract_kurtosis_during_inference(
        original_config, original_checkpoint, img_dir, single_image, original_dir
    )
    
    # 恢复原始方法，防止装饰器叠加
    GFocalHead.forward_single = original_forward_single
    
    # 提取改进模型的峰度信息
    print("\n====== 提取改进模型的峰度信息 ======")
    improved_results = extract_kurtosis_during_inference(
        improved_config, improved_checkpoint, img_dir, single_image, improved_dir
    )
    
    # 比较两个模型的峰度分布
    print("\n====== 比较峰度分布 ======")
    
    # 确保有处理结果
    if not original_results or not improved_results:
        print("没有有效的处理结果可以比较")
        return
    
    # 为每对图像创建比较结果
    for orig_dir in original_results:
        # 提取图像名称
        img_name = os.path.basename(orig_dir)
        improved_img_dir = os.path.join(improved_dir, img_name)
        
        if not os.path.exists(improved_img_dir):
            print(f"未找到对应的改进模型结果 {improved_img_dir}")
            continue
        
        # 创建比较目录
        compare_dir = os.path.join(save_dir, 'comparison', img_name)
        os.makedirs(compare_dir, exist_ok=True)
        
        print(f"\n比较图像 {img_name} 的结果")
        
        # 查找两个目录中的峰度数据文件
        original_files = {os.path.basename(f): f for f in glob.glob(os.path.join(orig_dir, "*.npy"))}
        improved_files = {os.path.basename(f): f for f in glob.glob(os.path.join(improved_img_dir, "*.npy"))}
        
        # 查找共同的文件名
        common_files = set(original_files.keys()) & set(improved_files.keys())
        
        if not common_files:
            print(f"图像 {img_name} 没有找到可比较的峰度数据文件")
            continue
        
        # 为每个共同文件进行比较
        for filename in common_files:
            original_data = np.load(original_files[filename])
            improved_data = np.load(improved_files[filename])
            
            basename = os.path.splitext(filename)[0]
            
            # 确保数据可比较
            if original_data.shape != improved_data.shape:
                print(f"警告: {basename} 数据形状不同，无法直接比较 ({original_data.shape} vs {improved_data.shape})")
                continue
            
            # 统计信息
            orig_flat = original_data.flatten()
            impr_flat = improved_data.flatten()
            
            # 绘制对比直方图
            plt.figure(figsize=(12, 8))
            
            # 计算共同的bin范围
            min_val = min(np.min(orig_flat), np.min(impr_flat))
            max_val = max(np.max(orig_flat), np.max(impr_flat))
            bins = np.linspace(min_val, max_val, 50)
            
            plt.hist(orig_flat, bins=bins, alpha=0.5, label='原始模型', color='blue')
            plt.hist(impr_flat, bins=bins, alpha=0.5, label='改进模型', color='green')
            
            plt.title(f'{basename} 分布对比')
            plt.xlabel('值')
            plt.ylabel('频率')
            plt.legend()
            
            # 添加统计信息
            orig_mean = np.mean(orig_flat)
            orig_std = np.std(orig_flat)
            try:
                from scipy.stats import kurtosis
                orig_kurt = kurtosis(orig_flat)
                impr_kurt = kurtosis(impr_flat)
                have_kurtosis = True
            except ImportError:
                have_kurtosis = False
                orig_kurt = "未安装scipy"
                impr_kurt = "未安装scipy"
            
            impr_mean = np.mean(impr_flat)
            impr_std = np.std(impr_flat)
            
            stats_text = (
                f"原始模型:\n"
                f"  均值: {orig_mean:.4f}\n"
                f"  标准差: {orig_std:.4f}\n"
                f"  峰度: {orig_kurt if isinstance(orig_kurt, str) else orig_kurt:.4f}\n\n"
                f"改进模型:\n"
                f"  均值: {impr_mean:.4f}\n"
                f"  标准差: {impr_std:.4f}\n"
                f"  峰度: {impr_kurt if isinstance(impr_kurt, str) else impr_kurt:.4f}\n\n"
                f"对比:\n"
                f"  均值差异: {impr_mean - orig_mean:.4f}\n"
                f"  标准差比值: {impr_std / orig_std:.4f}\n"
            )
            if have_kurtosis:
                stats_text += f"  峰度差异: {impr_kurt - orig_kurt:.4f}"
            
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(compare_dir, f'{basename}_comparison.png'))
            plt.close()
            print(f"已保存对比图: {os.path.join(compare_dir, f'{basename}_comparison.png')}")
        
        print(f"图像 {img_name} 的比较完成!")
    
    print(f"\n所有比较完成! 结果已保存到: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='提取和分析模型中的峰度信息')
    parser.add_argument('--config', help='模型配置文件路径')
    parser.add_argument('--checkpoint', help='模型检查点文件路径')
    parser.add_argument('--img-dir', help='测试图像目录')
    parser.add_argument('--single-image', help='单个图像文件路径，用于只处理一张图片')
    parser.add_argument('--save-dir', default='kurtosis_results', help='保存结果的目录')
    parser.add_argument('--compare', action='store_true', help='是否比较两个模型的峰度')
    parser.add_argument('--original-config', help='原始模型配置文件路径（用于比较）')
    parser.add_argument('--original-checkpoint', help='原始模型检查点文件路径（用于比较）')
    parser.add_argument('--improved-config', help='改进模型配置文件路径（用于比较）')
    parser.add_argument('--improved-checkpoint', help='改进模型检查点文件路径（用于比较）')
    
    args = parser.parse_args()
    
    # 确保提供了图像源
    if not args.img_dir and not args.single_image:
        print("错误: 必须提供--img-dir或--single-image参数")
        return
    
    if args.compare:
        # 比较两个模型的峰度
        compare_model_kurtosis(
            args.original_config, args.original_checkpoint,
            args.improved_config, args.improved_checkpoint,
            args.img_dir, args.single_image, args.save_dir
        )
    else:
        # 提取单个模型的峰度信息
        extract_kurtosis_during_inference(
            args.config, args.checkpoint, args.img_dir, args.single_image, args.save_dir
        )

if __name__ == '__main__':
    main() 