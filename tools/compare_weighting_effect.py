import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import torch
from scipy.stats import kurtosis, skew
import seaborn as sns

# 设置matplotlib字体
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def load_data(data_dir):
    """加载加权前后的数据"""
    before_path = os.path.join(data_dir, 'stat_before_weighting.npy')
    after_path = os.path.join(data_dir, 'stat_after_weighting.npy')
    attention_path = os.path.join(data_dir, 'attention_weights.npy')
    
    # 检查文件是否存在
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        raise FileNotFoundError(f"未找到必要的数据文件，请确保已运行提取脚本")
    
    # 加载数据
    before = np.load(before_path)
    after = np.load(after_path)
    attention = np.load(attention_path) if os.path.exists(attention_path) else None
    
    return before, after, attention

def visualize_feature_maps(before, after, attention, save_dir, layer_idx=0, feature_idx=0):
    """可视化特征图对比"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取一个示例特征图
    if before.ndim == 5:  # [N, C, K, H, W]
        # 选择第一个样本，特定通道和特定特征
        before_map = before[0, layer_idx, feature_idx]
        after_map = after[0, layer_idx, feature_idx]
    else:
        raise ValueError(f"不支持的数据维度: {before.shape}")
    
    # 创建对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制加权前
    im1 = axes[0].imshow(before_map, cmap='viridis')
    axes[0].set_title('加权前特征图', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 绘制加权后
    im2 = axes[1].imshow(after_map, cmap='viridis')
    axes[1].set_title('加权后特征图', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 绘制差异（相对增强）
    if np.any(before_map):  # 避免除以零
        rel_enhance = (after_map - before_map) / (np.abs(before_map) + 1e-6)
        vmax = min(np.percentile(rel_enhance, 95), 3)  # 限制色彩范围
        vmin = max(np.percentile(rel_enhance, 5), -3)
        im3 = axes[2].imshow(rel_enhance, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[2].set_title('相对增强图 (加权后-加权前)/加权前', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_map_c{layer_idx}_f{feature_idx}.png'), dpi=300)
    plt.close(fig)
    
    # 如果有注意力权重，也显示它
    if attention is not None:
        plt.figure(figsize=(8, 6))
        if attention.ndim >= 3:  # 多维注意力权重
            att_map = attention[0]  # 第一个样本
            plt.imshow(att_map, cmap='hot')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('注意力权重图', fontsize=14)
        else:
            plt.hist(attention.flatten(), bins=50, alpha=0.7)
            plt.title('注意力权重分布', fontsize=14)
            plt.xlabel('权重值')
            plt.ylabel('频率')
        plt.savefig(os.path.join(save_dir, 'attention_weights.png'), dpi=300)
        plt.close()

def calculate_statistics(before, after):
    """计算统计信息，显示加权前后的差异"""
    # 平坦化数据
    before_flat = before.flatten()
    after_flat = after.flatten()
    
    # 计算统计量
    stats = {
        '加权前均值': np.mean(before_flat),
        '加权后均值': np.mean(after_flat),
        '均值相对变化': (np.mean(after_flat) - np.mean(before_flat)) / np.abs(np.mean(before_flat) + 1e-8),
        '加权前标准差': np.std(before_flat),
        '加权后标准差': np.std(after_flat),
        '标准差相对变化': (np.std(after_flat) - np.std(before_flat)) / np.std(before_flat),
        '加权前峰度': kurtosis(before_flat),
        '加权后峰度': kurtosis(after_flat),
        '峰度差异': kurtosis(after_flat) - kurtosis(before_flat),
        '加权前偏度': skew(before_flat),
        '加权后偏度': skew(after_flat),
        '偏度差异': skew(after_flat) - skew(before_flat),
        '最大相对增强': np.max((after_flat - before_flat) / (np.abs(before_flat) + 1e-8)),
        '最大相对抑制': np.min((after_flat - before_flat) / (np.abs(before_flat) + 1e-8)),
    }
    
    return stats

def compare_distributions(before, after, attention, save_dir):
    """比较加权前后的分布"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 平坦化数据
    before_flat = before.flatten()
    after_flat = after.flatten()
    attention_flat = attention.flatten() if attention is not None else None
    
    # 创建分布对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 直方图对比
    ax1 = axes[0, 0]
    sns.histplot(before_flat, ax=ax1, label='加权前', color='blue', alpha=0.5, stat='density', kde=True)
    sns.histplot(after_flat, ax=ax1, label='加权后', color='red', alpha=0.5, stat='density', kde=True)
    ax1.set_title('特征值分布对比', fontsize=14)
    ax1.set_xlabel('特征值', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.legend()
    
    # 2. Q-Q图
    ax2 = axes[0, 1]
    quantiles = np.linspace(0, 1, 100)
    before_quantiles = np.quantile(before_flat, quantiles)
    after_quantiles = np.quantile(after_flat, quantiles)
    ax2.scatter(before_quantiles, after_quantiles, alpha=0.7)
    
    # 添加对角线
    min_val = min(np.min(before_quantiles), np.min(after_quantiles))
    max_val = max(np.max(before_quantiles), np.max(after_quantiles))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax2.set_title('Q-Q图：加权前 vs 加权后', fontsize=14)
    ax2.set_xlabel('加权前分位数', fontsize=12)
    ax2.set_ylabel('加权后分位数', fontsize=12)
    
    # 3. 特征相对变化
    ax3 = axes[1, 0]
    relative_change = (after_flat - before_flat) / (np.abs(before_flat) + 1e-8)
    vmax = min(np.percentile(relative_change, 99), 5)
    vmin = max(np.percentile(relative_change, 1), -5)
    relative_change = np.clip(relative_change, vmin, vmax)  # 去除极端值
    
    sns.histplot(relative_change, ax=ax3, color='green', alpha=0.7, bins=50)
    ax3.set_title('特征相对变化 (加权后-加权前)/加权前', fontsize=14)
    ax3.set_xlabel('相对变化', fontsize=12)
    ax3.set_ylabel('频率', fontsize=12)
    
    # 4. 如果有注意力权重，显示注意力分布
    ax4 = axes[1, 1]
    if attention_flat is not None:
        sns.histplot(attention_flat, ax=ax4, color='purple', alpha=0.7, bins=50)
        ax4.set_title('注意力权重分布', fontsize=14)
        ax4.set_xlabel('注意力权重', fontsize=12)
        ax4.set_ylabel('频率', fontsize=12)
    else:
        # 如果没有注意力权重，显示峰度变化情况
        bins = np.linspace(-0.5, 0.5, 50)
        before_kurt = np.zeros_like(before_flat)
        after_kurt = np.zeros_like(after_flat)
        
        if before.ndim == 5:  # [N, C, K, H, W]
            # 计算每个像素位置的峰度变化
            N, C, K, H, W = before.shape
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        before_kurt_val = kurtosis(before[0, c, :, h, w])
                        after_kurt_val = kurtosis(after[0, c, :, h, w])
                        idx = c * H * W + h * W + w
                        if idx < len(before_kurt):
                            before_kurt[idx] = before_kurt_val
                            after_kurt[idx] = after_kurt_val
        
        kurt_diff = after_kurt - before_kurt
        sns.histplot(kurt_diff, ax=ax4, color='orange', alpha=0.7, bins=bins)
        ax4.set_title('峰度变化分布', fontsize=14)
        ax4.set_xlabel('峰度变化 (加权后-加权前)', fontsize=12)
        ax4.set_ylabel('频率', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'), dpi=300)
    plt.close(fig)
    
    # 计算统计量
    stats = calculate_statistics(before, after)
    
    # 生成报告
    report_path = os.path.join(save_dir, 'weighting_effect_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("峰度加权效果分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("统计信息对比:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n峰度加权的影响分析:\n")
        if stats['峰度差异'] > 0:
            f.write("- 加权后峰度增加，说明特征分布更加集中\n")
            f.write("- 这表明模型更关注于显著特征，减少对不重要特征的响应\n")
        else:
            f.write("- 加权后峰度减少，说明特征分布更加分散\n")
            f.write("- 这可能表明模型在更广泛的特征上分配注意力\n")
        
        if stats['均值相对变化'] > 0:
            f.write("- 加权提高了特征的整体强度\n")
        else:
            f.write("- 加权降低了特征的整体强度\n")
        
        if stats['标准差相对变化'] > 0:
            f.write("- 加权增加了特征的对比度/变异性\n")
        else:
            f.write("- 加权减少了特征的对比度/变异性\n")
    
    print(f"分析报告已保存到: {report_path}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='比较峰度加权前后的概率分布差异')
    parser.add_argument('--data-dir', required=True, help='包含提取数据的目录')
    parser.add_argument('--output-dir', default='weighting_effect_analysis', help='输出目录')
    parser.add_argument('--layer', type=int, default=0, help='要可视化的层索引')
    parser.add_argument('--feature', type=int, default=0, help='要可视化的特征索引')
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        before, after, attention = load_data(args.data_dir)
        print(f"成功加载数据:")
        print(f"  加权前形状: {before.shape}")
        print(f"  加权后形状: {after.shape}")
        if attention is not None:
            print(f"  注意力权重形状: {attention.shape}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # 可视化特征图
    try:
        visualize_feature_maps(before, after, attention, args.output_dir, 
                              layer_idx=args.layer, feature_idx=args.feature)
        print(f"特征图可视化已保存到: {args.output_dir}")
    except Exception as e:
        print(f"可视化特征图时出错: {e}")
    
    # 比较分布
    try:
        stats = compare_distributions(before, after, attention, args.output_dir)
        print("统计信息摘要:")
        print(f"  峰度变化: {stats['峰度差异']:.4f}")
        print(f"  均值相对变化: {stats['均值相对变化']:.4f}")
        print(f"  标准差相对变化: {stats['标准差相对变化']:.4f}")
    except Exception as e:
        print(f"比较分布时出错: {e}")

if __name__ == '__main__':
    main() 