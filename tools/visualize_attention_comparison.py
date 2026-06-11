import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.font_manager import FontProperties
import platform

def set_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == 'Windows':
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':
        font_names = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
    else:
        font_names = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    for font_name in font_names:
        try:
            font = FontProperties(fname=font_name)
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [font_name]
            return True
        except:
            continue
    return False

def load_data(data_dir):
    """加载数据目录中的所有level数据"""
    level_dirs = [d for d in os.listdir(data_dir) if d.startswith('level_')]
    if not level_dirs:
        print(f"错误: 在 {data_dir} 中未找到任何level_X目录")
        return None
    
    data = {}
    for level_dir in level_dirs:
        level_path = os.path.join(data_dir, level_dir)
        level_num = int(level_dir.split('_')[1])
        
        # 加载加权前后的数据
        before_path = os.path.join(level_path, 'stat_before_weighting.npy')
        after_path = os.path.join(level_path, 'stat_after_weighting.npy')
        
        if os.path.exists(before_path) and os.path.exists(after_path):
            before_data = np.load(before_path)
            after_data = np.load(after_path)
            
            # 将数组展平并计算单个平均值
            before_mean = np.mean(before_data)  # 标量
            after_mean = np.mean(after_data)    # 标量
            
            data[level_num] = {
                'before': before_mean,
                'after': after_mean,
                'shape': before_data.shape,
                'before_data': before_data,  # 保存原始数据用于热度图
                'after_data': after_data     # 保存原始数据用于热度图
            }
    
    return data

def plot_comparison(data, save_dir, use_chinese=True):
    """绘制注意力前后的对比图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 为每个level创建对比图
    for level, values in data.items():
        before = values['before']  # 标量
        after = values['after']    # 标量
        shape = values['shape']    # shape元组
        before_data = values['before_data']  # 原始数据
        after_data = values['after_data']    # 原始数据
        
        # 打印调试信息
        print(f"\nDebug info for level {level}:")
        print(f"before value: {before}, type: {type(before)}")
        print(f"after value: {after}, type: {type(after)}")
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 设置标题
        if use_chinese:
            fig.suptitle(f'Level {level} 注意力机制前后对比 (特征图大小: {shape[1]}x{shape[2]})', fontsize=16)
        else:
            fig.suptitle(f'Level {level} Pre/Post Attention Comparison (Feature Map Size: {shape[1]}x{shape[2]})', fontsize=16)
        
        # 1. 柱状图对比 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        bar_width = 0.35
        x = 0  # 只有一个数据点
        
        # 绘制柱状图
        if use_chinese:
            ax1.bar(x - bar_width/2, before, width=bar_width, color='blue', alpha=0.7, label='加权前')
            ax1.bar(x + bar_width/2, after, width=bar_width, color='red', alpha=0.7, label='加权后')
        else:
            ax1.bar(x - bar_width/2, before, width=bar_width, color='blue', alpha=0.7, label='Before')
            ax1.bar(x + bar_width/2, after, width=bar_width, color='red', alpha=0.7, label='After')
        
        if use_chinese:
            ax1.set_xlabel('通道')
            ax1.set_ylabel('特征值')
            ax1.set_title('特征值对比')
        else:
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Feature Value')
            ax1.set_title('Feature Value Comparison')
        
        ax1.legend()
        ax1.set_xticks([x])
        ax1.set_xticklabels(['通道'])
        
        # 2. 变化百分比图 (左下)
        ax2 = fig.add_subplot(gs[1, 0])
        change = (after - before) / before * 100
        ax2.bar(x, change, width=bar_width, color='green')
        
        if use_chinese:
            ax2.set_xlabel('通道')
            ax2.set_ylabel('变化百分比 (%)')
            ax2.set_title('特征值变化百分比')
        else:
            ax2.set_xlabel('Channel')
            ax2.set_ylabel('Change Percentage (%)')
            ax2.set_title('Change Percentage')
        
        ax2.set_xticks([x])
        ax2.set_xticklabels(['通道'])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 处理热度图数据
        # 对多维数据取平均得到2D热度图
        if before_data.ndim > 2:
            before_heatmap = np.mean(before_data, axis=tuple(range(before_data.ndim - 2)))
        else:
            before_heatmap = before_data
            
        if after_data.ndim > 2:
            after_heatmap = np.mean(after_data, axis=tuple(range(after_data.ndim - 2)))
        else:
            after_heatmap = after_data
        
        # 计算热度图的共同范围，确保颜色映射一致
        vmin = min(np.min(before_heatmap), np.min(after_heatmap))
        vmax = max(np.max(before_heatmap), np.max(after_heatmap))
        
        # 使用相同的颜色映射
        same_cmap = 'viridis'  # 可选: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
        
        # 3. 加权前热度图 (中上)
        ax3 = fig.add_subplot(gs[0, 1])
        im3 = ax3.imshow(before_heatmap, cmap=same_cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im3, ax=ax3)
        if use_chinese:
            ax3.set_title('加权前特征热度图')
        else:
            ax3.set_title('Feature Heatmap Before Weighting')
        
        # 4. 加权后热度图 (中下)
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(after_heatmap, cmap=same_cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im4, ax=ax4)
        if use_chinese:
            ax4.set_title('加权后特征热度图')
        else:
            ax4.set_title('Feature Heatmap After Weighting')
        
        # 5. 热度图差异 (右上)
        ax5 = fig.add_subplot(gs[0, 2])
        # 计算加权前后的热度图差异
        diff_heatmap = after_heatmap - before_heatmap
        # 为差异图设置对称的颜色范围
        diff_abs_max = max(abs(np.min(diff_heatmap)), abs(np.max(diff_heatmap)))
        im5 = ax5.imshow(diff_heatmap, cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.colorbar(im5, ax=ax5)
        if use_chinese:
            ax5.set_title('加权前后热度图差异')
        else:
            ax5.set_title('Heatmap Difference (After-Before)')
        
        # 6. 热度图变化百分比 (右下)
        ax6 = fig.add_subplot(gs[1, 2])
        # 计算热度图变化百分比，避免除零错误
        eps = 1e-10  # 小常数防止除零
        percent_change = (after_heatmap - before_heatmap) / (np.abs(before_heatmap) + eps) * 100
        # 限制百分比范围，使其更加可读
        percent_clip = 100  # 限制在 ±100% 范围内
        im6 = ax6.imshow(np.clip(percent_change, -percent_clip, percent_clip), cmap='bwr', vmin=-percent_clip, vmax=percent_clip)
        plt.colorbar(im6, ax=ax6)
        if use_chinese:
            ax6.set_title('热度图变化百分比 (%)')
        else:
            ax6.set_title('Heatmap Percentage Change (%)')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'level_{level}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存对比图: {save_path}")
        
        # 计算并打印统计信息
        print(f"\nLevel {level} 统计信息:")
        print(f"特征图大小: {shape[1]}x{shape[2]}")
        print("\n变化统计:")
        print(f"加权前: {before:.4f}")
        print(f"加权后: {after:.4f}")
        print(f"变化百分比: {change:.2f}%")
        print(f"\n热度图统计:")
        print(f"热度图加权前均值: {np.mean(before_heatmap):.4f}")
        print(f"热度图加权后均值: {np.mean(after_heatmap):.4f}")
        print(f"热度图差异均值: {np.mean(diff_heatmap):.4f}")
        print(f"热度图变化百分比均值: {np.mean(percent_change):.2f}%")

def main():
    parser = argparse.ArgumentParser(description='可视化注意力机制前后的数值变化')
    parser.add_argument('--data-dir', required=True, help='包含提取数据的目录')
    parser.add_argument('--save-dir', default='attention_comparison', help='保存可视化结果的目录')
    
    args = parser.parse_args()
    
    # 设置中文字体
    use_chinese = set_chinese_font()
    
    # 加载数据
    data = load_data(args.data_dir)
    if data is None:
        return
    
    # 绘制对比图
    plot_comparison(data, args.save_dir, use_chinese)
    
    print(f"\n可视化完成! 结果已保存到: {args.save_dir}")

if __name__ == '__main__':
    main() 