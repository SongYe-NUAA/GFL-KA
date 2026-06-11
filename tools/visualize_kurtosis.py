import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pickle
import argparse
from scipy import stats
from scipy.stats import norm, kurtosis, skew

# 设置环境变量，避免OpenMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置matplotlib字体，支持中文显示
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置绘图样式和字体
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial', 'DejaVu Sans']  # 优先使用的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 修复中文环境下的负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 300

# 尝试从系统字体列表中查找可用的中文字体
try:
    from matplotlib.font_manager import findfont, FontProperties
    font_path = findfont(FontProperties(family=['SimHei', 'Microsoft YaHei', 'SimSun']))
    print(f"找到字体：{font_path}")
except:
    print("无法找到合适的中文字体，将使用默认字体")

def load_kurtosis_data(data_file):
    """加载峰度数据文件"""
    print(f"加载峰度数据: {data_file}")
    
    # 检查文件扩展名
    if data_file.endswith('.npy'):
        # 使用numpy加载.npy文件
        try:
            data = np.load(data_file)
            print(f"成功加载NumPy数据, 形状: {data.shape}")
        except Exception as e:
            print(f"加载NumPy文件时出错: {e}")
            raise
    else:
        # 尝试使用pickle加载
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print(f"成功加载Pickle数据")
        except Exception as e:
            print(f"加载Pickle文件时出错: {e}")
            # 尝试作为NumPy文件加载
            try:
                data = np.load(data_file)
                print(f"成功作为NumPy数据加载")
            except:
                print(f"无法以任何格式加载文件")
                raise
    
    # 处理可能的数据格式
    if isinstance(data, list):
        # 尝试将列表中的所有数组连接成一个大数组
        try:
            data = np.concatenate([d.flatten() for d in data if hasattr(d, 'flatten')])
        except:
            # 如果无法连接，则使用第一个元素
            print("警告: 无法连接数据，使用第一个元素")
            data = data[0].flatten() if hasattr(data[0], 'flatten') else np.array(data[0])
    elif isinstance(data, np.ndarray):
        data = data.flatten()
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy().flatten()
    else:
        raise ValueError(f"不支持的数据类型: {type(data)}")
        
    print(f"处理后的数据: 大小={len(data)}, 类型={type(data)}")
    print(f"数据统计: 均值={np.mean(data):.4f}, 标准差={np.std(data):.4f}, 峰度={kurtosis(data):.4f}")
    
    return data

def plot_kurtosis_analysis(data, method_name, save_dir, prefix=''):
    """绘制峰度分析图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算统计量
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_kurt = kurtosis(data)
    data_skew = skew(data)
    
    # 创建四张子图
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 直方图与正态分布拟合
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(np.min(data), np.max(data), 50)
    ax1.hist(data, bins=bins, alpha=0.7, color='steelblue', 
             label=f'峰度={data_kurt:.3f}', density=True)
    
    # 添加正态分布拟合曲线
    x = np.linspace(np.min(data), np.max(data), 1000)
    ax1.plot(x, norm.pdf(x, data_mean, data_std), 'r-', linewidth=2, 
             label='正态分布拟合')
    
    ax1.set_title(f'{method_name} - 直方图与正态分布拟合', fontsize=14, fontweight='bold')
    ax1.set_xlabel('值', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. KDE图
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data, ax=ax2, fill=True, color='forestgreen', alpha=0.5, 
                linewidth=2, label=f'KDE曲线 (峰度={data_kurt:.3f})')
    
    # 添加均值线
    ax2.axvline(data_mean, color='red', linestyle='--', 
                label=f'均值 = {data_mean:.3f}')
    
    ax2.set_title(f'{method_name} - 核密度估计图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('值', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 分位数-分位数图
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(data, dist="norm", plot=ax3)
    ax3.set_title(f'{method_name} - Q-Q图 (正态性检验)', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 累积分布函数
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_data = np.sort(data)
    ax4.plot(sorted_data, np.arange(1, len(sorted_data) + 1) / len(sorted_data), 
             label='经验CDF', linewidth=2, color='purple')
    
    # 添加正态分布CDF
    ax4.plot(sorted_data, norm.cdf(sorted_data, data_mean, data_std), 
             'r--', label='正态CDF', linewidth=2)
    
    ax4.set_title(f'{method_name} - 累积分布函数', fontsize=14, fontweight='bold')
    ax4.set_xlabel('值', fontsize=12)
    ax4.set_ylabel('累积概率', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 箱线图
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.boxplot(data, vert=False, widths=0.7, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    
    ax5.set_title(f'{method_name} - 箱线图', fontsize=14, fontweight='bold')
    ax5.set_xlabel('值', fontsize=12)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.set_yticks([])
    
    # 6. 统计信息
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')  # 隐藏坐标轴
    
    # 使用普通减号而不是特殊字符，避免编码问题
    stats_text = f"""
    {method_name} 统计信息:
    
    - 样本数: {len(data)}
    - 均值: {data_mean:.4f}
    - 标准差: {data_std:.4f}
    - 最小值: {np.min(data):.4f}
    - 25%分位数: {np.percentile(data, 25):.4f}
    - 中位数: {np.median(data):.4f}
    - 75%分位数: {np.percentile(data, 75):.4f}
    - 最大值: {np.max(data):.4f}
    
    形态统计:
    - 峰度 (正态分布=0): {data_kurt:.4f}
    - 偏度: {data_skew:.4f}
    
    峰度分析:
    {"- 分布呈尖峰分布，比正态分布更集中" if data_kurt > 0 else "- 分布呈平峰分布，比正态分布更分散"}
    {"- 数据更集中在均值附近，尾部更重" if data_kurt > 0 else "- 数据更均匀分布，尾部更轻"}
    
    偏度分析:
    {"- 分布右偏，有较长的右尾" if data_skew > 0 else "- 分布左偏，有较长的左尾" if data_skew < 0 else "- 分布对称"}
    """
    
    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=1', facecolor='#F0F0F0', alpha=0.8))
    
    # 保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 修复tight_layout警告
    plt.savefig(os.path.join(save_dir, f'{prefix}{method_name}_kurtosis_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 生成统计报告，使用UTF-8编码
    report_path = os.path.join(save_dir, f'{prefix}{method_name}_kurtosis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{method_name} 峰度分析报告\n")
        f.write("="*40 + "\n\n")
        f.write(stats_text)
    
    print(f"分析结果已保存到: {save_dir}")
    
    return {
        'mean': data_mean,
        'std': data_std,
        'kurtosis': data_kurt,
        'skewness': data_skew,
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75)
    }

def compare_kurtosis_distributions(data1, data2, name1, name2, save_dir):
    """比较两个分布的峰度"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建四个子图的对比图
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    
    # 1. 直方图对比
    ax1 = axs[0, 0]
    bins = np.linspace(min(np.min(data1), np.min(data2)), 
                      max(np.max(data1), np.max(data2)), 50)
    
    ax1.hist(data1, bins=bins, alpha=0.5, label=f'{name1} (峰度={kurtosis(data1):.3f})', 
             color='blue', density=True)
    ax1.hist(data2, bins=bins, alpha=0.5, label=f'{name2} (峰度={kurtosis(data2):.3f})', 
             color='green', density=True)
    
    ax1.set_title('直方图对比', fontsize=16, fontweight='bold')
    ax1.set_xlabel('值', fontsize=14)
    ax1.set_ylabel('密度', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. KDE图对比
    ax2 = axs[0, 1]
    sns.kdeplot(data1, ax=ax2, label=f'{name1}', color='blue', linewidth=2.5, fill=True, alpha=0.3)
    sns.kdeplot(data2, ax=ax2, label=f'{name2}', color='green', linewidth=2.5, fill=True, alpha=0.3)
    
    # 添加均值线
    ax2.axvline(np.mean(data1), color='blue', linestyle='--', label=f'{name1} 均值')
    ax2.axvline(np.mean(data2), color='green', linestyle='--', label=f'{name2} 均值')
    
    ax2.set_title('核密度估计图对比', fontsize=16, fontweight='bold')
    ax2.set_xlabel('值', fontsize=14)
    ax2.set_ylabel('密度', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 累积分布函数对比
    ax3 = axs[1, 0]
    sorted_data1 = np.sort(data1)
    sorted_data2 = np.sort(data2)
    
    ax3.plot(sorted_data1, np.arange(1, len(sorted_data1) + 1) / len(sorted_data1), 
            label=f'{name1}', linewidth=2, color='blue')
    ax3.plot(sorted_data2, np.arange(1, len(sorted_data2) + 1) / len(sorted_data2), 
            label=f'{name2}', linewidth=2, color='green')
    
    ax3.set_title('累积分布函数对比', fontsize=16, fontweight='bold')
    ax3.set_xlabel('值', fontsize=14)
    ax3.set_ylabel('累积概率', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 箱线图对比
    ax4 = axs[1, 1]
    box_data = [data1, data2]
    box_colors = ['lightblue', 'lightgreen']
    box_labels = [name1, name2]
    
    bplot = ax4.boxplot(box_data, vert=True, patch_artist=True, 
                      labels=box_labels, notch=True, widths=0.4)
    
    # 设置箱线图颜色
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
    
    ax4.set_title('箱线图对比', fontsize=16, fontweight='bold')
    ax4.set_ylabel('值', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计对比信息
    kurt1 = kurtosis(data1)
    kurt2 = kurtosis(data2)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    std1 = np.std(data1)
    std2 = np.std(data2)
    skew1 = skew(data1)
    skew2 = skew(data2)
    
    comparison_text = f"""统计对比:
    
{name1} vs {name2}
- 峰度: {kurt1:.4f} vs {kurt2:.4f} (差异: {kurt2 - kurt1:.4f})
- 均值: {mean1:.4f} vs {mean2:.4f} (差异: {mean2 - mean1:.4f})
- 标准差: {std1:.4f} vs {std2:.4f} (比值: {std2/std1:.4f})
- 偏度: {skew1:.4f} vs {skew2:.4f} (差异: {skew2 - skew1:.4f})

分析结论:
- {'改进方法的峰度值增加' if kurt2 > kurt1 else '改进方法的峰度值减少'} 
- {'分布更加集中在均值附近' if kurt2 > kurt1 else '分布更加分散'} 
- {'预测更加确定，模型对关键区域关注度更高' if kurt2 > kurt1 else '预测不确定性增加'}
"""
    
    # 添加文本注释到整个图表
    fig.text(0.5, 0.01, comparison_text, ha='center', va='bottom', fontsize=14,
            bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为底部文本留出空间
    
    # 保存对比图
    save_path = os.path.join(save_dir, f'{name1}_vs_{name2}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 生成对比报告，使用UTF-8编码
    report_path = os.path.join(save_dir, f'{name1}_vs_{name2}_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{name1} vs {name2} 峰度对比分析\n")
        f.write("="*40 + "\n\n")
        f.write(comparison_text)
        
        # 添加更详细的解释
        f.write("\n\n峰度解释:\n")
        f.write("峰度是描述概率分布尖峰程度的统计量，衡量概率分布的峰态和尾部权重。\n")
        f.write("- 峰度=0: 与正态分布的峰度相同（标准正态分布）\n")
        f.write("- 峰度>0: 分布比正态分布更尖锐（尖峰分布），尾部更重\n")
        f.write("- 峰度<0: 分布比正态分布更平坦（平峰分布），尾部更轻\n\n")
        
        f.write("在检测任务中的意义:\n")
        if kurt2 > kurt1:
            f.write("峰度增加表明模型对目标的置信度分布更加集中，减少了不确定性。\n")
            f.write("这通常意味着模型在预测时更加确定和一致，对目标区域有更强的关注。\n")
            f.write("在目标检测中，这可能导致更精确的定位和更高的召回率。\n")
        else:
            f.write("峰度降低表明模型的预测分布更加分散，不确定性增加。\n")
            f.write("这可能是由于模型考虑了更多的边缘情况或对不同特征赋予了更均衡的权重。\n")
            f.write("在某些情况下，这可能有助于减少过拟合，但也可能降低模型的确定性。\n")
    
    print(f"对比分析已保存到: {save_dir}")
    
    return {
        'kurtosis_diff': kurt2 - kurt1,
        'mean_diff': mean2 - mean1,
        'std_ratio': std2/std1,
        'skewness_diff': skew2 - skew1
    }

def main():
    parser = argparse.ArgumentParser(description='可视化和分析峰度数据')
    parser.add_argument('--data', help='单个数据文件路径')
    parser.add_argument('--name', default='Method', help='方法名称')
    parser.add_argument('--compare', action='store_true', help='比较两个分布')
    parser.add_argument('--data1', help='第一个数据文件路径（用于比较）')
    parser.add_argument('--data2', help='第二个数据文件路径（用于比较）')
    parser.add_argument('--name1', default='原始方法', help='第一个方法名称')
    parser.add_argument('--name2', default='改进方法', help='第二个方法名称')
    parser.add_argument('--output-dir', default='kurtosis_visualization', help='输出目录')
    parser.add_argument('--use-ascii', action='store_true', help='使用ASCII字符，避免中文编码问题')
    
    args = parser.parse_args()

    # 如果指定使用ASCII，替换中文名称
    if args.use_ascii:
        if args.name == '方法':
            args.name = 'Method'
        if args.name1 == '原始方法':
            args.name1 = 'Original'
        if args.name2 == '改进方法':
            args.name2 = 'Improved'
    
    if args.compare:
        # 比较两个分布
        if not (args.data1 and args.data2):
            print("错误: 进行比较时需要提供两个数据文件路径")
            return
        
        data1 = load_kurtosis_data(args.data1)
        data2 = load_kurtosis_data(args.data2)
        
        # 分别分析两个分布
        stats1 = plot_kurtosis_analysis(data1, args.name1, args.output_dir, prefix='1_')
        stats2 = plot_kurtosis_analysis(data2, args.name2, args.output_dir, prefix='2_')
        
        # 比较两个分布
        compare_kurtosis_distributions(data1, data2, args.name1, args.name2, args.output_dir)
    else:
        # 分析单个分布
        if not args.data:
            print("错误: 请提供数据文件路径")
            return
        
        data = load_kurtosis_data(args.data)
        plot_kurtosis_analysis(data, args.name, args.output_dir)

if __name__ == '__main__':
    main() 