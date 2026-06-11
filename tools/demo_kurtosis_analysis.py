import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from scipy.stats import norm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许使用多个OpenMP库

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Arial']  # 添加中文字体选项
plt.rcParams['axes.unicode_minus'] = False  # 修复中文环境下的负号显示问题

def generate_demo_data(output_dir='demo_data', seed=42):
    """
    生成用于演示的峰度数据
    
    Args:
        output_dir: 输出目录
        seed: 随机种子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 生成原始方法的数据：较低峰度，更接近正态分布
    # 使用正态分布
    data_original = np.random.normal(loc=0.7, scale=0.15, size=10000)
    
    # 生成改进方法的数据：较高峰度，更集中
    # 使用t分布，自由度较小时峰度较大
    data_improved = 0.7 + 0.1 * np.random.standard_t(df=5, size=10000)
    
    # 保存数据
    with open(os.path.join(output_dir, 'original_data.pkl'), 'wb') as f:
        pickle.dump(data_original, f)
    
    with open(os.path.join(output_dir, 'improved_data.pkl'), 'wb') as f:
        pickle.dump(data_improved, f)
    
    # 生成预览图
    plt.figure(figsize=(10, 6))
    plt.hist(data_original, bins=50, alpha=0.5, label='原始方法数据', color='blue', density=True)
    plt.hist(data_improved, bins=50, alpha=0.5, label='改进方法数据', color='green', density=True)
    plt.title('峰度数据预览')
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, 'data_preview.png'), dpi=300)
    plt.close()
    
    print(f"示例数据已生成并保存到: {output_dir}")
    
    return {
        'original_path': os.path.join(output_dir, 'original_data.pkl'),
        'improved_path': os.path.join(output_dir, 'improved_data.pkl')
    }

def generate_complex_data(output_dir='demo_data', seed=42):
    """
    生成更复杂的数据集，模拟实际检测场景中的峰度数据
    
    Args:
        output_dir: 输出目录
        seed: 随机种子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 模拟原始方法：混合分布，峰度较低
    # 模拟大量的背景噪声和少量的高置信度目标
    size = 10000
    background = np.random.normal(loc=0.3, scale=0.1, size=int(size*0.7))  # 背景，置信度较低
    foreground = np.random.normal(loc=0.7, scale=0.15, size=int(size*0.3))  # 前景目标，置信度中等
    
    # 组合
    data_original = np.concatenate([background, foreground])
    np.random.shuffle(data_original)
    
    # 模拟改进方法：更集中的分布，峰度更高
    # 减少中间状态，加强两极分化
    background_improved = np.random.normal(loc=0.2, scale=0.08, size=int(size*0.7))  # 背景，置信度更低
    foreground_improved = np.random.beta(6, 2, size=int(size*0.3)) * 0.3 + 0.7  # 前景目标，置信度更高，更集中
    
    # 组合
    data_improved = np.concatenate([background_improved, foreground_improved])
    np.random.shuffle(data_improved)
    
    # 保存数据
    with open(os.path.join(output_dir, 'original_complex.pkl'), 'wb') as f:
        pickle.dump(data_original, f)
    
    with open(os.path.join(output_dir, 'improved_complex.pkl'), 'wb') as f:
        pickle.dump(data_improved, f)
    
    # 生成预览图
    plt.figure(figsize=(12, 10))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 直方图对比
    bins = np.linspace(0, 1, 50)
    ax1.hist(data_original, bins=bins, alpha=0.5, label='原始方法', color='blue', density=True)
    ax1.hist(data_improved, bins=bins, alpha=0.5, label='改进方法', color='green', density=True)
    ax1.set_title('峰度数据预览 - 直方图对比')
    ax1.set_xlabel('置信度')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # KDE对比
    from scipy.stats import gaussian_kde
    
    x = np.linspace(0, 1, 1000)
    kde_original = gaussian_kde(data_original)
    kde_improved = gaussian_kde(data_improved)
    
    ax2.plot(x, kde_original(x), label='原始方法', color='blue', linewidth=2)
    ax2.plot(x, kde_improved(x), label='改进方法', color='green', linewidth=2)
    ax2.set_title('峰度数据预览 - KDE曲线对比')
    ax2.set_xlabel('置信度')
    ax2.set_ylabel('密度')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complex_data_preview.png'), dpi=300)
    plt.close()
    
    print(f"复杂示例数据已生成并保存到: {output_dir}")
    
    return {
        'original_path': os.path.join(output_dir, 'original_complex.pkl'),
        'improved_path': os.path.join(output_dir, 'improved_complex.pkl')
    }

def run_demo(use_ascii=False):
    """运行完整的演示流程"""
    # 1. 生成示例数据
    print("正在生成简单示例数据...")
    data_paths = generate_demo_data()
    
    # 2. 生成复杂示例数据
    print("\n正在生成复杂示例数据...")
    complex_data_paths = generate_complex_data()
    
    # 使用ASCII参数防止中文编码问题
    ascii_param = "--use-ascii" if use_ascii else ""
    
    # 3. 运行分析工具
    print("\n分析简单示例数据...")
    name1 = "Simple_Original" if use_ascii else "'简单原始方法'"
    name2 = "Simple_Improved" if use_ascii else "'简单改进方法'"
    
    os.system(f"python tools/visualize_kurtosis.py --compare --data1 {data_paths['original_path']} "
              f"--data2 {data_paths['improved_path']} --name1 {name1} --name2 {name2} "
              f"--output-dir demo_results/simple {ascii_param}")
    
    print("\n分析复杂示例数据...")
    name1 = "Complex_Original" if use_ascii else "'复杂原始方法'"
    name2 = "Complex_Improved" if use_ascii else "'复杂改进方法'"
    
    os.system(f"python tools/visualize_kurtosis.py --compare --data1 {complex_data_paths['original_path']} "
              f"--data2 {complex_data_paths['improved_path']} --name1 {name1} --name2 {name2} "
              f"--output-dir demo_results/complex {ascii_param}")
    
    print("\n演示完成！请查看 demo_results 目录下的结果。")
    print("\n使用方法总结:")
    print("1. 首先使用 tools/extract_kurtosis_data.py 从模型中提取峰度数据")
    print("2. 然后使用 tools/visualize_kurtosis.py 可视化和分析峰度数据")
    print("3. 对于更详细的网络分析，可以使用 tools/modify_gfocal_head.py 修改GFocalHead并提取更多信息")
    print("\n工具命令示例:")
    print("- 提取峰度数据: python tools/extract_kurtosis_data.py extract --config <配置文件> --checkpoint <检查点文件> --img-dir <图像目录> --save-path <保存路径>")
    print("- 分析单个数据: python tools/visualize_kurtosis.py --data <数据文件> --name <方法名称> --output-dir <输出目录>")
    print("- 比较两个分布: python tools/visualize_kurtosis.py --compare --data1 <数据文件1> --data2 <数据文件2> --name1 <方法1名称> --name2 <方法2名称> --output-dir <输出目录>")
    
def main():
    parser = argparse.ArgumentParser(description='峰度分析工具演示')
    parser.add_argument('--simple-only', action='store_true', help='只生成简单示例数据')
    parser.add_argument('--complex-only', action='store_true', help='只生成复杂示例数据')
    parser.add_argument('--no-analysis', action='store_true', help='不运行分析工具')
    parser.add_argument('--output-dir', default='demo_data', help='输出目录')
    parser.add_argument('--use-ascii', action='store_true', help='使用ASCII字符，避免中文编码问题')
    
    args = parser.parse_args()
    
    if args.simple_only:
        # 只生成简单示例数据
        generate_demo_data(args.output_dir)
    elif args.complex_only:
        # 只生成复杂示例数据
        generate_complex_data(args.output_dir)
    elif args.no_analysis:
        # 生成两种数据但不运行分析
        generate_demo_data(args.output_dir)
        generate_complex_data(args.output_dir)
    else:
        # 运行完整演示
        run_demo(args.use_ascii)

if __name__ == '__main__':
    main() 