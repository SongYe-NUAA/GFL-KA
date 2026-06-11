import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pickle
import json
import argparse
from scipy import stats
from scipy.stats import kurtosis, norm
import mmcv

# 设置绘图样式
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300

def load_prediction_data(pred_file):
    """加载预测结果文件"""
    print(f"加载预测结果: {pred_file}")
    if pred_file.endswith('.pkl'):
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
    else:
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    
    return predictions

def extract_distribution_data(predictions, method_name='Method'):
    """从预测结果中提取概率分布数据"""
    distribution_data = []
    
    if isinstance(predictions, list) and len(predictions) > 0:
        if isinstance(predictions[0], dict):
            if 'img_id' in predictions[0]:  # MMDetection格式
                for pred in predictions:
                    if 'pred_instances' in pred:
                        instances = pred['pred_instances']
                        
                        # 提取概率分布
                        if 'distributions' in instances:
                            # 如果模型保存了分布信息
                            distributions = instances['distributions']
                            if isinstance(distributions, torch.Tensor):
                                distributions = distributions.cpu().numpy()
                            
                            for dist in distributions:
                                distribution_data.append(dist.flatten())
                        
                        # 如果没有保存分布，尝试从scores提取信息
                        elif 'scores' in instances:
                            scores = instances['scores']
                            if isinstance(scores, torch.Tensor):
                                scores = scores.cpu().numpy()
                            
                            for score in scores:
                                distribution_data.append(np.array([score]))
    
    if not distribution_data:
        print(f"警告: 无法从{method_name}的预测结果中提取分布数据")
        return None
    
    return np.concatenate(distribution_data)

def calculate_kurtosis(data):
    """计算数据的峰度值"""
    k = stats.kurtosis(data, fisher=True)  # Fisher's definition (normal = 0)
    return k

def plot_distribution_comparison(data_original, data_improved, original_name, improved_name, save_path=None):
    """绘制原始方法和改进方法的分布对比图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # 计算峰度值
    k_original = calculate_kurtosis(data_original)
    k_improved = calculate_kurtosis(data_improved)
    
    # 计算正态分布参考
    mu_original, std_original = norm.fit(data_original)
    mu_improved, std_improved = norm.fit(data_improved)
    
    # 上面的图：直方图和正态分布
    bins = np.linspace(min(data_original.min(), data_improved.min()),
                       max(data_original.max(), data_improved.max()), 50)
    
    # 原始方法直方图
    ax1.hist(data_original, bins=bins, alpha=0.5, color='blue', 
             label=f'{original_name} (峰度={k_original:.3f})', density=True)
    
    # 改进方法直方图
    ax1.hist(data_improved, bins=bins, alpha=0.5, color='green', 
             label=f'{improved_name} (峰度={k_improved:.3f})', density=True)
    
    # 添加正态分布曲线
    x = np.linspace(bins[0], bins[-1], 100)
    ax1.plot(x, norm.pdf(x, mu_original, std_original), 'b--', linewidth=2, 
             label=f'{original_name} 正态拟合')
    ax1.plot(x, norm.pdf(x, mu_improved, std_improved), 'g--', linewidth=2, 
             label=f'{improved_name} 正态拟合')
    
    ax1.set_title('分布对比直方图与正态拟合', fontsize=16, fontweight='bold')
    ax1.set_xlabel('值', fontsize=14)
    ax1.set_ylabel('密度', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 下面的图：KDE图
    sns.kdeplot(data_original, ax=ax2, color='blue', label=f'{original_name} (峰度={k_original:.3f})', 
                linewidth=2.5, fill=True, alpha=0.3)
    sns.kdeplot(data_improved, ax=ax2, color='green', label=f'{improved_name} (峰度={k_improved:.3f})', 
                linewidth=2.5, fill=True, alpha=0.3)
    
    # 添加均值线
    ax2.axvline(np.mean(data_original), color='blue', linestyle='--', 
                label=f'{original_name} 均值')
    ax2.axvline(np.mean(data_improved), color='green', linestyle='--', 
                label=f'{improved_name} 均值')
    
    ax2.set_title('核密度估计图 (KDE)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('值', fontsize=14)
    ax2.set_ylabel('密度', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加解释文本
    improvement_text = f"""分布对比分析:
- {original_name} 峰度: {k_original:.3f}
- {improved_name} 峰度: {k_improved:.3f}
- 峰度变化: {k_improved - k_original:.3f}

峰度解释:
- 峰度值越高，分布越集中在中心，尾部越重
- 正态分布峰度为0（Fisher定义）
- 峰度增加表明模型预测更加集中/确定
"""
    
    fig.text(0.15, 0.02, improvement_text, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布对比图已保存到: {save_path}")
    
    plt.close(fig)
    return k_original, k_improved

def main():
    parser = argparse.ArgumentParser(description='分析原始方法和改进方法的峰度分布变化')
    parser.add_argument('original_pred_file', help='原始方法的预测结果文件路径')
    parser.add_argument('improved_pred_file', help='改进方法的预测结果文件路径')
    parser.add_argument('--original-name', default='原始方法', help='原始方法的名称')
    parser.add_argument('--improved-name', default='改进方法', help='改进方法的名称')
    parser.add_argument('--output-dir', default='kurtosis_analysis', help='输出目录路径')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载预测结果
    print(f"\n加载 {args.original_name} 的结果...")
    original_preds = load_prediction_data(args.original_pred_file)
    
    print(f"\n加载 {args.improved_name} 的结果...")
    improved_preds = load_prediction_data(args.improved_pred_file)
    
    # 提取分布数据
    data_original = extract_distribution_data(original_preds, args.original_name)
    data_improved = extract_distribution_data(improved_preds, args.improved_name)
    
    if data_original is not None and data_improved is not None:
        # 绘制分布对比图
        save_path = os.path.join(args.output_dir, 'kurtosis_distribution_comparison.png')
        k_original, k_improved = plot_distribution_comparison(
            data_original, data_improved, 
            args.original_name, args.improved_name, 
            save_path)
        
        # 打印峰度分析结果
        print("\n峰度分析结果:")
        print(f"{args.original_name}峰度值: {k_original:.4f}")
        print(f"{args.improved_name}峰度值: {k_improved:.4f}")
        print(f"峰度变化: {k_improved - k_original:.4f}")
        
        # 生成峰度分析报告
        report_path = os.path.join(args.output_dir, 'kurtosis_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("峰度分析报告\n")
            f.write("==============\n\n")
            f.write(f"{args.original_name}峰度值: {k_original:.4f}\n")
            f.write(f"{args.improved_name}峰度值: {k_improved:.4f}\n")
            f.write(f"峰度变化: {k_improved - k_original:.4f}\n\n")
            
            if k_improved > k_original:
                f.write("分析结论: 改进方法的峰度值增加，表明数据分布更加集中在均值附近，\n")
                f.write("预测更加确定，模型对于关键区域的关注度更高。\n")
            else:
                f.write("分析结论: 改进方法的峰度值降低，表明数据分布更加分散，\n")
                f.write("预测的不确定性增加。\n")
        
        print(f"\n峰度分析报告已保存到: {report_path}")
    else:
        print("\n分析失败: 无法提取有效的分布数据")

if __name__ == '__main__':
    main() 