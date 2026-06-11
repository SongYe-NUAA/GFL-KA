import json
import numpy as np
from pycocotools.coco import COCO
import pickle
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import gaussian_kde

# 设置OpenMP环境变量以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置Matplotlib后端，避免潜在的GUI问题
import matplotlib
matplotlib.use('Agg')

def load_predictions(pred_file):
    """加载预测结果文件"""
    print(f"加载预测结果: {pred_file}")
    if pred_file.endswith('.pkl'):
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
    else:
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    return predictions

def bbox_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 确保box格式统一：转换为[x1, y1, x2, y2]格式
    if len(box1) == 4:
        if box1[2] < box1[0] or box1[3] < box1[1]:  # 说明是[x, y, w, h]格式
            box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    if len(box2) == 4:
        if box2[2] < box2[0] or box2[3] < box2[1]:  # 说明是[x, y, w, h]格式
            box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

def plot_comparison(scores1, ious1, scores2, ious2, method1_name, method2_name, save_path=None):
    """绘制两种方法的对比散点图"""
    try:
        plt.style.use('seaborn')
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 第一个方法的散点图（左）
        scatter1 = ax1.scatter(scores1, ious1, 
                             c='blue', 
                             s=20, 
                             alpha=0.5, 
                             label=method1_name)
        ax1.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        ax1.set_title(f'{method1_name}\nScore vs IoU Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Prediction Score', fontsize=10)
        ax1.set_ylabel('IoU', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left', fontsize=10)
        
        # 计算第一个方法的统计信息
        corr1 = np.corrcoef(scores1, ious1)[0, 1]
        mean_iou1 = np.mean(ious1)
        median_iou1 = np.median(ious1)
        stats1 = (f'Correlation: {corr1:.3f}\n'
                 f'Mean IoU: {mean_iou1:.3f}\n'
                 f'Median IoU: {median_iou1:.3f}')
        ax1.text(0.05, 0.25, stats1,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10,
                verticalalignment='top')
        
        # 第二个方法的散点图（右）
        scatter2 = ax2.scatter(scores2, ious2, 
                             c='green', 
                             s=20, 
                             alpha=0.5, 
                             label=method2_name)
        ax2.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        ax2.set_title(f'{method2_name}\nScore vs IoU Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Score', fontsize=10)
        ax2.set_ylabel('IoU', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left', fontsize=10)
        
        # 计算第二个方法的统计信息
        corr2 = np.corrcoef(scores2, ious2)[0, 1]
        mean_iou2 = np.mean(ious2)
        median_iou2 = np.median(ious2)
        stats2 = (f'Correlation: {corr2:.3f}\n'
                 f'Mean IoU: {mean_iou2:.3f}\n'
                 f'Median IoU: {median_iou2:.3f}')
        ax2.text(0.05, 0.25, stats2,
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10,
                verticalalignment='top')
        
        # 调整布局
        plt.tight_layout()
        
        # 创建直接对比图
        plt.figure(figsize=(12, 10))
        plt.scatter(scores1, ious1, c='blue', s=20, alpha=0.5, label=method1_name)
        plt.scatter(scores2, ious2, c='green', s=20, alpha=0.5, label=method2_name)
        plt.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        plt.title('Direct Comparison of Methods', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Score', fontsize=12)
        plt.ylabel('IoU', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 将图例放在图形右上角外部
        plt.legend(loc='upper right', fontsize=10, framealpha=1.0)
        
        # 添加对比统计信息，放在左上角
        comparison_stats = (f'{method1_name} vs {method2_name}\n'
                          f'Correlation: {corr1:.3f} vs {corr2:.3f}\n'
                          f'Mean IoU: {mean_iou1:.3f} vs {mean_iou2:.3f}\n'
                          f'Median IoU: {median_iou1:.3f} vs {median_iou2:.3f}')

        # 将统计信息文本框放在左上角，避免与图例重叠
        plt.text(0.05, 0.25, comparison_stats,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white',
                         alpha=1.0,  # 增加不透明度
                         edgecolor='gray',
                         boxstyle='round,pad=0.5'),
                fontsize=10,
                verticalalignment='top')
        
        # 为了避免文本框与图例重叠，确保有足够的边距
        plt.subplots_adjust(top=0.9, right=0.9)
        
        # 保存图形
        if save_path:
            # 保存分开的对比图
            fig.savefig(save_path.replace('.png', '_separate.png'), 
                       dpi=300, bbox_inches='tight')
            # 保存直接对比图
            plt.savefig(save_path.replace('.png', '_combined.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
        
        plt.close('all')
        return True
        
    except Exception as e:
        warnings.warn(f"绘图时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_predictions(pred_file, anno_file):
    """分析单个预测结果文件"""
    try:
        # 加载COCO格式的标注文件
        coco_gt = COCO(anno_file)
        
        # 加载预测结果
        predictions = load_predictions(pred_file)
        
        # 收集预测分数和IoU值
        prediction_scores = []
        iou_values = []
        
        # 处理预测结果
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], dict):
                if 'img_id' in predictions[0]:  # MMDetection新格式
                    predictions_by_image = {}
                    for pred in predictions:
                        img_id = pred['img_id']
                        if 'pred_instances' in pred:
                            instances = pred['pred_instances']
                            scores = instances.get('scores', [])
                            if isinstance(scores, torch.Tensor):
                                scores = scores.cpu().numpy()
                            bboxes = instances.get('bboxes', [])
                            if isinstance(bboxes, torch.Tensor):
                                bboxes = bboxes.cpu().numpy()
                            
                            if len(scores) > 0:
                                max_score_idx = np.argmax(scores)
                                predictions_by_image[img_id] = {
                                    'bbox': bboxes[max_score_idx].tolist(),
                                    'score': float(scores[max_score_idx])
                                }
                
                # 计算IoU值
                for img_id in tqdm(coco_gt.getImgIds()):
                    if img_id in predictions_by_image:
                        pred = predictions_by_image[img_id]
                        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
                        annotations = coco_gt.loadAnns(ann_ids)
                        
                        max_iou = 0
                        for ann in annotations:
                            iou = bbox_iou(pred['bbox'], ann['bbox'])
                            max_iou = max(max_iou, iou)
                        
                        prediction_scores.append(float(pred['score']))
                        iou_values.append(float(max_iou))
        
        return np.array(prediction_scores), np.array(iou_values)
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        traceback.print_exc()
        return None, None

def plot_advanced_comparison(scores1, ious1, scores2, ious2, method1_name, method2_name, save_path=None):
    """创建高级对比图表，增加更直观的差异展示"""
    try:
        plt.style.use('seaborn')
        
        # 创建网格布局
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(4, 4)
        
        # 主散点图
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        
        # 绘制散点和理想线
        ax_main.scatter(scores1, ious1, c='blue', s=20, alpha=0.4, label=method1_name)
        ax_main.scatter(scores2, ious2, c='green', s=20, alpha=0.4, label=method2_name)
        ax_main.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        
        # 添加2D密度轮廓
        # 为每个方法创建2D密度估计
        def plot_density_contour(ax, x, y, color):
            try:
                # 创建网格
                xmin, xmax = 0, 1
                ymin, ymax = 0, 1
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                
                # 使用高斯KDE估计密度
                values = np.vstack([x, y])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # 绘制等高线
                ax.contour(xx, yy, f, levels=5, colors=color, alpha=0.6, linestyles='dashed')
            except Exception as e:
                print(f"密度图绘制错误: {e}")
        
        plot_density_contour(ax_main, scores1, ious1, 'blue')
        plot_density_contour(ax_main, scores2, ious2, 'green')
        
        # 计算并绘制分段平均值线
        def plot_segments(ax, scores, ious, color, linestyle='-'):
            # 划分分数区间
            bins = np.linspace(0, 1, 11)  # 0-0.1, 0.1-0.2, ...
            means = []
            bin_centers = []
            
            for i in range(len(bins)-1):
                # 获取该区间内的IoU值
                mask = (scores >= bins[i]) & (scores < bins[i+1])
                if np.sum(mask) > 0:
                    mean_iou = np.mean(ious[mask])
                    means.append(mean_iou)
                    bin_centers.append((bins[i] + bins[i+1])/2)
            
            if len(bin_centers) > 1:
                ax.plot(bin_centers, means, color=color, linestyle=linestyle, 
                        linewidth=3, alpha=0.7)
        
        plot_segments(ax_main, scores1, ious1, 'blue')
        plot_segments(ax_main, scores2, ious2, 'green')
        
        # 设置主图标题和标签
        ax_main.set_title('Score vs IoU Comparison', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('IoU', fontsize=12, fontweight='bold')
        ax_main.grid(True, linestyle='--', alpha=0.7)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.legend(loc='upper left', fontsize=10)
        
        # 计算统计信息
        corr1 = np.corrcoef(scores1, ious1)[0, 1]
        mean_iou1 = np.mean(ious1)
        median_iou1 = np.median(ious1)
        
        corr2 = np.corrcoef(scores2, ious2)[0, 1]
        mean_iou2 = np.mean(ious2)
        median_iou2 = np.median(ious2)
        
        # 添加统计信息
        stats_text = (f'{method1_name} vs {method2_name}\n'
                    f'Correlation: {corr1:.3f} vs {corr2:.3f}\n'
                    f'Mean IoU: {mean_iou1:.3f} vs {mean_iou2:.3f}\n'
                    f'Median IoU: {median_iou1:.3f} vs {median_iou2:.3f}')
        
        ax_main.text(0.05, 0.25, stats_text,
                transform=ax_main.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=10,
                verticalalignment='top')
        
        # 分数分布直方图（上方）
        ax_score = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_score.hist(scores1, bins=20, alpha=0.5, color='blue', density=True, label=method1_name)
        ax_score.hist(scores2, bins=20, alpha=0.5, color='green', density=True, label=method2_name)
        ax_score.set_ylabel('Density')
        ax_score.set_title('Score Distribution')
        ax_score.legend()
        ax_score.grid(True, linestyle='--', alpha=0.7)
        
        # IoU分布直方图（右侧）
        ax_iou = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        ax_iou.hist(ious1, bins=20, alpha=0.5, color='blue', density=True, 
                  orientation='horizontal', label=method1_name)
        ax_iou.hist(ious2, bins=20, alpha=0.5, color='green', density=True, 
                  orientation='horizontal', label=method2_name)
        ax_iou.set_xlabel('Density')
        ax_iou.set_title('IoU Distribution')
        ax_iou.legend()
        ax_iou.grid(True, linestyle='--', alpha=0.7)
        
        # 添加区间箱线图
        ax_box = fig.add_subplot(gs[0, 3])
        
        # 准备箱线图数据
        def prepare_boxplot_data(scores, ious):
            # 划分分数区间: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            data = []
            
            for i in range(len(bins)-1):
                mask = (scores >= bins[i]) & (scores < bins[i+1])
                data.append(ious[mask])
            
            return data
        
        # 绘制箱线图（并排比较）
        box_data1 = prepare_boxplot_data(scores1, ious1)
        box_data2 = prepare_boxplot_data(scores2, ious2)
        
        positions = np.array([1, 2, 3, 4])
        width = 0.35
        
        # 添加迷你箱线图作为总结
        ax_box.boxplot(box_data1, positions=positions-width/2, widths=width,
                     patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5))
        ax_box.boxplot(box_data2, positions=positions+width/2, widths=width,
                     patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5))
        
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0'])
        ax_box.set_title('IoU by Score Range')
        ax_box.set_ylabel('IoU')
        ax_box.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"高级对比图已保存到: {save_path}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        warnings.warn(f"绘图时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def plot_comparison_with_clean_segments(scores1, ious1, scores2, ious2, method1_name, method2_name, save_path=None):
    """绘制简洁版的对比图，只保留散点和一条加权平均线"""
    try:
        plt.style.use('seaborn')
        
        # 创建大图
        plt.figure(figsize=(14, 10))
        
        # 绘制散点
        plt.scatter(scores1, ious1, c='blue', s=20, alpha=0.3, label=f"{method1_name} data")
        plt.scatter(scores2, ious2, c='green', s=20, alpha=0.3, label=f"{method2_name} data")
        
        # 绘制理想线
        plt.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        
        # 计算并绘制统一的趋势线
        def plot_clean_trend(scores, ious, color, label):
            # 划分分数区间
            bins = np.linspace(0, 1, 11)  # 0-0.1, 0.1-0.2, ...
            means = []
            bin_centers = []
            sample_counts = []
            
            for i in range(len(bins)-1):
                # 获取该区间内的IoU值
                mask = (scores >= bins[i]) & (scores < bins[i+1])
                count = np.sum(mask)
                if count > 0:
                    mean_iou = np.mean(ious[mask])
                    means.append(mean_iou)
                    bin_centers.append((bins[i] + bins[i+1])/2)
                    sample_counts.append(count)
                    print(f"{label}, 区间 [{bins[i]:.1f}-{bins[i+1]:.1f}]: 样本数={count}, 平均IoU={mean_iou:.4f}")
            
            # 计算加权平均
            if len(sample_counts) > 0:
                total_samples = sum(sample_counts)
                print(f"{label} 总样本数: {total_samples}")
                # 只保留一条平滑曲线
                plt.plot(bin_centers, means, color=color, linewidth=4, label=f"{label} trend")
                
                # 添加采样点的大小信息
                for x, y, count in zip(bin_centers, means, sample_counts):
                    # 点的大小与样本数成正比
                    plt.scatter(x, y, s=count/2, color=color, alpha=0.7)
                    
                    # 对于大样本区间（超过100个样本），添加样本数标签
                    if count > 100:
                        plt.text(x, y+0.02, f"{count}", fontsize=9, ha='center', 
                                 bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        # 添加分段平均线
        plot_clean_trend(scores1, ious1, 'blue', method1_name)
        plot_clean_trend(scores2, ious2, 'green', method2_name)
        
        # 设置标题和标签
        plt.title('Score vs IoU Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Prediction Score', fontsize=14)
        plt.ylabel('IoU', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 放置图例在左上角
        legend = plt.legend(loc='upper left', fontsize=12)
        
        # 计算统计信息
        corr1 = np.corrcoef(scores1, ious1)[0, 1]
        mean_iou1 = np.mean(ious1)
        median_iou1 = np.median(ious1)
        
        corr2 = np.corrcoef(scores2, ious2)[0, 1]
        mean_iou2 = np.mean(ious2)
        median_iou2 = np.median(ious2)
        
        # 添加统计信息 - 放在图例下方
        legend_box = legend.get_window_extent().transformed(plt.gca().transAxes.inverted())
        legend_bottom = legend_box.y0  # 图例底部的y坐标
        
        stats_text = (f'{method1_name} vs {method2_name}\n'
                     f'Correlation: {corr1:.3f} vs {corr2:.3f}\n'
                     f'Mean IoU: {mean_iou1:.3f} vs {mean_iou2:.3f}\n'
                     f'Median IoU: {median_iou1:.3f} vs {median_iou2:.3f}')
        
        plt.text(0.05, legend_bottom - 0.02, stats_text,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
                 fontsize=12,
                 verticalalignment='top')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        warnings.warn(f"绘图时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='比较两种目标检测方法的预测结果')
    parser.add_argument('pred_file1', help='第一个方法的预测结果文件路径')
    parser.add_argument('pred_file2', help='第二个方法的预测结果文件路径')
    parser.add_argument('anno_file', help='COCO格式的标注文件路径')
    parser.add_argument('--method1-name', default='Method 1', help='第一个方法的名称')
    parser.add_argument('--method2-name', default='Method 2', help='第二个方法的名称')
    parser.add_argument('--output-dir', default='comparison_results', help='输出目录路径')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析两种方法的结果
    print(f"\n分析 {args.method1_name} 的结果...")
    scores1, ious1 = analyze_predictions(args.pred_file1, args.anno_file)
    
    print(f"\n分析 {args.method2_name} 的结果...")
    scores2, ious2 = analyze_predictions(args.pred_file2, args.anno_file)
    
    if scores1 is not None and scores2 is not None:
        # 绘制简洁的对比图
        save_path = os.path.join(args.output_dir, 'method_comparison_clean.png')
        plot_comparison_with_clean_segments(scores1, ious1, scores2, ious2, 
                                            args.method1_name, args.method2_name, 
                                            save_path)
        
        # 绘制改进的对比图，避免图例重叠
        advanced_save_path = os.path.join(args.output_dir, 'method_comparison_improved.png')
        plot_advanced_comparison(scores1, ious1, scores2, ious2, 
                               args.method1_name, args.method2_name, 
                               advanced_save_path)
        
        # 打印统计信息
        print("\n对比分析结果:")
        print(f"{args.method1_name}:")
        print(f"  相关系数: {np.corrcoef(scores1, ious1)[0, 1]:.4f}")
        print(f"  平均IoU: {np.mean(ious1):.4f}")
        print(f"  中位数IoU: {np.median(ious1):.4f}")
        
        print(f"\n{args.method2_name}:")
        print(f"  相关系数: {np.corrcoef(scores2, ious2)[0, 1]:.4f}")
        print(f"  平均IoU: {np.mean(ious2):.4f}")
        print(f"  中位数IoU: {np.median(ious2):.4f}")
    else:
        print("\n分析失败")

if __name__ == '__main__':
    main() 