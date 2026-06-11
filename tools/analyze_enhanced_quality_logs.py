# 增强质量损失日志分析工具

"""
分析增强质量损失训练日志，验证改动效果的工具脚本
"""

import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

def parse_training_log(log_file: str) -> Dict[str, List[float]]:
    """解析训练日志文件，提取增强质量损失相关指标
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        包含各种指标时间序列的字典
    """
    metrics = {
        # 基础损失指标
        'loss': [],
        'loss_cls': [],
        'loss_bbox': [],
        'loss_dfl': [],
        'loss_quality': [],
        
        # IoU相关指标
        'pred_iou': [],
        'gt_iou': [],
        'iou_alignment_error': [],
        
        # 增强质量损失专用指标
        'qfl_lqe_consistency_error': [],
        'iou_prediction_error': [],
        'ranking_consistency': [],
        'pred_quality_mean': [],
        'gt_iou_mean': [],
        'implicit_quality_mean': [],
        'improvement_ratio': [],
        
        # 课程学习指标
        'curriculum_offset': [],
        'kurtosis_scale': [],
        
        # 迭代信息
        'epoch': [],
        'iter': []
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析基础训练指标
            if 'loss:' in line and 'loss_cls:' in line:
                try:
                    # 提取epoch和iter
                    epoch_match = re.search(r'Epoch\(train\)\s*\[(\d+)\]', line)
                    iter_match = re.search(r'\[(\d+)/\d+\]', line)
                    
                    if epoch_match and iter_match:
                        epoch = int(epoch_match.group(1))
                        iter_num = int(iter_match.group(1))
                        metrics['epoch'].append(epoch)
                        metrics['iter'].append(iter_num)
                        
                        # 提取各种损失值
                        loss_patterns = {
                            'loss': r'loss:\s*([\d.]+)',
                            'loss_cls': r'loss_cls:\s*([\d.]+)',
                            'loss_bbox': r'loss_bbox:\s*([\d.]+)',
                            'loss_dfl': r'loss_dfl:\s*([\d.]+)',
                            'loss_quality': r'loss_quality:\s*([\d.]+)',
                            'pred_iou': r'pred_iou:\s*([\d.]+)',
                            'gt_iou': r'gt_iou:\s*([\d.]+)',
                            'iou_alignment_error': r'iou_alignment_error:\s*([\d.]+)',
                            'curriculum_offset': r'curriculum_offset:\s*([\d.]+)',
                            'kurtosis_scale': r'kurtosis_scale:\s*([\d.]+)'
                        }
                        
                        for metric, pattern in loss_patterns.items():
                            match = re.search(pattern, line)
                            if match:
                                metrics[metric].append(float(match.group(1)))
                            else:
                                metrics[metric].append(0.0)
                                
                except Exception as e:
                    continue
            
            # 解析增强质量损失专用指标
            enhanced_patterns = {
                'qfl_lqe_consistency_error': r'enhanced_quality/qfl_lqe_consistency_error.*?(\d+\.?\d*)',
                'iou_prediction_error': r'enhanced_quality/iou_prediction_error.*?(\d+\.?\d*)',
                'ranking_consistency': r'enhanced_quality/ranking_consistency.*?(\d+\.?\d*)',
                'pred_quality_mean': r'enhanced_quality/pred_quality_mean.*?(\d+\.?\d*)',
                'gt_iou_mean': r'enhanced_quality/gt_iou_mean.*?(\d+\.?\d*)',
                'implicit_quality_mean': r'enhanced_quality/implicit_quality_mean.*?(\d+\.?\d*)',
                'improvement_ratio': r'enhanced_quality/improvement_ratio.*?(\d+\.?\d*)'
            }
            
            for metric, pattern in enhanced_patterns.items():
                match = re.search(pattern, line)
                if match:
                    metrics[metric].append(float(match.group(1)))
    
    return metrics

def analyze_qfl_lqe_synergy(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """分析QFL-LQE协同效果
    
    Args:
        metrics: 解析的指标数据
        
    Returns:
        协同效果分析结果
    """
    analysis = {}
    
    if metrics['qfl_lqe_consistency_error']:
        consistency_errors = metrics['qfl_lqe_consistency_error']
        analysis['avg_consistency_error'] = np.mean(consistency_errors)
        analysis['consistency_improvement'] = consistency_errors[0] - consistency_errors[-1] if len(consistency_errors) > 1 else 0
        analysis['consistency_trend'] = 'improving' if analysis['consistency_improvement'] > 0 else 'stable'
    
    if metrics['improvement_ratio']:
        ratios = metrics['improvement_ratio']
        analysis['avg_improvement_ratio'] = np.mean(ratios)
        analysis['improvement_trend'] = 'positive' if analysis['avg_improvement_ratio'] > 0 else 'negative'
    
    return analysis

def analyze_ranking_consistency(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """分析排序一致性效果
    
    Args:
        metrics: 解析的指标数据
        
    Returns:
        排序一致性分析结果
    """
    analysis = {}
    
    if metrics['ranking_consistency']:
        rankings = metrics['ranking_consistency']
        analysis['avg_ranking_consistency'] = np.mean(rankings)
        analysis['ranking_improvement'] = rankings[-1] - rankings[0] if len(rankings) > 1 else 0
        analysis['ranking_stability'] = np.std(rankings)
    
    return analysis

def analyze_iou_prediction_accuracy(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """分析IoU预测精度
    
    Args:
        metrics: 解析的指标数据
        
    Returns:
        IoU预测精度分析结果
    """
    analysis = {}
    
    if metrics['iou_prediction_error']:
        errors = metrics['iou_prediction_error']
        analysis['avg_prediction_error'] = np.mean(errors)
        analysis['error_reduction'] = errors[0] - errors[-1] if len(errors) > 1 else 0
        analysis['error_trend'] = 'improving' if analysis['error_reduction'] > 0 else 'stable'
    
    if metrics['pred_iou'] and metrics['gt_iou']:
        pred_ious = metrics['pred_iou']
        gt_ious = metrics['gt_iou']
        if len(pred_ious) == len(gt_ious):
            correlation = np.corrcoef(pred_ious, gt_ious)[0, 1] if len(pred_ious) > 1 else 0
            analysis['iou_correlation'] = correlation
    
    return analysis

def plot_training_curves(metrics: Dict[str, List[float]], save_dir: str = './'):
    """绘制训练曲线图
    
    Args:
        metrics: 解析的指标数据
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 1. 基础损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Quality Loss Training Analysis', fontsize=16)
    
    # 损失值对比
    ax1 = axes[0, 0]
    if metrics['loss_cls']:
        ax1.plot(metrics['loss_cls'], label='QFL Loss', alpha=0.7)
    if metrics['loss_dfl']:
        ax1.plot(metrics['loss_dfl'], label='DFL Loss', alpha=0.7)
    if metrics['loss_quality']:
        ax1.plot(metrics['loss_quality'], label='Enhanced Quality Loss', alpha=0.7)
    ax1.set_title('Loss Components')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU预测精度
    ax2 = axes[0, 1]
    if metrics['pred_iou'] and metrics['gt_iou']:
        ax2.plot(metrics['pred_iou'], label='Predicted IoU', alpha=0.7)
        ax2.plot(metrics['gt_iou'], label='Ground Truth IoU', alpha=0.7)
    ax2.set_title('IoU Prediction vs Ground Truth')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('IoU Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # QFL-LQE协同效果
    ax3 = axes[1, 0]
    if metrics['qfl_lqe_consistency_error']:
        ax3.plot(metrics['qfl_lqe_consistency_error'], label='Consistency Error', color='red', alpha=0.7)
    if metrics['iou_prediction_error']:
        ax3.plot(metrics['iou_prediction_error'], label='IoU Prediction Error', color='blue', alpha=0.7)
    ax3.set_title('QFL-LQE Synergy Effect')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Error Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 排序一致性
    ax4 = axes[1, 1]
    if metrics['ranking_consistency']:
        ax4.plot(metrics['ranking_consistency'], label='Ranking Consistency', color='green', alpha=0.7)
    if metrics['iou_alignment_error']:
        ax4.plot(metrics['iou_alignment_error'], label='IoU Alignment Error', color='orange', alpha=0.7)
    ax4.set_title('Ranking Consistency & Alignment')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Metric Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'enhanced_quality_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report(metrics: Dict[str, List[float]], save_dir: str = './'):
    """生成分析报告
    
    Args:
        metrics: 解析的指标数据
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 进行各项分析
    synergy_analysis = analyze_qfl_lqe_synergy(metrics)
    ranking_analysis = analyze_ranking_consistency(metrics)
    iou_analysis = analyze_iou_prediction_accuracy(metrics)
    
    # 生成报告
    report = {
        'summary': {
            'total_iterations': len(metrics.get('loss', [])),
            'epochs_trained': max(metrics.get('epoch', [0])),
            'analysis_timestamp': str(np.datetime64('now'))
        },
        'qfl_lqe_synergy': synergy_analysis,
        'ranking_consistency': ranking_analysis,
        'iou_prediction': iou_analysis,
        'recommendations': []
    }
    
    # 生成建议
    if synergy_analysis.get('consistency_trend') == 'improving':
        report['recommendations'].append("✅ QFL-LQE协同效果良好，一致性在改善")
    else:
        report['recommendations'].append("⚠️ QFL-LQE协同效果需要调整，考虑调整consistency_weight")
    
    if ranking_analysis.get('avg_ranking_consistency', 0) > 0.8:
        report['recommendations'].append("✅ 排序一致性优秀，有利于NMS和课程学习")
    else:
        report['recommendations'].append("⚠️ 排序一致性需要改善，考虑增加ranking_weight")
    
    if iou_analysis.get('error_trend') == 'improving':
        report['recommendations'].append("✅ IoU预测精度在提升")
    else:
        report['recommendations'].append("⚠️ IoU预测精度改善缓慢，可能需要调整引导模式")
    
    # 保存报告
    with open(save_path / 'enhanced_quality_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印简要报告
    print("=" * 60)
    print("🔍 增强质量损失效果分析报告")
    print("=" * 60)
    
    print(f"\n📊 训练概况:")
    print(f"   总迭代数: {report['summary']['total_iterations']}")
    print(f"   训练轮数: {report['summary']['epochs_trained']}")
    
    print(f"\n🤝 QFL-LQE协同效果:")
    for key, value in synergy_analysis.items():
        print(f"   {key}: {value}")
    
    print(f"\n📈 排序一致性:")
    for key, value in ranking_analysis.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 IoU预测精度:")
    for key, value in iou_analysis.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 建议:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    print(f"\n📁 详细报告已保存到: {save_path / 'enhanced_quality_analysis_report.json'}")

def main():
    parser = argparse.ArgumentParser(description='分析增强质量损失训练日志')
    parser.add_argument('log_file', help='训练日志文件路径')
    parser.add_argument('--save_dir', default='./', help='分析结果保存目录')
    parser.add_argument('--plot', action='store_true', help='是否绘制图表')
    
    args = parser.parse_args()
    
    print("🔍 开始分析增强质量损失训练日志...")
    
    # 解析日志
    metrics = parse_training_log(args.log_file)
    
    if not metrics['loss']:
        print("❌ 未找到有效的训练日志数据")
        return
    
    print(f"✅ 成功解析 {len(metrics['loss'])} 条训练记录")
    
    # 生成分析报告
    generate_analysis_report(metrics, args.save_dir)
    
    # 绘制图表
    if args.plot:
        print("📊 正在生成训练曲线图...")
        plot_training_curves(metrics, args.save_dir)
        print("✅ 图表已保存")

if __name__ == "__main__":
    main()