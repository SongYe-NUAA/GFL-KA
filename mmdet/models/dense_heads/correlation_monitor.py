"""
🔥 特征-IOU相关性实时监控和可视化模块

核心功能：
1. 🔥 实时监控特征与IOU的相关性变化
2. 🔥 相关性可视化图表生成
3. 🔥 相关性统计分析和报告
4. 🔥 异常相关性检测和预警
5. 🔥 相关性学习效果评估

作者: AI助手
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureIOUCorrelationMonitor(nn.Module):
    """
    🔥 特征-IOU相关性实时监控器
    
    功能特点：
    1. 🔥 实时计算特征与IOU的皮尔逊相关系数
    2. 🔥 相关性历史趋势追踪
    3. 🔥 相关性稳定性分析
    4. 🔥 异常相关性检测
    5. 🔥 多维度相关性统计
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 history_length: int = 1000,
                 correlation_threshold: float = 0.1,
                 save_dir: str = "./correlation_logs",
                 enable_visualization: bool = True,
                 visualization_interval: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.correlation_threshold = correlation_threshold
        self.save_dir = save_dir
        self.enable_visualization = enable_visualization
        self.visualization_interval = visualization_interval
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)
        
        # === 🔥 相关性历史记录 ===
        self.correlation_history = deque(maxlen=history_length)
        self.feature_history = deque(maxlen=history_length)
        self.iou_history = deque(maxlen=history_length)
        self.timestamp_history = deque(maxlen=history_length)
        
        # === 🔥 统计信息 ===
        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('total_samples', torch.tensor(0))
        
        # 相关性统计
        self.register_buffer('running_correlation_mean', torch.zeros(feature_dim))
        self.register_buffer('running_correlation_std', torch.ones(feature_dim))
        self.register_buffer('positive_correlation_count', torch.zeros(feature_dim))
        self.register_buffer('negative_correlation_count', torch.zeros(feature_dim))
        self.register_buffer('stable_correlation_count', torch.zeros(feature_dim))
        
        # 异常检测
        self.register_buffer('correlation_flip_count', torch.zeros(feature_dim))
        self.register_buffer('last_correlation_sign', torch.zeros(feature_dim))
        
        # === 🔥 可视化设置 ===
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 监控日志
        self.monitoring_log = []
        
    def update_correlation(self, features: torch.Tensor, ious: torch.Tensor) -> Dict:
        """
        🔥 更新特征-IOU相关性统计
        
        Args:
            features: [N, feature_dim] 特征张量
            ious: [N] IOU值张量
            
        Returns:
            correlation_stats: Dict 相关性统计信息
        """
        batch_size = features.size(0)
        device = features.device
        
        # 确保输入格式正确
        if features.dim() == 3:  # [N, H*W, feature_dim]
            features = features.view(-1, features.size(-1))  # [N*H*W, feature_dim]
        if ious.dim() == 2:  # [N, H*W]
            ious = ious.view(-1)  # [N*H*W]
        
        # 确保特征和IOU数量匹配
        min_size = min(features.size(0), ious.size(0))
        features = features[:min_size]
        ious = ious[:min_size]
        
        # === 1. 🔥 计算皮尔逊相关系数 ===
        correlations = self._compute_pearson_correlation(features, ious)
        
        # === 2. 🔥 更新统计信息 ===
        self._update_running_stats(correlations)
        
        # === 3. 🔥 异常检测 ===
        anomalies = self._detect_correlation_anomalies(correlations)
        
        # === 4. 🔥 记录历史 ===
        self._record_history(features, ious, correlations)
        
        # === 5. 🔥 生成统计报告 ===
        stats = self._generate_correlation_stats(correlations, anomalies)
        
        # === 6. 🔥 可视化（定期） ===
        if (self.enable_visualization and 
            self.step_counter % self.visualization_interval == 0 and 
            len(self.correlation_history) > 10):
            self._generate_visualizations()
        
        self.step_counter += 1
        self.total_samples += batch_size
        
        return stats
    
    def _compute_pearson_correlation(self, features: torch.Tensor, ious: torch.Tensor) -> torch.Tensor:
        """计算皮尔逊相关系数"""
        # 标准化特征和IOU
        features_std = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-8)
        ious_std = (ious - ious.mean()) / (ious.std() + 1e-8)
        
        # 计算相关系数
        correlations = (features_std * ious_std.unsqueeze(-1)).mean(dim=0)
        
        # 处理NaN值
        correlations = torch.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        
        return correlations
    
    def _update_running_stats(self, correlations: torch.Tensor):
        """更新运行时统计"""
        momentum = 0.1
        
        # 更新均值和标准差
        self.running_correlation_mean.mul_(1 - momentum).add_(correlations, alpha=momentum)
        
        # 计算方差并更新标准差
        variance = (correlations - self.running_correlation_mean).pow(2)
        self.running_correlation_std.mul_(1 - momentum).add_(variance.sqrt(), alpha=momentum)
        
        # 更新正负相关计数
        positive_mask = (correlations > self.correlation_threshold).float()
        negative_mask = (correlations < -self.correlation_threshold).float()
        stable_mask = (torch.abs(correlations) <= self.correlation_threshold).float()
        
        self.positive_correlation_count += positive_mask
        self.negative_correlation_count += negative_mask
        self.stable_correlation_count += stable_mask
        
        # 检测相关性翻转
        if self.step_counter > 0:
            current_sign = torch.sign(correlations)
            flip_mask = (self.last_correlation_sign != current_sign).float()
            self.correlation_flip_count += flip_mask
            self.last_correlation_sign = current_sign
    
    def _detect_correlation_anomalies(self, correlations: torch.Tensor) -> Dict:
        """检测相关性异常"""
        anomalies = {
            'extreme_positive': [],
            'extreme_negative': [],
            'sudden_flip': [],
            'high_variance': []
        }
        
        # 极端正相关（> 0.8）
        extreme_pos_indices = torch.where(correlations > 0.8)[0].cpu().numpy()
        anomalies['extreme_positive'] = extreme_pos_indices.tolist()
        
        # 极端负相关（< -0.8）
        extreme_neg_indices = torch.where(correlations < -0.8)[0].cpu().numpy()
        anomalies['extreme_negative'] = extreme_neg_indices.tolist()
        
        # 突然翻转（如果有历史数据）
        if len(self.correlation_history) > 0:
            last_correlations = torch.tensor(self.correlation_history[-1])
            correlation_change = torch.abs(correlations - last_correlations)
            sudden_flip_indices = torch.where(correlation_change > 0.5)[0].cpu().numpy()
            anomalies['sudden_flip'] = sudden_flip_indices.tolist()
        
        # 高方差特征
        if self.step_counter > 10:
            high_var_indices = torch.where(self.running_correlation_std > 0.3)[0].cpu().numpy()
            anomalies['high_variance'] = high_var_indices.tolist()
        
        return anomalies
    
    def _record_history(self, features: torch.Tensor, ious: torch.Tensor, correlations: torch.Tensor):
        """记录历史数据"""
        # 记录相关性
        self.correlation_history.append(correlations.cpu().numpy())
        
        # 记录特征统计（均值和标准差）
        feature_stats = {
            'mean': features.mean(dim=0).cpu().numpy(),
            'std': features.std(dim=0).cpu().numpy(),
            'min': features.min(dim=0)[0].cpu().numpy(),
            'max': features.max(dim=0)[0].cpu().numpy()
        }
        self.feature_history.append(feature_stats)
        
        # 记录IOU统计
        iou_stats = {
            'mean': ious.mean().item(),
            'std': ious.std().item(),
            'min': ious.min().item(),
            'max': ious.max().item(),
            'median': ious.median().item()
        }
        self.iou_history.append(iou_stats)
        
        # 记录时间戳
        self.timestamp_history.append(time.time())
    
    def _generate_correlation_stats(self, correlations: torch.Tensor, anomalies: Dict) -> Dict:
        """生成相关性统计报告"""
        stats = {
            # 基本统计
            'step': self.step_counter.item(),
            'total_samples': self.total_samples.item(),
            'correlation_mean': correlations.mean().item(),
            'correlation_std': correlations.std().item(),
            'correlation_min': correlations.min().item(),
            'correlation_max': correlations.max().item(),
            
            # 分布统计
            'positive_correlation_ratio': (correlations > self.correlation_threshold).float().mean().item(),
            'negative_correlation_ratio': (correlations < -self.correlation_threshold).float().mean().item(),
            'stable_correlation_ratio': (torch.abs(correlations) <= self.correlation_threshold).float().mean().item(),
            
            # 异常统计
            'anomaly_count': sum(len(v) for v in anomalies.values()),
            'extreme_positive_count': len(anomalies['extreme_positive']),
            'extreme_negative_count': len(anomalies['extreme_negative']),
            'sudden_flip_count': len(anomalies['sudden_flip']),
            'high_variance_count': len(anomalies['high_variance']),
            
            # 历史统计（如果有足够历史数据）
            'correlation_stability': 0.0,
            'average_flip_rate': 0.0
        }
        
        # 计算稳定性指标
        if len(self.correlation_history) > 5:
            recent_correlations = np.array(list(self.correlation_history)[-5:])
            correlation_stability = 1.0 - np.std(recent_correlations, axis=0).mean()
            stats['correlation_stability'] = max(0.0, correlation_stability)
        
        # 计算平均翻转率
        if self.step_counter > 0:
            flip_rate = (self.correlation_flip_count / self.step_counter.float()).mean().item()
            stats['average_flip_rate'] = flip_rate
        
        # 记录到监控日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': self.step_counter.item(),
            'stats': stats,
            'anomalies': anomalies
        }
        self.monitoring_log.append(log_entry)
        
        return stats
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        try:
            if len(self.correlation_history) < 10:
                return
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'🔥 特征-IOU相关性监控 (Step {self.step_counter.item()})', fontsize=16)
            
            # === 1. 相关性时间序列 ===
            self._plot_correlation_timeseries(axes[0, 0])
            
            # === 2. 相关性分布直方图 ===
            self._plot_correlation_distribution(axes[0, 1])
            
            # === 3. 相关性热力图 ===
            self._plot_correlation_heatmap(axes[0, 2])
            
            # === 4. 特征稳定性分析 ===
            self._plot_feature_stability(axes[1, 0])
            
            # === 5. IOU分布变化 ===
            self._plot_iou_distribution(axes[1, 1])
            
            # === 6. 异常检测图 ===
            self._plot_anomaly_detection(axes[1, 2])
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.save_dir, "plots", f"correlation_monitor_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 相关性监控图表已保存: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
    
    def _plot_correlation_timeseries(self, ax):
        """绘制相关性时间序列"""
        if len(self.correlation_history) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        correlations = np.array(list(self.correlation_history))
        steps = np.arange(len(correlations))
        
        # 绘制均值和标准差
        mean_corr = correlations.mean(axis=1)
        std_corr = correlations.std(axis=1)
        
        ax.plot(steps, mean_corr, 'b-', linewidth=2, label='均值')
        ax.fill_between(steps, mean_corr - std_corr, mean_corr + std_corr, alpha=0.3, color='blue', label='±1σ')
        
        # 添加阈值线
        ax.axhline(y=self.correlation_threshold, color='red', linestyle='--', alpha=0.7, label='正相关阈值')
        ax.axhline(y=-self.correlation_threshold, color='red', linestyle='--', alpha=0.7, label='负相关阈值')
        
        ax.set_title('相关性时间序列')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('相关系数')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_distribution(self, ax):
        """绘制相关性分布直方图"""
        if len(self.correlation_history) == 0:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 获取最新的相关性数据
        latest_correlations = self.correlation_history[-1]
        
        ax.hist(latest_correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=self.correlation_threshold, color='red', linestyle='--', label='正阈值')
        ax.axvline(x=-self.correlation_threshold, color='red', linestyle='--', label='负阈值')
        ax.axvline(x=np.mean(latest_correlations), color='green', linewidth=2, label='均值')
        
        ax.set_title('当前相关性分布')
        ax.set_xlabel('相关系数')
        ax.set_ylabel('特征数量')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, ax):
        """绘制相关性热力图"""
        if len(self.correlation_history) < 10:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 获取最近10步的相关性数据
        recent_data = np.array(list(self.correlation_history)[-10:])
        
        # 只显示前32个特征（避免图表过于密集）
        display_features = min(32, recent_data.shape[1])
        heatmap_data = recent_data[:, :display_features].T
        
        im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('相关性热力图 (最近10步)')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('特征维度')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_feature_stability(self, ax):
        """绘制特征稳定性分析"""
        if len(self.correlation_history) < 5:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 计算特征稳定性（方差的倒数）
        correlations = np.array(list(self.correlation_history))
        feature_variance = np.var(correlations, axis=0)
        feature_stability = 1.0 / (feature_variance + 1e-8)
        
        # 显示前32个特征
        display_features = min(32, len(feature_stability))
        feature_indices = np.arange(display_features)
        
        bars = ax.bar(feature_indices, feature_stability[:display_features], 
                     color='lightgreen', alpha=0.7, edgecolor='black')
        
        # 标记最稳定和最不稳定的特征
        most_stable = np.argmax(feature_stability[:display_features])
        least_stable = np.argmin(feature_stability[:display_features])
        
        bars[most_stable].set_color('green')
        bars[least_stable].set_color('red')
        
        ax.set_title('特征稳定性分析')
        ax.set_xlabel('特征维度')
        ax.set_ylabel('稳定性分数')
        ax.grid(True, alpha=0.3)
    
    def _plot_iou_distribution(self, ax):
        """绘制IOU分布变化"""
        if len(self.iou_history) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取IOU统计
        iou_means = [stats['mean'] for stats in self.iou_history]
        iou_stds = [stats['std'] for stats in self.iou_history]
        steps = np.arange(len(iou_means))
        
        ax.plot(steps, iou_means, 'g-', linewidth=2, label='IOU均值')
        ax.fill_between(steps, 
                       np.array(iou_means) - np.array(iou_stds),
                       np.array(iou_means) + np.array(iou_stds),
                       alpha=0.3, color='green', label='±1σ')
        
        ax.set_title('IOU分布变化')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('IOU值')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_anomaly_detection(self, ax):
        """绘制异常检测图"""
        if len(self.monitoring_log) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取异常统计
        anomaly_counts = []
        flip_rates = []
        steps = []
        
        for log_entry in self.monitoring_log[-50:]:  # 最近50步
            stats = log_entry['stats']
            anomaly_counts.append(stats['anomaly_count'])
            flip_rates.append(stats['average_flip_rate'])
            steps.append(stats['step'])
        
        # 双y轴图
        ax2 = ax.twinx()
        
        line1 = ax.plot(steps, anomaly_counts, 'r-', linewidth=2, label='异常数量')
        line2 = ax2.plot(steps, flip_rates, 'b--', linewidth=2, label='翻转率')
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('异常数量', color='red')
        ax2.set_ylabel('相关性翻转率', color='blue')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.set_title('异常检测统计')
        ax.grid(True, alpha=0.3)
    
    def save_monitoring_data(self):
        """保存监控数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存监控日志
        log_path = os.path.join(self.save_dir, "data", f"monitoring_log_{timestamp}.json")
        with open(log_path, 'w') as f:
            json.dump(self.monitoring_log, f, indent=2)
        
        # 保存相关性历史
        if len(self.correlation_history) > 0:
            corr_path = os.path.join(self.save_dir, "data", f"correlation_history_{timestamp}.npy")
            correlations = np.array(list(self.correlation_history))
            np.save(corr_path, correlations)
        
        print(f"📁 监控数据已保存: {log_path}")
        return log_path
    
    def get_summary_report(self) -> Dict:
        """获取总结报告"""
        if len(self.monitoring_log) == 0:
            return {"message": "暂无监控数据"}
        
        latest_stats = self.monitoring_log[-1]['stats']
        
        # 计算历史趋势
        if len(self.monitoring_log) > 10:
            recent_stats = [log['stats'] for log in self.monitoring_log[-10:]]
            correlation_trend = np.mean([s['correlation_mean'] for s in recent_stats])
            stability_trend = np.mean([s['correlation_stability'] for s in recent_stats])
        else:
            correlation_trend = latest_stats['correlation_mean']
            stability_trend = latest_stats['correlation_stability']
        
        report = {
            "监控概要": {
                "总训练步数": latest_stats['step'],
                "总样本数": latest_stats['total_samples'],
                "监控历史长度": len(self.correlation_history)
            },
            "当前相关性状态": {
                "平均相关性": f"{latest_stats['correlation_mean']:.4f}",
                "相关性标准差": f"{latest_stats['correlation_std']:.4f}",
                "正相关比例": f"{latest_stats['positive_correlation_ratio']:.2%}",
                "负相关比例": f"{latest_stats['negative_correlation_ratio']:.2%}",
                "稳定相关比例": f"{latest_stats['stable_correlation_ratio']:.2%}"
            },
            "稳定性分析": {
                "相关性稳定性": f"{latest_stats['correlation_stability']:.4f}",
                "平均翻转率": f"{latest_stats['average_flip_rate']:.4f}",
                "历史趋势稳定性": f"{stability_trend:.4f}"
            },
            "异常检测": {
                "当前异常数量": latest_stats['anomaly_count'],
                "极端正相关": latest_stats['extreme_positive_count'],
                "极端负相关": latest_stats['extreme_negative_count'],
                "突然翻转": latest_stats['sudden_flip_count'],
                "高方差特征": latest_stats['high_variance_count']
            },
            "建议": self._generate_recommendations(latest_stats)
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 相关性强度建议
        if stats['correlation_mean'] < 0.1:
            recommendations.append("🔥 建议增强特征-IOU相关性学习，当前相关性较弱")
        
        # 稳定性建议
        if stats['correlation_stability'] < 0.7:
            recommendations.append("⚡ 建议提高相关性稳定性，当前波动较大")
        
        # 翻转率建议
        if stats['average_flip_rate'] > 0.3:
            recommendations.append("🎯 建议降低相关性翻转率，训练可能不稳定")
        
        # 异常检测建议
        if stats['anomaly_count'] > stats['step'] * 0.1:
            recommendations.append("⚠️ 异常相关性过多，建议检查特征质量")
        
        # 分布建议
        if stats['positive_correlation_ratio'] < 0.3:
            recommendations.append("📈 正相关特征偏少，建议优化特征提取")
        
        if len(recommendations) == 0:
            recommendations.append("✅ 相关性监控状态良好，继续保持")
        
        return recommendations

class CorrelationVisualizationDashboard:
    """
    🔥 相关性可视化仪表板
    
    提供更高级的可视化功能：
    1. 交互式图表
    2. 实时更新
    3. 多维度分析
    4. 对比分析
    """
    
    def __init__(self, monitor: FeatureIOUCorrelationMonitor):
        self.monitor = monitor
        
    def create_dashboard(self, output_file: str = "correlation_dashboard.html"):
        """创建HTML仪表板"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=['相关性时间序列', '相关性分布', '特征稳定性', 
                              'IOU趋势', '异常检测', '相关性热力图'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            # 添加各种图表...
            # (这里可以添加更详细的plotly图表代码)
            
            # 保存HTML文件
            dashboard_path = os.path.join(self.monitor.save_dir, output_file)
            pyo.plot(fig, filename=dashboard_path, auto_open=False)
            
            print(f"📊 交互式仪表板已创建: {dashboard_path}")
            return dashboard_path
            
        except ImportError:
            print("📋 需要安装plotly库来创建交互式仪表板: pip install plotly")
            return None

# === 🔥 使用示例 ===
def create_correlation_monitor_example():
    """创建监控器使用示例"""
    
    # 创建监控器
    monitor = FeatureIOUCorrelationMonitor(
        feature_dim=256,
        history_length=500,
        correlation_threshold=0.1,
        save_dir="./correlation_monitoring",
        enable_visualization=True,
        visualization_interval=50
    )
    
    # 模拟训练过程
    print("🔥 开始相关性监控示例...")
    
    for step in range(100):
        # 模拟特征和IOU数据
        batch_size = 32
        feature_dim = 256
        
        features = torch.randn(batch_size, feature_dim)
        ious = torch.sigmoid(torch.randn(batch_size))  # 模拟IOU值 [0,1]
        
        # 更新监控
        stats = monitor.update_correlation(features, ious)
        
        # 定期打印统计
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  平均相关性: {stats['correlation_mean']:.4f}")
            print(f"  正相关比例: {stats['positive_correlation_ratio']:.2%}")
            print(f"  异常数量: {stats['anomaly_count']}")
    
    # 生成总结报告
    report = monitor.get_summary_report()
    print("\n" + "="*50)
    print("🔥 相关性监控总结报告")
    print("="*50)
    for section, data in report.items():
        print(f"\n【{section}】")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(f"  • {item}")
        else:
            print(f"  {data}")
    
    # 保存数据
    monitor.save_monitoring_data()
    
    # 创建仪表板
    dashboard = CorrelationVisualizationDashboard(monitor)
    dashboard.create_dashboard()
    
    return monitor

if __name__ == "__main__":
    # 运行示例
    monitor = create_correlation_monitor_example()