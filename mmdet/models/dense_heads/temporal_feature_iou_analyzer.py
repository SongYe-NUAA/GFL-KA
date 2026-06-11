#!/usr/bin/env python3
"""
🕒 时序特征-IoU关系分析器：深度探索统计量与IoU动态关系

核心思想：
1. 系统性追踪不同训练阶段下特征-IoU的相关性变化
2. 发现特征重要性的时序演化模式
3. 构建动态映射学习器，自适应调整特征权重
4. 提供可视化分析工具，深入理解时序变化机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class FeatureIoUCorrelationTracker(nn.Module):
    """
    🔍 特征-IoU相关性追踪器：实时监控特征与IoU的关系变化
    
    核心功能：
    1. 实时计算特征-IoU相关性
    2. 追踪相关性的时序变化
    3. 检测显著变化点
    4. 提供统计分析和可视化
    """
    
    def __init__(self, feature_names: List[str], window_size: int = 1000):
        super().__init__()
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.window_size = window_size
        
        # 🕒 时序数据存储
        self.correlation_history = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_history = deque(maxlen=window_size)
        self.batch_history = deque(maxlen=window_size)
        
        # 📊 统计追踪
        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('current_batch', torch.tensor(0))
        
        # 🎯 相关性统计缓存
        self.correlation_cache = {}
        self.last_analysis_time = None
        
        # 📈 变化点检测
        self.change_points = defaultdict(list)  # 记录显著变化的时间点
        self.change_threshold = 0.3  # 相关性变化阈值
        
    def update(self, features_dict: Dict[str, torch.Tensor], iou_targets: torch.Tensor, 
               epoch: int, batch_idx: int):
        """
        更新特征-IoU相关性追踪
        
        Args:
            features_dict: 特征字典 {name: [N, C, H, W]}
            iou_targets: IoU目标值 [N, H, W]
            epoch: 当前epoch
            batch_idx: 当前batch索引
        """
        self.current_epoch = torch.tensor(epoch)
        self.current_batch = torch.tensor(batch_idx)
        self.update_count += 1
        
        # 记录时间戳
        self.epoch_history.append(epoch)
        self.batch_history.append(batch_idx)
        
        # 计算每个特征与IoU的相关性
        current_correlations = {}
        
        for name in self.feature_names:
            if name in features_dict:
                feature = features_dict[name]  # [N, C, H, W]
                
                # 展平特征和IoU用于相关性计算
                feature_flat = feature.view(-1).detach().cpu().numpy()
                iou_flat = iou_targets.view(-1).detach().cpu().numpy()
                
                # 计算皮尔逊相关系数
                if len(feature_flat) > 1 and np.std(feature_flat) > 1e-8 and np.std(iou_flat) > 1e-8:
                    correlation = np.corrcoef(feature_flat, iou_flat)[0, 1]
                    if not np.isnan(correlation):
                        current_correlations[name] = correlation
                        
                        # 检测显著变化
                        if len(self.correlation_history[name]) > 0:
                            prev_corr = self.correlation_history[name][-1]
                            change_magnitude = abs(correlation - prev_corr)
                            
                            if change_magnitude > self.change_threshold:
                                self.change_points[name].append({
                                    'epoch': epoch,
                                    'batch': batch_idx,
                                    'prev_corr': prev_corr,
                                    'new_corr': correlation,
                                    'change': change_magnitude
                                })
                        
                        # 更新历史记录
                        self.correlation_history[name].append(correlation)
        
        # 缓存当前相关性
        self.correlation_cache = current_correlations
        
    def get_correlation_trends(self, feature_name: str = None) -> Dict:
        """
        获取相关性趋势分析
        
        Args:
            feature_name: 特定特征名，如果为None则返回所有特征
            
        Returns:
            趋势分析结果
        """
        if feature_name:
            features_to_analyze = [feature_name]
        else:
            features_to_analyze = self.feature_names
        
        trends = {}
        
        for name in features_to_analyze:
            if name in self.correlation_history and len(self.correlation_history[name]) > 1:
                correlations = list(self.correlation_history[name])
                
                # 基础统计
                trends[name] = {
                    'current': correlations[-1] if correlations else 0.0,
                    'mean': np.mean(correlations),
                    'std': np.std(correlations),
                    'min': np.min(correlations),
                    'max': np.max(correlations),
                    'trend': self._calculate_trend(correlations),
                    'stability': self._calculate_stability(correlations),
                    'change_points': len(self.change_points.get(name, [])),
                    'recent_change': self._get_recent_change(correlations)
                }
        
        return trends
    
    def _calculate_trend(self, correlations: List[float]) -> str:
        """计算趋势方向"""
        if len(correlations) < 3:
            return "insufficient_data"
        
        # 使用线性回归计算趋势
        x = np.arange(len(correlations))
        y = np.array(correlations)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_stability(self, correlations: List[float]) -> float:
        """计算稳定性分数 (0-1, 1表示最稳定)"""
        if len(correlations) < 2:
            return 1.0
        
        # 使用变异系数的倒数作为稳定性指标
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        if abs(mean_corr) < 1e-8:
            return 0.0
        
        cv = std_corr / abs(mean_corr)
        stability = 1.0 / (1.0 + cv)
        return stability
    
    def _get_recent_change(self, correlations: List[float]) -> float:
        """获取最近的变化幅度"""
        if len(correlations) < 2:
            return 0.0
        
        recent_window = min(10, len(correlations) // 4)  # 最近25%的数据
        if recent_window < 2:
            return abs(correlations[-1] - correlations[0])
        
        recent = correlations[-recent_window:]
        older = correlations[-2*recent_window:-recent_window] if len(correlations) >= 2*recent_window else correlations[:-recent_window]
        
        return abs(np.mean(recent) - np.mean(older))

class DynamicMappingLearner(nn.Module):
    """
    🧠 动态映射学习器：学习时序变化的特征-IoU转换函数
    
    核心思想：
    1. 基于时序相关性分析，动态调整特征权重
    2. 学习不同训练阶段的最优特征组合
    3. 预测未来阶段的特征重要性变化
    4. 提供自适应校准机制
    """
    
    def __init__(self, num_features: int, temporal_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.temporal_dim = temporal_dim
        
        # 🕒 时序编码器：将训练阶段信息编码为向量
        self.temporal_encoder = nn.Sequential(
            nn.Linear(5, temporal_dim),  # [epoch_ratio, loss_trend, correlation_trend, stability, recent_change]
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 🎯 特征重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(temporal_dim + num_features, temporal_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim, num_features),
            nn.Sigmoid()  # 输出[0,1]的重要性权重
        )
        
        # 🔄 映射函数学习器：学习特征到IoU的非线性映射
        self.mapping_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, temporal_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(temporal_dim // 4, temporal_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(temporal_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_features)
        ])
        
        # 📈 时序状态追踪
        self.register_buffer('feature_importance_history', torch.zeros(100, num_features))
        self.register_buffer('history_ptr', torch.tensor(0))
        
        # 🎚️ 自适应调整参数
        self.adaptation_strength = nn.Parameter(torch.tensor(0.5))
        self.stability_weight = nn.Parameter(torch.tensor(0.3))
        
    def encode_temporal_state(self, epoch_ratio: float, loss_trend: float, 
                            correlation_trends: Dict, feature_correlations: Dict) -> torch.Tensor:
        """
        编码当前时序状态
        
        Args:
            epoch_ratio: 训练进度比例
            loss_trend: 损失变化趋势
            correlation_trends: 相关性趋势分析结果
            feature_correlations: 当前特征相关性
            
        Returns:
            temporal_encoding: 时序状态编码 [temporal_dim]
        """
        # 计算全局相关性统计
        all_correlations = list(feature_correlations.values())
        if all_correlations:
            avg_correlation_trend = np.mean([trends.get('recent_change', 0.0) 
                                           for trends in correlation_trends.values()])
            avg_stability = np.mean([trends.get('stability', 1.0) 
                                   for trends in correlation_trends.values()])
            recent_change_magnitude = np.mean([abs(corr) for corr in all_correlations])
        else:
            avg_correlation_trend = 0.0
            avg_stability = 1.0
            recent_change_magnitude = 0.0
        
        # 构建时序特征向量
        temporal_features = torch.tensor([
            epoch_ratio,
            loss_trend,
            avg_correlation_trend,
            avg_stability,
            recent_change_magnitude
        ], dtype=torch.float32)
        
        # 编码
        temporal_encoding = self.temporal_encoder(temporal_features)
        return temporal_encoding
    
    def predict_feature_importance(self, temporal_encoding: torch.Tensor, 
                                 current_correlations: torch.Tensor) -> torch.Tensor:
        """
        预测特征重要性权重
        
        Args:
            temporal_encoding: 时序编码 [temporal_dim]
            current_correlations: 当前相关性 [num_features]
            
        Returns:
            importance_weights: 特征重要性权重 [num_features]
        """
        # 组合时序信息和当前相关性
        combined_input = torch.cat([temporal_encoding, current_correlations])
        
        # 预测重要性权重
        importance_weights = self.importance_predictor(combined_input)
        
        # 更新历史记录
        ptr = self.history_ptr % self.feature_importance_history.size(0)
        self.feature_importance_history[ptr] = importance_weights.detach()
        self.history_ptr += 1
        
        return importance_weights
    
    def apply_dynamic_mapping(self, features: torch.Tensor, 
                            importance_weights: torch.Tensor) -> torch.Tensor:
        """
        应用动态映射变换
        
        Args:
            features: 原始特征 [N, num_features, H, W]
            importance_weights: 重要性权重 [num_features]
            
        Returns:
            mapped_features: 映射后的特征 [N, num_features, H, W]
        """
        N, C, H, W = features.shape
        mapped_features = []
        
        for i in range(C):
            # 提取单个特征通道
            feature_channel = features[:, i:i+1, :, :].view(-1, 1)  # [N*H*W, 1]
            
            # 应用对应的映射函数
            if i < len(self.mapping_layers):
                mapped_channel = self.mapping_layers[i](feature_channel)  # [N*H*W, 1]
                mapped_channel = mapped_channel.view(N, 1, H, W)
            else:
                mapped_channel = features[:, i:i+1, :, :]
            
            # 应用重要性权重
            importance_weight = importance_weights[i] if i < len(importance_weights) else 1.0
            weighted_channel = mapped_channel * importance_weight
            
            mapped_features.append(weighted_channel)
        
        return torch.cat(mapped_features, dim=1)
    
    def forward(self, features: torch.Tensor, temporal_state: Dict) -> torch.Tensor:
        """
        动态映射前向传播
        
        Args:
            features: 输入特征 [N, num_features, H, W]
            temporal_state: 时序状态信息
            
        Returns:
            adapted_features: 自适应调整后的特征 [N, num_features, H, W]
        """
        # 1. 编码时序状态
        temporal_encoding = self.encode_temporal_state(
            temporal_state['epoch_ratio'],
            temporal_state['loss_trend'],
            temporal_state['correlation_trends'],
            temporal_state['feature_correlations']
        )
        
        # 2. 构建当前相关性向量
        current_correlations = torch.tensor([
            temporal_state['feature_correlations'].get(f'feature_{i}', 0.0)
            for i in range(self.num_features)
        ], dtype=torch.float32)
        
        # 3. 预测特征重要性
        importance_weights = self.predict_feature_importance(temporal_encoding, current_correlations)
        
        # 4. 应用动态映射
        adapted_features = self.apply_dynamic_mapping(features, importance_weights)
        
        return adapted_features

class TemporalFeatureIoUAnalyzer:
    """
    🔬 时序特征-IoU分析器：统合分析和可视化系统
    
    核心功能：
    1. 整合相关性追踪和动态映射学习
    2. 提供全面的分析报告
    3. 生成可视化图表
    4. 导出分析结果和建议
    """
    
    def __init__(self, feature_names: List[str], save_dir: str = "./temporal_analysis"):
        self.feature_names = feature_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化组件
        self.tracker = FeatureIoUCorrelationTracker(feature_names)
        self.learner = DynamicMappingLearner(len(feature_names))
        
        # 分析结果存储
        self.analysis_history = []
        self.recommendations = []
        
    def analyze_batch(self, features_dict: Dict[str, torch.Tensor], 
                     iou_targets: torch.Tensor, epoch: int, batch_idx: int,
                     current_loss: float = 0.0):
        """
        分析单个batch的特征-IoU关系
        
        Args:
            features_dict: 特征字典
            iou_targets: IoU目标值
            epoch: 当前epoch
            batch_idx: batch索引
            current_loss: 当前损失值
        """
        # 更新相关性追踪
        self.tracker.update(features_dict, iou_targets, epoch, batch_idx)
        
        # 获取当前分析结果
        correlation_trends = self.tracker.get_correlation_trends()
        
        # 记录分析结果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch_idx,
            'loss': current_loss,
            'correlation_trends': correlation_trends,
            'feature_correlations': self.tracker.correlation_cache
        }
        
        self.analysis_history.append(analysis_result)
        
        # 生成建议（每100个batch生成一次）
        if batch_idx % 100 == 0:
            self._generate_recommendations(correlation_trends)
    
    def _generate_recommendations(self, correlation_trends: Dict):
        """生成优化建议"""
        recommendations = []
        timestamp = datetime.now().isoformat()
        
        for feature_name, trends in correlation_trends.items():
            # 检查不稳定的特征
            if trends['stability'] < 0.5:
                recommendations.append({
                    'type': 'instability_warning',
                    'feature': feature_name,
                    'message': f"特征 {feature_name} 相关性不稳定 (稳定性: {trends['stability']:.3f})",
                    'suggestion': "考虑降低该特征权重或增加正则化",
                    'timestamp': timestamp
                })
            
            # 检查相关性急剧下降的特征
            if trends['trend'] == 'decreasing' and trends['recent_change'] > 0.2:
                recommendations.append({
                    'type': 'correlation_drop',
                    'feature': feature_name,
                    'message': f"特征 {feature_name} 相关性急剧下降 (变化: {trends['recent_change']:.3f})",
                    'suggestion': "可能需要特征重新校准或替换",
                    'timestamp': timestamp
                })
            
            # 检查高价值稳定特征
            if trends['stability'] > 0.8 and abs(trends['current']) > 0.5:
                recommendations.append({
                    'type': 'valuable_feature',
                    'feature': feature_name,
                    'message': f"特征 {feature_name} 高价值且稳定 (相关性: {trends['current']:.3f})",
                    'suggestion': "建议增加该特征权重",
                    'timestamp': timestamp
                })
        
        self.recommendations.extend(recommendations)
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合分析报告"""
        if not self.analysis_history:
            return {"error": "没有分析数据"}
        
        latest_analysis = self.analysis_history[-1]
        correlation_trends = latest_analysis['correlation_trends']
        
        # 特征分类
        stable_features = []
        unstable_features = []
        high_correlation_features = []
        low_correlation_features = []
        
        for name, trends in correlation_trends.items():
            if trends['stability'] > 0.7:
                stable_features.append(name)
            else:
                unstable_features.append(name)
                
            if abs(trends['current']) > 0.4:
                high_correlation_features.append(name)
            else:
                low_correlation_features.append(name)
        
        # 生成报告
        report = {
            'analysis_summary': {
                'total_features': len(self.feature_names),
                'stable_features': len(stable_features),
                'unstable_features': len(unstable_features),
                'high_correlation_features': len(high_correlation_features),
                'low_correlation_features': len(low_correlation_features)
            },
            'feature_classification': {
                'stable': stable_features,
                'unstable': unstable_features,
                'high_correlation': high_correlation_features,
                'low_correlation': low_correlation_features
            },
            'recommendations': self.recommendations[-10:],  # 最近10条建议
            'correlation_trends': correlation_trends,
            'temporal_patterns': self._analyze_temporal_patterns()
        }
        
        return report
    
    def _analyze_temporal_patterns(self) -> Dict:
        """分析时序模式"""
        if len(self.analysis_history) < 10:
            return {"message": "数据不足，无法分析时序模式"}
        
        patterns = {}
        
        # 分析每个特征的时序模式
        for feature_name in self.feature_names:
            correlations = []
            epochs = []
            
            for analysis in self.analysis_history:
                if feature_name in analysis['feature_correlations']:
                    correlations.append(analysis['feature_correlations'][feature_name])
                    epochs.append(analysis['epoch'])
            
            if len(correlations) > 5:
                # 检测周期性
                patterns[feature_name] = {
                    'correlation_range': [min(correlations), max(correlations)],
                    'variance': np.var(correlations),
                    'trend_direction': self._detect_overall_trend(correlations),
                    'change_points': self._detect_change_points(correlations, epochs)
                }
        
        return patterns
    
    def _detect_overall_trend(self, correlations: List[float]) -> str:
        """检测整体趋势"""
        if len(correlations) < 3:
            return "insufficient_data"
        
        # 线性回归检测趋势
        x = np.arange(len(correlations))
        slope = np.polyfit(x, correlations, 1)[0]
        
        if slope > 0.05:
            return "strongly_increasing"
        elif slope > 0.01:
            return "increasing"
        elif slope < -0.05:
            return "strongly_decreasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_change_points(self, correlations: List[float], epochs: List[int]) -> List[Dict]:
        """检测变化点"""
        change_points = []
        threshold = 0.2
        
        for i in range(1, len(correlations)):
            change = abs(correlations[i] - correlations[i-1])
            if change > threshold:
                change_points.append({
                    'epoch': epochs[i],
                    'change_magnitude': change,
                    'before': correlations[i-1],
                    'after': correlations[i]
                })
        
        return change_points
    
    def save_analysis(self, filename: str = None):
        """保存分析结果"""
        if filename is None:
            filename = f"temporal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # 准备保存数据
        save_data = {
            'analysis_history': self.analysis_history,
            'recommendations': self.recommendations,
            'comprehensive_report': self.generate_comprehensive_report(),
            'metadata': {
                'feature_names': self.feature_names,
                'total_analyses': len(self.analysis_history),
                'save_timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {filepath}")
        return filepath
    
    def visualize_correlations(self, feature_names: List[str] = None, save_plot: bool = True):
        """可视化相关性变化"""
        if not self.analysis_history:
            print("没有可视化数据")
            return
        
        if feature_names is None:
            # 选择最有趣的特征进行可视化
            latest_trends = self.analysis_history[-1]['correlation_trends']
            # 选择相关性变化最大的前8个特征
            sorted_features = sorted(latest_trends.items(), 
                                   key=lambda x: x[1].get('recent_change', 0), 
                                   reverse=True)
            feature_names = [name for name, _ in sorted_features[:8]]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('时序特征-IoU相关性分析', fontsize=16)
        
        # 1. 相关性时序变化
        ax1 = axes[0, 0]
        for feature_name in feature_names[:4]:  # 显示前4个特征
            correlations = []
            epochs = []
            for analysis in self.analysis_history:
                if feature_name in analysis['feature_correlations']:
                    correlations.append(analysis['feature_correlations'][feature_name])
                    epochs.append(analysis['epoch'])
            
            if correlations:
                ax1.plot(epochs, correlations, label=feature_name, marker='o', markersize=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Correlation with IoU')
        ax1.set_title('特征-IoU相关性时序变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 相关性分布
        ax2 = axes[0, 1]
        current_correlations = []
        labels = []
        for name in feature_names[:8]:
            if name in self.analysis_history[-1]['feature_correlations']:
                current_correlations.append(abs(self.analysis_history[-1]['feature_correlations'][name]))
                labels.append(name.replace('_', '\n'))
        
        bars = ax2.bar(range(len(current_correlations)), current_correlations)
        ax2.set_xlabel('特征')
        ax2.set_ylabel('|相关性|')
        ax2.set_title('当前特征相关性强度')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # 为柱状图添加颜色编码
        for i, bar in enumerate(bars):
            if current_correlations[i] > 0.5:
                bar.set_color('green')
            elif current_correlations[i] > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 3. 稳定性分析
        ax3 = axes[1, 0]
        stability_scores = []
        stability_labels = []
        latest_trends = self.analysis_history[-1]['correlation_trends']
        
        for name in feature_names[:8]:
            if name in latest_trends:
                stability_scores.append(latest_trends[name].get('stability', 0))
                stability_labels.append(name.replace('_', '\n'))
        
        ax3.scatter(range(len(stability_scores)), stability_scores, 
                   s=100, c=stability_scores, cmap='RdYlGn', alpha=0.7)
        ax3.set_xlabel('特征')
        ax3.set_ylabel('稳定性分数')
        ax3.set_title('特征稳定性分析')
        ax3.set_xticks(range(len(stability_labels)))
        ax3.set_xticklabels(stability_labels, rotation=45, ha='right')
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='稳定性阈值')
        ax3.legend()
        
        # 4. 变化点检测
        ax4 = axes[1, 1]
        change_counts = {}
        for name in feature_names[:8]:
            change_counts[name] = len(self.tracker.change_points.get(name, []))
        
        names = list(change_counts.keys())
        counts = list(change_counts.values())
        
        ax4.bar(range(len(names)), counts)
        ax4.set_xlabel('特征')
        ax4.set_ylabel('显著变化次数')
        ax4.set_title('特征变化点统计')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, f"correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存到: {plot_path}")
        
        plt.show()
        
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        report = self.generate_comprehensive_report()
        suggestions = []
        
        # 基于分析结果生成具体建议
        if report['analysis_summary']['unstable_features'] > report['analysis_summary']['stable_features']:
            suggestions.append("🚨 不稳定特征过多，建议增加特征正则化或时序校准强度")
        
        if report['analysis_summary']['low_correlation_features'] > report['analysis_summary']['high_correlation_features']:
            suggestions.append("📉 低相关性特征过多，建议重新设计特征提取器或增加特征筛选")
        
        # 基于时序模式的建议
        temporal_patterns = report['temporal_patterns']
        for feature_name, pattern in temporal_patterns.items():
            if pattern['trend_direction'] == 'strongly_decreasing':
                suggestions.append(f"⚠️ 特征 {feature_name} 相关性持续下降，建议调整或替换")
            elif pattern['trend_direction'] == 'strongly_increasing' and pattern['variance'] < 0.1:
                suggestions.append(f"✅ 特征 {feature_name} 表现优秀且稳定，建议增加权重")
        
        return suggestions

# 使用示例和测试函数
def create_test_analyzer():
    """创建测试分析器"""
    # 模拟特征名称（基于LearnableFeatureCombiner的特征）
    feature_names = [
        'geometric_horizontal_balance', 'geometric_vertical_balance', 'geometric_aspect_ratio',
        'geometric_confidence', 'geometric_size_score', 'info_cross_entropy_quality',
        'info_kl_divergence', 'info_mutual_information', 'topo_perspective_invariance',
        'topo_connectivity', 'topo_constraint_satisfaction', 'topo_shape_regularity',
        'left_distance', 'top_distance', 'right_distance', 'bottom_distance',
        'distance_consistency', 'left_entropy', 'top_entropy', 'right_entropy', 'bottom_entropy'
    ]
    
    analyzer = TemporalFeatureIoUAnalyzer(feature_names)
    return analyzer

if __name__ == "__main__":
    # 创建测试分析器
    analyzer = create_test_analyzer()
    
    print("🕒 时序特征-IoU关系分析器已初始化")
    print(f"📊 追踪特征数量: {len(analyzer.feature_names)}")
    print("🔍 准备开始分析特征与IoU的动态关系...")
    
    # 生成使用说明
    print("\n📝 使用说明:")
    print("1. 在训练循环中调用 analyzer.analyze_batch() 来实时追踪")
    print("2. 使用 analyzer.generate_comprehensive_report() 获取分析报告")
    print("3. 使用 analyzer.visualize_correlations() 生成可视化图表")
    print("4. 使用 analyzer.get_optimization_suggestions() 获取优化建议")