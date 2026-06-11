#!/usr/bin/env python3
"""
🚀 动态特征选择器：根据输入内容自适应选择最优特征组合

核心创新：
1. 上下文感知特征选择：根据当前样本的特征分布动态调整权重
2. 多尺度特征重要性评估：在不同空间尺度上评估特征重要性
3. 自适应阈值特征过滤：动态过滤低质量特征
4. 特征互补性学习：学习特征间的互补关系而非简单加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class ContextAwareFeatureSelector(nn.Module):
    """
    🎯 上下文感知的动态特征选择器
    根据当前输入的特征分布动态调整特征权重
    """
    
    def __init__(self, num_features, context_dim=64, num_heads=4):
        super().__init__()
        self.num_features = num_features
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        # 上下文编码器：将输入特征编码为上下文向量
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(num_features, context_dim, 1),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_dim, context_dim, 1),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True)
        )
        
        # 多头注意力特征选择器
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 特征重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(context_dim // 2, num_features),
            nn.Sigmoid()
        )
        
        # 自适应阈值学习
        self.threshold_learner = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feature_stack):
        """
        Args:
            feature_stack: [N, num_features, H, W]
        Returns:
            selected_features: [N, num_features, H, W]
            selection_weights: [N, num_features]
            selection_mask: [N, num_features] (二值掩码)
        """
        N, C, H, W = feature_stack.shape
        
        # 1. 提取上下文信息
        context = self.context_encoder(feature_stack)  # [N, context_dim, 1, 1]
        context_flat = context.flatten(2).transpose(1, 2)  # [N, 1, context_dim]
        
        # 2. 多头注意力特征选择
        # 创建特征查询向量
        feature_queries = context_flat.expand(-1, self.num_features, -1)  # [N, num_features, context_dim]
        
        # 自注意力计算特征重要性
        attended_features, attention_weights = self.feature_attention(
            feature_queries, feature_queries, feature_queries
        )  # [N, num_features, context_dim], [N, num_features, num_features]
        
        # 3. 预测特征重要性权重
        context_summary = attended_features.mean(dim=1)  # [N, context_dim]
        importance_weights = self.importance_predictor(context_summary)  # [N, num_features]
        
        # 4. 自适应阈值过滤
        adaptive_threshold = self.threshold_learner(context_summary)  # [N, 1]
        selection_mask = (importance_weights > adaptive_threshold).float()  # [N, num_features]
        
        # 5. 应用选择权重
        final_weights = importance_weights * selection_mask
        
        # 归一化权重（确保至少有一个特征被选中）
        final_weights = final_weights + 1e-8  # 避免全零
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True)
        
        # 扩展权重到空间维度
        weight_expanded = final_weights.view(N, C, 1, 1).expand(-1, -1, H, W)
        selected_features = feature_stack * weight_expanded
        
        return selected_features, final_weights, selection_mask

class MultiScaleFeatureEvaluator(nn.Module):
    """
    🔍 多尺度特征重要性评估器
    在不同空间尺度上评估特征对IoU预测的贡献
    """
    
    def __init__(self, num_features, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.num_features = num_features
        
        # 为每个尺度创建评估器
        self.scale_evaluators = nn.ModuleList()
        for scale in scales:
            evaluator = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),  # 不同尺度的池化
                nn.Conv2d(num_features, num_features // 2, 1),
                nn.BatchNorm2d(num_features // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features // 2, num_features, 1),
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Flatten(),
                nn.Sigmoid()
            )
            self.scale_evaluators.append(evaluator)
        
        # 尺度融合网络
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_features * len(scales), num_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, feature_stack):
        """
        Args:
            feature_stack: [N, num_features, H, W]
        Returns:
            scale_aware_weights: [N, num_features]
        """
        scale_scores = []
        
        # 在不同尺度上评估特征重要性
        for evaluator in self.scale_evaluators:
            scale_score = evaluator(feature_stack)  # [N, num_features]
            scale_scores.append(scale_score)
        
        # 融合多尺度信息
        combined_scores = torch.cat(scale_scores, dim=1)  # [N, num_features * len(scales)]
        scale_aware_weights = self.scale_fusion(combined_scores)  # [N, num_features]
        
        return scale_aware_weights

class ComplementarityAwareSelector(nn.Module):
    """
    🤝 互补性感知特征选择器
    学习特征间的互补关系，选择最优特征组合
    """
    
    def __init__(self, num_features, hidden_dim=128):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # 特征关系建模
        self.relation_encoder = nn.Sequential(
            nn.Linear(num_features * num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 互补性评分网络
        self.complementarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_features),
            nn.Sigmoid()
        )
        
        # 冗余性惩罚网络
        self.redundancy_penalty = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, feature_stack):
        """
        Args:
            feature_stack: [N, num_features, H, W]
        Returns:
            complementarity_weights: [N, num_features]
        """
        N, C, H, W = feature_stack.shape
        
        # 计算特征间的相关性矩阵
        features_flat = feature_stack.view(N, C, -1)  # [N, C, H*W]
        
        # 计算特征相关性
        correlation_matrices = []
        for i in range(N):
            feat = features_flat[i]  # [C, H*W]
            # 计算特征间的余弦相似度
            feat_norm = F.normalize(feat, p=2, dim=1)
            corr_matrix = torch.mm(feat_norm, feat_norm.t())  # [C, C]
            correlation_matrices.append(corr_matrix.flatten())
        
        correlation_features = torch.stack(correlation_matrices, dim=0)  # [N, C*C]
        
        # 编码特征关系
        relation_features = self.relation_encoder(correlation_features)  # [N, hidden_dim//2]
        
        # 计算互补性分数和冗余性惩罚
        complementarity_scores = self.complementarity_scorer(relation_features)  # [N, C]
        redundancy_penalties = self.redundancy_penalty(relation_features)  # [N, C]
        
        # 结合互补性和冗余性
        complementarity_weights = complementarity_scores * (1 - redundancy_penalties)
        
        # 归一化
        complementarity_weights = complementarity_weights / (complementarity_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return complementarity_weights

class EnhancedDynamicFeatureSelector(nn.Module):
    """
    🚀 增强的动态特征选择器
    整合多种选择策略的综合特征选择器
    """
    
    def __init__(self, num_features, context_dim=64, num_heads=4, scales=[1, 2, 4]):
        super().__init__()
        self.num_features = num_features
        
        # 各种选择器组件
        self.context_selector = ContextAwareFeatureSelector(num_features, context_dim, num_heads)
        self.multiscale_evaluator = MultiScaleFeatureEvaluator(num_features, scales)
        self.complementarity_selector = ComplementarityAwareSelector(num_features)
        
        # 选择策略融合网络
        self.strategy_fusion = nn.Sequential(
            nn.Linear(num_features * 3, num_features * 2),
            nn.BatchNorm1d(num_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()
        )
        
        # 自适应策略权重
        self.strategy_weights = nn.Parameter(torch.ones(3) / 3)  # 三种策略的权重
        
    def forward(self, feature_stack):
        """
        Args:
            feature_stack: [N, num_features, H, W]
        Returns:
            enhanced_features: [N, num_features, H, W]
            selection_info: Dict containing selection details
        """
        N, C, H, W = feature_stack.shape
        
        # 1. 上下文感知选择
        _, context_weights, context_mask = self.context_selector(feature_stack)
        
        # 2. 多尺度评估
        scale_weights = self.multiscale_evaluator(feature_stack)
        
        # 3. 互补性感知选择
        complementarity_weights = self.complementarity_selector(feature_stack)
        
        # 4. 策略融合
        # 应用策略权重
        strategy_weights_norm = F.softmax(self.strategy_weights, dim=0)
        
        combined_weights = (
            strategy_weights_norm[0] * context_weights +
            strategy_weights_norm[1] * scale_weights +
            strategy_weights_norm[2] * complementarity_weights
        )
        
        # 进一步融合优化
        all_weights = torch.cat([context_weights, scale_weights, complementarity_weights], dim=1)
        refined_weights = self.strategy_fusion(all_weights)
        
        # 🚀 增强动态性：真正的自适应选择策略
        # 核心思想：不同样本应该有显著不同的选择策略
        
        # 1. 样本自适应锐化：根据权重分布特性调整锐化强度
        # 计算每个样本的权重分布熵，熵高说明分布平均，需要更强锐化
        weight_entropy = -(refined_weights * torch.log(refined_weights + 1e-8)).sum(dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(self.num_features, device=refined_weights.device))
        entropy_ratio = weight_entropy / max_entropy  # [N, 1] 归一化熵比例
        
        # 自适应锐化因子：熵高的样本需要更强锐化
        adaptive_sharpening = 1.0 + 2.0 * entropy_ratio  # 范围[1.0, 3.0]
        refined_weights_sharpened = torch.pow(refined_weights, 1.0 / adaptive_sharpening)
        combined_weights_sharpened = torch.pow(combined_weights, 1.0 / adaptive_sharpening)
        
        # 重新归一化锐化后的权重
        refined_weights_sharpened = refined_weights_sharpened / (refined_weights_sharpened.sum(dim=1, keepdim=True) + 1e-8)
        combined_weights_sharpened = combined_weights_sharpened / (combined_weights_sharpened.sum(dim=1, keepdim=True) + 1e-8)
        
        # 2. 动态Top-K选择：根据权重分布的集中度调整选择数量
        # 计算权重分布的集中度（基尼系数），集中度高的样本选择更少特征
        sorted_weights, _ = torch.sort(refined_weights_sharpened, dim=1, descending=True)
        cumsum_weights = torch.cumsum(sorted_weights, dim=1)
        total_weights = cumsum_weights[:, -1:] + 1e-8
        concentration = (cumsum_weights / total_weights)  # [N, num_features]
        
        # 找到累积权重达到80%的位置，作为该样本的动态K值
        threshold_80 = 0.8
        dynamic_k = torch.argmax((concentration >= threshold_80).float(), dim=1) + 1  # [N]
        dynamic_k = torch.clamp(dynamic_k, min=int(self.num_features * 0.3), 
                               max=int(self.num_features * 0.8))  # 限制在30%-80%范围
        
        # 为每个样本应用不同的Top-K
        top_k_mask = torch.zeros_like(refined_weights_sharpened)
        for i in range(N):
            k = dynamic_k[i].item()
            _, indices = torch.topk(refined_weights_sharpened[i], k)
            top_k_mask[i, indices] = 1.0
        
        # 3. 自适应阈值过滤：基于每个样本的权重分布特性
        # 使用分位数而非均值，避免被异常值影响
        weight_25th = torch.quantile(refined_weights_sharpened, q=0.25, dim=1, keepdim=True)
        weight_75th = torch.quantile(refined_weights_sharpened, q=0.75, dim=1, keepdim=True)
        adaptive_threshold = weight_25th + 0.7 * (weight_75th - weight_25th)  # 高于75%分位数附近
        
        threshold_mask = (refined_weights_sharpened > adaptive_threshold).float()
        
        # 4. 智能掩码组合：确保每个样本至少保留足够特征
        selection_mask = top_k_mask * threshold_mask
        
        # 如果过滤后特征太少，回退到Top-K掩码
        min_features = int(self.num_features * 0.25)  # 至少保留25%特征
        features_selected = selection_mask.sum(dim=1)  # [N]
        insufficient_mask = (features_selected < min_features).unsqueeze(1)  # [N, 1]
        selection_mask = torch.where(insufficient_mask, top_k_mask, selection_mask)
        
        # 5. 样本特定的权重混合：不同样本使用不同的策略权重
        # 基于上下文感知权重的方差调整混合比例
        context_variance = context_weights.var(dim=1, keepdim=True)  # [N, 1]
        adaptive_mix_ratio = 0.5 + 0.4 * torch.sigmoid(context_variance * 10)  # [N, 1] 范围[0.5, 0.9]
        
        final_weights = (
            (1 - adaptive_mix_ratio) * combined_weights_sharpened + 
            adaptive_mix_ratio * refined_weights_sharpened
        ) * selection_mask
        
        # 6. 避免过度平滑：使用更温和的归一化
        # 直接归一化而不使用softmax，保持权重的相对差异
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 6. 应用权重
        weight_expanded = final_weights.view(N, C, 1, 1).expand(-1, -1, H, W)
        enhanced_features = feature_stack * weight_expanded
        
        # 返回选择信息
        selection_info = {
            'context_weights': context_weights,
            'scale_weights': scale_weights,
            'complementarity_weights': complementarity_weights,
            'final_weights': final_weights,
            'strategy_weights': strategy_weights_norm,
            'selection_mask': selection_mask,
            'top_k_mask': top_k_mask,
            'threshold_mask': threshold_mask,
            'sharpened_weights': refined_weights_sharpened,
            # 🚀 新增：动态选择参数
            'dynamic_k': dynamic_k,  # 每个样本的动态K值
            'adaptive_sharpening': adaptive_sharpening,  # 每个样本的自适应锐化因子
            'entropy_ratio': entropy_ratio,  # 权重分布熵比例
            'adaptive_threshold': adaptive_threshold,  # 自适应阈值
            'adaptive_mix_ratio': adaptive_mix_ratio  # 样本特定的混合比例
        }
        
        return enhanced_features, selection_info
    
    def get_feature_importance_ranking(self, feature_stack):
        """获取特征重要性排序"""
        with torch.no_grad():
            _, selection_info = self.forward(feature_stack)
            final_weights = selection_info['final_weights']
            
            # 计算平均重要性
            avg_importance = final_weights.mean(dim=0)  # [num_features]
            
            # 排序
            sorted_importance, sorted_indices = torch.sort(avg_importance, descending=True)
            
            return sorted_indices, sorted_importance