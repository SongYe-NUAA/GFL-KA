"""
MoE增强的分布引导质量评估器
结合专家混合思想的轻量级高级LQE实现

Author: Advanced AI Assistant  
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class LightweightExpert(nn.Module):
    """
    轻量级专家网络
    每个专家专注于特定的质量评估任务
    """
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 32,
                 expert_type: str = "general"):
        super().__init__()
        self.expert_type = expert_type
        
        # 🎯 极简专家网络设计（显存友好）
        self.expert_layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=False),  # 1x1卷积降维
            nn.GroupNorm(min(8, hidden_dim//4), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, 1, bias=False),
            nn.GroupNorm(min(4, hidden_dim//8), hidden_dim//2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)  # 输出质量分数
        )
        
        # 专家特化初始化
        self._initialize_expert_weights()
    
    def _initialize_expert_weights(self):
        """根据专家类型进行特化初始化"""
        init_scale = {
            'semantic': 0.02,    # 语义专家：保守初始化
            'geometric': 0.01,   # 几何专家：更保守
            'distribution': 0.03, # 分布专家：稍激进
            'consistency': 0.015, # 一致性专家：中等
            'general': 0.02      # 通用专家：标准
        }.get(self.expert_type, 0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [N, C, H, W] 输入特征
        Returns:
            质量分数 [N, 1, H, W]
        """
        return torch.sigmoid(self.expert_layers(x))


class AdaptiveRouter(nn.Module):
    """
    自适应路由器
    根据输入特征智能选择专家组合
    """
    def __init__(self, 
                 input_dim: int = 128,
                 num_experts: int = 5,
                 top_k: int = 2,
                 temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature
        
        # 🎯 轻量级路由网络
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化降维
            nn.Conv2d(input_dim, input_dim//4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_dim//4, num_experts, 1)
        )
        
        # 🚀 可学习温度参数
        self.learnable_temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        路由决策
        Args:
            x: [N, C, H, W] 输入特征
        Returns:
            expert_weights: [N, num_experts, H, W] 专家权重
            routing_loss: 路由损失（用于负载均衡）
        """
        # 计算路由logits
        route_logits = self.router(x)  # [N, num_experts, 1, 1]
        
        # 温度缩放
        scaled_logits = route_logits / self.learnable_temperature.clamp(min=0.1, max=10.0)
        
        # Top-K门控（显存友好的稀疏激活）
        if self.top_k < self.num_experts:
            # 只激活Top-K个专家
            top_k_logits, top_k_indices = torch.topk(scaled_logits, self.top_k, dim=1)
            expert_weights = torch.zeros_like(scaled_logits)
            expert_weights.scatter_(1, top_k_indices, F.softmax(top_k_logits, dim=1))
        else:
            expert_weights = F.softmax(scaled_logits, dim=1)
        
        # 扩展到特征图尺寸
        N, _, H, W = x.shape
        expert_weights = expert_weights.expand(N, self.num_experts, H, W)
        
        # 计算负载均衡损失
        routing_loss = self._compute_load_balance_loss(expert_weights)
        
        return expert_weights, routing_loss
    
    def _compute_load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        mean_usage = expert_weights.mean(dim=(0, 2, 3))  # [num_experts]
        
        # 理想情况下每个专家使用率应该相等
        ideal_usage = 1.0 / self.num_experts
        
        # L2正则化鼓励均匀使用
        load_balance_loss = F.mse_loss(mean_usage, torch.full_like(mean_usage, ideal_usage))
        
        return load_balance_loss * 0.01  # 小权重避免干扰主任务


class MoEEnhancedDistributionGuidedQualityEstimator(nn.Module):
    """
    MoE增强的分布引导质量评估器
    轻量级设计，显存友好
    """
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 80,
                 reg_max: int = 16,
                 num_experts: int = 5,
                 expert_top_k: int = 2,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_experts = num_experts
        
        # 🎯 输入特征融合（轻量级设计）
        total_input_dim = in_channels + num_classes + 4*(reg_max+1)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_input_dim, hidden_dim*2, 1, bias=False),  # 1x1卷积降维
            nn.GroupNorm(32, hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim, 1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # 🚀 专家系统
        expert_types = ['semantic', 'geometric', 'distribution', 'consistency', 'general']
        self.experts = nn.ModuleList([
            LightweightExpert(
                input_dim=hidden_dim,
                hidden_dim=32,
                expert_type=expert_types[i % len(expert_types)]
            ) for i in range(num_experts)
        ])
        
        # 🎨 自适应路由器
        self.router = AdaptiveRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            top_k=expert_top_k
        )
        
        # 🔥 分布分析模块（复用现有逻辑）
        self.distribution_analyzer = nn.Sequential(
            nn.Conv2d(4*(reg_max+1), 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True)
        )
        
        # 🌟 最终质量合成（超轻量级）
        self.quality_synthesizer = nn.Sequential(
            nn.Conv2d(1 + 16, 16, 1, bias=False),  # MoE输出 + 分布特征
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 统计信息
        self.register_buffer('expert_usage_stats', torch.zeros(num_experts))
        self.register_buffer('forward_count', torch.tensor(0))
        
    def forward(self, 
                features: torch.Tensor,
                cls_logits: torch.Tensor, 
                bbox_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        Args:
            features: [N, in_channels, H, W] 骨干网络特征
            cls_logits: [N, num_classes, H, W] 分类logits
            bbox_pred: [N, 4*(reg_max+1), H, W] 边界框预测
        Returns:
            final_quality: [N, 1, H, W] 最终质量分数
            aux_info: 辅助信息字典
        """
        N, _, H, W = features.shape
        
        # 🎯 特征融合
        combined_features = torch.cat([
            features, 
            cls_logits, 
            bbox_pred
        ], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 🚀 路由决策
        expert_weights, routing_loss = self.router(fused_features)
        
        # 🎨 专家计算（稀疏激活，显存友好）
        expert_outputs = []
        active_experts = []
        
        for i, expert in enumerate(self.experts):
            # 检查专家是否被激活
            expert_weight = expert_weights[:, i:i+1, :, :]
            if expert_weight.max() > 1e-6:  # 只计算被激活的专家
                expert_output = expert(fused_features)
                expert_outputs.append(expert_output * expert_weight)
                active_experts.append(i)
        
        # MoE输出融合
        if expert_outputs:
            moe_output = sum(expert_outputs)
        else:
            moe_output = torch.zeros(N, 1, H, W, device=features.device)
        
        # 🔥 分布分析
        dist_features = self.distribution_analyzer(bbox_pred)
        
        # 🌟 最终质量合成
        synthesis_input = torch.cat([moe_output, dist_features], dim=1)
        final_quality = self.quality_synthesizer(synthesis_input)
        
        # 更新统计信息
        if self.training:
            self._update_stats(expert_weights, active_experts)
        
        # 构建辅助信息
        aux_info = {
            'expert_weights': expert_weights,
            'routing_loss': routing_loss,
            'active_experts': active_experts,
            'moe_output': moe_output,
            'distribution_features': dist_features,
            'expert_usage_stats': self.expert_usage_stats.clone()
        }
        
        return final_quality, aux_info
    
    def _update_stats(self, expert_weights: torch.Tensor, active_experts: list):
        """更新专家使用统计"""
        self.forward_count += 1
        
        # 更新专家使用统计
        usage = expert_weights.mean(dim=(0, 2, 3))  # [num_experts]
        self.expert_usage_stats = (self.expert_usage_stats * 0.99 + usage * 0.01)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取显存使用情况"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算显存使用（MB）
        param_memory = total_params * 4 / (1024**2)  # 假设float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_memory_mb': param_memory,
            'expert_count': self.num_experts,
            'active_expert_ratio': len([x for x in self.expert_usage_stats if x > 0.01]) / self.num_experts
        }
    
    def print_expert_stats(self):
        """打印专家使用统计"""
        print("\n🚀 MoE Expert Usage Statistics:")
        print("-" * 50)
        for i, usage in enumerate(self.expert_usage_stats):
            expert_type = ['semantic', 'geometric', 'distribution', 'consistency', 'general'][i % 5]
            print(f"Expert {i} ({expert_type}): {usage:.4f} ({usage*100:.2f}%)")
        print(f"Total Forward Calls: {self.forward_count}")
        print("-" * 50)


class MoEQualityLoss(nn.Module):
    """
    MoE质量损失函数
    包含质量损失和路由损失
    """
    def __init__(self, 
                 quality_loss_weight: float = 1.0,
                 routing_loss_weight: float = 0.1):
        super().__init__()
        self.quality_loss_weight = quality_loss_weight
        self.routing_loss_weight = routing_loss_weight
    
    def forward(self, 
                quality_pred: torch.Tensor,
                quality_target: torch.Tensor,
                routing_loss: torch.Tensor,
                pos_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算MoE质量损失
        Args:
            quality_pred: [N, 1, H, W] 预测质量
            quality_target: [N, H, W] 目标质量（IoU）
            routing_loss: 路由负载均衡损失
            pos_mask: [N, H, W] 正样本mask
        """
        # 质量损失（只在正样本上计算）
        if pos_mask.sum() > 0:
            pos_quality_pred = quality_pred.squeeze(1)[pos_mask]
            pos_quality_target = quality_target[pos_mask]
            
            # 使用平滑L1损失
            quality_loss = F.smooth_l1_loss(
                pos_quality_pred, 
                pos_quality_target, 
                reduction='mean'
            )
        else:
            quality_loss = torch.tensor(0.0, device=quality_pred.device)
        
        # 总损失
        total_loss = (self.quality_loss_weight * quality_loss + 
                     self.routing_loss_weight * routing_loss)
        
        loss_dict = {
            'quality_loss': quality_loss,
            'routing_loss': routing_loss,
            'total_moe_loss': total_loss
        }
        
        return total_loss, loss_dict


# 工具函数
def create_lightweight_moe_lqe(config: Dict) -> MoEEnhancedDistributionGuidedQualityEstimator:
    """
    创建轻量级MoE LQE的工厂函数
    """
    return MoEEnhancedDistributionGuidedQualityEstimator(
        in_channels=config.get('in_channels', 256),
        num_classes=config.get('num_classes', 80),
        reg_max=config.get('reg_max', 16),
        num_experts=config.get('num_experts', 5),
        expert_top_k=config.get('expert_top_k', 2),
        hidden_dim=config.get('hidden_dim', 128)
    )


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    config = {
        'in_channels': 256,
        'num_classes': 80, 
        'reg_max': 16,
        'num_experts': 5,
        'expert_top_k': 2,
        'hidden_dim': 128
    }
    
    model = create_lightweight_moe_lqe(config).to(device)
    
    # 测试输入
    batch_size, H, W = 2, 32, 32
    features = torch.randn(batch_size, 256, H, W).to(device)
    cls_logits = torch.randn(batch_size, 80, H, W).to(device)
    bbox_pred = torch.randn(batch_size, 4*17, H, W).to(device)
    
    # 前向传播
    with torch.no_grad():
        quality_pred, aux_info = model(features, cls_logits, bbox_pred)
    
    print("🚀 MoE Enhanced LQE Test Results:")
    print(f"Input shape: {features.shape}")
    print(f"Output quality shape: {quality_pred.shape}")
    print(f"Quality range: [{quality_pred.min():.4f}, {quality_pred.max():.4f}]")
    print(f"Active experts: {aux_info['active_experts']}")
    print(f"Routing loss: {aux_info['routing_loss']:.6f}")
    
    # 显存使用情况
    memory_info = model.get_memory_usage()
    print(f"\n💾 Memory Usage:")
    for key, value in memory_info.items():
        print(f"{key}: {value}")
    
    model.print_expert_stats()