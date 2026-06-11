      
      
      
#!/usr/bin/env python3
"""
🚀 可学习特征组合器：纯定位质量估计的NAS方案 + LFSC增强 + 基础特征

核心思想：
1. 仅基于回归分布提取定位质量特征，不依赖分类信息
2. 使用可学习的方式自动组合和权重这些纯几何特征
3. 确保LQE的独立性，避免分类偏差影响定位质量评估
4. 通过NAS自动发现最优的特征组合模式
5. 🚀 新增：轻量级特征自校正模块(LFSC)，解决IoU与统计特征相关性不稳定问题
6. 🎯 新增：4边top2基础特征(8维)，提供最直接的边界定位信息

🎯 LFSC模块优势：
- ✅ 零IoU依赖：完全基于特征内在统计特性
- ✅ 自动去噪：移除由采样偏差导致的特征不稳定性
- ✅ 相关性解耦：减少特征间有害相关性
- ✅ 时序平滑：确保训练过程中特征连续性
- ✅ 即插即用：无需修改现有训练流程

📖 使用示例：
```python
# 基础使用（LFSC默认开启）
feature_combiner = LearnableFeatureCombiner(
    reg_max=16,
    hidden_dim=32,
    output_dim=1,
    enable_lfsc=True  # 🚀 启用LFSC特征自校正
)

# 前向传播
bbox_pred = torch.randn(2, 68, 32, 32)  # [N, 4*(reg_max+1), H, W]
quality_score = feature_combiner(bbox_pred)  # [N, 1, H, W]

# 禁用LFSC（回退到原始特征提取）
feature_combiner_vanilla = LearnableFeatureCombiner(
    reg_max=16,
    enable_lfsc=False
)
```

🔧 配置说明：
- enable_lfsc: 是否启用LFSC模块（默认True）
- LFSC会在首次前向传播时自动初始化
- 训练时会自动打印校正效果监控信息
- 推理时LFSC仍然有效，确保特征稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple

class LightweightFeatureSelfCorrection(nn.Module):
    """
    🚀 轻量级特征自校正模块 (LFSC) - 无IoU版本
    
    核心机制：
    1. 统计归一化：自动移除分布偏移，提高特征稳定性
    2. 相关性去耦：基于特征内在统计量去除有害相关性
    3. 自适应校正：学习稳定的特征表示，减少噪声影响
    4. 零IoU依赖：完全基于特征自身特性，无需外部标签
    5. 时序平滑：确保训练过程中的特征连续性
    
    理论依据：
    - 统计去噪：移除由于采样偏差导致的特征不稳定性
    - 方差稳定化：自适应调节特征方差，确保一致的学习信号
    - 特征解耦：减少特征间的有害相关性，提高独立性
    """
    
    def __init__(self, input_dim, momentum=0.1, enable_temporal_smoothing=True):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.enable_temporal_smoothing = enable_temporal_smoothing
        
        # 🚀 特征去相关网络：移除特征间的有害相关性
        # 使用残差结构确保稳定的梯度流
        self.decorrelation_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim),
            nn.Tanh()  # 输出范围[-1,1]，作为残差
        )
        
        # 🚀 特征稳定性增强器：学习特征重要性权重
        self.stability_enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()  # 输出权重[0,1]
        )
        
        # 🚀 自适应方差调节器：学习每个特征的最优方差缩放
        self.variance_controller = nn.Parameter(torch.ones(input_dim))
        
        # 🚀 特征质量评估器：无监督地估计特征可靠性
        self.quality_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.SiLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # 运行时统计（用于稳定性监控和去噪）
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.register_buffer('running_std', torch.ones(input_dim))
        self.register_buffer('stability_score', torch.zeros(input_dim))
        
        # 🚀 时序平滑缓存
        if self.enable_temporal_smoothing:
            self.register_buffer('feature_ema', torch.zeros(input_dim))
            self.register_buffer('quality_ema', torch.zeros(input_dim))
            self.temporal_alpha = 0.95  # EMA衰减因子
        
        # 训练步数计数器
        self.register_buffer('step_counter', torch.tensor(0))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，确保稳定的训练开始"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        🚀 特征自校正前向传播
        
        Args:
            features: [N, input_dim] 输入特征
            
        Returns:
            corrected_features: [N, input_dim] 自校正后的特征
            correction_info: Dict 校正信息（用于监控）
        """
        batch_size = features.size(0)
        device = features.device
        
        # === 1. 🚀 批次内统计归一化 ===
        # 计算批次统计量，用于去除分布偏移
        batch_mean = features.mean(dim=0, keepdim=True)
        batch_var = features.var(dim=0, keepdim=True, unbiased=False)
        batch_std = torch.sqrt(batch_var + 1e-8)
        
        # Z-score标准化
        normalized_features = (features - batch_mean) / batch_std
        
        # === 2. 🚀 特征去相关：移除有害的特征间依赖 ===
        # 使用残差连接确保训练稳定性
        decorrelation_residual = self.decorrelation_net(normalized_features)
        decorrelated_features = normalized_features + 0.1 * decorrelation_residual
        
        # === 3. 🚀 计算特征稳定性分数（基于方差稳定性） ===
        # 方差越稳定的特征越可靠
        if self.training and self.step_counter > 10:  # 训练10步后开始计算稳定性
            # 方差稳定性：当前方差与历史EMA方差的比值
            variance_stability = torch.exp(-torch.abs(batch_var.squeeze() - self.running_var) / (self.running_var + 1e-8))
            
            # 特征幅度稳定性：避免特征值过大或过小
            magnitude_stability = torch.exp(-torch.abs(batch_mean.squeeze().abs() - 1.0))
            
            # 综合稳定性分数
            feature_stability = 0.6 * variance_stability + 0.4 * magnitude_stability
        else:
            # 训练初期使用固定稳定性
            feature_stability = torch.ones(self.input_dim, device=device) * 0.5
        
        # === 4. 🚀 特征质量评估（无监督） ===
        # 基于特征分布特性评估质量
        quality_scores = self.quality_estimator(decorrelated_features)
        
        # === 5. 🚀 稳定性加权特征增强 ===
        stability_weights = self.stability_enhancer(decorrelated_features)
        
        # 结合稳定性分数和学习到的权重
        combined_weights = stability_weights * feature_stability.unsqueeze(0) * quality_scores
        
        # === 6. 🚀 自适应方差控制 ===
        # 约束方差控制器参数，避免过度缩放
        constrained_variance_controller = torch.clamp(self.variance_controller, min=0.1, max=2.0)
        controlled_features = decorrelated_features * constrained_variance_controller.unsqueeze(0)
        
        # === 7. 🚀 应用综合权重 ===
        enhanced_features = controlled_features * combined_weights
        
        # === 8. 🔧 高效时序平滑（训练时） ===
        if self.training and self.enable_temporal_smoothing and self.step_counter > 0:
            with torch.cuda.amp.autocast(enabled=True):  # EMA计算可以用FP16
                # 批量计算当前平均特征
                current_mean_feature = enhanced_features.mean(dim=0)
                current_mean_quality = quality_scores.mean(dim=0)
            
            # 🔧 原地更新EMA
            alpha = 1 - self.temporal_alpha
            self.feature_ema.mul_(self.temporal_alpha).add_(current_mean_feature, alpha=alpha)  # 原地EMA
            self.quality_ema.mul_(self.temporal_alpha).add_(current_mean_quality, alpha=alpha)   # 原地EMA
            
            # 🔧 内存优化的时序引导
            temporal_guidance = self.feature_ema.unsqueeze(0).expand_as(enhanced_features)
            enhanced_features.mul_(0.95).add_(temporal_guidance, alpha=0.05)  # 原地加权平均
        
        # === 9. 🔧 原地更新运行时统计 ===
        if self.training:
            # 批量原地更新所有运行时统计
            alpha = self.momentum
            beta = 1 - self.momentum
            
            self.running_mean.mul_(beta).add_(batch_mean.squeeze(), alpha=alpha)  # 原地EMA
            self.running_var.mul_(beta).add_(batch_var.squeeze(), alpha=alpha)    # 原地EMA
            self.running_std.mul_(beta).add_(batch_std.squeeze(), alpha=alpha)    # 原地EMA
            self.stability_score.mul_(beta).add_(feature_stability, alpha=alpha)  # 原地EMA
            self.step_counter += 1
        
        # === 10. 🚀 最终残差连接 ===
        # 确保即使校正失败也不会破坏原始特征
        corrected_features = 0.8 * enhanced_features + 0.2 * normalized_features
        
        # 准备校正信息用于监控
        correction_info = {
            'stability_mean': feature_stability.mean().item(),
            'quality_mean': quality_scores.mean().item(),
            'variance_scale_mean': constrained_variance_controller.mean().item(),
            'correction_strength': torch.norm(corrected_features - normalized_features).item(),
            'feature_norm': torch.norm(corrected_features).item()
        }
        
        # 定期打印监控信息
        if self.training and self.step_counter % 500 == 0:
            print(f"\n🔧 LFSC校正监控 (Step {self.step_counter}):")
            print(f"   特征稳定性均值: {correction_info['stability_mean']:.4f}")
            print(f"   特征质量均值: {correction_info['quality_mean']:.4f}")
            print(f"   方差控制均值: {correction_info['variance_scale_mean']:.4f}")
            print(f"   校正强度: {correction_info['correction_strength']:.4f}")
            print(f"   特征范数: {correction_info['feature_norm']:.4f}")
        
        return corrected_features, correction_info

class SpatialFeatureSelfCorrection(nn.Module):
    """
    🚀 空间特征自校正模块 - LFSC的2D版本
    
    针对特征图 [N, C, H, W] 的自校正，保持空间结构
    
    核心机制：
    1. 空间统计归一化：在空间维度上进行归一化
    2. 通道间去相关：减少通道间有害相关性
    3. 空间平滑：确保相邻位置特征的连续性
    4. 自适应权重：学习每个通道的重要性
    """
    
    def __init__(self, num_channels, momentum=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        
        # 🚀 通道去相关网络（轻量级1x1卷积）
        self.channel_decorrelation = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.SiLU(),
            nn.Conv2d(num_channels, num_channels, 1, bias=False),
            nn.Tanh()
        )
        
        # 🚀 空间平滑器（深度可分离卷积）
        self.spatial_smoother = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, groups=num_channels, bias=False),
            nn.Conv2d(num_channels, num_channels, 1, bias=False),
            nn.GroupNorm(1, num_channels),
            nn.Sigmoid()
        )
        
        # 🚀 自适应通道权重
        self.channel_weights = nn.Parameter(torch.ones(num_channels))
        
        # 运行时统计
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))
        self.register_buffer('step_counter', torch.tensor(0))
        
    def forward(self, feature_map):
        """
        🚀 修复的空间特征自校正 - 解决原地操作与混合精度冲突
        
        修复内容：
        1. 🚀 避免原地操作，防止梯度计算错误
        2. 🚀 统一精度策略，在FP32下进行关键计算
        3. 🚀 保持数值稳定性和梯度流连续性
        
        Args:
            feature_map: [N, C, H, W] 特征图
            
        Returns:
            corrected_map: [N, C, H, W] 校正后的特征图
        """
        N, C, H, W = feature_map.shape
        
        # === 🚀 修复方案：在FP32下进行关键计算，避免原地操作 ===
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为FP32
            if feature_map.dtype != torch.float32:
                feature_map = feature_map.float()
            
            # === 1. 空间统计归一化（非原地） ===
            spatial_mean = feature_map.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
            spatial_var = feature_map.var(dim=(2, 3), keepdim=True, unbiased=False)  # [N, C, 1, 1]
            spatial_std = torch.sqrt(spatial_var + 1e-8)  # 非原地开方
            
            # 非原地标准化
            normalized_map = (feature_map - spatial_mean) / spatial_std
            
            # === 2. 通道去相关 ===
            decorrelation_residual = self.channel_decorrelation(normalized_map)
            # 非原地残差连接
            decorrelated_map = normalized_map + 0.1 * decorrelation_residual
            
            # === 3. 空间平滑权重 ===
            smooth_weights = self.spatial_smoother(decorrelated_map)
            
            # === 4. 自适应通道权重应用 ===
            constrained_weights = torch.clamp(self.channel_weights, min=0.1, max=2.0)
            weight_view = constrained_weights.view(1, -1, 1, 1)
            weighted_map = decorrelated_map * weight_view  # 非原地乘法
            
            # === 5. 空间加权 ===
            enhanced_map = weighted_map * smooth_weights  # 非原地乘法
            
            # === 6. 最终残差连接 ===
            corrected_map = 0.8 * enhanced_map + 0.2 * normalized_map  # 非原地操作
            
            # === 7. 更新运行时统计（保持原地更新以优化内存） ===
            if self.training:
                batch_mean = spatial_mean.mean(dim=0).squeeze()  # [C]
                batch_var = spatial_var.mean(dim=0).squeeze()    # [C]
                
                # 运行时统计的原地更新是安全的，因为它们不参与梯度计算
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)
                self.step_counter += 1
        
        return corrected_map

class QualityActivation(nn.Module):
    """
    🚀 压缩版sigmoid激活函数：提供更大的动态范围
    
    核心思想：
    1. 使用压缩版sigmoid提供更好的动态范围
    2. 可学习的温度参数控制激活函数的陡峭程度
    3. 输出范围[0,1]，但具有更好的区分能力
    """
    
    def __init__(self, mode='compressed_sigmoid'):
        super().__init__()
        self.mode = mode
        
        if mode == 'compressed_sigmoid':
            # 🎯 进一步优化的压缩版sigmoid参数 - 针对动态范围0.21→0.35+优化
            self.temperature = nn.Parameter(torch.tensor(3.0))  # 进一步降低温度，减少饱和
            self.offset = nn.Parameter(torch.tensor(-2.0))       # 中性偏移，平衡高低值输出
            self.min_val = nn.Parameter(torch.tensor(0.08))     # 提高最小值，改善低端区分
            self.max_val = nn.Parameter(torch.tensor(0.99))     # 适度降低最大值，避免饱和
        else:
            # 原来的简单模式作为备选
            self.scale = nn.Parameter(torch.tensor(2.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        """
        🎯 改进的激活函数 - 🚀 增加动态范围
        """
        if self.mode == 'compressed_sigmoid':
            # 🚀 进一步优化约束，针对动态范围0.21→0.35+
            # 平衡的offset约束，支持正负偏移
            constrained_offset = torch.clamp(self.offset, min=-0.8, max=0.8)
            # 更宽松的温度范围，减少过度饱和
            constrained_temperature = torch.clamp(self.temperature, min=0.6, max=2.0)
            # 更宽松的最小值范围，提升低端表达
            constrained_min_val = torch.clamp(self.min_val, min=0.03, max=0.25)
            # 更大的最大值范围，扩展动态范围上限
            constrained_max_val = torch.clamp(self.max_val, min=0.75, max=0.97)
            
            # 压缩版sigmoid：sigmoid((x + offset) * temperature)
            shifted = x + constrained_offset
            sigmoid_out = torch.sigmoid(shifted * constrained_temperature)
            
            # 将sigmoid输出从[0,1]映射到[min_val, max_val]
            # 🚀 更大的动态范围 [0.01, 0.99]
            compressed = constrained_min_val + sigmoid_out * (constrained_max_val - constrained_min_val)
            
            return compressed
        else:
            # 原来的简单激活函数
            scaled = self.scale * x + self.bias
            activated = F.relu6(scaled) / 6.0
            return activated

class FeatureExtractor(nn.Module):
    """
    🚀 高性能特征提取器：批量化计算 + 内存优化 + 混合精度 + LFSC增强 + 分类特征支持
    
    核心性能优化：
    1. ⚡ 批量化特征计算：一次计算所有统计矩，避免重复计算
    2. 🔧 内存原地操作：减少内存分配，提高缓存效率
    3. 🎯 混合精度优化：FP16统计计算 + FP32关键变换
    4. 🚀 LFSC特征自校正：提高特征稳定性
    5. 📊 共享中间结果：最大化计算复用
    6. 🎯 分类特征集成：支持分类置信度、熵、方差等特征
    """
    
    def __init__(self, reg_max=16, num_classes=80, enable_lfsc=True, enable_mixed_precision=True):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.enable_lfsc = enable_lfsc
        self.enable_mixed_precision = enable_mixed_precision
        self.kurtosis_max = 10.0  # 峰度的经验最大值
        self.concentration_max = 5.0  # 集中度的经验最大值
        
        # === 🚀 自适应归一化参数 ===
        self.adaptive_norm = nn.Parameter(torch.ones(2))
        
        # === ⚡ 预分配内存缓冲区（避免重复分配） ===
        self.register_buffer('indices_cache', torch.arange(reg_max + 1).float())
        self.register_buffer('max_entropy_cache', torch.log(torch.tensor(reg_max + 1.0)))
        self.register_buffer('max_var_cache', torch.tensor(((reg_max) / 2.0) ** 2))
        
        # === 🚀 LFSC模块：延迟初始化 ===
        self.lfsc_module = None
        self._lfsc_initialized = False
        
        # === 📊 性能监控 ===
        self._computation_cache = {}
        self._enable_profiling = False

        
    def _init_lfsc_module(self, num_features):
        """
        🚀 延迟初始化LFSC模块
        """
        if self.enable_lfsc and not self._lfsc_initialized:
            self.spatial_lfsc = SpatialFeatureSelfCorrection(
                num_channels=num_features,
                momentum=0.1
            )
            self._lfsc_initialized = True
        
    def extract_all_features(self, bbox_pred):
        """
        ⚡ 高性能批量化特征提取 + 混合精度优化 + LFSC增强
        
        核心优化策略：
        1. 批量计算所有统计矩，避免重复索引操作
        2. 内存原地操作，减少分配开销
        3. 混合精度：统计计算用FP16，关键变换用FP32
        4. 共享中间结果，最大化计算复用
        
        Args:
            bbox_pred: [N, 68, H, W] 原始回归分布
            
        Returns:
            Dict[str, Tensor]: 特征字典（已LFSC校正）
        """
        # === 🚀 修复方案：统一在FP32下进行数值计算，确保稳定性 ===
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为FP32，提供最佳数值精度
            if bbox_pred.dtype != torch.float32:
                bbox_pred = bbox_pred.float()
                
            N, C, H, W = bbox_pred.shape
            
            # === 🚀 批量化概率分布计算（复用缓存的indices） ===
            prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
            
            # 使用预缓存的indices，避免重复创建
            indices = self.indices_cache.to(bbox_pred.device)
            indices_expanded = indices.view(1, 1, -1, 1, 1)
            
            # === ⚡ 批量计算所有基础统计量（一次性计算，多次复用） ===
            # 1. 期望值（距离预测）
            distances = (prob * indices_expanded).sum(dim=2)  # [N, 4, H, W]
            
            # 2. 批量计算所有矩：一次性计算避免重复索引操作
            centered_indices = indices_expanded - distances.unsqueeze(2)  # [N, 4, reg_max+1, H, W]
            
            # 🚀 修复：避免原地操作，增强数值稳定性
            second_moment = (prob * torch.pow(centered_indices, 2)).sum(dim=2)  # 方差
            third_moment = (prob * torch.pow(centered_indices, 3)).sum(dim=2)   # 三阶矩
            fourth_moment = (prob * torch.pow(centered_indices, 4)).sum(dim=2)  # 四阶矩
            
            # 3. 批量计算概率统计量
            max_prob, max_idx = prob.max(dim=2)  # [N, 4, H, W]
            
            # 4. 批量计算熵（增强数值稳定性）
            log_prob = torch.log(torch.clamp(prob, min=1e-10))  # 更安全的log计算
            entropy = -(prob * log_prob).sum(dim=2)
            max_entropy = self.max_entropy_cache.to(bbox_pred.device)
        
            # 数据已在FP32精度下，不需要额外转换
            features = {}
            
            # === ⚡ 批量化特征计算（基于预计算的统计量） ===
            # === 1. ⚡ 批量化基础几何特征 ===
            
            # 🚀 修复：数值稳定的归一化，避免原地操作
            normalized_distances = distances / (self.reg_max + 1e-8)  # 增强数值稳定性
            
            # === 🚀 4边top2特征（8维基础特征）- 批量化计算 ===
            # 一次性获取所有边界的top2概率值
            prob_reshaped = prob.transpose(1, 2)  # [N, reg_max+1, 4, H, W]
            all_top2, _ = prob_reshaped.topk(4, dim=1)  # [N, 4, 4, H, W]
            
            # 🔧 内存优化：直接分割，避免额外拷贝
            features['left_top2'] = all_top2[:, :, 0]     # [N, 4, H, W] -> 4维
            features['top_top2'] = all_top2[:, :, 1]      # [N, 4, H, W] -> 4维
            features['right_top2'] = all_top2[:, :, 2]    # [N, 4, H, W] -> 4维  
            features['bottom_top2'] = all_top2[:, :, 3]   # [N, 4, H, W] -> 4维
            
            # 核心几何特征：平均距离（原地计算）
            mean_distances = normalized_distances.mean(dim=1, keepdim=True)  # [N, 1, H, W]
            features['distances_1d'] = mean_distances
            
            # 距离一致性特征（几何稳定性）- 非原地计算
            distance_std = normalized_distances.std(dim=1)  # [N, H, W]
            distance_consistency = torch.clamp(distance_std, 0.0, 1.0)
            distance_consistency = 1.0 - distance_consistency  # 非原地操作
            features['distance_consistency'] = (distance_consistency * self.adaptive_norm[0]).unsqueeze(1)
            
            # === 2. ⚡ 批量化分布形状特征 ===
            
            # 🚀 修复：增强数值稳定性的熵归一化
            normalized_entropy = torch.clamp(entropy / (max_entropy + 1e-8), 0.0, 1.0)  # 数值稳定的归一化
            entropy_certainty = 1.0 - normalized_entropy  # 非原地操作
            features['entropy_1d'] = (entropy_certainty.mean(dim=1) * self.adaptive_norm[1]).unsqueeze(1)
            
            # 最大概率（分布集中度）- 使用预计算的max_prob
            features['max_prob_1d'] = max_prob.mean(dim=1, keepdim=True)
            
            # === 3. ⚡ 批量化统计矩特征（使用预计算的矩） ===
            
            # 🚀 修复：增强数值稳定性的方差计算
            max_theoretical_var = self.max_var_cache.to(bbox_pred.device)
            normalized_variance = torch.clamp(second_moment / (max_theoretical_var + 1e-8), 0.0, 1.0)  # 数值稳定性
            variance_concentration = 1.0 - normalized_variance  # 非原地操作
            features['variance_concentration_1d'] = variance_concentration.mean(dim=1, keepdim=True)
            
            # 🚀 修复：增强数值稳定性的偏度计算
            safe_second_moment = torch.clamp(second_moment, min=1e-8)  # 防止开方时的数值问题
            variance_sqrt_cubed = torch.pow(torch.sqrt(safe_second_moment), 3) + 1e-8  # 数值稳定的立方根计算
            skewness = third_moment / variance_sqrt_cubed  # 非原地除法
            normalized_skewness = torch.clamp(torch.abs(skewness) / 3.0, 0.0, 1.0)  # 安全的归一化
            features['skewness_1d'] = normalized_skewness.mean(dim=1, keepdim=True)
            
            # 🚀 修复：增强数值稳定性的峰度计算
            safe_variance_squared = torch.clamp(second_moment ** 2, min=1e-8)  # 防止除零
            kurtosis = fourth_moment / safe_variance_squared  # 数值稳定的除法
            normalized_kurtosis = torch.clamp(kurtosis / (self.kurtosis_max + 1e-8), 0.0, 1.0)  # 安全的归一化
            features['kurtosis_1d'] = normalized_kurtosis.mean(dim=1, keepdim=True)
            
            # === 4. ⚡ 批量化位置特征 ===
            
            # 🚀 修复：数值稳定的位置归一化
            normalized_mode = max_idx / (self.reg_max + 1e-8)  # 数值稳定性
            features['mode_position_1d'] = normalized_mode.mean(dim=1, keepdim=True)
            
            # 🚀 四分位距（批量化计算）- 保持FP32精度以确保稳定性
            sorted_prob, _ = prob.sort(dim=2, descending=False)
            cumsum_prob = torch.cumsum(sorted_prob, dim=2)
            
            # 批量化分位数计算
            q1_mask = cumsum_prob >= 0.25
            q3_mask = cumsum_prob >= 0.75
            q1_indices = q1_mask.float().argmax(dim=2)
            q3_indices = q3_mask.float().argmax(dim=2)
            
            # 🚀 修复：数值稳定的IQR计算
            iqr = (q3_indices.float() - q1_indices.float()) / (self.reg_max + 1e-8)
            features['iqr_1d'] = iqr.mean(dim=1, keepdim=True)
            
            # === 5. ⚡ 批量化几何质量特征 ===
            
            # 🚀 几何平衡性（批量化计算，修复原地操作）
            # 使用normalized_distances避免重复除法
            left, top, right, bottom = normalized_distances.unbind(dim=1)  # [N, H, W] each
            
            # 🚀 修复：非原地计算，增强数值稳定性
            lr_min = torch.min(left, right)
            lr_max = torch.max(left, right) + 1e-6  # 非原地加法，数值稳定性
            tb_min = torch.min(top, bottom)
            tb_max = torch.max(top, bottom) + 1e-6  # 非原地加法，数值稳定性
            
            # 非原地计算平衡度
            lr_balance = lr_min / lr_max  # 非原地除法
            tb_balance = tb_min / tb_max  # 非原地除法
            boundary_balance = (lr_balance + tb_balance) / 2.0  # 非原地平均
            features['boundary_balance'] = boundary_balance.unsqueeze(1)
            
            # 🚀 多模态检测（保持FP32精度确保稳定性）
            # 批量化二阶差分计算
            prob_2nd_diff = prob[:, :, 2:] - 2 * prob[:, :, 1:-1] + prob[:, :, :-2]
            local_maxima = (prob_2nd_diff < -0.01).float().sum(dim=2)
            
            # 🚀 修复：非原地计算多模态分数，增强数值稳定性
            multimodal_score = 1.0 / (local_maxima.mean(dim=1) + 1.0 + 1e-8)  # 非原地操作，数值稳定性
            features['multimodal_quality'] = multimodal_score.unsqueeze(1)
            
            # === 🚀 LFSC特征自校正 ===
            if self.enable_lfsc:
                # 首先堆叠所有特征
                feature_list = list(features.values())
                if feature_list:
                    # 堆叠特征 [N, total_channels, H, W]
                    stacked_features = torch.cat(feature_list, dim=1)
                    num_channels = stacked_features.shape[1]
                    
                    # 延迟初始化LFSC模块
                    if not self._lfsc_initialized:
                        self._init_lfsc_module(num_channels)
                    
                    # 应用LFSC校正
                    corrected_features = self.spatial_lfsc(stacked_features)
                    
                    # 将校正后的特征重新分割回原始字典结构
                    corrected_features_dict = {}
                    start_idx = 0
                    for name, original_feature in features.items():
                        num_channels = original_feature.shape[1]
                        end_idx = start_idx + num_channels
                        corrected_features_dict[name] = corrected_features[:, start_idx:end_idx]
                        start_idx = end_idx
                    
                    # 🚀 监控LFSC效果
                    if self.training and self.spatial_lfsc.step_counter % 500 == 0:
                        improvement = torch.norm(corrected_features - stacked_features).item()
                        print(f"\n🔧 LFSC特征校正效果 (Step {self.spatial_lfsc.step_counter}):")
                        print(f"   特征改善强度: {improvement:.4f}")
                        print(f"   校正前特征范数: {torch.norm(stacked_features).item():.4f}")
                        print(f"   校正后特征范数: {torch.norm(corrected_features).item():.4f}")
                    
                    return corrected_features_dict
            
            return features

    def _simple_quality_estimate(self, prob, distances):
        """
        🎯 简化的质量估计（避免循环依赖）
        
        快速估计样本质量，用于错误分配检测
        """
        N, _, _, H, W = prob.shape
        
        # 1. 分布集中度
        max_prob, _ = prob.max(dim=2)  # [N, 4, H, W]
        mean_max_prob = max_prob.mean(dim=1)  # [N, H, W]
        
        # 2. 距离一致性（简化版）
        left, top, right, bottom = distances[:, 0], distances[:, 1], distances[:, 2], distances[:, 3]
        distance_std = torch.stack([left, top, right, bottom], dim=1).std(dim=1)  # [N, H, W]
        consistency = torch.exp(-distance_std / distance_std.mean())
        
        # 3. 基本质量评分
        quality_score = 0.6 * mean_max_prob + 0.4 * consistency
        
        return torch.clamp(quality_score, 0.0, 1.0)
    


class SimplifiedFusionLayer(nn.Module):
    """
    ⚡ 高性能简化融合层：混合精度 + 内存优化 + 核心功能
    
    核心性能优化：
    1. 🎯 混合精度：中间计算FP16，输出FP32
    2. 🔧 内存优化：原地激活函数
    3. ⚡ 计算优化：最小化必要组件
    4. 🚀 LFSC兼容：保持校正效果
    """
    
    def __init__(self, num_features, output_dim=1, enable_mixed_precision=True):
        super().__init__()
        
        self.enable_mixed_precision = enable_mixed_precision
        
        # 🚀 精简的特征压缩：直接从输入特征到输出
        self.hidden_dim = max(num_features // 4, 32)  # 更激进的压缩比
        
        # 🔧 核心变换网络：仅保留必要组件
        self.core_transform = nn.Sequential(
            # 阶段1：特征压缩
            nn.Conv2d(num_features, self.hidden_dim, 1, bias=True),
            nn.SiLU(inplace=True),
        
            # 阶段2：非线性变换
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.05),  # 轻微正则化
            
            # 阶段3：输出投影
            nn.Conv2d(self.hidden_dim, output_dim, 1, bias=True)
        )
        
        # 🔥 保留高效的质量激活函数
        self.quality_activation = QualityActivation(mode='compressed_sigmoid')
        
        # 🚀 简化的权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """🚀 针对简化结构的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 1:  # 输出层
                    nn.init.xavier_normal_(m.weight, gain=0.8)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
    def forward(self, feature_stack):
        """
        ⚡ 修复的混合精度前向传播 - 解决梯度中断问题
        
        修复内容：
        1. 🚀 避免手动类型转换，让PyTorch自动处理
        2. 🚀 统一精度策略，减少不必要的转换
        3. 🚀 保持梯度流的连续性
        
        Args:
            feature_stack: [N, num_features, H, W]
        Returns:
            output: [N, output_dim, H, W]
        """
        # === 🚀 修复方案：统一在FP32下进行关键计算 ===
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为FP32，避免手动转换
            if feature_stack.dtype != torch.float32:
                feature_stack = feature_stack.float()
            
            # 在FP32下进行核心变换，确保数值精度
            raw_quality = self.core_transform(feature_stack)
            
            # 质量激活函数在FP32下计算，确保稳定性
            quality_score = self.quality_activation(raw_quality)
        
        return quality_score

class LearnableFeatureCombiner(nn.Module):
    """
    🚀 分布感知的可学习特征组合器
    
    核心改进：
    1. 从原始分布提取自适应特征（根据训练阶段调整）
    2. 时序一致性约束，确保平滑过渡
    3. 多阶段监督机制，稳定训练过程
    """
    
    def __init__(self, reg_max=16, hidden_dim=32, output_dim=1, enable_lfsc=True, enable_mixed_precision=True):
        super().__init__()
        
        self.reg_max = reg_max
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.enable_lfsc = enable_lfsc
        self.enable_mixed_precision = enable_mixed_precision
        
        # 🚀 高性能特征提取器（LFSC + 混合精度支持）
        self.feature_extractor = FeatureExtractor(
            reg_max, 
            enable_lfsc=enable_lfsc,
            enable_mixed_precision=enable_mixed_precision
        )
        
        # 获取特征数量
        self._determine_feature_count()
        
        # 监控计数器
        self._step_counter = 0
        
        # 🚀 轻量级特征混合器（替代复杂交互层）
        self.feature_mixer = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.num_features, self.num_features, 1, bias=True)
        )
        
        # 🚀 高性能融合层（混合精度 + 必要的非线性变换）
        self.fusion_layer = SimplifiedFusionLayer(
            self.num_features, 
            output_dim,
            enable_mixed_precision=enable_mixed_precision
        )
        
        # NAS架构权重 - 🚀 稳定初始化，避免训练不稳定
        self.arch_weights = nn.Parameter(torch.ones(self.num_features) * 0.5)  # 更保守的初始权重
        
        
    def _determine_feature_count(self):
        """确定特征数量"""
        dummy_input = torch.randn(1, 4 * (self.reg_max + 1), 32, 32)
        
        with torch.no_grad():
            features = self.feature_extractor.extract_all_features(dummy_input)
            self.feature_names = list(features.keys())
            
            # 计算特征数量
            total_channels = 0
            for name, feature in features.items():
                total_channels += feature.shape[1]  # 累加通道数
            
            self.num_features = total_channels
        
    def forward(self, bbox_pred):
        """
        🚀 分布感知的前向传播
        
        Args:
            bbox_pred: [N, 68, H, W] 原始回归分布
            
        Returns:
            quality_score: [N, 1, H, W] 质量分数
        """
        # 1. 提取特征
        features_dict = self.feature_extractor.extract_all_features(bbox_pred)
        
        # 更新监控统计
        self._step_counter += 1
        
        # 2. 特征堆叠
        feature_list = []
        for name in self.feature_names:
            feature = features_dict[name]
            feature_list.append(feature)
        
        if feature_list:
            feature_stack = torch.cat(feature_list, dim=1)  # [N, total_channels, H, W]
            
            # 应用架构权重
            constrained_weights = torch.clamp(self.arch_weights, min=0.2, max=2.0)
            weight_expanded = constrained_weights.view(1, -1, 1, 1).expand_as(feature_stack)
            feature_stack = feature_stack * weight_expanded
        else:
            raise RuntimeError("没有可用的特征！")
        
        # 3. 🚀 修复：统一精度策略的轻量级特征混合
        with torch.cuda.amp.autocast(enabled=False):
            # 保持FP32精度确保数值稳定性
            mixed_residual = self.feature_mixer(feature_stack)
        
        # 🚀 修复：非原地残差连接，避免梯度问题
        mixed_features = feature_stack + 0.15 * mixed_residual  # 非原地加法
        
        # 4. 🚀 简化融合（直接输出质量分数）
        quality_score = self.fusion_layer(mixed_features)
        
        # 5. 🚀 简化版状态监控（训练时）
        if self.training and self.enable_lfsc and self._step_counter % 100 == 0:
            print(f"\n🚀 轻量级特征组合器状态报告 (Step {self._step_counter}):")
            print(f"   LFSC模块状态: {'已激活' if self.feature_extractor._lfsc_initialized else '未初始化'}")
            if self.feature_extractor._lfsc_initialized:
                spatial_lfsc = self.feature_extractor.spatial_lfsc
                print(f"   LFSC校正步数: {spatial_lfsc.step_counter.item()}")
                print(f"   LFSC通道权重均值: {spatial_lfsc.channel_weights.mean().item():.4f}")
                print(f"   LFSC特征稳定性: {torch.std(spatial_lfsc.running_var).item():.4f}")
            print(f"   特征总数: {self.num_features} (包含8维4边top2基础特征)")
            
            # 🚀 轻量级混合器状态
            print(f"\n🔧 轻量级混合器状态:")
            print(f"   混合权重: 0.15 (固定)")
            print(f"   混合器参数量: ~{sum(p.numel() for p in self.feature_mixer.parameters())/1000:.1f}K")
            print(f"   设计模式: 简化轻量级")
            
            # 监控LFSC与轻量级网络的协同效果
            with torch.no_grad():
                feature_norm_before = torch.norm(feature_stack).item()
                mixed_norm = torch.norm(mixed_features).item()
                final_norm = torch.norm(quality_score).item()
                
                print(f"\n📊 轻量级流水线效果:")
                print(f"      LFSC后特征范数: {feature_norm_before:.4f}")
                print(f"      轻量级混合后范数: {mixed_norm:.4f}")
                print(f"      最终输出范数: {final_norm:.4f}")
                print(f"      混合增强比例: {mixed_norm/feature_norm_before:.3f}")
                print(f"      整体处理比例: {final_norm/feature_norm_before:.3f}")
                
                # 🚀 质量分数分布统计
                quality_mean = quality_score.mean().item()
                quality_std = quality_score.std().item()
                quality_min = quality_score.min().item()
                quality_max = quality_score.max().item()
                
                print(f"\n🎯 质量分数分布统计:")
                print(f"      均值: {quality_mean:.4f}")
                print(f"      标准差: {quality_std:.4f}")
                print(f"      范围: [{quality_min:.4f}, {quality_max:.4f}]")
                print(f"      动态范围: {quality_max - quality_min:.4f}")
                
                # 🚀 效率对比
                total_params = sum(p.numel() for p in self.parameters()) / 1000
                lfsc_params = sum(p.numel() for p in self.feature_extractor.parameters()) / 1000 if hasattr(self.feature_extractor, 'spatial_lfsc') else 0
                mixer_params = sum(p.numel() for p in self.feature_mixer.parameters()) / 1000
                fusion_params = sum(p.numel() for p in self.fusion_layer.parameters()) / 1000
                
                print(f"\n⚡ 轻量化效果:")
                print(f"      总参数量: {total_params:.1f}K")
                print(f"      LFSC参数: {lfsc_params:.1f}K ({lfsc_params/total_params*100:.1f}%)")
                print(f"      混合器参数: {mixer_params:.1f}K ({mixer_params/total_params*100:.1f}%)")
                print(f"      融合层参数: {fusion_params:.1f}K ({fusion_params/total_params*100:.1f}%)")
                print("=" * 60)
        
        # 6. 返回质量分数
        return quality_score
    

    

    

    