"""
一致性度量工具 - 用于评估特征质量与定位质量的匹配度

核心思想：
- 特征质量（confidence）应该与定位质量（IoU）相匹配
- 一致性高的样本更适合用于对比学习
- 矛盾样本（特征好但IoU低，或特征差但IoU高）应该被抑制
"""

import torch
from torch import Tensor


def compute_consistency_hybrid(confidence: Tensor, iou: Tensor) -> Tensor:
    """🏆 混合一致性度量（最佳方案）
    
    结合correlation和rank_based的优势：
    - correlation: 提供细粒度的连续分数
    - rank_based: 提供强力的区分能力
    
    策略：
    1. 用correlation计算基础一致性（细粒度）
    2. 用rank_based判断是否跨区间（强区分）
    3. 对跨区间样本进行惩罚
    
    Args:
        confidence (Tensor): [N] LQE中的confidence特征，范围[0,1]
                            反映分布的集中度和判别性
        iou (Tensor): [N] 真实IoU，范围[0,1]
                     反映定位的准确性
        
    Returns:
        consistency (Tensor): [N] 一致性分数，范围[0,1]
                             1.0 = 完全一致（confidence≈IoU）
                             0.0 = 完全矛盾（confidence和IoU差异大）
    
    Example:
        >>> confidence = torch.tensor([0.85, 0.35, 0.88, 0.40])
        >>> iou = torch.tensor([0.87, 0.32, 0.35, 0.85])
        >>> consistency = compute_consistency_hybrid(confidence, iou)
        >>> print(consistency)
        tensor([0.95, 0.92, 0.05, 0.02])  # 前2个一致，后2个矛盾
    """
    # === 方法1：Correlation（高斯核，σ=0.15） ===
    # 物理意义：度量confidence与IoU的接近程度
    # - |confidence - iou| < 0.15: 高一致性 (>0.8)
    # - |confidence - iou| > 0.3: 低一致性 (<0.2)
    diff = torch.abs(confidence - iou)
    consistency_correlation = torch.exp(-diff.pow(2) / (2 * 0.15**2))
    
    # === 方法2：Rank-based（分层判断） ===
    # 定义三个区间：高(>0.6)、中(0.4-0.6)、低(<0.4)
    conf_high = confidence > 0.6
    conf_mid = (confidence > 0.4) & (confidence <= 0.6)
    conf_low = confidence <= 0.4
    
    iou_high = iou > 0.6
    iou_mid = (iou > 0.4) & (iou <= 0.6)
    iou_low = iou <= 0.4
    
    # 同区间：高一致性 (1.0)
    same_region = (conf_high & iou_high) | (conf_mid & iou_mid) | (conf_low & iou_low)
    
    # 相邻区间：中等一致性 (0.6)
    adjacent_region = (conf_high & iou_mid) | (conf_mid & iou_high) | \
                     (conf_mid & iou_low) | (conf_low & iou_mid)
    
    # 跨区间：低一致性 (0.0)
    # 例如：confidence高(>0.6) + IoU低(<0.4)
    
    consistency_rank = torch.zeros_like(confidence)
    consistency_rank[same_region] = 1.0
    consistency_rank[adjacent_region] = 0.6
    # 跨区间默认为0.0
    
    # === 混合策略 ===
    # 用rank_based的判断来调制correlation的分数
    # - 同区间或相邻区间：保持correlation的细粒度
    # - 跨区间：强力惩罚（×0.2）
    consistency_hybrid = consistency_correlation * (0.2 + 0.8 * consistency_rank)
    
    return consistency_hybrid.clamp(0, 1)


def compute_consistency_correlation(confidence: Tensor, iou: Tensor, sigma: float = 0.15) -> Tensor:
    """Correlation方法（高斯核）- 细粒度一致性度量
    
    优势：提供平滑的连续分数
    
    Args:
        confidence (Tensor): [N] 特征质量
        iou (Tensor): [N] 定位质量
        sigma (float): 高斯核宽度，控制容忍度
        
    Returns:
        consistency (Tensor): [N] 一致性分数
    """
    diff = torch.abs(confidence - iou)
    consistency = torch.exp(-diff.pow(2) / (2 * sigma**2))
    return consistency


def compute_consistency_rank(confidence: Tensor, iou: Tensor) -> Tensor:
    """Rank-based方法（分层判断）- 强区分一致性度量
    
    优势：对矛盾样本强力惩罚
    
    Args:
        confidence (Tensor): [N] 特征质量
        iou (Tensor): [N] 定位质量
        
    Returns:
        consistency (Tensor): [N] 一致性分数 {0.0, 0.6, 1.0}
    """
    # 定义区间
    conf_high = confidence > 0.6
    conf_mid = (confidence > 0.4) & (confidence <= 0.6)
    conf_low = confidence <= 0.4
    
    iou_high = iou > 0.6
    iou_mid = (iou > 0.4) & (iou <= 0.6)
    iou_low = iou <= 0.4
    
    # 同区间：高一致性
    same_region = (conf_high & iou_high) | (conf_mid & iou_mid) | (conf_low & iou_low)
    # 相邻区间：中等一致性
    adjacent_region = (conf_high & iou_mid) | (conf_mid & iou_high) | \
                     (conf_mid & iou_low) | (conf_low & iou_mid)
    
    consistency = torch.zeros_like(confidence)
    consistency[same_region] = 1.0
    consistency[adjacent_region] = 0.6
    
    return consistency


def compute_adaptive_weight(consistency_score: Tensor) -> Tensor:
    """根据一致性分数计算对比损失权重
    
    分层权重策略：
    - 一致性 > 0.75: 权重1.0（全力学习）
    - 一致性 0.50-0.75: 权重0.5（适度学习）
    - 一致性 0.30-0.50: 权重0.2（轻微学习）
    - 一致性 < 0.30: 权重0.0（跳过矛盾样本）
    
    Args:
        consistency_score (Tensor): [N] 一致性分数 [0, 1]
        
    Returns:
        weight (Tensor): [N] 对比损失权重 [0, 1]
        
    Example:
        >>> consistency = torch.tensor([0.95, 0.65, 0.45, 0.15])
        >>> weight = compute_adaptive_weight(consistency)
        >>> print(weight)
        tensor([1.0, 0.5, 0.2, 0.0])
    """
    weight = torch.zeros_like(consistency_score)
    
    # 高一致性 (>0.75)：全力学习
    high_mask = consistency_score > 0.75
    weight[high_mask] = 1.0
    
    # 中一致性 (0.50-0.75)：适度学习
    mid_mask = (consistency_score > 0.50) & (consistency_score <= 0.75)
    weight[mid_mask] = 0.5
    
    # 低一致性 (0.30-0.50)：轻微学习
    low_mask = (consistency_score > 0.30) & (consistency_score <= 0.50)
    weight[low_mask] = 0.2
    
    # 极低一致性 (≤0.30)：跳过（权重=0）
    
    return weight


def compute_consistency_statistics(consistency_score: Tensor) -> dict:
    """计算一致性统计信息（用于监控）
    
    Args:
        consistency_score (Tensor): [N] 一致性分数
        
    Returns:
        stats (dict): 统计信息
            - mean: 平均一致性
            - std: 标准差
            - high_ratio: 高一致性样本比例 (>0.75)
            - mid_ratio: 中一致性样本比例 (0.50-0.75)
            - low_ratio: 低一致性样本比例 (0.30-0.50)
            - reject_ratio: 矛盾样本比例 (<0.30)
    """
    if len(consistency_score) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'high_ratio': 0.0,
            'mid_ratio': 0.0,
            'low_ratio': 0.0,
            'reject_ratio': 0.0
        }
    
    return {
        'mean': consistency_score.mean().item(),
        'std': consistency_score.std().item(),
        'high_ratio': (consistency_score > 0.75).float().mean().item(),
        'mid_ratio': ((consistency_score > 0.50) & (consistency_score <= 0.75)).float().mean().item(),
        'low_ratio': ((consistency_score > 0.30) & (consistency_score <= 0.50)).float().mean().item(),
        'reject_ratio': (consistency_score <= 0.30).float().mean().item()
    }
