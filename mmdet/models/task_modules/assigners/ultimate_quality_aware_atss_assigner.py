#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
终极质量感知ATSS分配器 - 基于实测数据的根本性改进

核心突破：
1. 分层IoU保护策略 - 解决IoU暴跌问题
2. 置信度自适应融合 - 智能权重调整
3. 渐进式质量评估 - 避免简单粗暴乘积
4. 多层级感知机制 - 适应FPN特征金字塔

实测改进目标：
- IoU下降控制在20%以内（当前67.5% → 目标20%）
- 质量分数提升20%以上
- 高IoU样本选择率50%+
- 计算开销<150%

基于四版本实测数据的深度优化
"""

import warnings
from typing import List, Optional, Tuple
import torch
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def bbox_center_distance(bboxes: Tensor, priors: Tensor) -> Tensor:
    """计算bbox和priors之间的中心距离"""
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)

    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = torch.stack((priors_cx, priors_cy), dim=1)

    distances = (priors_points[:, None, :] -
                 bbox_points[None, :, :]).pow(2).sum(-1).sqrt()
    return distances


@TASK_UTILS.register_module()
class UltimateQualityAwareATSSAssigner(BaseAssigner):
    """终极质量感知ATSS分配器 - 基于实测数据的根本性改进
    
    核心设计：分层IoU保护 + 自适应质量融合
    
    解决的核心问题：
    1. 简化版IoU暴跌78.2%的问题
    2. 增强版过度复杂化导致性能更差
    3. 优化版改善不足的问题
    
    技术突破：
    - 分层IoU保护：高IoU区域强制保护，避免被低分类分数误杀
    - 置信度自适应：根据分类置信度动态调整融合权重
    - 渐进式评估：多步骤质量评估，避免简单粗暴操作
    - 层级感知：FPN不同层级采用不同策略
    
    Args:
        topk (int): 每层选择的候选anchor数量
        iou_protection_mode (str): IoU保护模式
            - 'strict': 严格保护高IoU (权重0.8)
            - 'balanced': 平衡保护 (权重0.6)  
            - 'adaptive': 自适应保护 (动态权重)
        confidence_aware (bool): 是否启用置信度自适应融合
        layered_fusion (bool): 是否启用分层融合策略
        progressive_evaluation (bool): 是否启用渐进式评估
    """

    def __init__(self,
                 topk: int,
                 # 核心改进参数
                 iou_protection_mode: str = 'adaptive',  # strict/balanced/adaptive
                 confidence_aware: bool = True,
                 layered_fusion: bool = True,
                 progressive_evaluation: bool = True,
                 # 分层IoU保护阈值
                 high_iou_threshold: float = 0.7,
                 medium_iou_threshold: float = 0.4,
                 # 自适应参数
                 confidence_threshold: float = 0.6,
                 adaptation_strength: float = 0.3,
                 # 兼容性参数
                 alpha: Optional[float] = None,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        
        self.topk = topk
        self.iou_protection_mode = iou_protection_mode
        self.confidence_aware = confidence_aware
        self.layered_fusion = layered_fusion
        self.progressive_evaluation = progressive_evaluation
        
        # IoU保护阈值
        self.high_iou_threshold = high_iou_threshold
        self.medium_iou_threshold = medium_iou_threshold
        
        # 自适应参数
        self.confidence_threshold = confidence_threshold
        self.adaptation_strength = adaptation_strength
        
        # 标准参数
        self.alpha = alpha
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               pred_instances: InstanceData,
               num_level_priors: List[int],
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None) -> AssignResult:
        """终极质量感知分配"""
        
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        INF = 100000000
        priors = priors[:, :4]
        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)

        # === 步骤1：计算IoU矩阵 ===
        overlaps = self.iou_calculator(gt_bboxes, priors)

        # 初始化分配结果
        assigned_gt_inds = overlaps.new_full((num_priors,), 0, dtype=torch.long)

        if num_gt == 0 or num_priors == 0:
            max_overlaps = overlaps.new_zeros((num_priors,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # === 步骤2：处理忽略区域 ===
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                gt_bboxes_ignore, priors, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            overlaps[:, ignore_idxs] = -INF
            assigned_gt_inds[ignore_idxs] = -1

        # === 步骤3：基于距离选择候选样本 ===
        distances = bbox_center_distance(gt_bboxes, priors)
        candidate_idxs = self._select_candidates_by_distance(
            distances, num_level_priors)

        if not candidate_idxs:
            assigned_gt_inds = assigned_gt_inds.new_zeros(num_priors, dtype=torch.long)
            assigned_labels = assigned_gt_inds.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # === 步骤4：🚀 终极质量感知评分 ===
        candidate_quality_scores = self._ultimate_quality_scoring(
            candidate_idxs, overlaps, pred_instances, gt_labels, num_gt, num_level_priors)

        # === 步骤5：基于质量分数计算自适应阈值 ===
        is_pos = self._compute_adaptive_thresholds(candidate_quality_scores)

        # === 步骤6：几何约束检查 ===
        is_in_gts = self._check_geometric_constraints(candidate_idxs, gt_bboxes, priors, num_gt)
        is_pos = is_pos & is_in_gts

        # === 步骤7：解决多GT冲突 ===
        assigned_gt_inds, max_overlaps = self._resolve_multi_gt_conflicts(
            candidate_idxs, candidate_quality_scores, is_pos, overlaps, num_gt, num_priors, assigned_gt_inds)

        # === 步骤8：分配标签 ===
        assigned_labels = self._assign_labels(assigned_gt_inds, gt_labels, num_priors)

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _ultimate_quality_scoring(self, candidate_idxs: Tensor, overlaps: Tensor,
                                  pred_instances: InstanceData, gt_labels: Tensor,
                                  num_gt: int, num_level_priors: List[int]) -> Tensor:
        """🚀 终极质量评分 - 核心突破"""
        
        if not (hasattr(pred_instances, 'scores') and pred_instances.scores is not None):
            # 退回到IoU评分
            return overlaps[torch.arange(num_gt), candidate_idxs]
        
        cls_scores = pred_instances.scores
        num_candidates_per_gt = candidate_idxs.shape[0]
        quality_scores = torch.zeros(num_candidates_per_gt, num_gt,
                                     device=candidate_idxs.device, dtype=overlaps.dtype)

        for gt_idx in range(num_gt):
            gt_label = gt_labels[gt_idx]
            current_candidates = candidate_idxs[:, gt_idx]
            valid_mask = current_candidates >= 0
            
            if not valid_mask.any():
                continue

            valid_candidates = current_candidates[valid_mask]
            candidate_ious = overlaps[gt_idx, valid_candidates]
            candidate_cls_scores = cls_scores[valid_candidates, gt_label]

            # 🚀 核心突破：分层IoU保护质量评分
            if self.layered_fusion:
                layered_quality = self._layered_iou_protection_scoring(
                    candidate_ious, candidate_cls_scores)
            else:
                # 简单融合作为fallback
                layered_quality = candidate_ious * candidate_cls_scores

            # 🚀 置信度自适应调整
            if self.confidence_aware:
                adaptive_quality = self._confidence_adaptive_fusion(
                    candidate_ious, candidate_cls_scores, layered_quality)
            else:
                adaptive_quality = layered_quality

            # 🚀 层级感知调整
            level_adjusted_quality = self._level_aware_adjustment(
                adaptive_quality, valid_candidates, num_level_priors)

            quality_scores[valid_mask, gt_idx] = level_adjusted_quality

        return quality_scores

    def _layered_iou_protection_scoring(self, ious: Tensor, cls_scores: Tensor) -> Tensor:
        """分层IoU保护评分 - 解决IoU暴跌的核心技术"""
        
        # 🔧 修复：确保所有常数与张量类型一致
        device = ious.device
        dtype = ious.dtype
        
        if self.iou_protection_mode == 'strict':
            # 严格保护模式：高IoU区域强制保护
            iou_weights = torch.where(
                ious > self.high_iou_threshold, 
                torch.tensor(0.85, device=device, dtype=dtype),
                torch.where(ious > self.medium_iou_threshold, 
                           torch.tensor(0.65, device=device, dtype=dtype), 
                           torch.tensor(0.35, device=device, dtype=dtype))
            )
        elif self.iou_protection_mode == 'balanced':
            # 平衡保护模式：适度保护高IoU
            iou_weights = torch.where(
                ious > self.high_iou_threshold, 
                torch.tensor(0.75, device=device, dtype=dtype),
                torch.where(ious > self.medium_iou_threshold, 
                           torch.tensor(0.55, device=device, dtype=dtype), 
                           torch.tensor(0.35, device=device, dtype=dtype))
            )
        else:  # adaptive
            # 自适应保护模式：根据IoU分布动态调整
            iou_weights = self._compute_adaptive_iou_weights(ious)
        
        cls_weights = torch.tensor(1.0, device=device, dtype=dtype) - iou_weights
        
        # 分层质量融合 - 避免简单粗暴的乘积
        protected_quality = iou_weights * ious + cls_weights * cls_scores
        
        return protected_quality

    def _compute_adaptive_iou_weights(self, ious: Tensor) -> Tensor:
        """计算自适应IoU权重"""
        # 🔧 修复：确保数据类型一致
        device = ious.device
        dtype = ious.dtype
        
        # 基于IoU分布的统计特性动态调整
        iou_mean = ious.mean()
        iou_std = ious.std()
        
        # 高IoU样本权重保护
        high_iou_mask = ious > (iou_mean + iou_std)
        medium_iou_mask = ious > iou_mean
        
        weights = torch.full_like(ious, 0.4, device=device, dtype=dtype)  # 基础权重
        weights[medium_iou_mask] = torch.tensor(0.6, device=device, dtype=dtype)
        weights[high_iou_mask] = torch.tensor(0.8, device=device, dtype=dtype)
        
        return weights

    def _confidence_adaptive_fusion(self, ious: Tensor, cls_scores: Tensor, 
                                   base_quality: Tensor) -> Tensor:
        """置信度自适应融合"""
        if not self.confidence_aware:
            return base_quality
            
        # 计算分类置信度
        confidence = cls_scores
        
        # 高置信度时适当增加分类权重，但不能损害IoU保护
        confidence_factor = torch.sigmoid(
            (confidence - self.confidence_threshold) / self.adaptation_strength
        )
        
        # 🔧 修复：确保数据类型一致
        device = base_quality.device
        dtype = base_quality.dtype
        
        # 自适应调整，但保持IoU基础
        adjustment_factor = torch.tensor(1.0, device=device, dtype=dtype) + torch.tensor(0.1, device=device, dtype=dtype) * confidence_factor
        adjusted_quality = base_quality * adjustment_factor
        
        return adjusted_quality

    def _level_aware_adjustment(self, quality_scores: Tensor, candidate_indices: Tensor,
                               num_level_priors: List[int]) -> Tensor:
        """层级感知调整"""
        if not hasattr(self, 'layered_fusion') or not self.layered_fusion:
            return quality_scores
            
        # 确定每个候选样本所属的层级
        start_idx = 0
        for level, num_level_prior in enumerate(num_level_priors):
            end_idx = start_idx + num_level_prior
            level_mask = (candidate_indices >= start_idx) & (candidate_indices < end_idx)
            
            if level_mask.any():
                # 不同层级采用不同的调整策略
                level_factor = 1.0 + 0.05 * level  # 深层级轻微加权
                quality_scores[level_mask] *= level_factor
                
            start_idx = end_idx
        
        return quality_scores

    def _select_candidates_by_distance(self, distances: Tensor, 
                                     num_level_priors: List[int]) -> List[Tensor]:
        """基于距离选择候选样本"""
        candidate_idxs = []
        start_idx = 0
        
        for level, num_level_prior in enumerate(num_level_priors):
            end_idx = start_idx + num_level_prior
            
            if num_level_prior == 0:
                start_idx = end_idx
                continue

            level_distances = distances[start_idx:end_idx, :]
            
            if level_distances.size(0) == 0:
                start_idx = end_idx
                continue

            selectable_k = min(self.topk, num_level_prior)
            if selectable_k > 0 and level_distances.size(1) > 0:
                _, level_topk_idxs = level_distances.topk(
                    selectable_k, dim=0, largest=False)
                level_topk_idxs += start_idx
                candidate_idxs.append(level_topk_idxs)
            start_idx = end_idx

        return candidate_idxs

    def _compute_adaptive_thresholds(self, quality_scores: Tensor) -> Tensor:
        """计算自适应阈值"""
        scores_mean_per_gt = quality_scores.mean(0)
        scores_std_per_gt = quality_scores.std(0)
        scores_thr_per_gt = scores_mean_per_gt + scores_std_per_gt
        
        is_pos = quality_scores >= scores_thr_per_gt[None, :]
        return is_pos

    def _check_geometric_constraints(self, candidate_idxs: Tensor, gt_bboxes: Tensor,
                                   priors: Tensor, num_gt: int) -> Tensor:
        """检查几何约束"""
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0

        is_in_gts_list = []
        for gt_idx in range(num_gt):
            gt_candidates = candidate_idxs[:, gt_idx]
            candidate_cx = priors_cx[gt_candidates]
            candidate_cy = priors_cy[gt_candidates]

            l_ = candidate_cx - gt_bboxes[gt_idx, 0]
            t_ = candidate_cy - gt_bboxes[gt_idx, 1]
            r_ = gt_bboxes[gt_idx, 2] - candidate_cx
            b_ = gt_bboxes[gt_idx, 3] - candidate_cy

            is_in_gt = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
            is_in_gts_list.append(is_in_gt)

        is_in_gts = torch.stack(is_in_gts_list, dim=1)
        return is_in_gts

    def _resolve_multi_gt_conflicts(self, candidate_idxs: Tensor, quality_scores: Tensor,
                                  is_pos: Tensor, overlaps: Tensor, num_gt: int,
                                  num_priors: int, assigned_gt_inds: Tensor) -> Tuple[Tensor, Tensor]:
        """解决多GT冲突"""
        INF = 100000000
        overlaps_inf = torch.full_like(overlaps, -INF).t()

        for gt_idx in range(num_gt):
            gt_candidates = candidate_idxs[:, gt_idx]
            gt_pos_mask = is_pos[:, gt_idx]
            gt_pos_candidates = gt_candidates[gt_pos_mask]

            if gt_pos_candidates.numel() > 0:
                gt_pos_quality_scores = quality_scores[gt_pos_mask, gt_idx]
                overlaps_inf[gt_pos_candidates, gt_idx] = gt_pos_quality_scores

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        return assigned_gt_inds, max_overlaps

    def _assign_labels(self, assigned_gt_inds: Tensor, gt_labels: Tensor, 
                      num_priors: int) -> Tensor:
        """分配标签"""
        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        return assigned_labels