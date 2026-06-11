# Copyright (c) OpenMMLab. All rights reserved.
"""
增强版质量感知ATSS分配器 - 多维度优化版本

核心创新：
1. 自适应质量权重：根据场景复杂度动态调整IoU和分类权重
2. 层级感知评估：不同FPN层级采用不同质量策略
3. 上下文感知机制：考虑周围anchor分布密度
4. 渐进式阈值：训练过程中动态调整质量阈值
5. 高效批量计算：优化大规模场景的计算性能

优势：
- 智能自适应：无需手动调参，自动适应不同场景
- 多维度质量感知：从IoU、分类、层级、上下文多个维度评估质量
- 高效稳定：优化计算性能，提升训练稳定性
- 渐进学习：配合训练进度逐步提升质量要求
"""

import warnings
import math
from typing import List, Optional
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
class EnhancedQualityAwareATSSAssigner(BaseAssigner):
    """增强版质量感知ATSS分配器
    
    主要改进：
    1. 自适应权重机制：根据场景复杂度自动调整IoU和分类权重
    2. 层级感知处理：不同FPN层级采用不同的质量评估策略
    3. 上下文感知：考虑anchor周围环境的复杂度
    4. 渐进式学习：配合训练进度动态调整质量要求
    5. 计算优化：批量化操作提升大规模场景性能

    Args:
        topk (int): 每层选择的候选anchor数量
        adaptive_weight (bool): 是否启用自适应权重机制
        level_aware (bool): 是否启用层级感知评估
        context_aware (bool): 是否启用上下文感知机制
        progressive_threshold (bool): 是否启用渐进式阈值
        iou_calculator: IoU计算器配置
        ignore_iof_thr (float): 忽略IoF阈值
    """

    def __init__(self,
                 topk: int,
                 # 核心功能开关
                 adaptive_weight: bool = True,      # 自适应权重
                 level_aware: bool = True,          # 层级感知
                 context_aware: bool = True,        # 上下文感知
                 progressive_threshold: bool = True, # 渐进式阈值
                 # 传统参数
                 alpha: Optional[float] = None,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        self.topk = topk
        self.alpha = alpha
        self.adaptive_weight = adaptive_weight
        self.level_aware = level_aware
        self.context_aware = context_aware
        self.progressive_threshold = progressive_threshold
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        
        # 训练进度追踪（用于渐进式阈值）
        self.training_iteration = 0
        self.total_iterations = 100000  # 可根据实际训练设置调整

    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """增强版质量感知ATSS分配"""
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

        # === 步骤3：基于距离选择初始候选样本 ===
        distances = bbox_center_distance(gt_bboxes, priors)
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

        if not candidate_idxs:
            assigned_gt_inds = assigned_gt_inds.new_zeros(num_priors, dtype=torch.long)
            assigned_labels = assigned_gt_inds.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # === 步骤4：增强版质量感知评分 ===
        candidate_overlaps = overlaps[torch.arange(num_gt), candidate_idxs]
        
        if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
            # 🚀 核心创新：多维度质量感知评分
            candidate_final_scores = self._compute_enhanced_quality_scores(
                candidate_idxs, overlaps, pred_instances, gt_labels, num_gt, num_level_priors)
            
            # 🚀 渐进式阈值计算
            if self.progressive_threshold:
                scores_thr_per_gt = self._compute_progressive_threshold(candidate_final_scores)
            else:
                scores_mean_per_gt = candidate_final_scores.mean(0)
                scores_std_per_gt = candidate_final_scores.std(0)
                scores_thr_per_gt = scores_mean_per_gt + scores_std_per_gt

            is_pos = candidate_final_scores >= scores_thr_per_gt[None, :]
        else:
            # 退回原始ATSS
            candidate_final_scores = candidate_overlaps
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # === 步骤5：几何约束检查 ===
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
        is_pos = is_pos & is_in_gts

        # === 步骤6：处理多GT分配冲突 ===
        overlaps_inf = torch.full_like(overlaps, -INF).t()

        for gt_idx in range(num_gt):
            gt_candidates = candidate_idxs[:, gt_idx]
            gt_pos_mask = is_pos[:, gt_idx]
            gt_pos_candidates = gt_candidates[gt_pos_mask]

            if gt_pos_candidates.numel() > 0:
                if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
                    gt_pos_final_scores = candidate_final_scores[gt_pos_mask, gt_idx]
                    overlaps_inf[gt_pos_candidates, gt_idx] = gt_pos_final_scores
                else:
                    overlaps_inf[gt_pos_candidates, gt_idx] = overlaps[gt_idx, gt_pos_candidates]

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        # 分配标签
        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        # 更新训练进度（用于渐进式阈值）
        self.training_iteration += 1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _compute_enhanced_quality_scores(self, candidate_idxs, overlaps, pred_instances,
                                        gt_labels, num_gt, num_level_priors):
        """计算增强版质量分数 - 多维度质量感知"""
        cls_scores = pred_instances.scores
        num_candidates_per_gt = candidate_idxs.shape[0]
        quality_scores = torch.zeros(num_candidates_per_gt, num_gt,
                                   device=candidate_idxs.device, dtype=overlaps.dtype)

        # 🚀 预计算层级信息（用于层级感知）
        level_indices = self._get_level_indices(candidate_idxs, num_level_priors) if self.level_aware else None
        
        # 🚀 预计算上下文复杂度（用于上下文感知）
        context_complexity = self._compute_context_complexity(candidate_idxs, overlaps) if self.context_aware else None

        for gt_idx in range(num_gt):
            gt_label = gt_labels[gt_idx]
            current_candidates = candidate_idxs[:, gt_idx]
            valid_mask = current_candidates >= 0
            if not valid_mask.any():
                continue

            valid_candidates = current_candidates[valid_mask]
            candidate_ious = overlaps[gt_idx, valid_candidates]
            candidate_cls_scores = cls_scores[valid_candidates, gt_label]

            # 🚀 多维度质量感知计算
            if self.adaptive_weight or self.level_aware or self.context_aware:
                enhanced_quality = self._compute_multi_dimensional_quality(
                    valid_candidates, candidate_ious, candidate_cls_scores,
                    level_indices, context_complexity, valid_mask)
            else:
                # 简单版本：IoU × 分类分数
                eps = 1e-8
                enhanced_quality = candidate_ious * torch.clamp(candidate_cls_scores, eps, 1.0)

            quality_scores[valid_mask, gt_idx] = enhanced_quality

        return quality_scores

    def _compute_multi_dimensional_quality(self, candidates, ious, cls_scores,
                                         level_indices, context_complexity, valid_mask):
        """多维度质量计算核心函数 - 优化版本"""
        eps = 1e-8
        
        # 🎯 核心改进：更保守的融合策略，确保IoU基础质量
        base_quality = ious * torch.clamp(cls_scores, eps, 1.0)
        
        if not (self.adaptive_weight or self.level_aware or self.context_aware):
            return base_quality

        # 🚀 自适应权重计算 - 更保守的调整
        if self.adaptive_weight:
            # 计算场景复杂度，但限制调整幅度
            iou_mean = ious.mean() + eps
            cls_mean = cls_scores.mean() + eps
            
            # 当IoU和分类分数都较高时，保持原始策略
            if iou_mean > 0.5 and cls_mean > 0.6:
                # 高质量场景：轻微调整
                adaptive_factor = 0.9 + 0.1 * cls_mean  # [0.9, 1.0]
            else:
                # 低质量场景：更依赖IoU稳定性
                iou_reliability = torch.clamp(iou_mean * 2.0, 0.0, 1.0)
                adaptive_factor = 0.7 + 0.3 * iou_reliability  # [0.7, 1.0]
            
            base_quality = base_quality * adaptive_factor

        # 🚀 层级感知调整 - 减少调整强度
        if self.level_aware and level_indices is not None:
            level_weights = self._compute_level_weights(level_indices[valid_mask])
            # 更温和的层级调整
            level_factor = 0.95 + 0.1 * level_weights.mean()  # [0.95, 1.05]
            base_quality = base_quality * level_factor

        # 🚀 上下文感知调整 - 仅在高复杂度时启用
        if self.context_aware and context_complexity is not None:
            context_weights = context_complexity[valid_mask]
            avg_complexity = context_weights.mean()
            
            # 只在复杂度超过阈值时调整
            if avg_complexity > 0.3:
                context_factor = 0.9 + 0.2 * (1 - avg_complexity)  # [0.9, 1.1]
                base_quality = base_quality * context_factor

        return base_quality

    def _get_level_indices(self, candidate_idxs, num_level_priors):
        """获取每个候选样本所属的FPN层级"""
        level_indices = torch.zeros(candidate_idxs.shape[0], device=candidate_idxs.device)
        start_idx = 0
        for level, num_level_prior in enumerate(num_level_priors):
            end_idx = start_idx + num_level_prior
            mask = (candidate_idxs >= start_idx) & (candidate_idxs < end_idx)
            level_indices[mask.any(dim=1)] = level
            start_idx = end_idx
        return level_indices

    def _compute_level_weights(self, level_indices):
        """计算层级权重：浅层偏重IoU，深层偏重分类分数"""
        # 归一化层级索引到[0,1]，然后计算权重
        max_level = level_indices.max() + 1e-8
        normalized_levels = level_indices / max_level
        # 浅层(0)权重接近1(偏重IoU)，深层权重接近0(偏重分类)
        return 1.0 - normalized_levels * 0.6  # 权重范围[0.4, 1.0]

    def _compute_context_complexity(self, candidate_idxs, overlaps):
        """计算上下文复杂度：分析候选样本周围的密度"""
        num_candidates = candidate_idxs.shape[0]
        complexity = torch.zeros(num_candidates, device=candidate_idxs.device)
        
        # 简化版本：基于IoU分布的方差计算复杂度
        for i in range(num_candidates):
            # 计算当前候选样本的IoU方差作为复杂度指标
            candidate_ious = overlaps[:, candidate_idxs[i]]
            complexity[i] = candidate_ious.std()
        
        # 归一化到[0,1]
        complexity = torch.sigmoid(complexity * 5.0 - 2.5)
        return complexity

    def _compute_progressive_threshold(self, candidate_scores):
        """计算渐进式质量阈值"""
        base_threshold = candidate_scores.mean(0) + candidate_scores.std(0)
        
        # 训练进度比例
        progress_ratio = min(1.0, self.training_iteration / self.total_iterations)
        
        # 训练初期放松阈值，后期收紧
        # 初期: 0.8x base_threshold, 后期: 1.2x base_threshold
        adaptive_factor = 0.8 + 0.4 * progress_ratio
        
        return base_threshold * adaptive_factor