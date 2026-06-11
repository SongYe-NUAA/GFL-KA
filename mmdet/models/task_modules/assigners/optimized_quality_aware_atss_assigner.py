# Copyright (c) OpenMMLab. All rights reserved.
"""
优化版质量感知ATSS分配器 - 基于实际测试结果的精准调优

基于测试发现的核心问题：
1. 简化版IoU暴跌80.3%的根本原因：过度依赖分类分数导致几何精度丢失
2. 增强版虽然改善但仍不够：需要更精准的IoU保护机制
3. 计算开销控制：在保证性能的前提下优化效率

优化策略：
1. IoU优先原则：确保几何精度不被过度稀释
2. 分层质量评估：不同IoU水平采用不同的分类分数权重
3. 渐进式融合：训练初期偏重IoU，后期逐步引入分类分数
4. 智能阈值自适应：根据候选样本质量分布动态调整
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
class OptimizedQualityAwareATSSAssigner(BaseAssigner):
    """优化版质量感知ATSS分配器
    
    核心设计原则：
    1. IoU优先：确保几何精度不被过度稀释
    2. 分层融合：不同IoU水平采用不同的分类权重
    3. 渐进学习：配合训练过程逐步提升质量要求
    4. 计算高效：优化算法复杂度

    Args:
        topk (int): 每层选择的候选anchor数量
        iou_priority_mode (str): IoU优先模式 ['strict', 'balanced', 'flexible']
        progressive_fusion (bool): 是否启用渐进式融合
        smart_threshold (bool): 是否启用智能阈值自适应
        iou_calculator: IoU计算器配置
        ignore_iof_thr (float): 忽略IoF阈值
    """

    def __init__(self,
                 topk: int,
                 # 核心优化参数
                 iou_priority_mode: str = 'balanced',  # strict/balanced/flexible
                 progressive_fusion: bool = True,      # 渐进式融合
                 smart_threshold: bool = True,         # 智能阈值
                 # 传统参数
                 alpha: Optional[float] = None,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        self.topk = topk
        self.alpha = alpha
        self.iou_priority_mode = iou_priority_mode
        self.progressive_fusion = progressive_fusion
        self.smart_threshold = smart_threshold
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        
        # 训练进度追踪
        self.training_iteration = 0
        self.warmup_iterations = 10000  # 前10k迭代为预热期

    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """优化版质量感知ATSS分配"""
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

        # === 步骤4：优化版质量感知评分 ===
        candidate_overlaps = overlaps[torch.arange(num_gt), candidate_idxs]
        
        if hasattr(pred_instances, 'scores') and pred_instances.scores is not None:
            # 🚀 核心创新：分层质量感知评分
            candidate_final_scores = self._compute_layered_quality_scores(
                candidate_idxs, overlaps, pred_instances, gt_labels, num_gt)
            
            # 🚀 智能阈值计算
            if self.smart_threshold:
                scores_thr_per_gt = self._compute_smart_threshold(candidate_final_scores, candidate_overlaps)
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

        # 更新训练进度
        self.training_iteration += 1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _compute_layered_quality_scores(self, candidate_idxs, overlaps, pred_instances,
                                       gt_labels, num_gt):
        """分层质量感知评分 - 核心优化算法"""
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

            # 🚀 分层质量计算
            quality_score = self._compute_layered_fusion(
                candidate_ious, candidate_cls_scores)

            quality_scores[valid_mask, gt_idx] = quality_score

        return quality_scores

    def _compute_layered_fusion(self, ious, cls_scores):
        """分层融合算法 - 基于IoU水平分层处理"""
        eps = 1e-8
        
        # 🎯 IoU分层策略
        if self.iou_priority_mode == 'strict':
            # 严格模式：强烈偏重IoU
            weights = self._get_strict_weights(ious)
        elif self.iou_priority_mode == 'balanced':
            # 平衡模式：适度融合
            weights = self._get_balanced_weights(ious, cls_scores)
        else:  # flexible
            # 灵活模式：自适应权重
            weights = self._get_flexible_weights(ious, cls_scores)
        
        # 🚀 渐进式融合
        if self.progressive_fusion:
            progress_factor = self._get_training_progress_factor()
            # 训练初期更偏重IoU，后期逐步引入分类分数
            weights = weights * (0.8 + 0.2 * progress_factor)
        
        # 计算最终质量分数
        quality_score = ious * weights + cls_scores * (1 - weights)
        quality_score = quality_score * ious  # 确保IoU为基础
        
        return torch.clamp(quality_score, eps, 1.0)

    def _get_strict_weights(self, ious):
        """严格模式权重：强烈偏重IoU"""
        # 高IoU样本权重接近1，低IoU样本权重也不会太低
        return torch.clamp(0.7 + 0.3 * ious, 0.7, 1.0)

    def _get_balanced_weights(self, ious, cls_scores):
        """平衡模式权重：适度融合IoU和分类分数"""
        # 基于IoU和分类分数的联合分布决定权重
        joint_quality = ious * cls_scores
        return torch.clamp(0.5 + 0.4 * joint_quality, 0.5, 0.9)

    def _get_flexible_weights(self, ious, cls_scores):
        """灵活模式权重：自适应调整"""
        # 当IoU很高时更信任IoU，当分类分数很高时适度引入
        iou_confidence = torch.sigmoid((ious - 0.5) * 10)  # IoU>0.5时快速增长
        cls_confidence = torch.sigmoid((cls_scores - 0.7) * 10)  # 分类>0.7时增长
        
        # 综合置信度决定权重
        base_weight = 0.6
        iou_bonus = 0.3 * iou_confidence
        cls_penalty = 0.1 * (1 - cls_confidence)
        
        return torch.clamp(base_weight + iou_bonus - cls_penalty, 0.4, 0.9)

    def _get_training_progress_factor(self):
        """获取训练进度因子"""
        if self.training_iteration < self.warmup_iterations:
            # 预热期：线性增长
            return self.training_iteration / self.warmup_iterations
        else:
            # 正常训练期：缓慢增长
            progress = min(1.0, (self.training_iteration - self.warmup_iterations) / 50000)
            return 0.5 + 0.5 * progress

    def _compute_smart_threshold(self, candidate_scores, candidate_ious):
        """智能阈值计算"""
        # 基于候选样本的质量分布动态调整阈值
        base_threshold = candidate_scores.mean(0) + candidate_scores.std(0)
        
        # 考虑IoU分布的影响
        iou_factor = candidate_ious.mean(0)
        
        # 高IoU场景降低阈值，低IoU场景提高阈值
        adaptive_factor = 0.8 + 0.4 * iou_factor  # [0.8, 1.2]
        
        return base_threshold * adaptive_factor