      
# Copyright (c) OpenMMLab. All rights reserved.
"""
质量感知ATSS分配器 - 直接融合几何距离和质量一致性

核心设计：
1. 统一评分机制：几何分数 ⊕ 质量一致性分数
2. 自适应权重：基于预测置信度动态调整
3. 直接质量筛选：一步到位的高质量样本选择

优势：
- 简洁高效，无复杂的阶段划分
- 自适应融合，无需手动调参
- 直接质量导向，提升训练效率
"""

import warnings
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
class QualityAwareATSSAssigner(BaseAssigner):
    """质量感知的ATSS分配器 - 简化版本

    核心改进：
    final_score = IoU × 分类分数
    
    优势：
    - 简单直接：避免复杂的权重调整和参数优化
    - 质量感知：同时考虑几何精度(IoU)和分类置信度
    - 兼容性好：可以无缝替换原始ATSS

    Args:
        topk (int): 每层选择的候选anchor数量
        alpha (float, optional): DDOD中的cost参数（保留兼容性）
        enable_quality_aware (bool): 是否启用质量感知评分
        iou_calculator: IoU计算器配置
        ignore_iof_thr (float): 忽略IoF阈值
    """

    def __init__(self,
                 topk: int,
                 alpha: Optional[float] = None,
                 # 简化参数，保留必要的配置
                 enable_quality_aware: bool = True,  # 是否启用质量感知（IoU×分类分数）
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        self.topk = topk
        self.alpha = alpha
        self.enable_quality_aware = enable_quality_aware
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr


    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """质量感知的ATSS分配 - 直接融合版本

        核心流程：
        1. 计算IoU和几何距离
        2. 基于距离选择初始候选样本
        3. 质量感知融合评分
        4. 应用自适应阈值选择正样本
        """
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
        # 🚀 修复：与原始ATSS保持一致的距离矩阵格式
        distances = bbox_center_distance(gt_bboxes, priors)  # [num_priors, num_gt]

        candidate_idxs = []
        start_idx = 0
        for level, num_level_prior in enumerate(num_level_priors):
            end_idx = start_idx + num_level_prior

            # 安全检查：跳过空层级
            if num_level_prior == 0:
                start_idx = end_idx
                continue

            # 🚀 修复：与原始ATSS保持一致的距离切片方式
            level_distances = distances[start_idx:end_idx, :]

            # 多重安全检查
            if level_distances.size(0) == 0:
                start_idx = end_idx
                continue

            # 🚀 修复：与原始ATSS保持一致的topk选择
            selectable_k = min(self.topk, num_level_prior)
            if selectable_k > 0 and level_distances.size(1) > 0:
                _, level_topk_idxs = level_distances.topk(
                    selectable_k, dim=0, largest=False)  # 沿priors维度选择
                level_topk_idxs += start_idx
                candidate_idxs.append(level_topk_idxs)
            start_idx = end_idx

        # 处理没有候选样本的情况
        if not candidate_idxs:
            # 如果没有有效候选样本，返回空分配结果
            assigned_gt_inds = assigned_gt_inds.new_zeros(num_priors, dtype=torch.long)
            assigned_labels = assigned_gt_inds.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 🚀 修复：与原始ATSS保持一致的候选索引格式 [total_candidates, num_gt]
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # === 步骤4：质量感知融合评分（移除无用的排序） ===
        # 直接进入质量感知自适应阈值计算

        # === 步骤5：计算自适应阈值并选择正样本 ===
        # 获取候选样本的IoU分数
        candidate_overlaps = overlaps[torch.arange(num_gt), candidate_idxs]

        # 🚀 简化方案：直接用 IoU × 分类分数 作为最终评分
        if (self.enable_quality_aware and hasattr(pred_instances, 'scores') 
                and pred_instances.scores is not None):
            # 计算 IoU × 分类分数
            candidate_final_scores = self._compute_iou_cls_product(
                candidate_idxs, overlaps, pred_instances, gt_labels, num_gt)
            
            # 基于 IoU × 分类分数 计算自适应阈值
            scores_mean_per_gt = candidate_final_scores.mean(0)
            scores_std_per_gt = candidate_final_scores.std(0)
            scores_thr_per_gt = scores_mean_per_gt + scores_std_per_gt

            is_pos = candidate_final_scores >= scores_thr_per_gt[None, :]
        else:
            # 原始ATSS：仅基于IoU
            candidate_final_scores = candidate_overlaps
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # === 步骤6：确保候选样本在GT框内 ===
        # 🚀 关键修复：使用原始索引避免CUDA索引越界
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0

        # 计算每个GT与其候选样本的几何约束
        is_in_gts_list = []
        for gt_idx in range(num_gt):
            gt_candidates = candidate_idxs[:, gt_idx]  # 当前GT的候选索引

            # 使用原始prior索引计算几何约束
            candidate_cx = priors_cx[gt_candidates]
            candidate_cy = priors_cy[gt_candidates]

            # 计算left, top, right, bottom距离
            l_ = candidate_cx - gt_bboxes[gt_idx, 0]
            t_ = candidate_cy - gt_bboxes[gt_idx, 1]
            r_ = gt_bboxes[gt_idx, 2] - candidate_cx
            b_ = gt_bboxes[gt_idx, 3] - candidate_cy

            # 检查是否在GT框内（所有距离都>0.01）
            is_in_gt = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
            is_in_gts_list.append(is_in_gt)

        # 将结果重新组织为与is_pos相同的形状
        is_in_gts = torch.stack(is_in_gts_list, dim=1)  # [num_candidates, num_gt]

        # 结合IoU阈值和GT框内约束
        is_pos = is_pos & is_in_gts

        # === 步骤7：处理多GT分配冲突 ===
        # 使用最终分数来解决多GT冲突，保持评分标准一致性
        overlaps_inf = torch.full_like(overlaps, -INF).t()  # [num_priors, num_gt]

        # 为每个GT填充其正样本的最终分数值
        for gt_idx in range(num_gt):
            gt_candidates = candidate_idxs[:, gt_idx]  # 当前GT的候选索引
            gt_pos_mask = is_pos[:, gt_idx]  # 当前GT的正样本掩码

            # 获取当前GT的正样本索引
            gt_pos_candidates = gt_candidates[gt_pos_mask]

            # 填充最终分数用于冲突解决
            if gt_pos_candidates.numel() > 0:
                if (self.enable_quality_aware and hasattr(pred_instances, 'scores') 
                        and pred_instances.scores is not None):
                    # 使用 IoU × 分类分数 解决冲突
                    gt_pos_final_scores = candidate_final_scores[gt_pos_mask, gt_idx]
                    overlaps_inf[gt_pos_candidates, gt_idx] = gt_pos_final_scores
                else:
                    # 退回到IoU（当没有分类分数时）
                    overlaps_inf[gt_pos_candidates, gt_idx] = overlaps[gt_idx, gt_pos_candidates]

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        # 分配标签
        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _compute_iou_cls_product(self, candidate_idxs, overlaps, pred_instances,
                                 gt_labels, num_gt):
        """计算 IoU × 分类分数 的简单质量评分"""
        cls_scores = pred_instances.scores
        # 创建正确维度的质量分数矩阵
        num_candidates_per_gt = candidate_idxs.shape[0]
        quality_scores = torch.zeros(num_candidates_per_gt, num_gt,
                                     device=candidate_idxs.device, dtype=overlaps.dtype)

        for gt_idx in range(num_gt):
            gt_label = gt_labels[gt_idx]
            current_candidates = candidate_idxs[:, gt_idx]

            # 过滤有效候选样本
            valid_mask = current_candidates >= 0
            if not valid_mask.any():
                continue

            valid_candidates = current_candidates[valid_mask]

            # 获取候选样本的IoU和分类分数
            candidate_ious = overlaps[gt_idx, valid_candidates]
            candidate_cls_scores = cls_scores[valid_candidates, gt_label]

            # 🚀 简化方案：直接计算 IoU × 分类分数
            # 添加小的epsilon避免数值问题
            eps = 1e-8
            iou_cls_product = candidate_ious * torch.clamp(candidate_cls_scores, eps, 1.0)

            # 存储质量分数到正确位置
            quality_scores[valid_mask, gt_idx] = iou_cls_product

        return quality_scores

    