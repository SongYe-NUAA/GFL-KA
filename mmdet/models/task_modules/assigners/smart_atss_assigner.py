# Copyright (c) OpenMMLab. All rights reserved.
"""
🚀 智能ATSS分配器 - 轻量级高效版本

核心创新：
1. 🎯 动态策略切换：基于场景特征自动选择最优策略
2. 🔄 在线学习：实时调整分配参数，无需预训练
3. 📊 多维度评估：综合IoU、置信度、几何特征的智能评分
4. ⚡ 轻量级设计：最小计算开销，最大性能提升

设计理念：
- 保持ATSS的简洁性，增强自适应能力
- 无需复杂元学习，使用简单有效的启发式算法
- 实时响应场景变化，动态优化分配策略
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from collections import deque
import math


def bbox_center_distance(bboxes: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
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


class SceneAnalyzer:
    """🎯 场景分析器：快速分析当前分配场景特征"""
    
    @staticmethod
    def analyze_scene(gt_bboxes, priors, cls_scores=None):
        """
        快速场景分析，返回场景类型和推荐策略
        
        Returns:
            scene_type: 'dense', 'sparse', 'mixed', 'challenging'
            strategy_params: dict with recommended parameters
        """
        num_gt = gt_bboxes.size(0)
        num_priors = priors.size(0)
        
        if num_gt == 0:
            return 'empty', {'strategy': 'standard'}
        
        # === 1. 密度分析 ===
        density_ratio = num_gt / num_priors
        
        # === 2. 尺度分析 ===
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        area_variance = gt_areas.std() / (gt_areas.mean() + 1e-8)
        
        # === 3. 分布分析 ===
        if num_gt > 1:
            gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
            distances = torch.cdist(gt_centers, gt_centers)
            distances = distances[distances > 0]
            spatial_variance = distances.std() / (distances.mean() + 1e-8)
        else:
            spatial_variance = 0.0
        
        # === 4. 置信度分析 ===
        if cls_scores is not None:
            conf_mean = cls_scores.sigmoid().mean()
            conf_std = cls_scores.sigmoid().std()
            conf_reliability = 1.0 - conf_std  # 标准差越小越可靠
        else:
            conf_mean = 0.5
            conf_reliability = 0.5
        
        # === 5. 场景分类 ===
        if density_ratio > 0.01:  # 高密度
            if area_variance > 1.0:  # 尺度差异大
                scene_type = 'challenging'
                strategy = 'multi_scale_aware'
            else:
                scene_type = 'dense'
                strategy = 'density_optimized'
        else:  # 低密度
            if spatial_variance > 1.0:  # 分布离散
                scene_type = 'sparse'
                strategy = 'distance_prioritized'
            else:
                scene_type = 'mixed'
                strategy = 'balanced'
        
        # === 6. 策略参数推荐 ===
        strategy_params = {
            'strategy': strategy,
            'quality_weight': min(0.4, conf_reliability * 0.6),  # 基于置信度可靠性
            'distance_weight': max(0.3, 1.0 - density_ratio * 10),  # 密度越高距离权重越低
            'iou_threshold_factor': 1.0 + area_variance * 0.2,  # 尺度差异大时提高阈值
            'topk_multiplier': max(1.0, density_ratio * 50),  # 密度高时增加候选数
            'confidence_threshold': max(0.1, conf_mean - conf_std)  # 动态置信度阈值
        }
        
        return scene_type, strategy_params


@TASK_UTILS.register_module()
class SmartATSSAssigner(BaseAssigner):
    """🚀 智能ATSS分配器 - 轻量级自适应版本
    
    核心特性：
    1. 🎯 场景感知：自动识别分配场景并调整策略
    2. 📊 多维度评分：IoU + 置信度 + 几何特征的智能融合
    3. 🔄 在线学习：基于历史性能实时调整参数
    4. ⚡ 高效实现：最小计算开销，最大性能收益
    """
    
    def __init__(self,
                 topk: int = 9,
                 # 🎯 自适应参数
                 enable_scene_adaptation: bool = True,
                 enable_online_learning: bool = True,
                 # 📊 评分权重 - 优化：增强IoU主导地位
                 base_iou_weight: float = 0.7,
                 base_quality_weight: float = 0.2,
                 base_distance_weight: float = 0.1,
                 # 🔄 学习参数
                 learning_rate: float = 0.01,
                 performance_window: int = 50,
                 adaptation_threshold: float = 0.02,
                 # 🛡️ 稳定性参数
                 min_quality_weight: float = 0.05,
                 max_quality_weight: float = 0.45,
                 warmup_iterations: int = 10,
                 # 基础参数
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        
        self.topk = topk
        self.enable_scene_adaptation = enable_scene_adaptation
        self.enable_online_learning = enable_online_learning
        
        # 基础权重
        self.base_iou_weight = base_iou_weight
        self.base_quality_weight = base_quality_weight
        self.base_distance_weight = base_distance_weight
        
        # 当前自适应权重
        self.current_iou_weight = base_iou_weight
        self.current_quality_weight = base_quality_weight
        self.current_distance_weight = base_distance_weight
        
        # 学习参数
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.min_quality_weight = min_quality_weight
        self.max_quality_weight = max_quality_weight
        self.warmup_iterations = warmup_iterations
        
        # 场景分析器
        self.scene_analyzer = SceneAnalyzer()
        
        # 性能跟踪
        self.performance_history = deque(maxlen=performance_window)
        self.iteration_count = 0
        
        # 基础组件
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
    
    def assign(self,
               pred_instances: InstanceData,
               num_level_priors: List[int],
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None) -> AssignResult:
        """🚀 智能自适应分配"""
        
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors[:, :4]
        gt_labels = gt_instances.labels
        cls_scores = getattr(pred_instances, 'scores', None)
        
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None
        
        self.iteration_count += 1
        
        # === 1. 场景分析与策略选择 ===
        if self.enable_scene_adaptation:
            scene_type, strategy_params = self.scene_analyzer.analyze_scene(
                gt_bboxes, priors, cls_scores
            )
            # 根据场景调整参数
            self._adapt_to_scene(strategy_params)
        
        # === 2. 基础ATSS分配 ===
        assign_result = self._enhanced_atss_assign(
            gt_bboxes, priors, gt_labels, num_level_priors, 
            cls_scores, gt_bboxes_ignore
        )
        
        # === 3. 性能评估与在线学习 ===
        if self.enable_online_learning and self.iteration_count > self.warmup_iterations:
            performance = self._evaluate_assignment_performance(assign_result, gt_bboxes, priors)
            self.performance_history.append(performance)
            self._online_parameter_update()
        
        return assign_result
    
    def _adapt_to_scene(self, strategy_params: Dict):
        """🎯 根据场景参数调整分配策略"""
        recommended_quality_weight = strategy_params.get('quality_weight', self.base_quality_weight)
        recommended_distance_weight = strategy_params.get('distance_weight', self.base_distance_weight)
        
        # 平滑调整权重，避免剧烈变化
        alpha = 0.1  # 调整速度
        self.current_quality_weight = (1 - alpha) * self.current_quality_weight + alpha * recommended_quality_weight
        self.current_distance_weight = (1 - alpha) * self.current_distance_weight + alpha * recommended_distance_weight
        
        # 确保权重和为1
        total_weight = self.current_quality_weight + self.current_distance_weight
        remaining_weight = 1.0 - total_weight
        self.current_iou_weight = max(0.3, remaining_weight)  # IoU权重至少保持0.3
        
        # 重新归一化
        total = self.current_iou_weight + self.current_quality_weight + self.current_distance_weight
        self.current_iou_weight /= total
        self.current_quality_weight /= total
        self.current_distance_weight /= total
    
    def _enhanced_atss_assign(self, gt_bboxes, priors, gt_labels, num_level_priors, cls_scores, gt_bboxes_ignore):
        """🚀 增强版ATSS分配算法"""
        INF = 100000000.0  # 🔧 修复：使用float类型的INF
        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)
        
        # 计算IoU矩阵
        overlaps = self.iou_calculator(gt_bboxes, priors)
        
        # 初始化分配结果
        assigned_gt_inds = overlaps.new_full((num_priors,), 0, dtype=torch.long)
        
        if num_gt == 0 or num_priors == 0:
            max_overlaps = overlaps.new_zeros((num_priors,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # 处理忽略区域
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, priors, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            overlaps[:, ignore_idxs] = -INF  # 这里使用的INF已经是float类型
            assigned_gt_inds[ignore_idxs] = -1
        
        # === 🚀 智能候选样本选择 ===
        candidate_idxs = self._smart_candidate_selection(
            gt_bboxes, priors, overlaps, num_level_priors, cls_scores
        )
        
        if candidate_idxs.numel() == 0:
            assigned_labels = assigned_gt_inds.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)
        
        # === 🚀 多维度评分与自适应阈值 ===
        # 注意：这里暂时不使用candidate_overlaps，因为索引操作复杂
        # candidate_overlaps = overlaps[torch.arange(num_gt)[:, None], candidate_idxs]
        
        # 计算综合分数
        composite_scores = self._compute_composite_scores(
            candidate_idxs, overlaps, gt_bboxes, priors, cls_scores, gt_labels
        )
        
        # 自适应阈值计算
        adaptive_thresholds = self._compute_adaptive_thresholds(composite_scores)
        
        # 正样本选择
        is_pos = composite_scores >= adaptive_thresholds[None, :]
        
        # === 🚀 几何约束检查 ===
        is_in_gts = self._check_geometric_constraints(candidate_idxs, gt_bboxes, priors, num_gt)
        is_pos = is_pos & is_in_gts
        
        # === 🚀 冲突解决 ===
        assigned_gt_inds = self._resolve_conflicts(
            candidate_idxs, is_pos, overlaps, num_gt, num_priors
        )
        
        # 计算最大IoU和分配标签
        max_overlaps = overlaps.new_zeros((num_priors,))
        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1, dtype=torch.long)
        
        for gt_idx in range(num_gt):
            gt_mask = assigned_gt_inds == (gt_idx + 1)
            if gt_mask.any():
                max_overlaps[gt_mask] = overlaps[gt_idx, gt_mask]
                assigned_labels[gt_mask] = gt_labels[gt_idx]
        
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
    
    def _smart_candidate_selection(self, gt_bboxes, priors, overlaps, num_level_priors, cls_scores):
        """🎯 智能候选样本选择 - 优化版：控制正样本数量"""
        distances = bbox_center_distance(gt_bboxes, priors)  # [num_priors, num_gt]
        num_gt = gt_bboxes.size(0)
        num_priors = priors.size(0)
        
        # === 🚀 优化1：更严格的topk控制 ===
        all_candidates = []
        start_idx = 0
        
        # 🎯 修复：收紧候选选择，提高质量
        base_topk = max(2, min(self.topk, 8))  # 降回到8，避免过多低质量候选
        adaptive_topk_per_gt = max(2, base_topk // max(1, num_gt))  # 每个GT至少2个候选
        
        for level, num_level_prior in enumerate(num_level_priors):
            end_idx = start_idx + num_level_prior
            
            if num_level_prior == 0:
                start_idx = end_idx
                continue
            
            # 获取当前层级的距离 [num_level_prior, num_gt]
            level_distances = distances[start_idx:end_idx, :]
            
            # 🎯 修复：收紧候选比例，专注高质量候选
            # 降低候选比例，只选择最优质的候选
            level_ratio = min(0.08, 0.3 / max(1, num_gt))  # 从15%降回到8%，提高选择性
            dynamic_topk = max(2, min(adaptive_topk_per_gt, int(num_level_prior * level_ratio)))
            dynamic_topk = min(dynamic_topk, num_level_prior)
            
            if level_distances.size(0) > 0 and level_distances.size(1) > 0:
                # === 🚀 优化3：基于IoU质量的候选筛选 ===
                level_overlaps = overlaps[:, start_idx:end_idx]  # [num_gt, num_level_prior]
                
                for gt_idx in range(num_gt):
                    gt_distances = level_distances[:, gt_idx]  # [num_level_prior]
                    gt_overlaps = level_overlaps[gt_idx, :]  # [num_level_prior]
                    
                    # 组合距离和IoU进行候选选择
                    # 距离越小越好，IoU越大越好
                    normalized_distances = gt_distances / (gt_distances.max() + 1e-8)
                    normalized_overlaps = gt_overlaps / (gt_overlaps.max() + 1e-8)
                    
                    # 综合分数：距离权重0.6，IoU权重0.4
                    composite_scores = (1.0 - normalized_distances) * 0.6 + normalized_overlaps * 0.4
                    
                    # 选择综合分数最高的候选
                    _, topk_indices = composite_scores.topk(dynamic_topk, largest=True)
                    
                    # 🎯 修复：提高IoU筛选阈值，只保留高质量候选
                    valid_mask = gt_overlaps[topk_indices] > 0.1  # 恢复到0.1，过滤低质量候选
                    if valid_mask.any():
                        filtered_indices = topk_indices[valid_mask]
                        # 转换为全局索引
                        global_indices = filtered_indices + start_idx
                        all_candidates.append(global_indices)
            
            start_idx = end_idx
        
        if all_candidates:
            # 将所有候选者合并为一个列表
            all_candidate_indices = torch.cat(all_candidates, dim=0)
            # 去重并排序
            unique_candidates = torch.unique(all_candidate_indices)
            
            # === 🔧 修复4：放宽最终候选数量限制 ===
            # 允许更多候选样本参与质量评估
            max_candidates = min(len(unique_candidates), num_priors // 5, num_gt * 100)  # 从//10改为//5，从*50改为*100
            if len(unique_candidates) > max_candidates:
                # 基于IoU重新筛选最优候选
                candidate_scores = []
                for candidate_idx in unique_candidates:
                    max_iou = overlaps[:, candidate_idx].max()
                    candidate_scores.append(max_iou)
                
                candidate_scores = torch.tensor(candidate_scores, device=gt_bboxes.device)
                _, top_indices = candidate_scores.topk(max_candidates, largest=True)
                unique_candidates = unique_candidates[top_indices]
            
            return unique_candidates
        else:
            return torch.empty(0, dtype=torch.long, device=gt_bboxes.device)
    
    def _compute_composite_scores(self, candidate_idxs, overlaps, gt_bboxes, priors, cls_scores, gt_labels):
        """📊 计算多维度综合分数"""
        num_candidates = candidate_idxs.size(0)
        num_gt = gt_bboxes.size(0)
        composite_scores = torch.zeros((num_candidates, num_gt), device=candidate_idxs.device)
        
        for gt_idx in range(num_gt):
            # IoU分数 - 使用所有候选者
            iou_scores = overlaps[gt_idx, candidate_idxs]
            
            # 距离分数（归一化后取倒数）
            gt_center = (gt_bboxes[gt_idx, :2] + gt_bboxes[gt_idx, 2:]) / 2
            candidate_centers = (priors[candidate_idxs, :2] + priors[candidate_idxs, 2:]) / 2
            distances = torch.norm(candidate_centers - gt_center, dim=1)
            max_dist = distances.max() + 1e-8
            distance_scores = 1.0 - (distances / max_dist)
            
            # 🚀 全新质量分数计算：确保与IoU排序一致
            if cls_scores is not None and cls_scores.size(0) > 0:
                gt_label = gt_labels[gt_idx]
                # 确保标签索引在有效范围内
                if gt_label < cls_scores.size(1):
                    raw_quality_scores = cls_scores[candidate_idxs, gt_label].sigmoid()
                    
                    # 🎯 核心修复：IoU主导的质量分数计算
                    # 使用IoU作为主要因子，分类分数作为调节因子
                    iou_weight = 0.8  # IoU占主导
                    cls_weight = 0.2  # 分类分数作为微调
                    
                    # 对低质量样本施加强惩罚
                    iou_penalty = torch.where(
                        iou_scores < 0.1,
                        iou_scores * 0.1,  # 极低IoU样本质量分数严重压制
                        iou_scores
                    )
                    
                    # 组合计算：IoU主导，分类分数微调
                    quality_scores = iou_weight * iou_penalty + cls_weight * raw_quality_scores
                    
                    # 额外的低质量惩罚
                    very_low_mask = iou_scores < 0.05
                    quality_scores = torch.where(
                        very_low_mask,
                        quality_scores * 0.1,  # 极低质量样本进一步压制
                        quality_scores
                    )
                    
                    quality_scores = torch.clamp(quality_scores, 0.0, 1.0)
                else:
                    quality_scores = iou_scores * 0.6  # 使用IoU的60%作为质量分数
            else:
                quality_scores = iou_scores * 0.6  # 没有分类分数时，使用IoU的60%
            
            # 加权融合
            composite_scores[:, gt_idx] = (
                self.current_iou_weight * iou_scores +
                self.current_quality_weight * quality_scores +
                self.current_distance_weight * distance_scores
            )
        
        return composite_scores
    
    def _compute_adaptive_thresholds(self, composite_scores):
        """🔄 计算自适应阈值 - 优化版：更严格的阈值控制"""
        means = composite_scores.mean(0)
        stds = composite_scores.std(0)
        
        # === 🔧 修复5：优化阈值计算，提高灵敏性 ===
        # 基于历史性能调整阈值严格程度
        if len(self.performance_history) > 5:
            # 🔧 修复：deque对象不支持切片，需要转换为list
            history_list = list(self.performance_history)
            recent_performance = sum(history_list[-5:]) / 5
            if recent_performance > 0.75:  # 性能非常好，适度提高阈值
                threshold_factor = 1.1  # 从1.2降低到1.1
            elif recent_performance < 0.4:  # 性能较差，降低阈值
                threshold_factor = 0.9  # 从1.0降低到0.9
            else:
                threshold_factor = 1.0  # 从1.1降低到1.0，更温和
        else:
            # 训练初期使用中等阈值
            threshold_factor = 1.0  # 从1.15降低到1.0
        
        # === 🔧 修复6：优化候选数量调整逻辑 ===
        num_candidates = composite_scores.size(0)
        if num_candidates > 500:  # 提高触发阈值，从200改为500
            density_factor = 1.05 + (num_candidates - 500) * 0.0005  # 降低调整幅度
            threshold_factor *= density_factor
        
        # 🎯 修复：收紧阈值范围，提高选择性
        thresholds = means + threshold_factor * stds
        # 提高最低阈值，确保只选择高质量正样本
        return torch.clamp(thresholds, 0.25, 0.75)  # 收紧到[0.25, 0.75]，提高选择性
    
    def _check_geometric_constraints(self, candidate_idxs, gt_bboxes, priors, num_gt):
        """🛡️ 几何约束检查"""
        num_candidates = candidate_idxs.size(0)
        is_in_gts = torch.zeros((num_candidates, num_gt), dtype=torch.bool, device=candidate_idxs.device)
        
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
        
        for gt_idx in range(num_gt):
            # 使用所有候选者检查几何约束
            candidate_cx = priors_cx[candidate_idxs]
            candidate_cy = priors_cy[candidate_idxs]
            
            l_ = candidate_cx - gt_bboxes[gt_idx, 0]
            t_ = candidate_cy - gt_bboxes[gt_idx, 1]
            r_ = gt_bboxes[gt_idx, 2] - candidate_cx
            b_ = gt_bboxes[gt_idx, 3] - candidate_cy
            
            is_in_gt = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
            is_in_gts[:, gt_idx] = is_in_gt
        
        return is_in_gts
    
    def _resolve_conflicts(self, candidate_idxs, is_pos, overlaps, num_gt, num_priors):
        """🔄 解决多GT分配冲突"""
        INF = 100000000.0  # 🔧 修复：使用float类型的INF
        assigned_gt_inds = torch.zeros(num_priors, dtype=torch.long, device=candidate_idxs.device)
        
        # 🔧 修复：使用与overlaps相同的数据类型
        overlaps_inf = torch.full((num_priors, num_gt), -INF, 
                                 dtype=overlaps.dtype, device=candidate_idxs.device)
        
        for gt_idx in range(num_gt):
            # 获取当前GT的正样本mask
            gt_pos_mask = is_pos[:, gt_idx]
            gt_pos_candidates = candidate_idxs[gt_pos_mask]
            
            if gt_pos_candidates.numel() > 0:
                overlaps_inf[gt_pos_candidates, gt_idx] = overlaps[gt_idx, gt_pos_candidates]
        
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        return assigned_gt_inds
    
    def _evaluate_assignment_performance(self, assign_result, gt_bboxes, priors):
        """📊 评估分配性能"""
        if assign_result.max_overlaps is None:
            return 0.0
        
        pos_mask = assign_result.gt_inds > 0
        if pos_mask.sum() == 0:
            return 0.0
        
        # 计算平均IoU作为性能指标
        avg_iou = assign_result.max_overlaps[pos_mask].mean().item()
        
        # 计算正样本比例
        pos_ratio = pos_mask.sum().float() / pos_mask.numel()
        
        # 综合性能分数
        performance = 0.7 * avg_iou + 0.3 * min(pos_ratio.item(), 0.1) * 10
        return performance
    
    def _online_parameter_update(self):
        """🔄 在线参数更新"""
        if len(self.performance_history) < 10:
            return
        
        # 🔧 修复：deque对象不支持切片，需要转换为list
        history_list = list(self.performance_history)
        
        # 计算性能趋势
        recent_perf = sum(history_list[-5:]) / 5
        older_perf = sum(history_list[-10:-5]) / 5
        
        performance_trend = recent_perf - older_perf
        
        # 如果性能下降，调整参数
        if performance_trend < -self.adaptation_threshold:
            # 增加IoU权重，减少质量权重
            adjustment = self.learning_rate * abs(performance_trend)
            self.current_quality_weight = max(
                self.min_quality_weight,
                self.current_quality_weight - adjustment
            )
            
            # 重新归一化权重
            total_non_iou = self.current_quality_weight + self.current_distance_weight
            if total_non_iou < 0.7:  # 确保IoU权重不超过0.7
                self.current_iou_weight = 1.0 - total_non_iou
            else:
                # 按比例缩放
                scale = 0.7 / total_non_iou
                self.current_quality_weight *= scale
                self.current_distance_weight *= scale
                self.current_iou_weight = 0.3
    
    def get_current_status(self) -> Dict:
        """📊 获取当前状态信息"""
        # 🔧 修复：deque对象不支持切片，需要转换为list
        if self.performance_history:
            history_list = list(self.performance_history)
            recent_performance = sum(history_list[-5:]) / min(5, len(self.performance_history))
        else:
            recent_performance = 0.0
            
        return {
            'iteration_count': self.iteration_count,
            'current_weights': {
                'iou': self.current_iou_weight,
                'quality': self.current_quality_weight,
                'distance': self.current_distance_weight
            },
            'performance_history_length': len(self.performance_history),
            'recent_performance': recent_performance,
            'scene_adaptation_enabled': self.enable_scene_adaptation,
            'online_learning_enabled': self.enable_online_learning
        }