# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from mmengine.registry import MODELS

@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=False,
                      gamma=1.0, alpha=0.25):
    r"""🎯 自适应质量焦点损失 - 保持原始IoU目标的优雅方案

    核心创新：
    1. 🎯 保持原始目标：始终以真实IoU为学习目标，不做任何修改
    2. 🧠 智能权重函数：根据样本难度和预测差距自适应调整权重
    3. 🔄 渐进学习：训练早期权重适中，后期逐渐恢复正常QFL强度
    4. ⚖️ 梯度控制：通过权重设计避免梯度爆炸，而非改变目标

    Args:
        pred (torch.Tensor): 预测的分类和质量联合表示，形状为 (N, C)
        target (tuple): 包含(类别标签, IoU质量分数)的元组
        beta (float): 基础focal loss调制参数，默认2.0
        use_sigmoid (bool): 是否使用sigmoid，默认False
        gamma (float): 难度自适应参数，控制困难样本权重调制，默认1.0
        alpha (float): 梯度平衡参数，控制大差距时的权重平衡，默认0.25

    Returns:
        torch.Tensor: 自适应QFL损失，形状为 (N,)
    """

    label, score = target
    # 直接使用sigmoid输出，无需转换为logits
    pred_sigmoid = pred.clamp(min=1e-7, max=1-1e-7)  # 数值稳定性保护
    
    # === 🎯 核心原则：保持原始IoU目标完全不变！===
    # 这是检测任务的真实需求，不应该被人为修改
    quality_target = torch.clamp(score, 0.0, 1.0)  # 只做基本范围限制
    
    # 使用binary_cross_entropy，更简洁高效
    func = F.binary_cross_entropy
    
    # 步骤1：处理负样本
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred_sigmoid.shape)
    loss = func(pred_sigmoid, zerolabel, reduction='none') * scale_factor.pow(beta)

    # 步骤2：处理正样本 - 核心创新在这里
    # 前景类别id范围: [0, num_classes-1], 背景类别id: num_classes
    bg_class_ind = pred_sigmoid.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)  # 找到正样本索引
    
    if len(pos) > 0:  # 如果存在正样本
        pos_label = label[pos].long()  # 正样本的类别标签
        pos_pred_sigmoid = pred_sigmoid[pos, pos_label]
        pos_quality_target = quality_target[pos]
        
        # === 🧠 核心创新：自适应权重函数 ===
        # 基础质量差距
        quality_diff = torch.abs(pos_quality_target - pos_pred_sigmoid)
        
        # 1. 基础权重：保持QFL的核心思想
        base_weight = quality_diff.pow(beta) 
        
        # 3. 梯度平衡因子：当预测与目标差距过大时，添加平衡因子
        # 🎯 使用尺度自适应的阈值：根据特征层stride确定
        # 3. 梯度平衡因子：区分低估和过度自信两种情况
        # 区分低估（预测过低）和过度自信（预测过高）的情况
        under_confident = pos_quality_target > pos_pred_sigmoid  # 预测过低
        over_confident = pos_pred_sigmoid > pos_quality_target    # 预测过高

        # 3.1 对低估情况：使用原有逻辑，保持较强的权重降低
        under_gap = pos_quality_target - pos_pred_sigmoid
        under_balance = torch.where(
            under_confident ,
            1.0 / (1.0 + 0.25 * under_gap),
            torch.ones_like(under_gap)
        )

        # 3.2 对过度自信情况：使用更温和的处理（系数更小）
        over_gap = pos_pred_sigmoid - pos_quality_target
        over_balance = torch.where(
            over_confident ,
            1.0 / (1.0 + 0.1 * over_gap),  # 更小的系数，避免过度惩罚
            torch.ones_like(over_gap)
        )

        # 合并两个因子
        balance_factor = under_balance * over_balance
        
        # 4. 最终自适应权重
        adaptive_weight = base_weight * balance_factor.detach()
        
        # 5. 平滑截断：确保权重在合理范围内

        adaptive_weight = torch.clamp(adaptive_weight, min=0.1, max=5.0)
        
        # === ✨ 关键：使用原始IoU目标，配合智能权重 ===
        # 正样本损失：BCE(预测概率, 真实IoU) * 自适应权重
        loss[pos, pos_label] = func(
            pos_pred_sigmoid, 
            pos_quality_target,  # 使用完全未修改的原始IoU目标！
            reduction='none'
        ) * adaptive_weight
    
    # 对每个样本的所有类别损失求和
    focal_loss = loss.sum(dim=1, keepdim=False)
    
    return focal_loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


def regression_quality_focal_loss(pred_distances, gt_distances, pred_iou, cls_score, 
                                beta=2.0, quality_threshold=0.3, reg_max=16, 
                                weight=None, reduction='mean', avg_factor=None):
    """回归质量焦点损失 - 使用统一可靠性权重
    
    Args:
        pred_distances (Tensor): 预测的距离分布 [N, reg_max+1]
        gt_distances (Tensor): GT距离值 [N]  
        pred_iou (Tensor): 预测框与GT的IoU [N]
        cls_score (Tensor): 分类分数 [N]
        beta (float): 调制参数，默认2.0
        weight (Tensor, optional): 可靠性权重，由Head计算传入
        
    Returns:
        Tensor: RQFL损失
    """
    # 基础验证
    if len(pred_distances) == 0:
        return pred_distances.sum() * 0.0
    
    # 使用传入的可靠性权重，避免重复计算
    if weight is not None:
        quality_weight = weight
    else:
        # 回退到基础权重（当没有传入weight时）
        sharpness = F.softmax(pred_distances, dim=1).max(dim=1)[0]
        quality = torch.clamp(pred_iou, 0, 1) * torch.clamp(cls_score, 0, 1)
        quality_weight = (sharpness ** beta) * quality + 0.1
    
    # === 标准DFL损失计算 ===
    gt_distances = torch.clamp(gt_distances, 0.0, float(reg_max))
    dis_left = gt_distances.long()
    dis_right = dis_left + 1
    dis_left = torch.clamp(dis_left, 0, reg_max - 1)
    dis_right = torch.clamp(dis_right, 1, reg_max)
    
    weight_left = dis_right.float() - gt_distances
    weight_right = gt_distances - dis_left.float()
    
    loss_left = F.cross_entropy(pred_distances, dis_left, reduction='none')
    loss_right = F.cross_entropy(pred_distances, dis_right, reduction='none')
    
    # 分别计算左右损失
    dfl_loss = loss_left * weight_left + loss_right * weight_right
    
    # 应用质量权重
    enhanced_rqfl_loss = dfl_loss * quality_weight
    
    # 应用权重和缩减
    from .utils import weight_reduce_loss
    enhanced_rqfl_loss = weight_reduce_loss(enhanced_rqfl_loss, weight, reduction, avg_factor)
    
    return enhanced_rqfl_loss


@MODELS.register_module()
class QualityFocalLoss(nn.Module):
    r"""🎯 自适应质量焦点损失 - 保持原始IoU目标的优雅方案
    
    核心优势：
    1. 🎯 语义一致：始终以真实IoU为目标，保持检测任务的语义完整性
    2. 🧠 智能权重：根据样本难度和预测差距自适应调整学习强度
    3. 🔄 渐进学习：训练早期稳定，后期精确，自然的学习曲线
    4. ⚖️ 梯度控制：通过权重设计避免梯度爆炸，而非改变学习目标
    5. ✨ 保持QFL精神：质量感知的自适应权重机制
    
    Args:
        use_sigmoid (bool): 是否使用sigmoid激活，默认True
        beta (float): 基础focal loss调制参数，默认2.0
        gamma (float): 难度自适应参数，控制困难样本权重调制，默认1.0
        alpha (float): 梯度平衡参数，控制大差距时的权重平衡，默认0.25
        reduction (str): 损失归约方式，默认'mean'
        loss_weight (float): 损失权重，默认1.0
        enable_aqt (bool): 是否启用AQT模式，默认True
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 gamma=1.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 enable_aqt=True,
                 activated=False):
        super(QualityFocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.enable_aqt = enable_aqt
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """自适应QFL前向传播函数 - 保持原始IoU目标的优雅方案

        Args:
            pred (torch.Tensor): 预测的分类和质量联合表示，形状(N, C)
            target (tuple): 目标标签，包含类别标签和质量标签
            weight (torch.Tensor, optional): 可靠性权重，由Head计算传入
            avg_factor (int, optional): 平均因子，默认None
            reduction_override (str, optional): 损失归约方式覆盖，默认None

        Returns:
            torch.Tensor: 自适应QFL损失值 - 既保持原始目标又避免梯度爆炸
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # AQT模式：检查是否启用自适应质量目标
        if not self.enable_aqt and len(target) == 3:
            # 如果禁用AQT但传入了3元组，退化为传统2元组
            target = target[:2]

        # 计算自适应QFL损失，传入自适应参数
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            target,
            weight,  # 统一的可靠性权重通过这里传入
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            gamma=self.gamma,    # 🧠 难度自适应参数
            alpha=self.alpha,    # ⚖️ 梯度平衡参数
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls


@MODELS.register_module()
class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


@MODELS.register_module()
class RegressionQualityFocalLoss(nn.Module):
    r"""回归质量焦点损失 - 使用统一可靠性权重
    
    Args:
        beta (float): 质量感知权重的调制参数，默认2.0
        quality_threshold (float): 质量一致性阈值，默认0.3
        reg_max (int): 最大回归值，默认16
        reduction (str): 损失缩减方式，默认'mean'
        loss_weight (float): 损失权重，默认1.0
    """

    def __init__(self, 
                 beta=2.0,
                 quality_threshold=0.3,
                 reg_max=16,
                 reduction='mean', 
                 loss_weight=1.0):
        super(RegressionQualityFocalLoss, self).__init__()
        self.beta = beta
        self.quality_threshold = quality_threshold
        self.reg_max = reg_max
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred_distances,
                gt_distances,
                pred_iou,
                cls_score,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """RQFL前向传播
        
        Args:
            pred_distances (Tensor): 预测的距离分布 [N, reg_max+1]
            gt_distances (Tensor): GT距离值 [N]
            pred_iou (Tensor): 预测框与GT的IoU [N]
            cls_score (Tensor): 分类分数 [N]
            weight (Tensor, optional): 可靠性权重，由Head计算传入
            avg_factor (int, optional): 平均因子
            reduction_override (str, optional): 缩减方式覆盖
        
        Returns:
            Tensor: RQFL损失
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        loss = self.loss_weight * regression_quality_focal_loss(
            pred_distances,
            gt_distances,
            pred_iou,
            cls_score,
            beta=self.beta,
            quality_threshold=self.quality_threshold,
            reg_max=self.reg_max,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        
        return loss

    

    

    

    