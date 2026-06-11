import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     unmap)
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import (bbox2distance, bbox_overlaps, distance2bbox)

from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import ( multi_apply)
from mmdet.models.utils import images_to_levels, unmap
from .anchor_head import AnchorHead
from mmdet.models.layers import multiclass_nms
from mmengine.registry import MODELS  # 3.x 新注册器
from typing import List, Optional, Tuple
from torch import Tensor
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
class Integral(nn.Module):
    """
    这是一个用于计算分布积分结果的固定层。
    通过公式 sum{P(y_i) * y_i} 计算目标位置，
    其中 P(y_i) 是表示离散分布的softmax向量，
    y_i 是离散集合，通常是 {0, 1, 2, ..., reg_max}
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()# 继承nn.Module的初始化
        self.reg_max = reg_max# 设置最大回归值（默认16）
        # 创建一个从0到reg_max的等间距向量
        # 例如：[0, 1, 2, ..., 16]
        # register_buffer将tensor注册到模块，但不会被认为是可训练参数
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """
           前向传播函数，将回归头的特征转换为边界框位置的积分结果，这四个偏移量代表的是相对于锚框中心点到目标框四个边界的距离。

           Args:
               x (Tensor): 回归头的特征，形状为(N, 4*(n+1))
                          其中n是self.reg_max
                          4代表边界框的4个方向(左,上,右,下)

           Returns:
               x (Tensor): 边界框位置的积分结果，形状为(N, 4)
                          表示框中心在四个方向上的偏移距离
           """
        # 1. 重塑输入并应用softmax
        # 将输入重塑为(-1, reg_max + 1)的形状，并在每个预测上应用softmax
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # 2. 计算积分（加权和）
        # F.linear实现了 sum{P(y_i) * y_i}
        # project提供了y_i值[0,1,2,...,reg_max]
        # 结果重塑为(-1, 4)，对应四个方向的预测
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFocalHead(AnchorHead):
    """Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.

    GFocal head structure is similar with GFL head, however GFocal uses
    the statistics of learned distribution to guide the
    localization quality estimation (LQE)

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
        reg_topk (int): top-k statistics of distribution to guide LQE
        reg_channels (int): hidden layer unit to generate LQE
    Example:
        >>> self = GFocalHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,# 类别数（不包括背景）
                 in_channels,# 输入特征图的通道数
                 stacked_convs=4,# 分类和回归分支的卷积层数
                 conv_cfg=None,# 卷积层配置
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),# 归一化层配置
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),# DFL损失配置
                 reg_max=16, # 回归编码的最大值
                 reg_topk=4, # 用于LQE的top-k统计量
                 reg_channels=64, # LQE的隐藏层单元数
                 add_mean=True,# 是否添加均值特征
                 **kwargs):
        # 初始化类属性
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        if add_mean:
            self.total_dim += 1# 如果使用均值，维度+1
        print('total dim = ', self.total_dim * 4)# 4个方向的总维度
        # 调用父类初始化
        super().__init__(num_classes, in_channels, **kwargs)#进行初始化
        # 设置采样策略
        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)# 构建分配器
            # 使用PseudoSampler（不进行采样）
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)

        # 构建积分层和DFL损失
        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        # 分类和回归的卷积层列表
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # 构建堆叠的卷积层
        for i in range(self.stacked_convs):
            # 第一层使用输入通道数，其他层使用特征通道数
            chn = self.in_channels if i == 0 else self.feat_channels
            # 添加分类卷积层
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            # 添加回归卷积层
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # 确保是anchor-free版本
        assert self.num_anchors == 1, 'anchor free version'
        # 最终的分类和回归预测层
        self.gfl_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels,
            4 * (self.reg_max + 1),
            3,
            padding=1)
        # 特征层的尺度因子
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        # 构建定位质量估计(LQE)网络
        # 输入通道数是4 * total_dim，因为有4个方向
        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
        self.reg_conf = nn.Sequential(*conf_vector)

    def init_weights(self):
        """初始化检测头的权重"""
        # 旧版错误代码
        # init_cfg = dict(type='Normal', ...)
        # normal_init(m.conv, **init_cfg)  # ❌ 错误传递方式

        # 新版正确初始化方式
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # 分类层特殊初始化
        bias_cls = bias_init_with_prob(0.1)  # 增大初始概率值
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)

        # 回归层初始化
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):
        """前向传播处理来自主干网络的特征

           参数:
               feats (tuple[Tensor]): 主干网络输出的特征图元组，每个元素是4D张量

           返回:
               tuple: 通常包含分类分数和边界框预测的元组
                   cls_scores: 所有尺度级别的分类和质量(IoU)联合分数
                   bbox_preds: 所有尺度级别的边界框分布logits
           """
        return multi_apply(self.forward_single, feats, self.scales)    # 对每个特征层级应用forward_single,函数feats 是来自特征金字塔网络(FPN)的特征图列表,self.scales 是每个特征层级对应的尺度因子列表

    def forward_single(self, x, scale):
        """处理单个特征图层级的前向传播

        Args:
            x (Tensor): 输入特征图, 形状为 (N, C, H, W)
            scale (nn.Module): 用于回归预测的尺度因子

        Returns:
            tuple:
                cls_score (Tensor): 分类得分, 形状为 (N, num_classes, H, W)
                bbox_pred (Tensor): 边界框预测, 形状为 (N, 4*(reg_max+1), H, W)
        """
        # 复制输入特征用于分类和回归分支
        cls_feat = x  # 分类特征
        reg_feat = x  # 回归特征

        # 分类分支: 通过多个卷积层处理特征
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        # 回归分支: 通过多个卷积层处理特征
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # 生成分类得分和边界框特征
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()  # 通过回归层并应用尺度因子得到边界框预测

        # 计算定位质量分数
        N, C, H, W = bbox_pred.size()  # 获取预测张量的维度
        # 将边界框预测重塑并应用softmax得到概率分布
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        # 获取每个位置概率最高的top-k个值
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        # 如果启用add_mean, 将均值特征与top-k特征拼接
        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2)
        else:
            stat = prob_topk

        # 通过回归置信度网络生成质量分数
        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))

        # 将质量分数与原始分类得分相乘得到最终分类得分
        # quality_score.sigmoid()将质量分数压缩到(0,1)区间
        cls_score = self.gfl_cls(cls_feat).sigmoid() * quality_score

        return cls_score, bbox_pred

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_by_feat_single(self,
                           cls_score: Tensor,
                           bbox_pred: Tensor,
                           anchors: Tensor,
                           labels: Tensor,
                           label_weights: Tensor,
                           bbox_targets: Tensor,
                           stride: Tuple[int],
                           num_total_samples: int = 1.0):
        """计算单个特征层级的损失"""
        # 确保特征图的水平和垂直步长相同
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'

        # 重塑输入张量为2D形式，方便后续处理
        anchors = anchors.reshape(-1, 4)  # [n, 4] 所有anchor展平
        # [B,C,H,W] -> [B*H*W, C] 将特征图重排为2D形式
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        # [B,4*(reg_max+1),H,W] -> [B*H*W, 4*(reg_max+1)]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))

        # 重塑目标张量
        bbox_targets = bbox_targets.reshape(-1, 4)  # [n, 4] 边界框目标
        labels = labels.reshape(-1)                 # [n] 类别标签
        label_weights = label_weights.reshape(-1)   # [n] 标签权重

        # 获取正样本索引（类别标签在有效范围内的样本）
        bg_class_ind = self.num_classes  # 背景类的索引
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # 初始化分数张量（用于QFL损失）
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:  # 如果有正样本
            # 提取正样本相关数据
            pos_bbox_targets = bbox_targets[pos_inds]  # 正样本的边界框目标
            pos_bbox_pred = bbox_pred[pos_inds]        # 正样本的边界框预测
            pos_anchors = anchors[pos_inds]            # 正样本的anchor
            # 计算anchor中心点并除以stride归一化
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # 计算分类权重（基于预测分数）
            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_corners = self.integral(pos_bbox_pred)

            # 修改解码方法，避免重复
            pos_decode_bbox_pred = distance2bbox(
                pos_anchor_centers, 
                pos_bbox_pred_corners
            )

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_anchor_centers,
                pos_decode_bbox_targets, 
                self.reg_max
            ).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        # 打印标签信息
       #print("Labels:", labels)
        # print("Labels dtype:", labels.dtype)
        # print("Labels range:", labels.min(), labels.max())
        #print("Background class index:", self.num_classes)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # Get classification channels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # Get training targets
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            gt_labels_list=[gt_instances.labels for gt_instances in batch_gt_instances],
            label_channels=label_channels)

        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        # 计算平均样本数
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        # 计算损失
        losses_cls, losses_bbox, losses_dfl, \
            avg_factor = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides)

        # 处理平均因子
        avg_factor = sum(avg_factor)# 重新计算平均因子
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()# 确保平均因子不小于1
        # 使用平均因子归一化边界框损失和DFL损失
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        # Detailed logging for debugging
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj: `ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape
              [num_bboxes, 5], where the first 4 columns are bounding
              box positions (tl_x, tl_y, br_x, br_y) and the 5-th
              column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding
              box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = distance2bbox(
                self.anchor_center(priors), 
                bbox_pred, 
                max_shape=img_shape
            )
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def get_targets(self,
                    anchor_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    gt_labels_list: List[Tensor] = None,
                    label_channels: int = 1,
                    unmap_outputs: bool = True) -> tuple:
        """获取训练目标"""
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # 合并每张图像的多层级锚框
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # 处理忽略的实例
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # 计算每张图像的目标
        results = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = results
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # 计算正负样本数量
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        # 将结果重新分配到不同特征层级
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           flat_anchors,      # 展平的所有anchor框，形状为(N, 4)
                           valid_flags,       # anchor的有效标志，形状为(N,)
                           num_level_anchors, # 每个特征层级的anchor数量列表
                           gt_instances,      # InstanceData对象，包含bboxes和labels
                           img_meta,          # 图像的元信息
                           gt_instances_ignore=None,  # 需要忽略的实例
                           unmap_outputs=True): # 是否需要将输出映射回原始anchor空间
        """为单张图像中的anchor计算回归和分类目标"""
        
        # 从 img_meta 中安全地获取图像形状
        img_shape = img_meta.get('img_shape', img_meta.get('pad_shape', img_meta.get('ori_shape')))
        if img_shape is None:
            raise ValueError("Cannot find image shape in metadata")
        
        # 检查 anchor 是否在原始输入图像的有效范围内
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_shape[:2],
                                           self.train_cfg.allowed_border)
        
        # 如果没有有效的anchor，返回7个None
        if not inside_flags.any():
            return (None, ) * 7
        
        # 只保留在图像内部的anchor
        anchors = flat_anchors[inside_flags, :]

        # 计算每个特征层级内部的anchor数量
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        # 创建预测实例，包含anchor信息
        pred_instances = InstanceData(priors=anchors)
        
        # 使用分配器将GT分配给anchor
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            num_level_priors=num_level_anchors_inside,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        # 使用采样器采样正负样本
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        # 初始化目标张量
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # 获取正负样本索引
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 处理正样本
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            # 如果标签为 None，默认从 0 开始
            if gt_instances.labels is None:
                labels[pos_inds] = 0
            else:
                # 确保标签从 0 开始，且在有效范围内
                labels[pos_inds] = torch.clamp(
                    gt_instances.labels[sampling_result.pos_assigned_gt_inds], 
                    min=0, 
                    max=self.num_classes - 1
                )

            # 设置正样本的分类权重
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        # 处理负样本
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # 如果需要，将结果映射回原始anchor空间
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        # 返回计算得到的所有目标值
        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
