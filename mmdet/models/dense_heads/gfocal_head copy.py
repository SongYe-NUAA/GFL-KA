
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
from ..task_modules.samplers import PseudoSampler
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import (multi_apply)
from mmdet.models.utils import images_to_levels, unmap
from .anchor_head import AnchorHead
from mmengine.registry import MODELS  # 3.x 新注册器
from typing import List, Optional, Sequence, Tuple
from torch import Tensor
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.logging import MMLogger
from mmengine import MessageHub


class Integral(nn.Module):
    """
    这是一个用于计算分布积分结果的固定层。
    通过公式 sum{P(y_i) * y_i} 计算目标位置，
    其中 P(y_i) 是表示离散分布的softmax向量，
    y_i 是离散集合，通常是 {0, 1, 2, ..., reg_max}
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()  # 继承nn.Module的初始化
        self.reg_max = reg_max  # 设置最大回归值（默认16）
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
                 num_classes,  # 类别数（不包括背景）
                 in_channels,  # 输入特征图的通道数
                 stacked_convs=4,  # 分类和回归分支的卷积层数
                 conv_cfg=None,  # 卷积层配置
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # 归一化层配置
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),  # DFL损失配置
                 reg_max=16,  # 回归编码的最大值
                 reg_topk=4,  # 用于LQE的top-k统计量
                 reg_channels=64,  # LQE的隐藏层单元数
                 add_mean=True,  # 是否添加均值特征
                 lqe_use_1x1_conv=True,  # 🎯 LQE网络是否使用1x1卷积
                 use_channel_attention=True,  # 🎯 是否使用通道注意力模块
                 use_top3_plus_top4=True,  # 🎯 是否使用top3+top4融合特征（True=一维融合，False=二维独立）
                 use_kurtosis_weight=True,  # 🎯 是否使用峰度加权特征
                 log_gt_top5_positive=True,  # 🎯 是否记录GT Top5正样本日志
                 use_original_lqe=False,  # 🎯 是否使用原始LQE网络设计
                 **kwargs):
        # 获取日志记录器
        self.logger = MMLogger.get_current_instance()

        # 初始化类属性
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.lqe_use_1x1_conv = lqe_use_1x1_conv  # 🎯 LQE网络卷积核大小选择
        self.use_channel_attention = use_channel_attention  # 🎯 通道注意力开关
        self.use_top3_plus_top4 = use_top3_plus_top4  # 🎯 top3+top4融合特征开关
        self.use_kurtosis_weight = use_kurtosis_weight  # 🎯 峰度加权开关
        self.log_gt_top5_positive = log_gt_top5_positive  # 🎯 GT Top5正样本日志开关
        self.use_original_lqe = use_original_lqe  # 🎯 原始LQE网络开关

        # 🚀 使用增强版峰度加权特征
        # 根据开关计算特征维度
        if self.use_top3_plus_top4:
            # top3+top4作为一维融合特征 → 4维特征
            # [top1, top2, (top3+top4), mean] = 4维
            self.total_dim = 4 if self.add_mean else 3
        else:
            # top3和top4作为独立的二维特征 → 5维特征
            # [top1, top2, top3, top4, mean] = 5维
            self.total_dim = 5 if self.add_mean else 4
        self.lqe_in_channels = 4 * self.total_dim  # 4个方向 * 特征维度
        # 调用父类初始化
        super().__init__(num_classes, in_channels, **kwargs)  # 进行初始化

        # 🔧 峰度相关可学习参数在所有配置下都需要，提前初始化避免属性缺失
        self._init_kurtosis_parameters()
        # 设置采样策略,不使用传统的标签分配策略
        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])  # 构建分配器
            # 构建采样器
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)  # TASK_UTILS任务相关工具
            self.initial_epoch = self.train_cfg['initial_epoch']

        # 构建将离散的分类预测转换为连续的边界框坐标值层和DFL损失（这里对于integral还需要再研究一下）
        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)  # MODELS 注册器：用于核心模型组件

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        # 分类和回归的卷积层列表
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # 定义SiLU激活函数配置
        act_cfg = dict(type='SiLU')
        # 构建堆叠的卷积层
        for i in range(self.stacked_convs):
            # 第一层使用输入通道数，其他层使用特征通道数
            chn = self.in_channels if i == 0 else self.feat_channels
            # 添加回归卷积层
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg))
        # 最后两层使用SiLU激活函数
        for i in range(self.stacked_convs - 2):
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
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        # 最终的分类和回归预测层
        self.gfl_cls = nn.Conv2d(
            self.feat_channels,
            self.num_classes,
            3,
            padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels,
            4 * (self.reg_max + 1),
            3,
            padding=1)
        # 特征层的尺度因子,创建了一个可学习的尺度因子列表，用于调整不同特征层级的预测。
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])  # Scale(1.0)：为每个特征层级创建一个初始值为1.0的可学习尺度因子

        # 🚀 构建传统对称LQE网络
        self._build_traditional_symmetric_lqe_network()

    def _init_kurtosis_parameters(self):
        """初始化峰度相关可学习参数，确保任意配置下都可用。"""
        self.kurtosis_scale_factor = nn.Parameter(torch.tensor(0.05))
        self.kurtosis_bias = nn.Parameter(torch.tensor(0.0))
        self.kurtosis_fusion_weight = nn.Parameter(torch.tensor(1.0))

        # === 🎯 通道注意力模块（可选） ===
        if self.use_channel_attention:
            # Global Average Pooling + Global Max Pooling
            self.channel_attn_avg = nn.AdaptiveAvgPool2d(1)
            self.channel_attn_max = nn.AdaptiveMaxPool2d(1)

            # 共享MLP：先压缩再扩张
            channel_hidden_dim = max(self.lqe_in_channels // 16, 8)  # 压缩比16，最小8维
            self.channel_attn_mlp = nn.Sequential(
                nn.Conv2d(self.lqe_in_channels, channel_hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_hidden_dim, self.lqe_in_channels, 1)
            )

    def _extract_enhanced_statistics(self, bbox_pred_fp32):
        """🎯 峰度加权的复合特征提取 - 完整17维峰度版本

        🔄 核心思路：使用峰度作为质量权重，对复合特征进行加权
        - 提取 top1, top2, (top3+top4), mean 特征（用于LQE输入）
        - 🚀 使用完整17维概率分布计算峰度权重（更准确的质量评估）
        - 用峰度权重对特征进行加权，突出高质量预测

        Args:
            bbox_pred_fp32 (Tensor): 边界框预测 [N, 4*(reg_max+1), H, W]

        Returns:
            Tensor: 峰度加权特征 [N, 4*total_dim, H, W] = [N, 16, H, W] (4方向×4特征)
        """
        N, C, H, W = bbox_pred_fp32.size()

        # 将边界框预测重塑并应用softmax得到概率分布
        # [N, 4*(reg_max+1), H, W] → [N, 4, reg_max+1, H, W]
        prob = F.softmax(bbox_pred_fp32.reshape(N, 4, self.reg_max + 1, H, W), dim=2)

        # === 🎯 改进特征提取：top1, top2, (top3+top4), mean ===
        # 获取每个方向的top4最大概率值（用于特征提取）
        # [N, 4, reg_max+1, H, W] → [N, 4, 4, H, W]
        top4_values, _ = torch.topk(prob, k=4, dim=2)

        # 提取 top1 和 top2：[N, 4, 2, H, W]
        top2_values = top4_values[:, :, :2, :, :]

        # 🎯 根据开关选择 top3+top4 特征
        if self.use_top3_plus_top4:
            # top3 + top4 作为一维融合特征：[N, 4, 1, H, W]
            top3_plus_top4 = (top4_values[:, :, 2:3, :, :] + top4_values[:, :, 3:4, :, :])
        else:
            # top3 和 top4 作为独立的二维特征：[N, 4, 2, H, W]
            top3_top4 = top4_values[:, :, 2:4, :, :]

        # 计算均值特征：[N, 4, 1, H, W]
        mean_feature = top4_values.mean(dim=2, keepdim=True)

        # 组合特征
        if self.use_top3_plus_top4:
            # [top1, top2, (top3+top4), mean] = [N, 4, 4, H, W]
            original_features = torch.cat([top2_values, top3_plus_top4, mean_feature], dim=2)
        else:
            # [top1, top2, top3, top4, mean] = [N, 4, 5, H, W]
            original_features = torch.cat([top2_values, top3_top4, mean_feature], dim=2)

        # === 🔥 计算完整17维峰度权重 ===
        # 🚀 关键改进：传入完整概率分布而非仅TOP4
        # 完整分布能更准确地反映预测质量（捕捉多峰、平坦分布等特征）
        kurtosis_weights = self._compute_top4_kurtosis_weights(prob)  # [N, 4, H, W]
        # 注意：prob是[N, 4, 17, H, W]，包含完整概率分布

        # 🎯 可学习自适应因子：让网络自己学习如何调控峰度权重影响
        # 🔥 添加可学习偏置，提供更灵活的权重控制
        kurtosis_weights = torch.exp(kurtosis_weights * self.kurtosis_scale_factor + self.kurtosis_bias)

        # === ⚡ 峰度加权：突出高质量预测 ===
        kurtosis_expanded = kurtosis_weights.mean(dim=1, keepdim=True)  # [N, 1, H, W]

        # 重塑为最终输出格式：[N, 4×(4/5), H, W] = [N, 16/20, H, W]
        if self.use_kurtosis_weight:
            # 🎯 使用峰度加权：突出高质量预测
            # 🔥 使用可学习融合权重，让网络自己学习峰度权重的最优影响强度
            enhanced_stat = original_features.reshape(N, -1, H, W) * (1 + kurtosis_expanded * self.kurtosis_fusion_weight)
        else:
            # 🎯 不使用峰度加权：使用原始特征
            enhanced_stat = original_features.reshape(N, -1, H, W)

        # === 🎯 通道注意力模块（可选） ===
        if self.use_channel_attention:
            # 使用全局平均池化和最大池化捕获通道统计信息
            avg_out = self.channel_attn_avg(enhanced_stat)  # [N, C, 1, 1]
            max_out = self.channel_attn_max(enhanced_stat)  # [N, C, 1, 1]

            # 通过共享MLP获取注意力权重
            avg_weight = self.channel_attn_mlp(avg_out)  # [N, C, 1, 1]
            max_weight = self.channel_attn_mlp(max_out)  # [N, C, 1, 1]

            # 合并平均和最大池化的注意力权重
            channel_weight = torch.sigmoid(avg_weight + max_weight)  # [N, C, 1, 1]

            # 应用通道注意力：自适应调整不同特征通道的重要性
            enhanced_stat = enhanced_stat * channel_weight

        return enhanced_stat

    def _build_traditional_symmetric_lqe_network(self):
        """🎯 构建LQE网络（支持原始/改进两种模式）

        Args:
            use_original: 是否使用原始LQE设计

        原始设计 (use_original_lqe=True):
            Conv(lqe_in_channels→reg_channels, 1x1) → ReLU → Conv(reg_channels→1, 1x1) → Sigmoid

        改进设计 (use_original_lqe=False):
            ConvModule(lqe_in_channels→64, 1x1, GN, SiLU) × 2 → Conv(64→1, 1x1) → Sigmoid
        """
        if self.use_original_lqe:
            # 🎯 原始LQE网络设计：Conv + ReLU + Conv + Sigmoid
            self.original_lqe = nn.Sequential(
                nn.Conv2d(4 * self.total_dim, self.reg_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.reg_channels, 1, 1),
                nn.Sigmoid()
            )
        else:
            # 🎯 改进版LQE网络：ConvModule × 2 + Conv
            self._build_enhanced_lqe_network()

    def _build_enhanced_lqe_network(self):
        """🎯 构建改进版LQE网络（ConvModule + GN + SiLU）

        架构：ConvModule × 2 + Conv
        """
        # 定义SiLU激活函数配置
        act_cfg = dict(type='SiLU')

        # LQE卷积层列表
        self.lqe_convs = nn.ModuleList()

        # 构建与cls分支相同层数的卷积层
        for i in range(self.stacked_convs - 2):
            if i == 0:
                chn = self.lqe_in_channels
            else:
                chn = 64

            kernel_size = 1 if self.lqe_use_1x1_conv else 3
            padding = 0 if self.lqe_use_1x1_conv else 1
            self.lqe_convs.append(
                ConvModule(
                    chn,
                    64,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg))

        # 最终的LQE预测层
        kernel_size = 1 if self.lqe_use_1x1_conv else 3
        padding = 0 if self.lqe_use_1x1_conv else 1
        self.gfl_lqe = nn.Conv2d(
            64,
            1,
            kernel_size,
            padding=padding)

    def _forward_traditional_lqe(self, stat):
        """🎯 传统对称LQE前向传播

        Args:
            stat (Tensor): 输入统计特征 [N, 4*total_dim, H, W]

        Returns:
            Tensor: 质量分数 [N, 1, H, W]
        """
        # 通过LQE卷积层，与cls_convs相同的结构
        x = stat
        for lqe_conv in self.lqe_convs:
            x = lqe_conv(x)

        # 最终质量预测层
        quality_score = self.gfl_lqe(x)
        quality_score = quality_score.sigmoid()  # 确保输出在[0,1]范围

        return quality_score

    def _compute_top4_kurtosis_weights(self, top4_values):
        """🔥 计算完整17维峰度权重 - 数值稳定版本

        🎯 核心改进：使用完整概率分布计算峰度，而非仅TOP4
        - 优势1：捕捉完整分布形态，更准确反映预测质量
        - 优势2：能识别多峰分布（低质量预测的典型特征）
        - 优势3：理论正确 - 峰度定义就是基于完整分布

        Args:
            top4_values (Tensor): TOP4概率值 [N, 4, 4, H, W] - 仅用于特征提取
                                 注意：此参数名保留向后兼容，实际内部使用完整17维

        Returns:
            Tensor: 峰度权重 [N, 4, H, W]
        """
        # ⚠️ 注意：这里需要完整17维概率分布，而非TOP4
        # 由于函数接口限制，我们需要在调用处传入完整分布
        # 为了向后兼容，此函数保留top4_values参数名，但期望接收完整分布

        # 🎯 使用完整17维计算峰度
        # top4_values 应该被重命名为 prob_distribution: [N, 4, 17, H, W]
        prob_distribution = top4_values  # [N, 4, K, H, W] where K=17

        # 计算期望值（均值）
        mean_val = prob_distribution.mean(dim=2, keepdim=True)  # [N, 4, 1, H, W]

        # 中心化
        centered = prob_distribution - mean_val  # [N, 4, K, H, W]

        # 计算二阶矩（方差）
        var = (centered ** 2).mean(dim=2, keepdim=True)  # [N, 4, 1, H, W]

        # 计算四阶矩
        fourth_moment = (centered ** 4).mean(dim=2, keepdim=True)  # [N, 4, 1, H, W]

        # Fisher峰度 = E[(X-μ)⁴]/σ⁴
        # 数值稳定性：添加小常数避免除零
        kurtosis = fourth_moment / (var ** 2 + 1e-8)  # [N, 4, 1, H, W]
        kurtosis = kurtosis.squeeze(2)  # [N, 4, H, W]

        return kurtosis



    def init_weights(self):
        """初始化检测头的权重 - 优化版本，减少初期分类损失"""
        # 新版正确初始化方式
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        # 🚀 LQE网络初始化（根据开关选择）
        self._init_lqe_weights()

        # 🚀 优化分类层初始化：减少初期损失
        bias_cls = bias_init_with_prob(0.1)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)

        # 回归层初始化
        normal_init(self.gfl_reg, std=0.01)

    def _init_lqe_weights(self):
        """🎯 初始化LQE网络权重（支持原始/改进两种模式）"""
        if self.use_original_lqe:
            # 原始LQE网络初始化
            for m in self.original_lqe:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            # 输出层偏置初始化
            if self.original_lqe[-2].bias is not None:
                nn.init.constant_(self.original_lqe[-2].bias, -0.2)
        else:
            # 改进版LQE网络初始化
            for lqe_conv in self.lqe_convs:
                normal_init(lqe_conv.conv, std=0.01)
            normal_init(self.gfl_lqe, std=0.01)
            if self.gfl_lqe.bias is not None:
                nn.init.constant_(self.gfl_lqe.bias, -0.2)

    def forward(self, feats):
        """前向传播处理来自主干网络的特征

           参数:
               feats (tuple[Tensor]): 主干网络输出的特征图元组，每个元素是4D张量

           返回:
               tuple: 包含分类分数、边界框预测和质量分数的元组
                   cls_scores: 所有尺度级别的分类和质量(IoU)联合分数
                   bbox_preds: 所有尺度级别的边界框分布logits
                   quality_scores: 所有尺度级别的定位质量分数
           """
        results = multi_apply(self.forward_single, feats, self.scales)
        # multi_apply返回的是元组，包含三个列表
        cls_scores, bbox_preds, quality_scores = results
        return cls_scores, bbox_preds, quality_scores

    def forward_single(self, x, scale):
        """处理单个特征图层级的前向传播 - 测试时使用FP32确保跨设备一致性

        Args:
            x (Tensor): 输入特征图, 形状为 (N, C, H, W)
            scale (nn.Module): 用于回归预测的尺度因子

        Returns:
            tuple:
                cls_score (Tensor): 分类得分, 形状为 (N, num_classes, H, W)
                bbox_pred (Tensor): 边界框预测, 形状为 (N, 4*(reg_max+1), H, W)
        """
        # === 🔧 关键修复：测试时强制使用FP32，确保不同设备结果一致 ===
        # 解决：测试时完全禁用AMP，使用FP32确保跨设备一致性
        use_amp = self.training  # 训练时使用AMP，测试时禁用

        input_dtype = x.dtype

        # 复制输入特征用于分类和回归分支
        cls_feat = x  # 分类特征
        reg_feat = x  # 回归特征

        # 分类分支: 通过多个卷积层处理特征
        with torch.cuda.amp.autocast(enabled=use_amp):
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)

        # 回归分支: 通过多个卷积层处理特征
        with torch.cuda.amp.autocast(enabled=use_amp):
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

        # 获取分类输出
        with torch.cuda.amp.autocast(enabled=use_amp):
            cls_score = self.gfl_cls(cls_feat)

        # 获取epoch数
        message_hub = MessageHub.get_current_instance()
        self.epoch = message_hub.get_info('epoch')

        # 回归预测
        with torch.cuda.amp.autocast(enabled=use_amp):
            bbox_pred = scale(self.gfl_reg(reg_feat))
        bbox_pred = bbox_pred.float()  # 🔧 确保回归预测使用FP32精度

        # === 🚀 关键计算：强制FP32精度确保数值稳定性 ===
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为FP32进行关键计算
            bbox_pred_fp32 = bbox_pred.float()

            # 🚀 使用增强版峰度加权特征提取（4维训练自适应特征：top1, top2, top3+top4, mean）
            stat = self._extract_enhanced_statistics(bbox_pred_fp32)

            # 🚀 根据开关选择LQE网络前向传播方式
            if self.use_original_lqe:
                # 原始LQE网络：Conv → ReLU → Conv → Sigmoid
                quality_score = self.original_lqe(stat)
            else:
                # 改进版LQE网络：ConvModule × 2 + Conv + Sigmoid
                quality_score = self._forward_traditional_lqe(stat)

            # 分类分数处理 (FP32精度确保sigmoid稳定性)
            cls_score_fp32 = cls_score.float()
            cls_score_fp32 = cls_score_fp32.sigmoid()
            cls_score_fp32 = cls_score_fp32 * quality_score

        # 🔧 测试时返回FP32确保跨设备一致性，训练时保持原始dtype以优化性能
        if self.training:
            return cls_score_fp32.to(input_dtype), bbox_pred.to(input_dtype), quality_score.to(input_dtype)
        else:
            # 测试时返回FP32，避免不同GPU架构的FP16精度差异
            return cls_score_fp32, bbox_pred, quality_score

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

    @torch.cuda.amp.autocast(enabled=False)  # 🚀 损失计算必须使用FP32确保数值稳定性
    def loss_by_feat_single(self,
                            cls_score: Tensor,  # cls的得分
                            bbox_pred: Tensor,  # 预测框
                            quality_score: Tensor,  # 质量分数
                            anchors: Tensor,  # 锚框
                            labels: Tensor,  # 锚框标签
                            label_weights: Tensor,  # 锚框权重
                            bbox_targets: Tensor,  # 真实边界框
                            stride: Tuple[int],  # 步长
                            num_total_samples: int = 1.0):  # 正样本
        """计算单个特征层级的损失 - 统一可靠性权重版本"""
        # 确保特征图的水平和垂直步长相同
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'

        # === 🚀 混合精度优化：统一转换为FP32进行损失计算 ===
        cls_score = cls_score.float()
        bbox_pred = bbox_pred.float()
        quality_score = quality_score.float()
        anchors = anchors.float()
        bbox_targets = bbox_targets.float()

        # 重塑输入张量为2D形式，方便后续处理
        anchors = anchors.reshape(-1, 4)  # [n, 4] 所有anchor展平
        # [B,C,H,W] -> [B*H*W, C] 将特征图重排为2D形式
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        # [B,4*(reg_max+1),H,W] -> [B*H*W, 4*(reg_max+1)]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        # [B,1,H,W] -> [B*H*W, 1] 将质量分数重排为2D形式
        quality_score = quality_score.permute(0, 2, 3, 1).reshape(-1, 1)

        # 重塑目标张量
        bbox_targets = bbox_targets.reshape(-1, 4)  # [n, 4] 锚框对应的真实边界框目标
        labels = labels.reshape(-1)  # [n] 类别标签
        label_weights = label_weights.reshape(-1)  # [n] 标签权重

        # 获取本层中正样本数量
        bg_class_ind = self.num_classes  # 背景类的索引
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # 初始化分数张量（用于QFL损失）
        score = label_weights.new_zeros(labels.shape)

        # 初始化变量，避免在条件块外引用未定义的变量
        pos_anchors = None
        pos_bbox_pred = None

        if len(pos_inds) > 0:  # 如果有正样本
            # 提取正样本相关数据
            pos_bbox_targets = bbox_targets[pos_inds]  # 真实值的边界框目标
            pos_bbox_pred = bbox_pred[pos_inds]  # 预测偏移值
            pos_anchors = anchors[pos_inds]  # 代表的是与正样本匹配的锚框
            # 计算anchor中心点并除以stride回到特征图，得到锚框的在特征图的中心点
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # 4. 检查分类分数 - 添加数值稳定性保护
            weight_targets = cls_score.detach()  # 从计算图中分离张量 cls_score。这意味着后续的操作不会影响到梯度计算。
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            # 🔧 数值稳定性：防止权重过小导致梯度消失
            weight_targets = torch.clamp(weight_targets, min=1e-6)

            pos_bbox_pred = pos_bbox_pred.reshape(-1, 4, self.reg_max + 1)  # 变回4x17
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)  #

            # 生成最终的预测边界框 - 添加数值稳定性保护
            pos_decode_bbox_pred = distance2bbox(
                pos_anchor_centers,
                pos_bbox_pred_corners
            )
            # 🔧 数值稳定性：确保边界框坐标在合理范围内
            pos_decode_bbox_pred = torch.clamp(pos_decode_bbox_pred, min=-1e6, max=1e6)

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]  # 回到特征图
            pos_decode_bbox_targets = torch.clamp(pos_decode_bbox_targets, min=-1e6, max=1e6)

            # 计算IoU分数 - 添加数值稳定性保护
            iou_scores = bbox_overlaps(  # 计算正样本边界框预测与真实边界框目标之间的重叠度（IoU）
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            # 🔧 数值稳定性：确保IoU分数在[0,1]范围内
            score[pos_inds] = torch.clamp(iou_scores, min=0.0, max=1.0)

            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)

            # 将真实边界框，转为相对锚框的偏移量
            target_corners = bbox2distance(
                pos_anchor_centers,
                pos_decode_bbox_targets,
                self.reg_max
            ).reshape(-1)  # 不知道为什么要重塑

            # 🔧 数值稳定性：确保目标角点在合理范围内
            target_corners = torch.clamp(target_corners, min=0.0, max=float(self.reg_max))

            # 🏆 BBOX损失 - 应用统一可靠性权重
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,  # 预测框
                pos_decode_bbox_targets,  # 真实框
                weight=weight_targets,  # 🚀 应用可靠性权重
                avg_factor=1.0)

            # 🏆 DFL损失 - 应用统一可靠性权重
            dfl_reliability_weights = (weight_targets )[:, None].expand(-1, 4).reshape(-1)
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=dfl_reliability_weights,  # 🚀 应用可靠性权重
                avg_factor=4.0)

            # 🔧 数值稳定性：检查损失是否为有限值
            if not torch.isfinite(loss_bbox):
                loss_bbox = bbox_pred.sum() * 0
            if not torch.isfinite(loss_dfl):
                loss_dfl = bbox_pred.sum() * 0

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0.0)

        # 🎯 获取当前特征层的尺度自适应阈值

        # 🏆 分类损失计算（使用标准QFL）
        loss_cls = self.loss_cls(
            cls_score, (labels, score),  # 传入LQE预测用于AQT
            weight=label_weights,
            avg_factor=num_total_samples)  # 🎯 传入尺度自适应阈值

        # 🔧 数值稳定性：检查分类损失是否为有限值
        if not torch.isfinite(loss_cls):
            loss_cls = cls_score.sum() * 0

        # 只在有正样本且开启日志开关时才记录
        if self.log_gt_top5_positive and len(pos_inds) > 0 and pos_anchors is not None and pos_bbox_pred is not None:
            self._log_gt_top5_positive_samples(
                pos_consistency_score=quality_score[pos_inds].squeeze(-1),
                pos_gt_iou=score[pos_inds],
                pos_cls_score=cls_score[pos_inds],
                pos_anchors=pos_anchors,
                pos_bbox_pred=pos_bbox_pred,  # 添加17维特征数据
                stride=stride[0]
            )
        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    # 在头函数的forward之后，自然进入到损失计算环节，
    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            quality_scores: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute losses of the head."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  # 获取特征图的空间尺寸
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        # 根据特征图尺度以及步长，生成对应的锚框列表并返回验证锚框是否在图像内的结果
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # 为每个预测框生成相应的目标标签和回归目标。
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            gt_labels_list=[gt_instances.labels for gt_instances in batch_gt_instances],
        )

        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        # 计算总正样本
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        # 计算损失
        losses_cls, losses_bbox, losses_dfl, \
            avg_factor = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            quality_scores,
            anchor_list,  # 处理五次，每次四个batch一起，根据不同维度来
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            num_total_samples=num_total_samples)

        # 处理平均因子
        avg_factor = sum(avg_factor)  # 重新计算平均因子
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()  #
        # 使用平均因子归一化边界框损失和DFL损失
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))  # 对于每个损失值除以avg_factor
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
                -1, self.num_classes)

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
                    unmap_outputs: bool = True) -> tuple:
        """获取训练目标"""
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]  # 计算每个特征层的锚框数量
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
                           flat_anchors,  # 展平的所有anchor框，形状为(N, 4)
                           valid_flags,  # anchor的有效标志，形状为(N,)
                           num_level_anchors,  # 每个特征层级的anchor数量列表
                           gt_instances,  # InstanceData对象，包含bboxes和labels
                           img_meta,  # 图像的元信息
                           gt_instances_ignore=None,  # 需要忽略的实例
                           unmap_outputs=True):  # 是否需要将输出映射回原始anchor空间
        """为单张图像中的anchor计算回归和分类目标"""

        # 从 img_meta 中安全地获取图像形状，具有先后关系
        img_shape = img_meta.get('img_shape', img_meta.get('pad_shape', img_meta.get('ori_shape')))
        if img_shape is None:
            raise ValueError("Cannot find image shape in metadata")

        # 检查 anchor 是否在原始输入图像的有效范围内，等于是再检查一边
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_shape[:2],
                                           self.train_cfg.allowed_border)

        # 如果没有有效的anchor，返回7个None
        if not inside_flags.any():
            return (None,) * 7

        # 只保留在图像内部的anchor
        anchors = flat_anchors[inside_flags, :]

        # 计算每个特征层级内部的anchor数量
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        # 将 anchors 作为 priors 参数传递给 InstanceData，表示这些锚框是模型在进行目标检测时的先验框
        pred_instances = InstanceData(priors=anchors)

        # 使用分配器将GT分配给anchor
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            num_level_priors=num_level_anchors_inside,  # 它帮助分配器了解每个特征层中有多少锚框可供使用。
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
        bbox_weights = torch.zeros_like(anchors)  # 边界框权重，用于计算损失
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)  # 标签权重，用于计算损失

        # 获取正负样本索引
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 处理正样本
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes  # 获取真实边界框

            bbox_targets[pos_inds, :] = pos_bbox_targets  # Tensor张量不需要for循环语句
            bbox_weights[pos_inds, :] = 1.0

            # 如果标签为 None，默认从 0 开始
            if gt_instances.labels is None:
                labels[pos_inds] = 0
            else:
                # 确保标签从 0 开始，且使用clamp来确保在有效范围内
                labels[pos_inds] = torch.clamp(
                    gt_instances.labels[sampling_result.pos_assigned_gt_inds],
                    min=0,
                    max=self.num_classes - 1
                )

            # 设置正样本的分类权重，如果输入为负的，表示不想增加正样本的权重，使用默认为1
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        # 处理负样本
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0  # 设置负样本权重为1

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

    def _log_gt_top5_positive_samples(self, pos_consistency_score, pos_gt_iou, pos_cls_score, pos_anchors,
                                      pos_bbox_pred, stride):
        """
        🔍 记录正样本的IoU极值分析 - 显示最高和最低IoU样本的详细信息

        Args:
            pos_consistency_score (Tensor): 正样本的LQE质量预测 [num_pos]
            pos_gt_iou (Tensor): 正样本的真实IoU [num_pos]
            pos_cls_score (Tensor): 正样本的分类分数 [num_pos, num_classes]
            pos_anchors (Tensor): 正样本的anchor坐标 [num_pos, 4]
            pos_bbox_pred (Tensor): 正样本的17维特征数据 [num_pos, 4, 17]
            stride (int): 当前层的步长
        """
        if not hasattr(self, '_gt_log_counter'):
            self._gt_log_counter = 0

        self._gt_log_counter += 1

        # 每20次记录一次，避免日志过多
        if self._gt_log_counter % 20 != 0:
            return

        try:
            with torch.no_grad():
                # 获取当前epoch
                current_epoch = getattr(self, 'epoch', 0)

                # 初始化日志保存
                import os

                # 创建日志目录 - 与gfocal_head.py保持一致
                log_dir = 'log'
                os.makedirs(log_dir, exist_ok=True)

                print(f"🔍 GT正样本IoU极值监控 - Epoch {current_epoch}, Step {self._gt_log_counter}, Stride {stride}")
                print(f"   🎯 峰度缩放因子: {self.kurtosis_scale_factor.item():.4f} (可学习参数)")
                print(f"   🎯 峰度偏置: {self.kurtosis_bias.item():.4f} (可学习参数)")
                print(f"   🔥 峰度融合权重: {self.kurtosis_fusion_weight.item():.4f} (可学习参数 - 控制峰度影响强度)")
                print(f"   🚀 峰度计算方式: 完整17维概率分布（更准确反映预测质量）")

                # === 🎯 计算所有正样本的分布统计特征 ===
                # pos_bbox_pred: [num_pos, 4, 17]
                num_pos = pos_bbox_pred.shape[0]  # 正样本数量
                prob = F.softmax(pos_bbox_pred, dim=2)  # [num_pos, 4, 17]

                # 🎯 提取TOP4特征
                top4_values, _ = torch.topk(prob, k=4, dim=2)  # [num_pos, 4, 4]
                top1_mean = top4_values[:, :, 0].mean(dim=1)  # [num_pos] - top1在4个方向的平均
                top2_mean = top4_values[:, :, 1].mean(dim=1)  # [num_pos] - top2在4个方向的平均
                top3_mean = top4_values[:, :, 2].mean(dim=1)  # [num_pos] - top3在4个方向的平均
                top4_mean_val = top4_values[:, :, 3].mean(dim=1)  # [num_pos] - top4在4个方向的平均

                # 🎯 计算top3+top4融合特征
                top3_plus_top4 = (top4_values[:, :, 2] + top4_values[:, :, 3]).mean(dim=1)  # [num_pos]

                # 🎯 计算TOP4均值
                top4_avg = top4_values.mean(dim=2).mean(dim=1)  # [num_pos]

                # 🔥 计算峰度权重 - 使用完整17维分布
                # 🚀 改进：使用完整概率分布计算峰度，而非仅TOP4
                mean_val = prob.mean(dim=2, keepdim=True)  # [num_pos, 4, 1] - 完整17维均值
                centered = prob - mean_val  # [num_pos, 4, 17] - 完整分布中心化
                var = (centered ** 2).mean(dim=2, keepdim=True)  # [num_pos, 4, 1] - 方差
                fourth_moment = (centered ** 4).mean(dim=2, keepdim=True)  # [num_pos, 4, 1] - 四阶矩
                kurtosis_raw = fourth_moment / (var ** 2 + 1e-8)  # [num_pos, 4, 1] - Fisher峰度
                kurtosis_mean = kurtosis_raw.squeeze(2).mean(dim=1)  # [num_pos] - 原始峰度值

                # 🎯 应用缩放因子和偏置后的峰度权重（模拟前向传播中的计算）
                kurtosis_scaled = torch.exp \
                    (kurtosis_mean * self.kurtosis_scale_factor.item() + self.kurtosis_bias.item())  # [num_pos]

                # 📊 峰度权重全局统计
                print(f"\n📊 峰度权重全局统计 (所有正样本):")
                print \
                    (f"   🔢 原始峰度值: 最小={kurtosis_mean.min().item():.2f}, 最大={kurtosis_mean.max().item():.2f}, 均值={kurtosis_mean.mean().item():.2f}")
                print \
                    (f"   ⚡ 缩放后权重: 最小={kurtosis_scaled.min().item():.3f}, 最大={kurtosis_scaled.max().item():.3f}, 均值={kurtosis_scaled.mean().item():.3f}")

                # 🎯 计算分布期望值和标准差
                positions = torch.arange(self.reg_max + 1, device=prob.device, dtype=prob.dtype)
                positions = positions.view(1, 1, -1)  # [1, 1, 17]
                expected_position = (prob * positions).sum(dim=2)  # [num_pos, 4]
                variance = ((positions - expected_position.unsqueeze(2)) ** 2 * prob).sum(dim=2)  # [num_pos, 4]
                std_dev = torch.sqrt(variance + 1e-8).mean(dim=1)  # [num_pos]
                expected_position_mean = expected_position.mean(dim=1)  # [num_pos]

                # 🔧 按真实IoU排序，分别取最高和最低样本
                _, sorted_indices_desc = torch.sort(pos_gt_iou, descending=True)  # 降序：最高IoU
                _, sorted_indices_asc = torch.sort(pos_gt_iou, descending=False)  # 升序：最低IoU

                num_high_samples = min(15, len(sorted_indices_desc))
                num_low_samples = min(15, len(sorted_indices_asc))

                if num_high_samples > 0:
                    # === 🏆 最高IoU样本分析 ===
                    high_indices = sorted_indices_desc[:num_high_samples]

                    high_lqe_pred = pos_consistency_score[high_indices]  # [num_high_samples]
                    high_gt_iou = pos_gt_iou[high_indices]  # [num_high_samples]
                    high_cls_score = pos_cls_score[high_indices]  # [num_high_samples, num_classes]
                    high_top1 = top1_mean[high_indices]  # [num_high_samples]
                    high_top2 = top2_mean[high_indices]  # [num_high_samples]
                    high_top34 = top3_plus_top4[high_indices]  # [num_high_samples]
                    high_top4_avg = top4_avg[high_indices]  # [num_high_samples]
                    high_kurtosis = kurtosis_mean[high_indices]  # [num_high_samples] - 原始峰度
                    high_kurtosis_scaled = kurtosis_scaled[high_indices]  # [num_high_samples] - 缩放后
                    high_std_dev = std_dev[high_indices]  # [num_high_samples]
                    high_expected_pos = expected_position_mean[high_indices]  # [num_high_samples]

                    # 获取最大分类分数
                    high_cls_score_max = high_cls_score.max(dim=1)[0]  # [num_high_samples]

                    print(f"\n🏆 最高IoU样本 (Top {num_high_samples}):")
                    print(f"   📊 真实IoU:    {[f'{x:.3f}' for x in high_gt_iou.cpu().numpy()]}")
                    print(f"   🔮 LQE预测:    {[f'{x:.3f}' for x in high_lqe_pred.cpu().numpy()]}")
                    print(f"   📈 分类分数:   {[f'{x:.3f}' for x in high_cls_score_max.cpu().numpy()]}")
                    print(f"   🎯 Top1均值:   {[f'{x:.3f}' for x in high_top1.cpu().numpy()]}")
                    print(f"   🎯 Top2均值:   {[f'{x:.3f}' for x in high_top2.cpu().numpy()]}")
                    print(f"   🔀 Top3+4融合: {[f'{x:.3f}' for x in high_top34.cpu().numpy()]}")
                    print(f"   📊 TOP4均值:   {[f'{x:.3f}' for x in high_top4_avg.cpu().numpy()]}")
                    print(f"   🔥 原始峰度:   {[f'{x:.2f}' for x in high_kurtosis.cpu().numpy()]}")
                    print(f"   ⚡ 缩放权重:   {[f'{x:.3f}' for x in high_kurtosis_scaled.cpu().numpy()]}")
                    print(f"   📐 标准差:     {[f'{x:.3f}' for x in high_std_dev.cpu().numpy()]}")
                    print(f"   🎲 期望位置:   {[f'{x:.2f}' for x in high_expected_pos.cpu().numpy()]}")

                if num_low_samples > 0:
                    # === 📉 最低IoU样本分析 ===
                    low_indices = sorted_indices_asc[:num_low_samples]

                    low_lqe_pred = pos_consistency_score[low_indices]  # [num_low_samples]
                    low_gt_iou = pos_gt_iou[low_indices]  # [num_low_samples]
                    low_cls_score = pos_cls_score[low_indices]  # [num_low_samples, num_classes]
                    low_top1 = top1_mean[low_indices]  # [num_low_samples]
                    low_top2 = top2_mean[low_indices]  # [num_low_samples]
                    low_top34 = top3_plus_top4[low_indices]  # [num_low_samples]
                    low_top4_avg = top4_avg[low_indices]  # [num_low_samples]
                    low_kurtosis = kurtosis_mean[low_indices]  # [num_low_samples] - 原始峰度
                    low_kurtosis_scaled = kurtosis_scaled[low_indices]  # [num_low_samples] - 缩放后
                    low_std_dev = std_dev[low_indices]  # [num_low_samples]
                    low_expected_pos = expected_position_mean[low_indices]  # [num_low_samples]

                    # 获取最大分类分数
                    low_cls_score_max = low_cls_score.max(dim=1)[0]  # [num_low_samples]

                    print(f"\n📉 最低IoU样本 (Bottom {num_low_samples}):")
                    print(f"   📊 真实IoU:    {[f'{x:.3f}' for x in low_gt_iou.cpu().numpy()]}")
                    print(f"   🔮 LQE预测:    {[f'{x:.3f}' for x in low_lqe_pred.cpu().numpy()]}")
                    print(f"   📈 分类分数:   {[f'{x:.3f}' for x in low_cls_score_max.cpu().numpy()]}")
                    print(f"   🎯 Top1均值:   {[f'{x:.3f}' for x in low_top1.cpu().numpy()]}")
                    print(f"   🎯 Top2均值:   {[f'{x:.3f}' for x in low_top2.cpu().numpy()]}")
                    print(f"   🔀 Top3+4融合: {[f'{x:.3f}' for x in low_top34.cpu().numpy()]}")
                    print(f"   📊 TOP4均值:   {[f'{x:.3f}' for x in low_top4_avg.cpu().numpy()]}")
                    print(f"   🔥 原始峰度:   {[f'{x:.2f}' for x in low_kurtosis.cpu().numpy()]}")
                    print(f"   ⚡ 缩放权重:   {[f'{x:.3f}' for x in low_kurtosis_scaled.cpu().numpy()]}")
                    print(f"   📐 标准差:     {[f'{x:.3f}' for x in low_std_dev.cpu().numpy()]}")
                    print(f"   🎲 期望位置:   {[f'{x:.2f}' for x in low_expected_pos.cpu().numpy()]}")

                print(f"{'=' * 100}")

                # 📝 使用详细数据记录方法（与gfocal_head.py一致）
                try:
                    # 获取当前epoch和iteration信息
                    current_epoch = getattr(self, 'epoch', 0)
                    current_iter = self._gt_log_counter

                    # 构建详细的数据记录
                    data_log = f"KURTOSIS_DATA: epoch={current_epoch}, iter={current_iter}, stride={stride}"
                    data_log += f", total_pos={num_pos}"

                    # 🎯 记录峰度缩放因子（可学习参数）
                    data_log += f", kurtosis_scale_factor={self.kurtosis_scale_factor.item():.4f}"
                    # 🎯 记录峰度偏置（可学习参数）
                    data_log += f", kurtosis_bias={self.kurtosis_bias.item():.4f}"
                    # 🔥 记录峰度融合权重（可学习参数）
                    data_log += f", kurtosis_fusion_weight={self.kurtosis_fusion_weight.item():.4f}"

                    # 🎯 记录峰度全局统计
                    data_log += f", kurtosis_raw_min={kurtosis_mean.min().item():.4f}"
                    data_log += f", kurtosis_raw_max={kurtosis_mean.max().item():.4f}"
                    data_log += f", kurtosis_raw_mean={kurtosis_mean.mean().item():.4f}"
                    data_log += f", kurtosis_scaled_min={kurtosis_scaled.min().item():.4f}"
                    data_log += f", kurtosis_scaled_max={kurtosis_scaled.max().item():.4f}"
                    data_log += f", kurtosis_scaled_mean={kurtosis_scaled.mean().item():.4f}"

                    # 🏆 高IoU样本统计
                    if num_high_samples > 0:
                        data_log += f", high_iou_count={num_high_samples}"
                        data_log += f", high_gt_iou_mean={high_gt_iou.mean().item():.4f}"
                        data_log += f", high_lqe_pred_mean={high_lqe_pred.mean().item():.4f}"
                        data_log += f", high_cls_score_mean={high_cls_score_max.mean().item():.4f}"
                        data_log += f", high_top1_mean={high_top1.mean().item():.4f}"
                        data_log += f", high_top2_mean={high_top2.mean().item():.4f}"
                        data_log += f", high_top34_mean={high_top34.mean().item():.4f}"
                        data_log += f", high_top4_avg_mean={high_top4_avg.mean().item():.4f}"
                        data_log += f", high_kurtosis_raw_mean={high_kurtosis.mean().item():.4f}"
                        data_log += f", high_kurtosis_scaled_mean={high_kurtosis_scaled.mean().item():.4f}"
                        data_log += f", high_std_dev_mean={high_std_dev.mean().item():.4f}"
                        data_log += f", high_expected_pos_mean={high_expected_pos.mean().item():.4f}"

                    # 📉 低IoU样本统计
                    if num_low_samples > 0:
                        data_log += f", low_iou_count={num_low_samples}"
                        data_log += f", low_gt_iou_mean={low_gt_iou.mean().item():.4f}"
                        data_log += f", low_lqe_pred_mean={low_lqe_pred.mean().item():.4f}"
                        data_log += f", low_cls_score_mean={low_cls_score_max.mean().item():.4f}"
                        data_log += f", low_top1_mean={low_top1.mean().item():.4f}"
                        data_log += f", low_top2_mean={low_top2.mean().item():.4f}"
                        data_log += f", low_top34_mean={low_top34.mean().item():.4f}"
                        data_log += f", low_top4_avg_mean={low_top4_avg.mean().item():.4f}"
                        data_log += f", low_kurtosis_raw_mean={low_kurtosis.mean().item():.4f}"
                        data_log += f", low_kurtosis_scaled_mean={low_kurtosis_scaled.mean().item():.4f}"
                        data_log += f", low_std_dev_mean={low_std_dev.mean().item():.4f}"
                        data_log += f", low_expected_pos_mean={low_expected_pos.mean().item():.4f}"

                    # 记录到MMLogger
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.info(data_log)
                    else:
                        print(f"📊 {data_log}")

                except Exception as log_err:
                    print(f"⚠️  记录详细数据失败（不影响训练）: {log_err}")

        except Exception as e:
            # 日志记录失败不应该影响训练
            print(f"❌ GT正样本监控日志失败: {e}")
            import traceback
            traceback.print_exc()
            pass



