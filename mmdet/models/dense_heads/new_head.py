      
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from ..utils import (filter_scores_and_topk, multi_apply)
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import (bbox2distance, bbox_overlaps, distance2bbox)
from ..task_modules.samplers import PseudoSampler
from ..task_modules.prior_generators import anchor_inside_flags
from ..task_modules.assigners.assign_result import AssignResult
from mmdet.models.utils import images_to_levels, unmap
from .anchor_head import AnchorHead
from typing import List, Optional, Sequence, Tuple
from torch import Tensor
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.logging import MMLogger
from mmengine import MessageHub


class ECABlock(nn.Module):
    """
    🚀 ECA注意力机制：高效的通道注意力，参数极少
    基于论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应kernel size计算
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1  # 确保奇数
        k = max(3, k)  # 最小为3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化 [B, C, 1, 1]
        y = self.avg_pool(x)
        # 转换为1D卷积输入 [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D卷积学习通道相关性 [B, 1, C]
        y = self.conv(y)
        # 转换回通道注意力权重 [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Sigmoid激活
        y = self.sigmoid(y)
        # 应用注意力权重
        return x * y.expand_as(x)


class LayerNorm2d(nn.Module):
    """
    🚀 2D LayerNorm：用于ConvNet的LayerNorm实现
    比BatchNorm更稳定，适合小batch训练
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


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
        # 🚀 混合精度训练：在积分计算中使用32位精度确保准确性
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入为32位精度
            x_float32 = x.float()

            # 1. 重塑输入并应用softmax
            # 将输入重塑为(-1, reg_max + 1)的形状，并在每个预测上应用softmax
            x_float32 = F.softmax(x_float32.reshape(-1, self.reg_max + 1), dim=1)

            # 2. 计算积分（加权和）
            # F.linear实现了 sum{P(y_i) * y_i}
            # project提供了y_i值[0,1,2,...,reg_max]
            # 结果重塑为(-1, 4)，对应四个方向的预测
            x_float32 = F.linear(x_float32, self.project.type_as(x_float32)).reshape(-1, 4)

            # 转换回原始精度
            return x_float32.to(x.dtype)


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
                 dfl_quality_aware=False,  # 是否启用质量感知DFL权重
                 # === 🚀 新增：RQFL配置参数 ===
                 enable_rqfl=False,  # 是否启用回归质量焦点损失
                 rqfl_beta=2.0,  # RQFL的beta参数
                 rqfl_quality_threshold=0.3,  # 质量一致性阈值
                 rqfl_loss_weight=0.25,  # RQFL损失权重
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
        self.dfl_quality_aware = dfl_quality_aware  # 质量感知DFL权重开关

        # === 🚀 新增：RQFL相关属性 ===
        self.enable_rqfl = enable_rqfl
        self.rqfl_beta = rqfl_beta
        self.rqfl_quality_threshold = rqfl_quality_threshold
        self.rqfl_loss_weight = rqfl_loss_weight
        
        # === 🔄 训练阶段自适应：获取总训练轮次 ===
        self.total_epochs = None  # 将在训练时从train_cfg获取

        self.total_dim = reg_topk
        if add_mean:
            self.total_dim += 1  # 如果使用均值，维度+1
        # self.logger.info(f'total dim = {self.total_dim * 4}')# 4个方向的总维度
        # 调用父类初始化
        super().__init__(num_classes, in_channels, **kwargs)  # 进行初始化
        # 设置采样策略,不使用传统的标签分配策略
        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)  # 构建分配器
            # 构建采样器
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)  # TASK_UTILS任务相关工具
            self.initial_epoch = self.train_cfg['initial_epoch']
            # === 🔄 训练阶段自适应：获取总训练轮次 ===
            self.total_epochs = self.train_cfg.get('max_epochs', 300)
        self.attention_weight_exp = nn.Parameter(torch.tensor(0.5))
        self.attention_weight = nn.Parameter(torch.tensor(0.5))
        self.kurtosis_scale = nn.Parameter(torch.tensor(1.0))
        # 构建将离散的分类预测转换为连续的边界框坐标值层和DFL损失（这里对于integral还需要再研究一下）
        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)  # MODELS 注册器：用于核心模型组件

        # === 🚀 新增：构建Ultra-Simple RQFL损失 ===
        if self.enable_rqfl:
            rqfl_cfg = dict(
                type='RegressionQualityFocalLoss',
                beta=self.rqfl_beta,
                # quality_threshold已不再需要（Ultra-Simple版本零参数调优）
                reg_max=self.reg_max,
                loss_weight=self.rqfl_loss_weight
            )
            self.loss_rqfl = MODELS.build(rqfl_cfg)
            self.logger.info(
                f"🚀 Ultra-Simple RQFL v2.0已启用: beta={self.rqfl_beta}, weight={self.rqfl_loss_weight} (零调参设计)")
        else:
            self.loss_rqfl = None

        # === 注释：移除LQE独立质量损失 ===
        # 原因：quality_score已经通过 cls_score * quality_score 参与QFL损失计算
        # QFL本身就是质量感知的，提供了足够的训练信号，无需额外的质量损失

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

        quality_lqe_14_hidden_dim = 64  # 增大隐藏维度

        self.quality_lqe_14 = nn.Sequential(
            # 🔥 Stage 1: 特征归一化 + 跨通道融合
            # 先对特征进行归一化（动态通道数）
            nn.BatchNorm2d(20),
            # 使用标准卷积进行跨通道信息融合（而非深度可分离卷积）
            nn.Conv2d(20, quality_lqe_14_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(quality_lqe_14_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # 🆕 添加轻微dropout防止过拟合

            # 🔥 Stage 2: 特征增强 + 注意力机制
            nn.Conv2d(quality_lqe_14_hidden_dim, quality_lqe_14_hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(quality_lqe_14_hidden_dim * 2),
            nn.ReLU(inplace=True),

            # ECA注意力机制
            ECABlock(quality_lqe_14_hidden_dim * 2),

            # 🔥 Stage 3: 质量预测输出
            nn.Conv2d(quality_lqe_14_hidden_dim * 2, quality_lqe_14_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(quality_lqe_14_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),  # 🆕 输出前轻微dropout
            nn.Conv2d(quality_lqe_14_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.quality_lqe_cnn = nn.Sequential(
            # 输入特征归一化层 - 14维特征输入
            nn.BatchNorm2d(20),
            # 第一层：质量特征提取 - 增大通道数提升表达能力
            nn.Conv2d(20, self.reg_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.reg_channels * 2),  # 更多组归一化
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.08),  # 降低dropout防止过拟合

            # 第二层：质量融合计算 - 保持较大通道数
            nn.Conv2d(self.reg_channels * 2, self.reg_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.reg_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.04),

            # 第三层：质量精炼 - 渐进式降维
            nn.Conv2d(self.reg_channels, self.reg_channels // 2, kernel_size=1, padding=0),
            nn.GroupNorm(2, self.reg_channels // 2),
            nn.SiLU(inplace=True),
            # 🚀 新增：残差连接层提升梯度流
            nn.Conv2d(self.reg_channels // 2, self.reg_channels // 4, kernel_size=1, padding=0),
            nn.GroupNorm(1, self.reg_channels // 4),
            nn.SiLU(inplace=True),

            # 输出层：生成质量一致性分数
            nn.Conv2d(self.reg_channels // 4, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 🆕 温度校准参数：帮助模型输出更校准的概率
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始温度1.5，降低过度自信

        # 移除复杂的第二阶段，直接使用简化网络
        self.lqe_stage2 = None

        # === 8D LQE特征：移除可学习权重，让网络自主学习 ===
        # 不再使用手工设计的权重参数，直接输入原始特征

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
        # 初始化10维质量预测LQE网络 - 修复版
        for m in self.quality_lqe_14:
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 1:  # 输出层
                    normal_init(m, std=0.1, bias=0.0)  # 增大输出层初始化
                else:  # 隐藏层
                    normal_init(m, std=0.02)  # 增大隐藏层初始化
        for m in self.quality_lqe_cnn:
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 1:  # 输出层
                    normal_init(m, std=0.1, bias=0.0)  # 增大输出层初始化
                else:  # 隐藏层
                    normal_init(m, std=0.02)  # 增大隐藏层初始化
        # 分类层特殊初始化
        bias_cls = bias_init_with_prob(0.01)  # 增大初始概率值
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)

        # 回归层初始化
        normal_init(self.gfl_reg, std=0.01)

    def _compute_14d_quality_features(self, bbox_pred, cls_score, reg_max=16, feature_mode="full"):
        """
        计算质量特征 - 支持多种模式的特征组合

        支持的模式：
        1. "full": 完整14维特征 (4维HQK + 4维Top1 + 4维Top4 + 1维熵 + 1维方差)
        2. "shape": 形状特征 (4维HQK + 1维熵 + 1维方差) - 6维
        3. "prob": 概率特征 (4维Top1 + 4维Top4) - 8维
        4. "minimal": 最小特征 (1维熵 + 1维方差) - 2维
        5. "hqk_only": 仅HQK特征 (4维HQK) - 4维
        6. "top_only": 仅Top特征 (4维Top1 + 4维Top4) - 8维
        7. "entropy_only": 仅熵特征 (1维熵) - 1维
        8. "variance_only": 仅方差特征 (1维方差) - 1维

        关键改进：
        1. 使用稳定的归一化方法（Tanh + 分位数裁剪）
        2. 减少极值敏感性
        3. 提高训练稳定性
        4. 支持多种特征组合模式

        Args:
            bbox_pred (Tensor): 边界框预测，形状为 (N, 4*(reg_max+1), H, W)
            cls_score (Tensor): 分类分数，形状为 (N, num_classes, H, W)
            reg_max (int): 最大回归值，默认16
            feature_mode (str): 特征模式，支持 "full", "shape", "prob", "minimal", 
                              "hqk_only", "top_only", "entropy_only", "variance_only"

        Returns:
            Tensor: 质量特征，维度根据模式而定
        """
        # 🚀 混合精度训练：在质量特征计算中使用32位精度确保准确性
        with torch.cuda.amp.autocast(enabled=False):
            N, C, H, W = bbox_pred.shape

            # === 1. 计算分类置信度（稳定化处理） ===
            cls_confidence = cls_score.float().sigmoid().max(dim=1, keepdim=True)[0]  # [N, 1, H, W]

            # === 2. 计算概率分布 ===
            prob = F.softmax(bbox_pred.float().reshape(N, 4, reg_max + 1, H, W), dim=2)  # [N, 4, reg_max+1, H, W]
            indices = torch.arange(reg_max + 1, device=prob.device, dtype=torch.float32).view(1, 1, -1, 1, 1)

            # === 3. 计算稳定的HQK特征 ===
            mean = torch.sum(prob * indices, dim=2, keepdim=True)  # [N, 4, 1, H, W]
            var = torch.sum(prob * (indices - mean) ** 2, dim=2, keepdim=True)  # [N, 4, 1, H, W]
            fourth_moment = torch.sum(prob * (indices - mean) ** 4, dim=2)  # [N, 4, H, W]

            # 🆕 稳定的峰度计算：使用log变换 + 裁剪
            safe_var = var.squeeze(2).clamp(min=1e-6)  # 避免除零
            raw_kurtosis = fourth_moment / (safe_var ** 2)

            # 🆕 使用分位数裁剪代替全局Min-Max，更稳定
            kurtosis_flat = raw_kurtosis.view(N, -1)
            q25 = torch.quantile(kurtosis_flat, 0.25, dim=1, keepdim=True)
            q75 = torch.quantile(kurtosis_flat, 0.75, dim=1, keepdim=True)
            iqr = q75 - q25 + 1e-8

            # IQR标准化 + Tanh饱和函数
            hqk_normalized = (raw_kurtosis.view(N, 4, H, W) - q25.view(N, 1, 1, 1)) / iqr.view(N, 1, 1, 1)
            hqk_4d = torch.tanh(hqk_normalized * 0.5)  # 限制在[-1,1]，然后映射到[0,1]
            hqk_4d = (hqk_4d + 1.0) * 0.5  # 映射到[0,1]

            # === 4. 计算稳定的Top1特征（分类相关，需要加权） ===
            top1_values, _ = torch.topk(prob, k=1, dim=2)  # [N, 4, 1, H, W]
            top1_4d = top1_values.squeeze(2)  # [N, 4, H, W]

            # === 5. 计算稳定的Top4均值（分类相关，需要加权） ===
            top4_values, _ = torch.topk(prob, k=4, dim=2)  # [N, 4, 4, H, W]
            top4_mean = top4_values.mean(dim=2)  # [N, 4, H, W]

            # === 6. 计算聚合特征（分布形状特征，不依赖分类） ===
            # 🆕 添加分布的熵特征，衡量不确定性
            entropy = -torch.sum(prob * torch.log(prob.clamp(min=1e-8)), dim=2)  # [N, 4, H, W]
            # 熵归一化：完全均匀分布时熵最大为log(reg_max+1)
            max_entropy = torch.log(torch.tensor(reg_max + 1.0, device=prob.device))
            entropy_normalized = (entropy / max_entropy).mean(dim=1, keepdim=True)  # [N, 1, H, W]

            # === 7. 计算方差特征（分布形状特征，不依赖分类） ===
            variance_mean = var.squeeze(2).mean(dim=1, keepdim=True)  # [N, 1, H, W]
            # 方差归一化：使用Sigmoid函数
            variance_normalized = torch.sigmoid(variance_mean - 2.0)  # 中心在方差=2处

            # === 8. 智能权重策略：根据特征语义分类加权 ===
            # 🎯 策略1：概率相关特征 × 分类置信度 (Top1, Top4)
            # 理由：这些特征直接反映分类概率，应该受分类置信度影响
            # 🚀 渐进式权重策略：在训练初期更保守，避免过度抑制
            # 使用软权重而不是硬截断，减少训练波动
            cls_weight = torch.sigmoid(cls_confidence * 2.0 - 1.0) * 0.8 + 0.2  # 范围[0.2, 1.0]
            top1_weighted = top1_4d * cls_weight  # [N, 4, H, W] - 加权Top1
            top4_weighted = top4_mean * cls_weight  # [N, 4, H, W] - 加权Top4

            # 🎯 策略2：形状特征保持原始 (HQK, 熵, 方差)
            # 理由：这些特征反映分布形状，与分类置信度独立，有助于提供互补信息
            hqk_unweighted = hqk_4d  # [N, 4, H, W] - 原始HQK
            entropy_unweighted = entropy_normalized  # [N, 1, H, W] - 原始熵
            variance_unweighted = variance_normalized  # [N, 1, H, W] - 原始方差

            prob_topk, _ = prob.topk(self.reg_topk, dim=2)
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2).reshape(N, -1, H, W)
            stat_cls = stat * cls_confidence
            stat_k = stat * raw_kurtosis.mean(dim=1, keepdim=True)
            stat_cls_k = stat * raw_kurtosis.mean(dim=1, keepdim=True) * cls_confidence
            # === 9. 根据模式组合特征 ===
            feature_components = {
                'hqk': hqk_unweighted,  # [N, 4, H, W] - 形状特征（不加权）
                'top1': top1_weighted,  # [N, 4, H, W] - 概率特征（加权）
                'top4': top4_weighted,  # [N, 4, H, W] - 概率特征（加权）
                'entropy': entropy_unweighted,  # [N, 1, H, W] - 形状特征（不加权）
                'variance': variance_unweighted,  # [N, 1, H, W] - 形状特征（不加权）
                'top4+mean': stat,
                'top4*cls': stat_cls,
                'top4*k': stat_k,
                'top4*cls*k': stat_cls_k,
            }

            # 🎯 模式识别：根据feature_mode返回不同的特征组合
            if feature_mode == "full":
                # 完整14维特征：4+4+4+1+1=14维
                quality_features = torch.cat([
                    feature_components['hqk'],
                    feature_components['top1'],
                    feature_components['top4'],
                    feature_components['entropy'],
                    feature_components['variance'],
                ], dim=1)
            elif feature_mode == "regression_only":
                # 🚀 新增：仅回归质量特征（不含分类偏差）
                # 专注于纯粹的定位质量，避免分类置信度干扰
                quality_features = torch.cat([
                    feature_components['hqk'],      # 4维：分布形状特征
                    feature_components['variance'], # 1维：分布集中度
                ], dim=1)  # 总共5维，更简洁更专注
            if feature_mode == "top4":
                # 完整14维特征：4+4+4+1+1=14维
                quality_features = torch.cat([
                    feature_components['top4+mean'],
                ], dim=1)
            if feature_mode == "top4*k":
                # 完整14维特征：4+4+4+1+1=14维
                quality_features = torch.cat([
                    feature_components['top4*k'],
                ], dim=1)
            if feature_mode == "top4_cls":
                # 完整14维特征：4+4+4+1+1=14维
                quality_features = torch.cat([
                    feature_components['top4*cls'],
                ], dim=1)
            if feature_mode == "top4_cls*k":
                # 完整14维特征：4+4+4+1+1=14维
                quality_features = torch.cat([
                    feature_components['top4*cls*k'],
                ], dim=1)
            # 转换回原始精度
            return quality_features.to(bbox_pred.dtype)

    def forward(self, feats):
        """前向传播处理来自主干网络的特征

           参数:
               feats (tuple[Tensor]): 主干网络输出的特征图元组，每个元素是4D张量

           返回:
               tuple: 通常包含分类分数和边界框预测的元组
                   cls_scores: 所有尺度级别的分类和质量(IoU)联合分数
                   bbox_preds: 所有尺度级别的边界框分布logits
           """
        results = multi_apply(self.forward_single, feats,
                              self.scales)  # 对每个特征层级应用forward_single,函数feats 是来自特征金字塔网络(FPN)的特征图列表,self.scales 是每个特征层级对应的尺度因子列表
        cls_scores, bbox_preds, consistency_scores = results
        return cls_scores, bbox_preds, consistency_scores

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
        # 为了学习特征，从而进行分开训练
        # 分类分支: 通过多个卷积层处理特征
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        # 回归分支: 通过多个卷积层处理特征
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # 获取分类输出
        cls_score = self.gfl_cls(cls_feat)
        # 获取epoch数
        message_hub = MessageHub.get_current_instance()
        self.epoch = message_hub.get_info('epoch')

        bbox_pred = scale(self.gfl_reg(reg_feat)).float()  # 通过回归层并应用尺度因子得到边界框预测

        # === 🚀 使用质量预测LQE网络生成质量一致性分数 ===
        # 🎯 智能梯度耦合策略：根据配置选择最优策略
        feature_mode = getattr(self, 'feature_mode', 'top4*k')  # 获取特征模式，默认为'full'
        coupling_strategy = getattr(self, 'LQE_strategy', 'CNN')
        # 特征选择
        features = self._compute_14d_quality_features(bbox_pred, cls_score, self.reg_max, feature_mode)

        # LQE网络 - 确保在所有情况下都定义raw_consistency_score
        if coupling_strategy == '14d':
            # 🚀 混合精度训练：在LQE网络前向传播中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                # 通过14维智能加权LQE网络生成质量一致性分数
                raw_consistency_score = self.quality_lqe_14(features.float())  # [N, 1, H, W]
        elif coupling_strategy == 'CNN':
            # 🚀 混合精度训练：在LQE网络前向传播中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                # 通过CNN LQE网络生成质量一致性分数
                raw_consistency_score = self.quality_lqe_cnn(features.float())  # [N, 1, H, W]

        # 🔥 修复：确保分类分数也使用相同精度进行计算
        cls_score_float32 = cls_score.float().sigmoid()
        # 质量感知分类：分类分数 × 一致性分数（都是float32）
        cls_score_fused = cls_score_float32 * raw_consistency_score
        # 转换回原始精度以保持一致性
        cls_score = cls_score_fused.to(cls_score.dtype)

        return cls_score, bbox_pred, raw_consistency_score

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
                            cls_score: Tensor,  # cls的得分
                            bbox_pred: Tensor,  # 预测框
                            consistency_score: Tensor,  # 一致性分数
                            anchors: Tensor,  # 锚框
                            labels: Tensor,  # 锚框标签
                            label_weights: Tensor,  # 锚框权重
                            bbox_targets: Tensor,  # 真实边界框
                            stride: Tuple[int],  # 步长
                            num_total_samples: int = 1.0):  # 正样本
        """计算单个特征层级的损失"""
        # 确保特征图的水平和垂直步长相同
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'

        # 重塑输入张量为2D形式，方便后续处理
        anchors = anchors.reshape(-1, 4)  # [n, 4] 所有anchor展平
        # [B,C,H,W] -> [B*H*W, C] 将特征图重排为2D形式
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        # [B,4*(reg_max+1),H,W] -> [B*H*W, 4*(reg_max+1)]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        # [B,1,H,W] -> [B*H*W, 1] 将一致性分数重排为2D形式
        consistency_score = consistency_score.permute(0, 2, 3, 1).reshape(-1, 1)

        # 重塑目标张量
        bbox_targets = bbox_targets.reshape(-1, 4)  # [n, 4] 锚框对应的真实边界框目标
        labels = labels.reshape(-1)  # [n] 类别标签
        label_weights = label_weights.reshape(-1)  # [n] 标签权重

        # 获取本层中正样本数量
        bg_class_ind = self.num_classes  # 背景类的索引
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # 初始化分数张量（用于QFL损失）
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:  # 如果有正样本
            # 提取正样本相关数据
            pos_bbox_targets = bbox_targets[pos_inds]  # 真实值的边界框目标
            pos_bbox_pred = bbox_pred[pos_inds]  # 预测偏移值
            pos_anchors = anchors[pos_inds]  # 代表的是与正样本匹配的锚框
            # 计算anchor中心点并除以stride回到特征图，得到锚框的在特征图的中心点
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # 4. 检查分类分数
            weight_targets = cls_score.detach()  # 从计算图中分离张量 cls_score。这意味着后续的操作不会影响到梯度计算。
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_pred_iou = consistency_score[pos_inds].mean()  # 使用consistency_score作为预测一致性

            pos_bbox_pred = pos_bbox_pred.reshape(-1, 4, self.reg_max + 1)  # 变回4x17

            # 🚀 混合精度训练：在积分计算中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                pos_bbox_pred_corners = self.integral(pos_bbox_pred.float()).to(pos_bbox_pred.dtype)

            # 生成最终的预测边界框
            pos_decode_bbox_pred = distance2bbox(
                pos_anchor_centers,
                pos_bbox_pred_corners
            )

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]  # 回到特征图

            # 🚀 混合精度训练：在IoU计算中使用32位精度确保准确性
            with torch.cuda.amp.autocast(enabled=False):
                score[pos_inds] = bbox_overlaps(  # 计算正样本边界框预测与真实边界框目标之间的重叠度（IoU）
                    pos_decode_bbox_pred.detach().float(),
                    pos_decode_bbox_targets.float(),
                    is_aligned=True).to(score.dtype)
            pos_gt_iou = score[pos_inds].mean()

            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)

            # 🚀 混合精度训练：在bbox2distance计算中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                # 将真实边界框，转为相对锚框的偏移量
                target_corners = bbox2distance(
                    pos_anchor_centers.float(),
                    pos_decode_bbox_targets.float(),
                    self.reg_max
                ).reshape(-1).to(pos_bbox_pred.dtype)  # 不知道为什么要重塑

            # 🚀 混合精度训练：在损失计算中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred.float(),  # 预测框
                    pos_decode_bbox_targets.float(),  # 真实框
                    weight=weight_targets.float(),  # 传入分类分数
                    avg_factor=1.0)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners.float(),
                    target_corners.float(),
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1).float(),
                    avg_factor=4.0)

            # === 🚀 新增：RQFL损失计算 ===
            if self.enable_rqfl and self.loss_rqfl is not None:
                # 🚀 混合精度训练：在RQFL损失计算中使用32位精度
                with torch.cuda.amp.autocast(enabled=False):
                    # 计算RQFL需要的输入
                    # 获取正样本的分类分数 (已经是sigmoid后的概率)
                    pos_cls_score_sigmoid = weight_targets.float()  # 这个就是sigmoid后的分类分数

                    # 使用真实IoU作为pred_iou（或者可以用quality_score）
                    pos_gt_iou_for_rqfl = score[pos_inds].float()  # 真实IoU

                    # 计算RQFL损失 - 对四个边分别计算
                    rqfl_losses = []

                    # 重塑预测和目标数据以便按边处理
                    pred_reshaped = pred_corners.reshape(-1, 4, self.reg_max + 1).float()  # [num_pos, 4, reg_max+1]
                    target_reshaped = target_corners.reshape(-1, 4).float()  # [num_pos, 4]

                    for edge_idx in range(4):
                        edge_pred = pred_reshaped[:, edge_idx, :]  # [num_pos, reg_max+1]
                        edge_target = target_reshaped[:, edge_idx]  # [num_pos]

                        if len(edge_pred) > 0:
                            rqfl_loss_edge = self.loss_rqfl(
                                pred_distances=edge_pred,
                                gt_distances=edge_target,
                                pred_iou=pos_gt_iou_for_rqfl,
                                cls_score=pos_cls_score_sigmoid
                            )
                            rqfl_losses.append(rqfl_loss_edge)

                    # 平均四个边的RQFL损失
                    if rqfl_losses:
                        loss_rqfl = sum(rqfl_losses) / len(rqfl_losses)
                    else:
                        loss_rqfl = pred_corners.sum() * 0
            else:
                loss_rqfl = pred_corners.sum() * 0

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_rqfl = bbox_pred.sum() * 0  # === 🚀 新增：RQFL损失初始化 ===
            weight_targets = bbox_pred.new_tensor(0)
            pos_pred_iou = bbox_pred.new_tensor(0.0)
            pos_gt_iou = bbox_pred.new_tensor(0.0)

        # 🚀 混合精度训练：在分类损失计算中使用32位精度
        with torch.cuda.amp.autocast(enabled=False):
            # === 🔄 训练阶段自适应：计算epoch_ratio ===
            epoch_ratio = None
            if self.training and self.total_epochs is not None:
                try:
                    message_hub = MessageHub.get_current_instance()
                    current_epoch = message_hub.get_info('epoch')
                    epoch_ratio = current_epoch / self.total_epochs
                    epoch_ratio = max(0.0, min(1.0, epoch_ratio))
                except Exception:
                    epoch_ratio = None
            
            loss_cls = self.loss_cls(
                cls_score.float(), (labels, score.float()),
                weight=label_weights.float(),
                avg_factor=num_total_samples,
                epoch_ratio=epoch_ratio)  # 🔄 传递训练阶段参数

        # === 🚀 新增：GT前五正样本IoU监控 ===
        if self.training and len(pos_inds) > 0:
            # 获取正样本的正确类别分数
            pos_labels = labels[pos_inds]
            pos_cls_scores_all = cls_score[pos_inds]  # [num_pos, num_classes]
            pos_correct_cls_score = pos_cls_scores_all.gather(1, pos_labels.unsqueeze(1)).squeeze(1)

            self._log_gt_top5_positive_samples(
                pos_consistency_score=consistency_score[pos_inds].squeeze(-1),
                pos_gt_iou=score[pos_inds],
                pos_cls_score=pos_correct_cls_score,
                pos_anchors=pos_anchors,
                pos_bbox_pred=pos_bbox_pred,  # 添加17维特征数据
                stride=stride[0]
            )

        return loss_cls, loss_bbox, loss_dfl, loss_rqfl, weight_targets.sum(), pos_pred_iou, pos_gt_iou

    # 在头函数的forward之后，自然进入到损失计算环节，
    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            consistency_scores: List[Tensor],
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
            cls_scores=cls_scores,  # 🚀 传递分类分数给分配器
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
        losses_cls, losses_bbox, losses_dfl, losses_rqfl, \
            avg_factor, pos_pred_ious, pos_gt_ious = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            consistency_scores,
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
        # === 🚀 新增：归一化RQFL损失 ===
        losses_rqfl = list(map(lambda x: x / avg_factor, losses_rqfl))

        # 计算平均IOU值并添加到损失字典中
        if self.training and pos_pred_ious and pos_gt_ious:
            # 确保所有张量都是至少一维的，然后连接
            pos_pred_ious_1d = [iou.flatten() if iou.dim() > 0 else iou.unsqueeze(0) for iou in pos_pred_ious]
            pos_gt_ious_1d = [iou.flatten() if iou.dim() > 0 else iou.unsqueeze(0) for iou in pos_gt_ious]

            # 将所有尺度的IoU连接起来
            all_pos_pred_ious = torch.cat(pos_pred_ious_1d, dim=0)
            all_pos_gt_ious = torch.cat(pos_gt_ious_1d, dim=0)
            avg_pos_pred_iou = torch.mean(all_pos_pred_ious)
            avg_pos_gt_iou = torch.mean(all_pos_gt_ious)

            # 注意：已移除质量指标监控以简化日志输出

        else:
            avg_pos_pred_iou = torch.tensor(0.0, device=cls_scores[0].device)
            avg_pos_gt_iou = torch.tensor(0.0, device=cls_scores[0].device)

        # 简化的损失字典返回（移除统计信息）
        loss_dict = dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,

            # 保留基本指标
            pred_iou=avg_pos_pred_iou,
            gt_iou=avg_pos_gt_iou,

            # 注意：已移除所有统计信息以避免日志污染
        )

        # === 🚀 新增：添加RQFL损失到返回字典（如果启用） ===
        if self.enable_rqfl:
            loss_dict['loss_rqfl'] = losses_rqfl

        return loss_dict

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

            # 🚀 混合精度训练：在预测阶段的积分计算中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                bbox_pred_float32 = self.integral(bbox_pred.float()) * stride[0]
                # 🔥 修复：保持原始变量引用，避免类型推断问题
                bbox_pred = bbox_pred_float32.to(bbox_pred.dtype)

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.num_classes)

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            # 🚀 混合精度训练：在预测阶段的bbox计算中使用32位精度
            with torch.cuda.amp.autocast(enabled=False):
                bboxes = distance2bbox(
                    self.anchor_center(priors).float(),
                    bbox_pred.float(),
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
                    cls_scores: List[Tensor] = None,
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

        # 🚀 处理分类分数：为每张图像准备分类分数
        if cls_scores is not None:
            # cls_scores是List[Tensor]，每个Tensor形状为[N, num_classes, H, W]
            # 需要为每张图像展平分类分数
            cls_scores_list = []
            for i in range(num_imgs):
                # 合并所有层级的分类分数并展平
                img_cls_scores = []
                for level_cls_score in cls_scores:
                    # level_cls_score形状: [N, num_classes, H, W]
                    # 取第i张图像的分数: [num_classes, H, W]
                    img_level_score = level_cls_score[i]  # [num_classes, H, W]
                    # 展平为 [H*W, num_classes]
                    img_level_score = img_level_score.permute(1, 2, 0).reshape(-1, self.num_classes)
                    img_cls_scores.append(img_level_score)
                # 合并所有层级: [total_anchors, num_classes]
                img_cls_scores = torch.cat(img_cls_scores, dim=0)
                cls_scores_list.append(img_cls_scores)
        else:
            cls_scores_list = [None] * num_imgs

        # 计算每张图像的目标
        results = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            cls_scores_list,
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
                           cls_scores=None,  # 🚀 分类分数，形状为(N, num_classes)
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

        # 🚀 处理分类分数：只保留有效anchor对应的分类分数
        if cls_scores is not None:
            cls_scores = cls_scores[inside_flags, :]  # 只保留有效anchor的分类分数

        # 计算每个特征层级内部的anchor数量
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        # 将 anchors 作为 priors 参数传递给 InstanceData，表示这些锚框是模型在进行目标检测时的先验框
        pred_instances = InstanceData(priors=anchors)

        # 🚀 添加分类分数到pred_instances，供质量感知分配器使用
        if cls_scores is not None:
            pred_instances.scores = cls_scores

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

            # 💡 设置正样本的分类权重，支持质量感知权重
            base_pos_weight = self.train_cfg.pos_weight if self.train_cfg.pos_weight > 0 else 1.0

            # 💡 如果分配结果包含样本权重，使用质量感知权重
            if hasattr(assign_result, 'sample_weights') and assign_result.sample_weights is not None:
                # 获取正样本的质量感知权重
                quality_weights = assign_result.sample_weights[pos_inds]
                # 结合基础权重和质量权重
                label_weights[pos_inds] = base_pos_weight * quality_weights
            else:
                # 使用默认权重
                label_weights[pos_inds] = base_pos_weight

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
                                      pos_bbox_pred,
                                      stride):
        """
        🔍 记录每个GT的前5个正样本的一致性预测和真实IoU，以及排序分析

        Args:
            pos_consistency_score (Tensor): 正样本的一致性预测 [num_pos]
            pos_gt_iou (Tensor): 正样本的真实IoU [num_pos]
            pos_cls_score (Tensor): 正样本的正确类别分数 [num_pos]
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

                # 创建日志目录 - 保存在log目录中
                log_dir = 'log'
                os.makedirs(log_dir, exist_ok=True)

                print(f"🔍 GT前5正样本一致性监控 - Epoch {current_epoch}, Step {self._gt_log_counter}, Stride {stride}")

                # 按GT IoU值进行聚类分组（简化版本）
                unique_ious = torch.unique(torch.round(pos_gt_iou * 100) / 100)

                for i, target_iou in enumerate(unique_ious[:5]):  # 最多分析5个不同的IoU组
                    # 找到IoU接近的样本作为同一个GT的正样本
                    iou_mask = torch.abs(pos_gt_iou - target_iou) < 0.05

                    if iou_mask.sum() == 0:
                        continue

                    group_consistency_pred = pos_consistency_score[iou_mask]
                    group_gt_iou = pos_gt_iou[iou_mask]
                    group_cls_score = pos_cls_score[iou_mask]

                    # 取前5个样本（如果不足5个则取全部）
                    num_samples = min(5, len(group_consistency_pred))

                    if num_samples == 0:
                        continue

                    # 按真实IoU排序，取前num_samples个
                    _, top_indices = torch.topk(group_gt_iou, num_samples)

                    top_consistency_pred = group_consistency_pred[top_indices]
                    top_gt_iou = group_gt_iou[top_indices]
                    top_cls_score = group_cls_score[top_indices]

                    # 🚀 计算两种关键乘积
                    # 1. 分类分数与预测一致性乘积（模型预测的质量分数）
                    cls_consistency_product = top_cls_score * top_consistency_pred

                    # 2. 真实IoU与分类分数乘积（真实质量分数）
                    cls_gt_iou_product = top_cls_score * top_gt_iou

                    # 计算两种乘积的排序
                    cls_consistency_ranking = torch.argsort(cls_consistency_product, descending=True)
                    cls_gt_iou_ranking = torch.argsort(cls_gt_iou_product, descending=True)

                    # 计算两种排序的一致性（这是我们要比较的核心指标）
                    ranking_matches = (cls_consistency_ranking == cls_gt_iou_ranking).sum().item()
                    ranking_accuracy = ranking_matches / num_samples

                    # 计算预测误差统计
                    prediction_errors = torch.abs(top_consistency_pred - top_gt_iou)
                    avg_error = prediction_errors.mean().item()
                    max_error = prediction_errors.max().item()

                    # 计算相对误差（百分比）
                    relative_errors = prediction_errors / (top_gt_iou + 1e-8) * 100
                    avg_relative_error = relative_errors.mean().item()

                    print(f"\n📊 GT组 #{i + 1} (目标IoU≈{target_iou.item():.3f}, {num_samples}个样本):")
                    print(f"   🎯 真实IoU:    {[f'{x:.3f}' for x in top_gt_iou.cpu().numpy()]}")
                    print(f"   🔮 预测一致性: {[f'{x:.3f}' for x in top_consistency_pred.cpu().numpy()]}")
                    print(f"   📈 正确类分数: {[f'{x:.3f}' for x in top_cls_score.cpu().numpy()]}")

                    # 🚀 两种关键乘积分析
                    print(f"   🔄 分类×预测一致性: {[f'{x:.3f}' for x in cls_consistency_product.cpu().numpy()]}")
                    print(f"   🔄 分类×真实IoU:    {[f'{x:.3f}' for x in cls_gt_iou_product.cpu().numpy()]}")

                    # 预测误差分析
                    print(f"   📏 预测误差:   {[f'{x:.3f}' for x in prediction_errors.cpu().numpy()]}")
                    print(f"   📊 误差统计:   平均{avg_error:.3f}, 最大{max_error:.3f}, 相对{avg_relative_error:.1f}%")

                    # 🚀 两种乘积排序比较
                    print(f"   🏆 分类×预测一致性排序: {cls_consistency_ranking.cpu().numpy().tolist()}")
                    print(f"   🎯 分类×真实IoU排序:    {cls_gt_iou_ranking.cpu().numpy().tolist()}")
                    print(f"   ✅ 排序一致性: {ranking_accuracy:.1%} ({ranking_matches}/{num_samples})")
                    print(f"      💡 表示模型预测的质量排序与真实质量排序的匹配程度")

                print(f"{'=' * 80}")

        except Exception as e:
            # 日志记录失败不应该影响训练
            print(f"❌ GT前5正样本监控日志失败: {e}")
            pass




    