from __future__ import print_function, division
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from einops.layers.torch import Rearrange
from mmcv.ops import DeformConv2d


class MSDC(nn.Module):
    def __init__(self, dim_in, dim_out, bn_norm = 0.1):
        super(MSDC, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1,bias=True),
            nn.BatchNorm2d(dim_out,momentum=bn_norm),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=2, dilation=2,bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True),
        )

        self.branch_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch_bn = nn.BatchNorm2d(dim_out, momentum=bn_norm)
        self.branch_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1 = self.branch1(x)
        conv2 = self.branch2(x)
        conv3 = self.branch3(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch_conv(global_feature)
        global_feature = self.branch_bn(global_feature)
        global_feature = self.branch_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv1, conv2, conv3, global_feature], dim=1)
        result1 = self.conv_cat(feature_cat)

        return result1


class DCN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DCN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(dim_in, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # self.p_conv.register_backward_hook(self._set_lr)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(dim_in, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # self.m_conv.register_backward_hook(self._set_lr)
            self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batchsize, -1, height, width)

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        # conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        # conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        # conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).to(conv_weight.device)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class IDConvio(nn.Module):
    def __init__(self, indim, outdim):
        super(IDConvio, self).__init__()
        self.conv1_1 = Conv2d_cd(indim, outdim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(indim, outdim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(indim, outdim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(indim, outdim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(indim, outdim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res


class GateSelector(nn.Module):
    def __init__(self, in_channels):
        super(GateSelector, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, out_features=3, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        weights = self.fc(y).view(b, 3, 1, 1)
        return weights


class LiteConvolutionSelector(nn.Module):
    """轻量级的多感受野特征交互卷积模块
    
    相比原始ConvolutionSelector，这个轻量级版本:
    1. 只使用两种卷积类型(标准卷积和空洞卷积)
    2. 简化了门控机制
    3. 减少了中间通道数
    4. 使用了更高效的实现
    
    Args:
        dim_in (int): 输入通道数
        dim_out (int): 输出通道数
        reduction_ratio (int): 通道数缩减比例，用于降低计算量
    """
    def __init__(self, dim_in, dim_out, reduction_ratio=4):
        super(LiteConvolutionSelector, self).__init__()
        
        # 计算中间通道数，用于降低计算量
        mid_channels = max(dim_in // reduction_ratio, 32)
        
        # 标准卷积分支
        self.standard_conv = nn.Sequential(
            nn.Conv2d(dim_in, mid_channels, 1),  # 先降维
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim_out, 3, padding=1),  # 3x3卷积
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积分支
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(dim_in, mid_channels, 1),  # 先降维
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim_out, 3, padding=2, dilation=2),  # 空洞卷积
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
        
        # 简化的注意力门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim_in, 2, 1),  # 直接用1x1卷积替代FC层
            nn.Softmax(dim=1)  # 确保两个权重和为1
        )
    
    def forward(self, x):
        # 计算门控权重
        weights = self.gate(x)
        
        # 分别应用两种卷积
        standard_out = self.standard_conv(x)
        dilated_out = self.dilated_conv(x)
        
        # 加权融合
        out = standard_out * weights[:, 0:1] + dilated_out * weights[:, 1:2]
        
        return out


class LiteMSDC(nn.Module):
    """轻量级的多尺度空洞卷积模块
    
    相比原始MSDC，这个轻量级版本:
    1. 只使用两个空洞卷积分支
    2. 移除了全局特征分支
    3. 使用1x1卷积进行特征融合
    
    Args:
        dim_in (int): 输入通道数
        dim_out (int): 输出通道数
        bn_norm (float): BatchNorm的momentum参数
    """
    def __init__(self, dim_in, dim_out, bn_norm=0.1):
        super(LiteMSDC, self).__init__()
        
        # 第一个分支: 标准3x3卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True)
        )
        
        # 第二个分支: 空洞卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 2, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_norm),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 应用两个分支
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # 拼接并融合
        out = torch.cat([out1, out2], dim=1)
        out = self.fusion(out)
        
        return out


# 更高效的可变形卷积实现，使用mmcv的DCN
class EfficientDCN(nn.Module):
    """高效的可变形卷积实现
    
    使用mmcv的DeformConv2d替代自定义实现，提高效率
    
    Args:
        dim_in (int): 输入通道数
        dim_out (int): 输出通道数
    """
    def __init__(self, dim_in, dim_out):
        super(EfficientDCN, self).__init__()
        # 偏移量预测
        self.offset_conv = nn.Conv2d(dim_in, 18, 3, 1, 1)
        # 可变形卷积
        self.dcn = DeformConv2d(dim_in, dim_out, 3, 1, 1)
        # 添加BN和ReLU
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.dcn(x, offset)
        out = self.bn(out)
        out = self.relu(out)
        return out


# 超轻量级的ConvolutionSelector，进一步减少计算量
class UltraLiteConvolutionSelector(nn.Module):
    """超轻量级的多感受野特征交互卷积模块
    
    这是最轻量级的版本，适用于计算资源极为有限的场景
    
    Args:
        dim_in (int): 输入通道数
        dim_out (int): 输出通道数
    """
    def __init__(self, dim_in, dim_out):
        super(UltraLiteConvolutionSelector, self).__init__()
        
        # 标准卷积
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)
        
        # 轻量级空洞卷积注意力
        self.attn = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=2, dilation=2, groups=dim_in),
            nn.BatchNorm2d(dim_out),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 标准卷积
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        
        # 空洞卷积注意力
        attn = self.attn(x)
        
        # 注意力加权
        out = conv_out * attn
        out = self.relu(out)
        
        return out
