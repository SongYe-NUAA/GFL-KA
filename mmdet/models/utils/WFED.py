import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def frequency_highpass_filter(x):
    """高通滤波器，带有数值稳定性保护"""
    B, C, H, W = x.shape
    
    # 使用更稳定的FFT实现
    try:
        freq = torch.fft.fft2(x, norm='ortho')
        freq_shift = torch.fft.fftshift(freq)

        u = torch.arange(H).to(x.device) - H // 2
        v = torch.arange(W).to(x.device) - W // 2
        U, V = torch.meshgrid(u, v, indexing='ij')
        D = torch.sqrt(U ** 2 + V ** 2)

        D0 = max(H, W) // 20
        highpass_filter = 1 - torch.exp(- (D ** 2) / (2 * D0 ** 2))
        highpass_filter = highpass_filter.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

        filtered_freq = freq_shift * highpass_filter
        filtered_freq = torch.fft.ifftshift(filtered_freq)
        x_filtered = torch.fft.ifft2(filtered_freq, norm='ortho').real
        
        # 确保数值稳定性
        x_filtered = torch.clamp(x_filtered, min=-1.0, max=1.0)
        
        # 归一化到原始输入的范围
        x_min = x.min()
        x_max = x.max()
        x_filtered = (x_filtered - x_filtered.min()) / (x_filtered.max() - x_filtered.min() + 1e-10)
        x_filtered = x_filtered * (x_max - x_min) + x_min
        
        return x_filtered
    except Exception:
        # 如果FFT失败，返回原始输入的安全副本
        return x.clone()


def frequency_lowpass_filter(x):
    """低通滤波器，带有数值稳定性保护"""
    B, C, H, W = x.shape
    
    # 使用更稳定的FFT实现
    try:
        freq = torch.fft.fft2(x, norm='ortho')
        freq_shift = torch.fft.fftshift(freq)

        u = torch.arange(H).to(x.device) - H // 2
        v = torch.arange(W).to(x.device) - W // 2
        U, V = torch.meshgrid(u, v, indexing='ij')
        D = torch.sqrt(U ** 2 + V ** 2)

        D0 = max(H, W) // 20
        lowpass_filter = torch.exp(- (D ** 2) / (2 * D0 ** 2))
        lowpass_filter = lowpass_filter.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

        filtered_freq = freq_shift * lowpass_filter
        filtered_freq = torch.fft.ifftshift(filtered_freq)
        x_filtered = torch.fft.ifft2(filtered_freq, norm='ortho').real
        
        # 确保数值稳定性
        x_filtered = torch.clamp(x_filtered, min=-1.0, max=1.0)
        
        # 归一化到原始输入的范围
        x_min = x.min()
        x_max = x.max()
        x_filtered = (x_filtered - x_filtered.min()) / (x_filtered.max() - x_filtered.min() + 1e-10)
        x_filtered = x_filtered * (x_max - x_min) + x_min
        
        return x_filtered
    except Exception:
        # 如果FFT失败，返回原始输入的安全副本
        return x.clone()


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 确保reduction不会导致通道数为0
        reduction_channels = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # 确保输入通道数与初始化时一致
        if c != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {c} channels")
        
        # 添加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 如果有NaN或Inf值，用0替换
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        se_weight = self.global_avg_pool(x).view(b, c)
        se_weight = self.fc(se_weight).view(b, c, 1, 1)
        return x * se_weight


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 添加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        attention = self.sigmoid(self.conv(x))  # 计算每个像素的重要性
        return x * attention


class UltraLiteWFED(nn.Module):
    """超轻量级WFED模块，不使用频域滤波，只使用空间域注意力机制"""
    def __init__(self, in_channel=64, out_channel=64):
        super(UltraLiteWFED, self).__init__()
        self.se = SEBlock(in_channel)
        self.pa = PixelAttention(in_channel)
        
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 确保数值稳定性
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 应用注意力机制
        enhanced = self.pa(self.se(x))
        
        # 卷积处理
        output = self.conv_bn_relu(enhanced)
        
        return output


class WFED(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(WFED, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # 使用超轻量级WFED，避免频域操作
        self.ultra_lite_wfed = UltraLiteWFED(in_channel, out_channel)
        
    def forward(self, x):
        # 使用超轻量级WFED处理
        output = self.ultra_lite_wfed(x)
        
        # 添加残差连接
        if self.in_channel == self.out_channel:
            output = output + x
        
        return output
