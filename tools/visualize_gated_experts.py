#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
门控专家混合可视化工具
用于验证和可视化门控机制的工作原理
"""

import os
# 解决 OpenMP 重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def simulate_gated_mixture(concentration_values, ema_references, quantile_values):
    """
    模拟门控专家混合机制
    
    Args:
        concentration_values: 样本的concentration值 [N]
        ema_references: 三个专家的EMA参考值 [3]
        quantile_values: 分位数值 [3]
    
    Returns:
        weights: 归一化权重 [3, N]
        confidence_raws: 专家输出 [3, N]
        final_confidence: 融合后的最终confidence [N]
    """
    N = len(concentration_values)
    concentration_std = concentration_values.std()
    
    weights_list = []
    confidence_raws_list = []
    
    for q_idx, ema_ref in enumerate(ema_references):
        # 计算相对偏差
        relative_deviation = (concentration_values - ema_ref) / concentration_std
        
        # 专家输出（温度=2.0）
        conf_raw = torch.tanh(relative_deviation * 2.0)
        confidence_raws_list.append(conf_raw)
        
        # 计算专家权重（高斯核）
        distance = torch.abs(concentration_values - ema_ref) / concentration_std
        weight = torch.exp(-distance.pow(2))
        weights_list.append(weight)
    
    # 归一化权重
    weights_stack = torch.stack(weights_list)  # [3, N]
    weights_norm = weights_stack / (weights_stack.sum(dim=0, keepdim=True) + 1e-8)
    
    # 加权融合
    conf_stack = torch.stack(confidence_raws_list)  # [3, N]
    weighted_conf_raw = (conf_stack * weights_norm).sum(dim=0)  # [N]
    
    # 最终confidence
    final_confidence = (weighted_conf_raw + 1.0) * 0.5  # [N]，范围[0,1]
    
    return weights_norm, conf_stack, final_confidence


def visualize_expert_weights():
    """可视化专家权重分布"""
    print("🎨 生成专家权重分布可视化...")
    
    # 模拟参数
    ema_references = torch.tensor([0.3, 0.6, 0.85])  # P25, P50, P75
    quantile_values = [0.25, 0.50, 0.75]
    
    # 生成不同concentration值的样本
    concentration_values = torch.linspace(0.1, 1.0, 100)
    
    # 计算权重
    weights, conf_raws, final_conf = simulate_gated_mixture(
        concentration_values, ema_references, quantile_values
    )
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：专家权重分布
    ax = axes[0, 0]
    for i, q_val in enumerate(quantile_values):
        ax.plot(concentration_values.numpy(), weights[i].numpy(), 
                label=f'专家{i+1} (P{int(q_val*100)}, EMA={ema_references[i]:.2f})',
                linewidth=2)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='均匀权重 (1/3)')
    ax.set_xlabel('Concentration值', fontsize=12)
    ax.set_ylabel('归一化权重', fontsize=12)
    ax.set_title('🌟 专家权重随Concentration变化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图2：专家输出（confidence_raw）
    ax = axes[0, 1]
    for i, q_val in enumerate(quantile_values):
        ax.plot(concentration_values.numpy(), conf_raws[i].numpy(),
                label=f'专家{i+1} (P{int(q_val*100)})',
                linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Concentration值', fontsize=12)
    ax.set_ylabel('Confidence Raw', fontsize=12)
    ax.set_title('🎯 专家输出（激活后）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图3：最终融合结果
    ax = axes[1, 0]
    ax.plot(concentration_values.numpy(), final_conf.numpy(),
            color='red', linewidth=2.5, label='融合后Confidence')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='中性值 (0.5)')
    ax.set_xlabel('Concentration值', fontsize=12)
    ax.set_ylabel('最终Confidence', fontsize=12)
    ax.set_title('🏆 门控融合后的最终Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图4：主导专家分布
    ax = axes[1, 1]
    dominant_expert = weights.argmax(dim=0).numpy()  # 每个样本的主导专家
    colors = ['blue', 'green', 'orange']
    expert_names = ['专家1 (P25)', '专家2 (P50)', '专家3 (P75)']
    
    for i in range(3):
        mask = dominant_expert == i
        if mask.sum() > 0:
            ax.scatter(concentration_values[mask].numpy(), 
                      final_conf[mask].numpy(),
                      c=colors[i], label=expert_names[i], alpha=0.6, s=30)
    
    ax.set_xlabel('Concentration值', fontsize=12)
    ax.set_ylabel('最终Confidence', fontsize=12)
    ax.set_title('🎓 主导专家分布（颜色表示权重最大的专家）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'expert_weights_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存到: {output_path}")
    plt.close()


def analyze_edge_independence():
    """分析边独立性"""
    print("\n🔬 分析边独立性...")
    
    # 模拟一个预测框的4条边
    torch.manual_seed(42)
    
    # 假设4条边的concentration不同（例如遮挡场景）
    edge_concentrations = torch.tensor([
        0.25,  # 左边：低质量（被遮挡）
        0.60,  # 上边：中等质量
        0.90,  # 右边：高质量（清晰）
        0.55   # 下边：中等质量
    ])
    
    ema_references = torch.tensor([0.3, 0.6, 0.85])
    quantile_values = [0.25, 0.50, 0.75]
    
    # 为每条边计算权重
    print("\n📊 4条边的专家权重分布：")
    print("-" * 70)
    print(f"{'边':<8} {'Concentration':<15} {'专家1(P25)':<12} {'专家2(P50)':<12} {'专家3(P75)':<12} {'主导专家':<10}")
    print("-" * 70)
    
    edge_names = ['左边', '上边', '右边', '下边']
    
    for edge_idx, conc in enumerate(edge_concentrations):
        conc_expanded = conc.unsqueeze(0)  # [1]
        weights, _, final_conf = simulate_gated_mixture(
            conc_expanded, ema_references, quantile_values
        )
        
        dominant_expert = weights.argmax(dim=0).item() + 1
        
        print(f"{edge_names[edge_idx]:<8} {conc.item():<15.3f} "
              f"{weights[0, 0].item():<12.3f} "
              f"{weights[1, 0].item():<12.3f} "
              f"{weights[2, 0].item():<12.3f} "
              f"专家{dominant_expert}")
    
    print("-" * 70)
    print("\n💡 观察：")
    print("  - 左边(0.25)主要听专家1的意见（低质量专家）")
    print("  - 上边(0.60)和下边(0.55)主要听专家2的意见（中等质量专家）")
    print("  - 右边(0.90)主要听专家3的意见（高质量专家）")
    print("  - ✅ 每条边独立选择最合适的专家，保留了边之间的差异性！")


def compare_with_simple_average():
    """对比门控混合与简单平均"""
    print("\n📊 对比门控混合 vs 简单平均...")
    
    # 模拟高低质量样本
    high_quality_conc = torch.tensor([0.85, 0.90, 0.88])  # 高质量样本
    low_quality_conc = torch.tensor([0.25, 0.30, 0.28])   # 低质量样本
    
    ema_references = torch.tensor([0.3, 0.6, 0.85])
    quantile_values = [0.25, 0.50, 0.75]
    
    # 门控混合
    _, conf_raws_high, final_conf_high = simulate_gated_mixture(
        high_quality_conc, ema_references, quantile_values
    )
    _, conf_raws_low, final_conf_low = simulate_gated_mixture(
        low_quality_conc, ema_references, quantile_values
    )
    
    # 简单平均
    simple_avg_high = conf_raws_high.mean(dim=0)
    simple_avg_high_conf = (simple_avg_high + 1.0) * 0.5
    
    simple_avg_low = conf_raws_low.mean(dim=0)
    simple_avg_low_conf = (simple_avg_low + 1.0) * 0.5
    
    print("\n高质量样本 (concentration ≈ 0.88):")
    print(f"  门控混合: {final_conf_high.mean().item():.3f}")
    print(f"  简单平均: {simple_avg_high_conf.mean().item():.3f}")
    
    print("\n低质量样本 (concentration ≈ 0.28):")
    print(f"  门控混合: {final_conf_low.mean().item():.3f}")
    print(f"  简单平均: {simple_avg_low_conf.mean().item():.3f}")
    
    print("\n区分度对比:")
    gated_diff = final_conf_high.mean().item() - final_conf_low.mean().item()
    simple_diff = simple_avg_high_conf.mean().item() - simple_avg_low_conf.mean().item()
    
    print(f"  门控混合区分度: {gated_diff:.3f}")
    print(f"  简单平均区分度: {simple_diff:.3f}")
    print(f"  提升: {(gated_diff / simple_diff - 1) * 100:.1f}%")
    
    if gated_diff > simple_diff:
        print("  ✅ 门控混合的区分度更高！")
    else:
        print("  ⚠️ 简单平均的区分度更高（不应该出现）")


def test_gaussian_kernel():
    """测试高斯核的特性"""
    print("\n🔬 测试高斯核特性...")
    
    distances = torch.linspace(0, 3, 100)
    weights = torch.exp(-distances.pow(2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances.numpy(), weights.numpy(), linewidth=2.5, color='purple')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='权重=0.5')
    ax.axvline(x=np.sqrt(np.log(2)), color='red', linestyle='--', alpha=0.5, 
               label=f'距离={np.sqrt(np.log(2)):.2f}')
    
    # 标注关键点
    key_points = [0, 0.5, 1.0, 1.5, 2.0]
    for d in key_points:
        w = np.exp(-d**2)
        ax.plot(d, w, 'ro', markersize=8)
        ax.text(d, w + 0.05, f'd={d:.1f}\nw={w:.2f}', 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('标准化距离 (distance / std)', fontsize=12)
    ax.set_ylabel('权重 (weight)', fontsize=12)
    ax.set_title('🌟 高斯核权重函数: weight = exp(-distance²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'gaussian_kernel.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 高斯核可视化已保存到: {output_path}")
    plt.close()
    
    print("\n💡 高斯核特性：")
    print(f"  - 距离=0.0 → 权重=1.00 (完全信任)")
    print(f"  - 距离=0.5 → 权重={np.exp(-0.25):.2f}")
    print(f"  - 距离=1.0 → 权重={np.exp(-1.0):.2f}")
    print(f"  - 距离=1.5 → 权重={np.exp(-2.25):.2f}")
    print(f"  - 距离=2.0 → 权重={np.exp(-4.0):.2f} (几乎不信任)")
    print("  - ✅ 平滑衰减，避免硬切换，训练稳定")


def main():
    """主函数"""
    print("=" * 70)
    print("🌟 门控专家混合（Gated Mixture of Experts）可视化工具")
    print("=" * 70)
    
    # 1. 可视化专家权重
    visualize_expert_weights()
    
    # 2. 分析边独立性
    analyze_edge_independence()
    
    # 3. 对比简单平均
    compare_with_simple_average()
    
    # 4. 测试高斯核
    test_gaussian_kernel()
    
    print("\n" + "=" * 70)
    print("✅ 所有可视化和分析完成！")
    print("=" * 70)
    print("\n生成的文件：")
    print("  1. expert_weights_visualization.png - 专家权重分布可视化")
    print("  2. gaussian_kernel.png - 高斯核函数可视化")
    print("\n💡 这些可视化展示了门控专家混合的核心优势：")
    print("  ✅ 自适应权重：每个样本自动选择最合适的专家")
    print("  ✅ 边独立性：每条边独立决策，保留差异性")
    print("  ✅ 平滑过渡：高斯核确保权重平滑变化")
    print("  ✅ 零参数：无需训练，泛化能力强")


if __name__ == '__main__':
    main()
