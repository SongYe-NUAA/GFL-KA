#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专家竞争机制可视化工具
对比改进前后的专家权重分布和融合效果
"""

import os
# 解决 OpenMP 重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def simulate_expert_weights_comparison(concentration_values, ema_references):
    """
    对比改进前后的专家权重计算
    
    Args:
        concentration_values: concentration值范围 (numpy array)
        ema_references: 三个专家的EMA阈值 [P50, P75, P90]
    
    Returns:
        tuple: (原始高斯权重, 改进后Softmax权重)
    """
    concentration = torch.tensor(concentration_values, dtype=torch.float32)
    emas = torch.tensor(ema_references, dtype=torch.float32)
    
    # 计算concentration的标准差（模拟）
    concentration_std = concentration.std().item() + 1e-8
    
    # === 改进前：高斯核权重 ===
    gaussian_weights = []
    for ema in emas:
        distance = torch.abs(concentration - ema) / concentration_std
        weight = torch.exp(-distance.pow(2))  # 高斯权重
        gaussian_weights.append(weight)
    
    gaussian_weights = torch.stack(gaussian_weights)  # [3, N]
    gaussian_weights_norm = gaussian_weights / (gaussian_weights.sum(dim=0, keepdim=True) + 1e-8)
    
    # === 改进后：Softmax竞争 ===
    softmax_scores = []
    for ema in emas:
        distance = torch.abs(concentration - ema) / concentration_std
        score = -distance  # 负距离作为分数
        softmax_scores.append(score)
    
    softmax_scores = torch.stack(softmax_scores)  # [3, N]
    softmax_temperature = 0.5  # 竞争强度
    softmax_weights_norm = F.softmax(softmax_scores / softmax_temperature, dim=0)
    
    return gaussian_weights_norm.numpy(), softmax_weights_norm.numpy()


def simulate_expert_outputs(concentration_values, ema_references, temperatures):
    """
    模拟专家输出（差异化激活）
    
    Args:
        concentration_values: concentration值范围
        ema_references: 三个专家的EMA阈值
        temperatures: 三个专家的激活温度
    
    Returns:
        expert_outputs: 每个专家的输出 [3, N]
    """
    concentration = torch.tensor(concentration_values, dtype=torch.float32)
    emas = torch.tensor(ema_references, dtype=torch.float32)
    concentration_std = concentration.std().item() + 1e-8
    
    expert_outputs = []
    for ema, temp in zip(emas, temperatures):
        relative_deviation = (concentration - ema) / concentration_std
        output = torch.tanh(relative_deviation * temp)
        expert_outputs.append(output)
    
    return torch.stack(expert_outputs).numpy()  # [3, N]


def plot_comparison():
    """生成对比图表"""
    # === 设置参数 ===
    concentration_values = np.linspace(2.0, 8.0, 200)  # Concentration范围
    ema_references = [3.5, 5.0, 6.5]  # P50, P75, P90的EMA阈值
    
    # 改进前：统一温度
    temperatures_old = [2.0, 2.0, 2.0]
    
    # 改进后：差异化温度
    temperatures_new = [1.5, 3.0, 0.8]  # P50, P75, P90
    
    # === 计算权重 ===
    gaussian_weights, softmax_weights = simulate_expert_weights_comparison(
        concentration_values, ema_references
    )
    
    # === 计算专家输出 ===
    expert_outputs_old = simulate_expert_outputs(
        concentration_values, ema_references, temperatures_old
    )
    
    expert_outputs_new = simulate_expert_outputs(
        concentration_values, ema_references, temperatures_new
    )
    
    # === 计算融合结果 ===
    # 改进前：高斯权重 + 统一温度
    fused_output_old = (expert_outputs_old * gaussian_weights).sum(axis=0)
    
    # 改进后：Softmax权重 + 差异化温度
    fused_output_new = (expert_outputs_new * softmax_weights).sum(axis=0)
    
    # === 绘图 ===
    fig = plt.figure(figsize=(20, 12))
    
    # === 图1：专家权重对比（改进前：高斯核） ===
    ax1 = plt.subplot(3, 3, 1)
    for i, (label, ema) in enumerate(zip(['专家1(P50)', '专家2(P75)', '专家3(P90)'], ema_references)):
        ax1.plot(concentration_values, gaussian_weights[i], label=label, linewidth=2)
    ax1.axvline(ema_references[0], color='C0', linestyle='--', alpha=0.5, label='P50 EMA')
    ax1.axvline(ema_references[1], color='C1', linestyle='--', alpha=0.5, label='P75 EMA')
    ax1.axvline(ema_references[2], color='C2', linestyle='--', alpha=0.5, label='P90 EMA')
    ax1.set_xlabel('Concentration')
    ax1.set_ylabel('专家权重')
    ax1.set_title('改进前：高斯核权重（过于平滑）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 图2：专家权重对比（改进后：Softmax竞争） ===
    ax2 = plt.subplot(3, 3, 2)
    for i, (label, ema) in enumerate(zip(['专家1(P50)', '专家2(P75)', '专家3(P90)'], ema_references)):
        ax2.plot(concentration_values, softmax_weights[i], label=label, linewidth=2)
    ax2.axvline(ema_references[0], color='C0', linestyle='--', alpha=0.5)
    ax2.axvline(ema_references[1], color='C1', linestyle='--', alpha=0.5)
    ax2.axvline(ema_references[2], color='C2', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Concentration')
    ax2.set_ylabel('专家权重')
    ax2.set_title('改进后：Softmax竞争（明确选择）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === 图3：权重熵对比 ===
    ax3 = plt.subplot(3, 3, 3)
    # 计算每个样本的权重熵（熵越小越明确）
    gaussian_entropy = -(gaussian_weights * np.log(gaussian_weights + 1e-8)).sum(axis=0)
    softmax_entropy = -(softmax_weights * np.log(softmax_weights + 1e-8)).sum(axis=0)
    
    ax3.plot(concentration_values, gaussian_entropy, label='高斯核（改进前）', linewidth=2)
    ax3.plot(concentration_values, softmax_entropy, label='Softmax（改进后）', linewidth=2)
    ax3.set_xlabel('Concentration')
    ax3.set_ylabel('权重熵')
    ax3.set_title('专家选择确定性（熵越低越明确）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === 图4：专家输出对比（改进前：统一温度） ===
    ax4 = plt.subplot(3, 3, 4)
    for i, (label, temp) in enumerate(zip(['专家1(T=2.0)', '专家2(T=2.0)', '专家3(T=2.0)'], temperatures_old)):
        ax4.plot(concentration_values, expert_outputs_old[i], label=label, linewidth=2)
    ax4.set_xlabel('Concentration')
    ax4.set_ylabel('专家输出 (conf_raw)')
    ax4.set_title('改进前：统一温度（输出相似）')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # === 图5：专家输出对比（改进后：差异化温度） ===
    ax5 = plt.subplot(3, 3, 5)
    for i, (label, temp) in enumerate(zip(['专家1(T=1.5)', '专家2(T=3.0)', '专家3(T=0.8)'], temperatures_new)):
        ax5.plot(concentration_values, expert_outputs_new[i], label=label, linewidth=2)
    ax5.set_xlabel('Concentration')
    ax5.set_ylabel('专家输出 (conf_raw)')
    ax5.set_title('改进后：差异化温度（输出差异大）')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # === 图6：专家输出差异性 ===
    ax6 = plt.subplot(3, 3, 6)
    # 计算专家输出的标准差（差异性指标）
    expert_std_old = expert_outputs_old.std(axis=0)
    expert_std_new = expert_outputs_new.std(axis=0)
    
    ax6.plot(concentration_values, expert_std_old, label='改进前（统一温度）', linewidth=2)
    ax6.plot(concentration_values, expert_std_new, label='改进后（差异化温度）', linewidth=2)
    ax6.set_xlabel('Concentration')
    ax6.set_ylabel('专家输出标准差')
    ax6.set_title('专家多样性（标准差越大越多样）')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # === 图7：融合结果对比 ===
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(concentration_values, fused_output_old, label='改进前', linewidth=2)
    ax7.plot(concentration_values, fused_output_new, label='改进后', linewidth=2)
    ax7.set_xlabel('Concentration')
    ax7.set_ylabel('融合输出 (conf_raw)')
    ax7.set_title('最终融合结果对比')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # === 图8：融合结果差异 ===
    ax8 = plt.subplot(3, 3, 8)
    fusion_diff = fused_output_new - fused_output_old
    ax8.plot(concentration_values, fusion_diff, linewidth=2, color='red')
    ax8.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Concentration')
    ax8.set_ylabel('融合差异（改进后 - 改进前）')
    ax8.set_title('改进带来的变化')
    ax8.grid(True, alpha=0.3)
    
    # === 图9：特定样本的专家贡献分析 ===
    ax9 = plt.subplot(3, 3, 9)
    # 选择3个代表性样本
    sample_indices = [50, 100, 150]  # 低/中/高 concentration
    x_pos = np.arange(3)
    width = 0.35
    
    for idx, sample_idx in enumerate(sample_indices):
        conc_val = concentration_values[sample_idx]
        
        # 改进前的专家权重
        old_weights = gaussian_weights[:, sample_idx]
        # 改进后的专家权重
        new_weights = softmax_weights[:, sample_idx]
        
        if idx == 0:
            ax9.bar(x_pos - width/2, old_weights, width, 
                   label=f'改进前 (Conc={conc_val:.1f})', alpha=0.7)
            ax9.bar(x_pos + width/2, new_weights, width, 
                   label=f'改进后 (Conc={conc_val:.1f})', alpha=0.7)
    
    ax9.set_ylabel('专家权重')
    ax9.set_title('典型样本的专家贡献（低Conc样本）')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(['专家1(P50)', '专家2(P75)', '专家3(P90)'])
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expert_competition_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 对比图已保存: expert_competition_comparison.png")
    
    # === 生成量化分析报告 ===
    print("\n" + "="*80)
    print("📊 量化分析报告")
    print("="*80)
    
    # 1. 权重确定性
    avg_gaussian_entropy = gaussian_entropy.mean()
    avg_softmax_entropy = softmax_entropy.mean()
    entropy_reduction = (avg_gaussian_entropy - avg_softmax_entropy) / avg_gaussian_entropy * 100
    
    print(f"\n1️⃣ 专家选择确定性:")
    print(f"   改进前平均熵: {avg_gaussian_entropy:.4f}")
    print(f"   改进后平均熵: {avg_softmax_entropy:.4f}")
    print(f"   熵降低率: {entropy_reduction:.1f}% ({'✅ 选择更明确' if entropy_reduction > 0 else '❌ 选择更模糊'})")
    
    # 2. 专家多样性
    avg_std_old = expert_std_old.mean()
    avg_std_new = expert_std_new.mean()
    diversity_increase = (avg_std_new - avg_std_old) / avg_std_old * 100
    
    print(f"\n2️⃣ 专家输出多样性:")
    print(f"   改进前平均标准差: {avg_std_old:.4f}")
    print(f"   改进后平均标准差: {avg_std_new:.4f}")
    print(f"   多样性提升: {diversity_increase:.1f}% ({'✅ 更多样' if diversity_increase > 0 else '❌ 更单一'})")
    
    # 3. 融合结果差异
    max_fusion_diff = np.abs(fusion_diff).max()
    avg_fusion_diff = np.abs(fusion_diff).mean()
    
    print(f"\n3️⃣ 融合结果变化:")
    print(f"   最大差异: {max_fusion_diff:.4f}")
    print(f"   平均差异: {avg_fusion_diff:.4f}")
    
    # 4. 专家利用率
    print(f"\n4️⃣ 专家平均利用率:")
    for i in range(3):
        gaussian_util = gaussian_weights[i].mean()
        softmax_util = softmax_weights[i].mean()
        print(f"   专家{i+1}: 改进前={gaussian_util:.3f}, 改进后={softmax_util:.3f}, "
              f"变化={softmax_util-gaussian_util:+.3f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    print("🚀 门控专家混合改进效果可视化")
    print("="*80)
    plot_comparison()
    print("\n✅ 可视化完成！请查看生成的图表文件。")
