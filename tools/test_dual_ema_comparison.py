#!/usr/bin/env python
"""
🚀 双EMA系统 vs 单一EMA对比测试脚本

功能：
1. 模拟训练过程中的concentration分布变化
2. 对比单一EMA和双EMA系统的响应速度和稳定性
3. 可视化EMA跟踪曲线

使用方法：
    python tools/test_dual_ema_comparison.py

预期结果：
- 快速EMA：快速响应，但可能过度敏感
- 慢速EMA：稳定可靠，但响应慢
- 双EMA系统：自适应结合两者优势
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class EMATracker:
    """EMA跟踪器基类"""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema = None
    
    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * value
        return self.ema


class DualEMATracker:
    """双EMA跟踪器"""
    
    def __init__(self, fast_alpha: float = 0.7, slow_alpha: float = 0.95, 
                 variance_threshold: float = 0.3):
        self.fast_ema = EMATracker(fast_alpha)
        self.slow_ema = EMATracker(slow_alpha)
        self.variance_threshold = variance_threshold
        self.ema = None
        self.variance_history = []
    
    def update(self, value: float, variance: float) -> Tuple[float, str]:
        """
        更新双EMA系统
        
        Returns:
            (ema_value, strategy): EMA值和使用的策略 ("slow" or "mixed")
        """
        fast_val = self.fast_ema.update(value)
        slow_val = self.slow_ema.update(value)
        
        self.variance_history.append(variance)
        
        if variance < self.variance_threshold:
            # 分布稳定：使用慢速EMA
            self.ema = slow_val
            strategy = "slow"
        else:
            # 分布波动：混合快慢EMA
            self.ema = 0.7 * slow_val + 0.3 * fast_val
            strategy = "mixed"
        
        return self.ema, strategy


def simulate_concentration_distribution(num_batches: int = 500) -> Tuple[List[float], List[float]]:
    """
    模拟训练过程中concentration的P75分位数和方差变化
    
    训练阶段：
    - Epoch 1-3 (0-150 batch): 初期震荡，方差大
    - Epoch 4-8 (150-400 batch): 逐渐稳定，方差中等
    - Epoch 9-12 (400-500 batch): 收敛稳定，方差小
    """
    p75_values = []
    variances = []
    
    for i in range(num_batches):
        # 整体趋势：从2.0逐渐增长到2.8
        trend = 2.0 + 0.8 * (i / num_batches)
        
        # 训练初期：大幅波动
        if i < 150:
            noise = np.random.normal(0, 0.2)
            variance = np.random.uniform(0.35, 0.6)
        # 训练中期：中等波动
        elif i < 400:
            noise = np.random.normal(0, 0.1)
            variance = np.random.uniform(0.25, 0.35)
        # 训练后期：小幅波动
        else:
            noise = np.random.normal(0, 0.05)
            variance = np.random.uniform(0.15, 0.25)
        
        # 添加周期性扰动（模拟不同难度的batch）
        periodic = 0.15 * np.sin(i / 20)
        
        p75 = trend + noise + periodic
        p75_values.append(p75)
        variances.append(variance)
    
    return p75_values, variances


def run_comparison():
    """运行对比实验"""
    print("🚀 开始双EMA vs 单一EMA对比测试...\n")
    
    # 生成模拟数据
    print("📊 生成模拟concentration分布...")
    p75_values, variances = simulate_concentration_distribution(num_batches=500)
    
    # 初始化跟踪器
    single_ema_slow = EMATracker(alpha=0.9)  # 传统慢速EMA
    single_ema_fast = EMATracker(alpha=0.7)  # 传统快速EMA
    dual_ema = DualEMATracker(fast_alpha=0.7, slow_alpha=0.95, variance_threshold=0.3)
    
    # 跟踪结果
    single_slow_history = []
    single_fast_history = []
    dual_ema_history = []
    dual_strategy_history = []
    
    # 运行模拟
    print("🔄 运行EMA跟踪...")
    for i, (p75, var) in enumerate(zip(p75_values, variances)):
        single_slow = single_ema_slow.update(p75)
        single_fast = single_ema_fast.update(p75)
        dual_val, strategy = dual_ema.update(p75, var)
        
        single_slow_history.append(single_slow)
        single_fast_history.append(single_fast)
        dual_ema_history.append(dual_val)
        dual_strategy_history.append(1 if strategy == "mixed" else 0)
    
    # 计算评估指标
    print("\n📈 评估指标：")
    
    # 1. 跟踪误差 (MAE)
    mae_slow = np.mean(np.abs(np.array(single_slow_history) - np.array(p75_values)))
    mae_fast = np.mean(np.abs(np.array(single_fast_history) - np.array(p75_values)))
    mae_dual = np.mean(np.abs(np.array(dual_ema_history) - np.array(p75_values)))
    
    print(f"   跟踪误差 (MAE):")
    print(f"      单一EMA (α=0.9):  {mae_slow:.4f}")
    print(f"      单一EMA (α=0.7):  {mae_fast:.4f}")
    print(f"      双EMA系统:         {mae_dual:.4f} {'✅最优' if mae_dual < min(mae_slow, mae_fast) else ''}")
    
    # 2. 稳定性 (EMA值的标准差)
    std_slow = np.std(np.diff(single_slow_history))
    std_fast = np.std(np.diff(single_fast_history))
    std_dual = np.std(np.diff(dual_ema_history))
    
    print(f"\n   稳定性 (变化标准差，越小越稳定):")
    print(f"      单一EMA (α=0.9):  {std_slow:.4f}")
    print(f"      单一EMA (α=0.7):  {std_fast:.4f}")
    print(f"      双EMA系统:         {std_dual:.4f} {'✅最优' if std_dual < min(std_slow, std_fast) else ''}")
    
    # 3. 响应速度 (前150个batch的跟踪误差)
    mae_slow_early = np.mean(np.abs(np.array(single_slow_history[:150]) - np.array(p75_values[:150])))
    mae_fast_early = np.mean(np.abs(np.array(single_fast_history[:150]) - np.array(p75_values[:150])))
    mae_dual_early = np.mean(np.abs(np.array(dual_ema_history[:150]) - np.array(p75_values[:150])))
    
    print(f"\n   初期响应速度 (前150 batch的MAE，越小越好):")
    print(f"      单一EMA (α=0.9):  {mae_slow_early:.4f}")
    print(f"      单一EMA (α=0.7):  {mae_fast_early:.4f}")
    print(f"      双EMA系统:         {mae_dual_early:.4f} {'✅最优' if mae_dual_early < min(mae_slow_early, mae_fast_early) else ''}")
    
    # 4. 后期稳定性 (后100个batch的标准差)
    std_slow_late = np.std(np.diff(single_slow_history[-100:]))
    std_fast_late = np.std(np.diff(single_fast_history[-100:]))
    std_dual_late = np.std(np.diff(dual_ema_history[-100:]))
    
    print(f"\n   后期稳定性 (后100 batch的变化std，越小越好):")
    print(f"      单一EMA (α=0.9):  {std_slow_late:.4f}")
    print(f"      单一EMA (α=0.7):  {std_fast_late:.4f}")
    print(f"      双EMA系统:         {std_dual_late:.4f} {'✅最优' if std_dual_late < min(std_slow_late, std_fast_late) else ''}")
    
    # 5. 双EMA策略统计
    mixed_ratio = np.mean(dual_strategy_history)
    slow_ratio = 1 - mixed_ratio
    
    print(f"\n   双EMA策略使用统计:")
    print(f"      Slow策略 (方差<0.3):  {slow_ratio*100:.1f}%")
    print(f"      Mixed策略 (方差≥0.3): {mixed_ratio*100:.1f}%")
    
    # 可视化
    print("\n📊 生成对比图表...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: EMA跟踪曲线
    ax1 = axes[0]
    ax1.plot(p75_values, label='真实P75', color='black', alpha=0.3, linewidth=1)
    ax1.plot(single_slow_history, label='单一EMA (α=0.9)', color='blue', linewidth=1.5)
    ax1.plot(single_fast_history, label='单一EMA (α=0.7)', color='red', linewidth=1.5)
    ax1.plot(dual_ema_history, label='双EMA系统', color='green', linewidth=2)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Concentration P75')
    ax1.set_title('EMA跟踪对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加阶段分界线
    ax1.axvline(x=150, color='gray', linestyle='--', alpha=0.5, label='训练阶段')
    ax1.axvline(x=400, color='gray', linestyle='--', alpha=0.5)
    ax1.text(75, ax1.get_ylim()[1]*0.95, '初期震荡', ha='center')
    ax1.text(275, ax1.get_ylim()[1]*0.95, '中期过渡', ha='center')
    ax1.text(450, ax1.get_ylim()[1]*0.95, '后期稳定', ha='center')
    
    # 图2: 跟踪误差对比
    ax2 = axes[1]
    errors_slow = np.abs(np.array(single_slow_history) - np.array(p75_values))
    errors_fast = np.abs(np.array(single_fast_history) - np.array(p75_values))
    errors_dual = np.abs(np.array(dual_ema_history) - np.array(p75_values))
    
    # 使用滑动窗口平滑误差曲线
    window_size = 20
    errors_slow_smooth = np.convolve(errors_slow, np.ones(window_size)/window_size, mode='valid')
    errors_fast_smooth = np.convolve(errors_fast, np.ones(window_size)/window_size, mode='valid')
    errors_dual_smooth = np.convolve(errors_dual, np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(errors_slow_smooth, label='单一EMA (α=0.9)', color='blue', linewidth=1.5)
    ax2.plot(errors_fast_smooth, label='单一EMA (α=0.7)', color='red', linewidth=1.5)
    ax2.plot(errors_dual_smooth, label='双EMA系统', color='green', linewidth=2)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('跟踪误差 (MAE)')
    ax2.set_title('跟踪误差对比 (20-batch滑动平均)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 方差和策略选择
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # 绘制方差
    ax3.plot(variances, label='Concentration方差', color='orange', alpha=0.7, linewidth=1)
    ax3.axhline(y=0.3, color='purple', linestyle='--', alpha=0.5, label='方差阈值 (0.3)')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('方差', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.set_ylim([0, max(variances)*1.1])
    
    # 绘制策略选择
    ax3_twin.fill_between(range(len(dual_strategy_history)), 
                           dual_strategy_history, 
                           color='green', alpha=0.3, label='Mixed策略')
    ax3_twin.set_ylabel('使用Mixed策略 (1=是, 0=否)', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.set_ylim([0, 1.2])
    
    ax3.set_title('双EMA自适应策略选择')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'dual_ema_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图表已保存到: {output_path}")
    
    # 显示图表
    try:
        plt.show()
    except:
        print("⚠️  无法显示图表（可能是无GUI环境），但图表已保存")
    
    print("\n✨ 测试完成！")
    
    # 总结建议
    print("\n💡 结论和建议:")
    print("=" * 60)
    
    if mae_dual < mae_slow and std_dual < std_fast:
        print("✅ 双EMA系统综合表现最优！")
        print("   - 比慢速EMA响应更快")
        print("   - 比快速EMA更稳定")
        print("   - 建议在实际训练中启用")
    elif mae_dual < mae_slow:
        print("⚡ 双EMA系统响应速度优秀！")
        print("   - 可考虑在训练初期使用")
        print("   - 后期可切换到单一慢速EMA")
    else:
        print("🤔 单一EMA在当前设置下表现更好")
        print("   - 建议调整双EMA参数:")
        print("     * 降低fast_alpha (当前0.7 → 0.65)")
        print("     * 降低variance_threshold (当前0.3 → 0.25)")
    
    print("=" * 60)


if __name__ == '__main__':
    # 设置随机种子保证可复现
    np.random.seed(42)
    
    # 运行对比
    run_comparison()

"""
🚀 双EMA系统 vs 单一EMA对比测试脚本

功能：
1. 模拟训练过程中的concentration分布变化
2. 对比单一EMA和双EMA系统的响应速度和稳定性
3. 可视化EMA跟踪曲线

使用方法：
    python tools/test_dual_ema_comparison.py

预期结果：
- 快速EMA：快速响应，但可能过度敏感
- 慢速EMA：稳定可靠，但响应慢
- 双EMA系统：自适应结合两者优势
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class EMATracker:
    """EMA跟踪器基类"""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema = None
    
    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * value
        return self.ema


class DualEMATracker:
    """双EMA跟踪器"""
    
    def __init__(self, fast_alpha: float = 0.7, slow_alpha: float = 0.95, 
                 variance_threshold: float = 0.3):
        self.fast_ema = EMATracker(fast_alpha)
        self.slow_ema = EMATracker(slow_alpha)
        self.variance_threshold = variance_threshold
        self.ema = None
        self.variance_history = []
    
    def update(self, value: float, variance: float) -> Tuple[float, str]:
        """
        更新双EMA系统
        
        Returns:
            (ema_value, strategy): EMA值和使用的策略 ("slow" or "mixed")
        """
        fast_val = self.fast_ema.update(value)
        slow_val = self.slow_ema.update(value)
        
        self.variance_history.append(variance)
        
        if variance < self.variance_threshold:
            # 分布稳定：使用慢速EMA
            self.ema = slow_val
            strategy = "slow"
        else:
            # 分布波动：混合快慢EMA
            self.ema = 0.7 * slow_val + 0.3 * fast_val
            strategy = "mixed"
        
        return self.ema, strategy


def simulate_concentration_distribution(num_batches: int = 500) -> Tuple[List[float], List[float]]:
    """
    模拟训练过程中concentration的P75分位数和方差变化
    
    训练阶段：
    - Epoch 1-3 (0-150 batch): 初期震荡，方差大
    - Epoch 4-8 (150-400 batch): 逐渐稳定，方差中等
    - Epoch 9-12 (400-500 batch): 收敛稳定，方差小
    """
    p75_values = []
    variances = []
    
    for i in range(num_batches):
        # 整体趋势：从2.0逐渐增长到2.8
        trend = 2.0 + 0.8 * (i / num_batches)
        
        # 训练初期：大幅波动
        if i < 150:
            noise = np.random.normal(0, 0.2)
            variance = np.random.uniform(0.35, 0.6)
        # 训练中期：中等波动
        elif i < 400:
            noise = np.random.normal(0, 0.1)
            variance = np.random.uniform(0.25, 0.35)
        # 训练后期：小幅波动
        else:
            noise = np.random.normal(0, 0.05)
            variance = np.random.uniform(0.15, 0.25)
        
        # 添加周期性扰动（模拟不同难度的batch）
        periodic = 0.15 * np.sin(i / 20)
        
        p75 = trend + noise + periodic
        p75_values.append(p75)
        variances.append(variance)
    
    return p75_values, variances


def run_comparison():
    """运行对比实验"""
    print("🚀 开始双EMA vs 单一EMA对比测试...\n")
    
    # 生成模拟数据
    print("📊 生成模拟concentration分布...")
    p75_values, variances = simulate_concentration_distribution(num_batches=500)
    
    # 初始化跟踪器
    single_ema_slow = EMATracker(alpha=0.9)  # 传统慢速EMA
    single_ema_fast = EMATracker(alpha=0.7)  # 传统快速EMA
    dual_ema = DualEMATracker(fast_alpha=0.7, slow_alpha=0.95, variance_threshold=0.3)
    
    # 跟踪结果
    single_slow_history = []
    single_fast_history = []
    dual_ema_history = []
    dual_strategy_history = []
    
    # 运行模拟
    print("🔄 运行EMA跟踪...")
    for i, (p75, var) in enumerate(zip(p75_values, variances)):
        single_slow = single_ema_slow.update(p75)
        single_fast = single_ema_fast.update(p75)
        dual_val, strategy = dual_ema.update(p75, var)
        
        single_slow_history.append(single_slow)
        single_fast_history.append(single_fast)
        dual_ema_history.append(dual_val)
        dual_strategy_history.append(1 if strategy == "mixed" else 0)
    
    # 计算评估指标
    print("\n📈 评估指标：")
    
    # 1. 跟踪误差 (MAE)
    mae_slow = np.mean(np.abs(np.array(single_slow_history) - np.array(p75_values)))
    mae_fast = np.mean(np.abs(np.array(single_fast_history) - np.array(p75_values)))
    mae_dual = np.mean(np.abs(np.array(dual_ema_history) - np.array(p75_values)))
    
    print(f"   跟踪误差 (MAE):")
    print(f"      单一EMA (α=0.9):  {mae_slow:.4f}")
    print(f"      单一EMA (α=0.7):  {mae_fast:.4f}")
    print(f"      双EMA系统:         {mae_dual:.4f} {'✅最优' if mae_dual < min(mae_slow, mae_fast) else ''}")
    
    # 2. 稳定性 (EMA值的标准差)
    std_slow = np.std(np.diff(single_slow_history))
    std_fast = np.std(np.diff(single_fast_history))
    std_dual = np.std(np.diff(dual_ema_history))
    
    print(f"\n   稳定性 (变化标准差，越小越稳定):")
    print(f"      单一EMA (α=0.9):  {std_slow:.4f}")
    print(f"      单一EMA (α=0.7):  {std_fast:.4f}")
    print(f"      双EMA系统:         {std_dual:.4f} {'✅最优' if std_dual < min(std_slow, std_fast) else ''}")
    
    # 3. 响应速度 (前150个batch的跟踪误差)
    mae_slow_early = np.mean(np.abs(np.array(single_slow_history[:150]) - np.array(p75_values[:150])))
    mae_fast_early = np.mean(np.abs(np.array(single_fast_history[:150]) - np.array(p75_values[:150])))
    mae_dual_early = np.mean(np.abs(np.array(dual_ema_history[:150]) - np.array(p75_values[:150])))
    
    print(f"\n   初期响应速度 (前150 batch的MAE，越小越好):")
    print(f"      单一EMA (α=0.9):  {mae_slow_early:.4f}")
    print(f"      单一EMA (α=0.7):  {mae_fast_early:.4f}")
    print(f"      双EMA系统:         {mae_dual_early:.4f} {'✅最优' if mae_dual_early < min(mae_slow_early, mae_fast_early) else ''}")
    
    # 4. 后期稳定性 (后100个batch的标准差)
    std_slow_late = np.std(np.diff(single_slow_history[-100:]))
    std_fast_late = np.std(np.diff(single_fast_history[-100:]))
    std_dual_late = np.std(np.diff(dual_ema_history[-100:]))
    
    print(f"\n   后期稳定性 (后100 batch的变化std，越小越好):")
    print(f"      单一EMA (α=0.9):  {std_slow_late:.4f}")
    print(f"      单一EMA (α=0.7):  {std_fast_late:.4f}")
    print(f"      双EMA系统:         {std_dual_late:.4f} {'✅最优' if std_dual_late < min(std_slow_late, std_fast_late) else ''}")
    
    # 5. 双EMA策略统计
    mixed_ratio = np.mean(dual_strategy_history)
    slow_ratio = 1 - mixed_ratio
    
    print(f"\n   双EMA策略使用统计:")
    print(f"      Slow策略 (方差<0.3):  {slow_ratio*100:.1f}%")
    print(f"      Mixed策略 (方差≥0.3): {mixed_ratio*100:.1f}%")
    
    # 可视化
    print("\n📊 生成对比图表...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: EMA跟踪曲线
    ax1 = axes[0]
    ax1.plot(p75_values, label='真实P75', color='black', alpha=0.3, linewidth=1)
    ax1.plot(single_slow_history, label='单一EMA (α=0.9)', color='blue', linewidth=1.5)
    ax1.plot(single_fast_history, label='单一EMA (α=0.7)', color='red', linewidth=1.5)
    ax1.plot(dual_ema_history, label='双EMA系统', color='green', linewidth=2)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Concentration P75')
    ax1.set_title('EMA跟踪对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加阶段分界线
    ax1.axvline(x=150, color='gray', linestyle='--', alpha=0.5, label='训练阶段')
    ax1.axvline(x=400, color='gray', linestyle='--', alpha=0.5)
    ax1.text(75, ax1.get_ylim()[1]*0.95, '初期震荡', ha='center')
    ax1.text(275, ax1.get_ylim()[1]*0.95, '中期过渡', ha='center')
    ax1.text(450, ax1.get_ylim()[1]*0.95, '后期稳定', ha='center')
    
    # 图2: 跟踪误差对比
    ax2 = axes[1]
    errors_slow = np.abs(np.array(single_slow_history) - np.array(p75_values))
    errors_fast = np.abs(np.array(single_fast_history) - np.array(p75_values))
    errors_dual = np.abs(np.array(dual_ema_history) - np.array(p75_values))
    
    # 使用滑动窗口平滑误差曲线
    window_size = 20
    errors_slow_smooth = np.convolve(errors_slow, np.ones(window_size)/window_size, mode='valid')
    errors_fast_smooth = np.convolve(errors_fast, np.ones(window_size)/window_size, mode='valid')
    errors_dual_smooth = np.convolve(errors_dual, np.ones(window_size)/window_size, mode='valid')
    
    ax2.plot(errors_slow_smooth, label='单一EMA (α=0.9)', color='blue', linewidth=1.5)
    ax2.plot(errors_fast_smooth, label='单一EMA (α=0.7)', color='red', linewidth=1.5)
    ax2.plot(errors_dual_smooth, label='双EMA系统', color='green', linewidth=2)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('跟踪误差 (MAE)')
    ax2.set_title('跟踪误差对比 (20-batch滑动平均)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 方差和策略选择
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # 绘制方差
    ax3.plot(variances, label='Concentration方差', color='orange', alpha=0.7, linewidth=1)
    ax3.axhline(y=0.3, color='purple', linestyle='--', alpha=0.5, label='方差阈值 (0.3)')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('方差', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.set_ylim([0, max(variances)*1.1])
    
    # 绘制策略选择
    ax3_twin.fill_between(range(len(dual_strategy_history)), 
                           dual_strategy_history, 
                           color='green', alpha=0.3, label='Mixed策略')
    ax3_twin.set_ylabel('使用Mixed策略 (1=是, 0=否)', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.set_ylim([0, 1.2])
    
    ax3.set_title('双EMA自适应策略选择')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'dual_ema_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图表已保存到: {output_path}")
    
    # 显示图表
    try:
        plt.show()
    except:
        print("⚠️  无法显示图表（可能是无GUI环境），但图表已保存")
    
    print("\n✨ 测试完成！")
    
    # 总结建议
    print("\n💡 结论和建议:")
    print("=" * 60)
    
    if mae_dual < mae_slow and std_dual < std_fast:
        print("✅ 双EMA系统综合表现最优！")
        print("   - 比慢速EMA响应更快")
        print("   - 比快速EMA更稳定")
        print("   - 建议在实际训练中启用")
    elif mae_dual < mae_slow:
        print("⚡ 双EMA系统响应速度优秀！")
        print("   - 可考虑在训练初期使用")
        print("   - 后期可切换到单一慢速EMA")
    else:
        print("🤔 单一EMA在当前设置下表现更好")
        print("   - 建议调整双EMA参数:")
        print("     * 降低fast_alpha (当前0.7 → 0.65)")
        print("     * 降低variance_threshold (当前0.3 → 0.25)")
    
    print("=" * 60)


if __name__ == '__main__':
    # 设置随机种子保证可复现
    np.random.seed(42)
    
    # 运行对比
    run_comparison()

 