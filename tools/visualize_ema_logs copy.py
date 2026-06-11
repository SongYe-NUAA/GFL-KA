#!/usr/bin/env python3
"""📊 EMA阈值训练日志可视化工具

功能：
1. 从训练日志文件中提取 EMA_THRESHOLD_DATA 记录
2. 可视化EMA阈值、温度、样本选择等指标的变化趋势
3. 生成训练报告

使用方法：
    python visualize_ema_logs.py [log_file_path]
    
    默认读取最新的日志文件：
    - 优先查找 work_dirs/*/[timestamp]/vis_data/[timestamp].log
    - 或者指定日志文件路径
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def find_latest_log_file():
    """自动查找最新的训练日志文件"""
    # 查找 work_dirs 下的所有日志文件
    log_patterns = [
        "work_dirs/*/*/vis_data/*.log",
        "work_dirs/*/*.log",
        "logs/*.log",
        "*.log"
    ]
    
    all_logs = []
    for pattern in log_patterns:
        all_logs.extend(glob.glob(pattern))
    
    if not all_logs:
        return None
    
    # 返回最新的日志文件
    latest_log = max(all_logs, key=os.path.getmtime)
    return latest_log


def parse_ema_log_line(line):
    """解析单行 EMA_THRESHOLD_DATA 日志"""
    # 示例格式：
    # EMA_THRESHOLD_DATA: epoch: 0/12, stage: Foundation, ema_threshold: 0.8523, ...
    
    if "EMA_THRESHOLD_DATA:" not in line:
        return None
    
    try:
        # 提取数据部分
        data_part = line.split("EMA_THRESHOLD_DATA:")[1].strip()
        
        # 解析各个字段
        log_entry = {}
        
        # epoch: 0/12
        epoch_match = re.search(r'epoch:\s*(\d+)/(\d+)', data_part)
        if epoch_match:
            log_entry['epoch'] = int(epoch_match.group(1))
            log_entry['total_epochs'] = int(epoch_match.group(2))
        
        # stage: Foundation
        stage_match = re.search(r'stage:\s*(\w+(?:-\w+)?)', data_part)
        if stage_match:
            log_entry['stage'] = stage_match.group(1)
        
        # ema_threshold: 0.8523
        ema_threshold_match = re.search(r'ema_threshold:\s*([\d.]+)', data_part)
        if ema_threshold_match:
            log_entry['ema_threshold'] = float(ema_threshold_match.group(1))
        
        # batch_threshold: 0.8567
        batch_threshold_match = re.search(r'batch_threshold:\s*([\d.]+)', data_part)
        if batch_threshold_match:
            log_entry['current_batch_threshold'] = float(batch_threshold_match.group(1))
        
        # temperature: 0.0
        temperature_match = re.search(r'temperature:\s*([\d.]+)', data_part)
        if temperature_match:
            log_entry['temperature'] = float(temperature_match.group(1))
        
        # selected: 245/1024 (23.93%)
        selected_match = re.search(r'selected:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)', data_part)
        if selected_match:
            log_entry['selected_samples'] = int(selected_match.group(1))
            log_entry['total_samples'] = int(selected_match.group(2))
            log_entry['selection_ratio'] = float(selected_match.group(3))
        
        # iou_mean: 0.8942
        iou_mean_match = re.search(r'iou_mean:\s*([\d.]+)', data_part)
        if iou_mean_match:
            log_entry['selected_iou_mean'] = float(iou_mean_match.group(1))
        
        # iou_range: [0.8523, 0.9562]
        iou_range_match = re.search(r'iou_range:\s*\[([\d.]+),\s*([\d.]+)\]', data_part)
        if iou_range_match:
            log_entry['selected_iou_min'] = float(iou_range_match.group(1))
            log_entry['selected_iou_max'] = float(iou_range_match.group(2))
        
        # target_ratio: 0.20
        target_ratio_match = re.search(r'target_ratio:\s*([\d.]+)', data_part)
        if target_ratio_match:
            log_entry['top_k_ratio'] = float(target_ratio_match.group(1))
        
        return log_entry if log_entry else None
        
    except Exception as e:
        print(f"⚠️ 解析日志行失败: {e}")
        return None


def load_logs(log_file=None):
    """加载并解析日志文件"""
    # 如果没有指定日志文件，自动查找
    if log_file is None:
        log_file = find_latest_log_file()
        if log_file is None:
            print(f"❌ 未找到训练日志文件")
            print(f"   请指定日志文件路径，或确保训练已生成日志")
            return None
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return None
    
    print(f"📂 读取日志文件: {log_file}")
    
    # 读取并解析日志
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            log_entry = parse_ema_log_line(line)
            if log_entry:
                logs.append(log_entry)
    
    if logs:
        print(f"✅ 成功解析 {len(logs)} 条 EMA 日志记录")
    else:
        print(f"❌ 未找到 EMA_THRESHOLD_DATA 记录")
        print(f"   请确保训练已启用 EMA 自适应阈值功能")
    
    return logs if logs else None


def plot_ema_training(logs, output_dir="ema_threshold_logs"):
    """绘制EMA训练过程可视化图表"""
    
    if not logs:
        print("❌ 没有日志数据可绘制")
        return
    
    # 提取数据
    epochs = [log['epoch'] for log in logs]
    ema_thresholds = [log['ema_threshold'] for log in logs]
    batch_thresholds = [log['current_batch_threshold'] for log in logs]
    temperatures = [log['temperature'] for log in logs]
    selection_ratios = [log['selection_ratio'] for log in logs]
    iou_means = [log['selected_iou_mean'] for log in logs]
    stages = [log['stage'] for log in logs]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EMA Adaptive Threshold Training Visualization', fontsize=16, fontweight='bold')
    
    # === 子图1: EMA阈值变化 ===
    ax1 = axes[0, 0]
    ax1.plot(epochs, ema_thresholds, 'b-o', linewidth=2, markersize=6, label='EMA Threshold')
    ax1.plot(epochs, batch_thresholds, 'g--', alpha=0.5, linewidth=1, label='Current Batch Threshold')
    ax1.axhline(y=0.4, color='r', linestyle=':', alpha=0.3, label='Lower Bound (0.4)')
    ax1.axhline(y=0.95, color='r', linestyle=':', alpha=0.3, label='Upper Bound (0.95)')
    
    # 标注训练阶段
    stage_colors = {'Foundation': 'yellow', 'Warmup': 'orange', 'Full-Power': 'lightgreen'}
    for i, stage in enumerate(stages):
        ax1.axvspan(epochs[i]-0.4, epochs[i]+0.4, alpha=0.2, color=stage_colors.get(stage, 'gray'))
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('IoU Threshold', fontsize=12)
    ax1.set_title('EMA Threshold Learning Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # === 子图2: 温度退火 ===
    ax2 = axes[0, 1]
    ax2.plot(epochs, temperatures, 'r-o', linewidth=2, markersize=6)
    ax2.fill_between(epochs, 0, temperatures, alpha=0.3, color='red')
    
    # 标注阶段
    for i, stage in enumerate(stages):
        ax2.axvspan(epochs[i]-0.4, epochs[i]+0.4, alpha=0.2, color=stage_colors.get(stage, 'gray'))
        if i == 0 or stages[i] != stages[i-1]:
            ax2.text(epochs[i], temperatures[i] + 0.05, stage, 
                    fontsize=9, ha='center', fontweight='bold')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Temperature', fontsize=12)
    ax2.set_title('Temperature Annealing Strategy', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # === 子图3: 样本选择比例 ===
    ax3 = axes[1, 0]
    ax3.plot(epochs, selection_ratios, 'g-o', linewidth=2, markersize=6)
    ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Target Ratio (20%)')
    ax3.fill_between(epochs, 15, 30, alpha=0.1, color='green', label='Ideal Range (15-30%)')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Selection Ratio (%)', fontsize=12)
    ax3.set_title('High-Quality Sample Selection Ratio', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # === 子图4: 选中样本IoU均值 ===
    ax4 = axes[1, 1]
    ax4.plot(epochs, iou_means, 'm-o', linewidth=2, markersize=6)
    ax4.fill_between(epochs, [log['selected_iou_min'] for log in logs],
                     [log['selected_iou_max'] for log in logs],
                     alpha=0.2, color='magenta', label='IoU Range')
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('IoU Value', fontsize=12)
    ax4.set_title('Selected Samples IoU Distribution', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "ema_training_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已保存: {output_path}")
    
    plt.show()


def generate_report(logs, output_dir="ema_threshold_logs"):
    """生成训练报告"""
    
    if not logs:
        print("❌ 没有日志数据可生成报告")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 EMA自适应阈值训练报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基本信息
    report_lines.append(f"🔢 总训练轮数: {logs[-1]['total_epochs']}")
    report_lines.append(f"📝 记录条数: {len(logs)}")
    report_lines.append(f"⏰ 训练时间: {logs[0]['timestamp']} → {logs[-1]['timestamp']}")
    report_lines.append("")
    
    # EMA阈值统计
    ema_thresholds = [log['ema_threshold'] for log in logs]
    report_lines.append("📈 EMA阈值变化：")
    report_lines.append(f"   - 初始值: {ema_thresholds[0]:.4f}")
    report_lines.append(f"   - 最终值: {ema_thresholds[-1]:.4f}")
    report_lines.append(f"   - 最大值: {max(ema_thresholds):.4f}")
    report_lines.append(f"   - 最小值: {min(ema_thresholds):.4f}")
    report_lines.append(f"   - 平均值: {np.mean(ema_thresholds):.4f}")
    report_lines.append(f"   - 标准差: {np.std(ema_thresholds):.4f}")
    report_lines.append("")
    
    # 温度退火统计
    temperatures = [log['temperature'] for log in logs]
    report_lines.append("🌡️ 温度退火：")
    foundation_epochs = sum(1 for log in logs if log['stage'] == 'Foundation')
    warmup_epochs = sum(1 for log in logs if log['stage'] == 'Warmup')
    fullpower_epochs = sum(1 for log in logs if log['stage'] == 'Full-Power')
    report_lines.append(f"   - Foundation阶段: {foundation_epochs} epochs")
    report_lines.append(f"   - Warmup阶段: {warmup_epochs} epochs")
    report_lines.append(f"   - Full-Power阶段: {fullpower_epochs} epochs")
    report_lines.append("")
    
    # 样本选择统计
    selection_ratios = [log['selection_ratio'] for log in logs]
    report_lines.append("🎯 样本选择统计：")
    report_lines.append(f"   - 平均选择比例: {np.mean(selection_ratios):.2f}%")
    report_lines.append(f"   - 目标比例: {logs[0]['top_k_ratio']*100:.0f}%")
    report_lines.append(f"   - 选择比例范围: [{min(selection_ratios):.2f}%, {max(selection_ratios):.2f}%]")
    report_lines.append("")
    
    # 各阶段详细信息
    report_lines.append("📋 各Epoch详细信息：")
    report_lines.append("")
    report_lines.append(f"{'Epoch':<8}{'Stage':<15}{'EMA阈值':<12}{'温度':<10}{'选择比例':<12}{'IoU均值':<10}")
    report_lines.append("-" * 80)
    
    for log in logs:
        report_lines.append(
            f"{log['epoch']:<8}"
            f"{log['stage']:<15}"
            f"{log['ema_threshold']:<12.4f}"
            f"{log['temperature']:<10.3f}"
            f"{log['selection_ratio']:<12.2f}%"
            f"{log['selected_iou_mean']:<10.4f}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ 报告生成完成")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "ema_training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n📄 报告已保存: {report_path}")


def main():
    """主函数"""
    import sys
    
    print("=" * 80)
    print("📊 EMA阈值训练日志分析工具")
    print("=" * 80)
    print()
    
    # 检查命令行参数
    log_file = None
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        print(f"📂 使用指定日志文件: {log_file}")
    else:
        print(f"🔍 自动查找最新日志文件...")
    
    # 加载日志
    logs = load_logs(log_file)
    
    if logs:
        # 生成可视化
        print("\n🎨 生成可视化图表...")
        plot_ema_training(logs)
        
        # 生成报告
        print("\n📝 生成训练报告...")
        generate_report(logs)
        
        print("\n✅ 所有分析完成！")
    else:
        print("\n❌ 无法加载日志数据")
        print("\n💡 使用方法:")
        print("   1. 自动查找: python visualize_ema_logs.py")
        print("   2. 指定文件: python visualize_ema_logs.py <log_file_path>")
        print("\n📝 确保训练日志中包含 EMA_THRESHOLD_DATA 记录")


if __name__ == "__main__":
    main()
