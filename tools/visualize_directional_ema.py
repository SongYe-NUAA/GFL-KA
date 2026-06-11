import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# 在这里直接配置默认日志路径，便于脚本直接运行
DEFAULT_LOG_PATH = Path('log/DIR.log')

LOG_PATTERN = re.compile(
    r'\[DirectionalEMA\]\[(?P<mode>[^\]]+)\]\s+epoch=(?P<epoch>-?\d+),\s*'
    r'iter=(?P<iter>-?\d+),\s*stride=(?P<stride>\d+)\s*->\s*(?P<ema>.+)')

DIRECTIONS = ('Left', 'Top', 'Right', 'Bottom')


def parse_direction_values(raw_text: str) -> Dict[str, float]:
    values = {}
    for item in raw_text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            continue
        key, val = item.split(':', 1)
        key = key.strip()
        val = val.strip()
        try:
            values[key] = float(val)
        except ValueError:
            continue
    return values


def parse_log_file(log_path: Path,
                   target_modes: Tuple[str, ...],
                   target_strides: Tuple[int, ...]) -> Dict[int, List[Dict]]:
    stride_records: Dict[int, List[Dict]] = defaultdict(list)

    with log_path.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if '[DirectionalEMA]' not in line:
                continue
            match = LOG_PATTERN.search(line)
            if not match:
                continue

            mode = match.group('mode').strip()
            if target_modes and mode not in target_modes:
                continue

            stride = int(match.group('stride'))
            if target_strides and stride not in target_strides:
                continue

            ema_text = match.group('ema')
            quantile_text = None
            if '| current_quantile=' in ema_text:
                ema_text, quantile_text = ema_text.split('| current_quantile=')

            ema_values = parse_direction_values(ema_text)
            quantile_values = (parse_direction_values(quantile_text)
                               if quantile_text else None)

            epoch = int(match.group('epoch'))
            iteration = int(match.group('iter'))
            global_step = epoch * 1_000_000 + iteration

            stride_records[stride].append({
                'mode': mode,
                'epoch': epoch,
                'iter': iteration,
                'global_step': global_step,
                'ema': ema_values,
                'quantile': quantile_values
            })

    # 排序，确保可视化时曲线有序
    for stride in stride_records:
        stride_records[stride].sort(key=lambda item: item['global_step'])

    return stride_records


def plot_directional_curves(stride_records: Dict[int, List[Dict]],
                            output_path: Path,
                            show_quantile: bool,
                            dpi: int):
    if not stride_records:
        raise RuntimeError('未在日志中找到符合条件的 DirectionalEMA 记录。')

    strides = sorted(stride_records.keys())
    n_strides = len(strides)
    n_cols = 2 if n_strides > 1 else 1
    n_rows = math.ceil(n_strides / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4 * n_rows),
        dpi=dpi,
        squeeze=False)

    for idx, stride in enumerate(strides):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        records = stride_records[stride]

        steps = [rec['iter'] for rec in records]

        for direction in DIRECTIONS:
            values = [rec['ema'].get(direction, float('nan')) for rec in records]
            ax.plot(
                steps,
                values,
                label=f'EMA-{direction}',
                linewidth=1.6)

        if show_quantile:
            for direction in DIRECTIONS:
                values = [
                    (rec['quantile'].get(direction, float('nan'))
                     if rec['quantile'] else float('nan')) for rec in records
                ]
                ax.plot(
                    steps,
                    values,
                    linestyle='--',
                    linewidth=1.0,
                    label=f'Q-{direction}')

        ax.set_title(f'Stride {stride}')
        ax.set_xlabel('Iteration (iter)')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)

    # 隐藏未使用的子图
    for empty_idx in range(n_strides, n_rows * n_cols):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes[row][col].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f'✅ 已保存图像到: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='可视化 Directional EMA 日志的四方向变化趋势')
    parser.add_argument('--log-path',
                        type=Path,
                        default=DEFAULT_LOG_PATH,
                        help='包含 DirectionalEMA 日志的文件')
    parser.add_argument('--mode',
                        choices=('train', 'val', 'all'),
                        default='all',
                        help='筛选日志模式')
    parser.add_argument('--stride',
                        type=str,
                        default='',
                        help='筛选 stride，如 "8,16,32"')
    parser.add_argument('--show-quantile',
                        action='store_true',
                        help='叠加当前 batch quantile 曲线')
    parser.add_argument('--output',
                        type=Path,
                        default=Path('directional_ema.png'),
                        help='输出图片路径')
    parser.add_argument('--dpi', type=int, default=150, help='图片 DPI')

    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        raise FileNotFoundError(
            f'日志文件不存在: {log_path}，请在 DEFAULT_LOG_PATH 或 --log-path 中配置')

    if args.mode == 'all':
        target_modes: Tuple[str, ...] = ()
    else:
        target_modes = (args.mode,)

    if args.stride.strip():
        target_strides = tuple(
            sorted({int(s.strip()) for s in args.stride.split(',') if s.strip()}))
    else:
        target_strides = ()

    stride_records = parse_log_file(log_path, target_modes, target_strides)
    plot_directional_curves(
        stride_records=stride_records,
        output_path=args.output,
        show_quantile=args.show_quantile,
        dpi=args.dpi)


if __name__ == '__main__':
    main()

