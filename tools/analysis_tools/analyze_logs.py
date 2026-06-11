# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        if not all_times:
            raise KeyError(
                'Please reduce the log interval in the config so that'
                'interval is less than iterations of one epoch.')
        epoch_ave_time = np.array(list(map(lambda x: np.mean(x), all_times)))
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f} s/iter')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f} s/iter')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(epoch_ave_time):.4f} s/iter\n')


def plot_curve(log_dicts, args):
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20  # 设置全局字体大小
    plt.rcParams['axes.titlesize'] = 20  # 设置标题字体大小
    plt.rcParams['axes.labelsize'] = 24  # 设置轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 24  # 设置x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 24  # 设置y轴刻度标签字体大小
    plt.rcParams['legend.fontsize'] = 24  # 设置图例字体大小

    if args.backend is not None:
        plt.switch_backend(args.backend)
    # sns.set_style(args.style)  # 这行注释掉
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    # TODO: support dynamic eval interval(e.g. RTMDet) when plotting mAP.
    num_metrics = len(metrics)
    plt.figure(figsize=(10, 8))  # 增加图像尺寸
    plt.subplots_adjust(bottom=0.15)  # 增加底部边距
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')

            if not epochs:
                raise ValueError(f'No epochs found in log file: {args.json_logs[i]}')

            # Handle the case where eval_interval is out of bounds
            eval_epoch_idx = int(args.eval_interval) - 1
            if eval_epoch_idx >= len(epochs):
                # If the specified eval_interval is greater than the number of epochs,
                # check the last available epoch instead of crashing.
                eval_epoch_idx = -1
            
            target_epoch = epochs[eval_epoch_idx]

            if metric not in log_dict[target_epoch]:
                if 'mAP' in metric or 'Avg Quality Score' in metric:
                    raise KeyError(
                        f'{args.json_logs[i]} does not contain metric '
                        f'{metric}. Please check if "--no-validate" is '
                        'specified when you trained the model. Or check '
                        f'if the eval_interval {args.eval_interval} in args '
                        'is equal to the eval_interval during training.')
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}. '
                    'Please reduce the log interval in the config so that '
                    'interval is less than iterations of one epoch.')

            if 'mAP' in metric or 'Avg Quality Score' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    if log_dict[epoch][metric]:
                        xs += [epoch]
                max_epoch = max(epochs)
                xs_filtered = []
                ys_filtered = []
                for x, y in zip(xs, ys):
                    if x <= max_epoch:
                        xs_filtered.append(x)
                        ys_filtered.append(y)
                plt.xlabel('epoch', fontsize=24, loc='left', fontweight='bold')
                plt.ylabel(metric if args.ylabel is None else args.ylabel, fontsize=24)
                plt.plot(xs_filtered, ys_filtered, label=legend[i * num_metrics + j], marker='o', linewidth=2.8)
                ax = plt.gca()
                ax.set_xlim(left=0)
                if args.y_max is not None:
                    ax.set_ylim(bottom=args.y_min, top=args.y_max)
                else:
                    ax.set_ylim(bottom=args.y_min)
                ax.set_xlabel('epoch', fontsize=24, loc='left', fontweight='bold')
                ax.set_ylabel(metric if args.ylabel is None else args.ylabel, fontsize=24)
                ax.xaxis.set_label_coords(0, -0.08)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
            else:
                xs = []
                ys = []
                for epoch in epochs:
                    # Use 'step' as global iteration number for x-axis
                    steps = log_dict[epoch].get('step')
                    if not steps:
                        # Fallback for older logs without 'step' key
                        steps = log_dict[epoch].get('iter', [])
                    
                    batch_values = log_dict[epoch].get(metric, [])
                    num_points = min(len(steps), len(batch_values))
                    xs.extend(steps[:num_points])
                    ys.extend(batch_values[:num_points])

                if not xs:
                    print(f'Cannot find value of {metric} in log.')
                    continue
                
                # Set Y-axis label
                if num_metrics > 1:
                    plt.ylabel('Entropy', fontsize=28, fontweight='bold')
                else:
                    plt.ylabel(metric if args.ylabel is None else args.ylabel, fontsize=28, fontweight='bold')

                plt.xlabel('Iteration', fontsize=28, fontweight='bold')
                plt.plot(xs, ys, linewidth=2.8, alpha=0.5, label=legend[i * num_metrics + j])
                plt.legend(fontsize=20, loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'weight': 'bold'})
                plt.grid(True, linestyle='--', alpha=0.7, color='gray')

                # === Updated styling ===
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.8)
                ax.tick_params(direction='out', length=6, width=1)
                ax.set_xlim(left=0)
                if args.y_max is not None:
                    ax.set_ylim(bottom=args.y_min, top=args.y_max)
                else:
                    ax.set_ylim(bottom=args.y_min)
                # Use a more adaptive locator for iterations
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

                def format_func(value, tick_number):
                    num = int(value)
                    if num >= 1000:
                        return f'{num // 1000}k'
                    return str(num)

                ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                # =======================
            if args.title is not None:
                plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def plot_epoch_wise(log_dicts, args):
    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24

    if args.backend is not None:
        plt.switch_backend(args.backend)
    
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys
    num_metrics = len(metrics)

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)
    
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot epoch-wise curve of {args.json_logs[i]}, metric is {metric}')
            if not epochs:
                raise ValueError(f'No epochs found in log file: {args.json_logs[i]}')

            xs = []
            ys_mean = []
            ys_std = []
            for epoch in sorted(epochs):
                values = log_dict[epoch].get(metric)
                if values:
                    xs.append(epoch)
                    ys_mean.append(np.mean(values))
                    ys_std.append(np.std(values))
            
            if not xs:
                print(f'Cannot find value of {metric} in log.')
                continue
            
            ys_mean = np.array(ys_mean)
            ys_std = np.array(ys_std)
            
            plt.plot(xs, ys_mean, marker='o', linestyle='-', label=legend[i * num_metrics + j])
            plt.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)

    plt.xlabel('epoch', fontsize=28, fontweight='bold')
    plt.ylabel(args.ylabel if args.ylabel else 'value', fontsize=28, fontweight='bold')
    plt.legend(fontsize=16, prop={'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7, color='gray')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
    ax.tick_params(direction='out', length=6, width=1)
    ax.set_xlim(left=0)
    if args.y_max is not None:
        ax.set_ylim(top=args.y_max)
    ax.set_ylim(bottom=args.y_min)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    def format_func(value, tick_number):
        num = int(value)
        if num >= 1000:
            return f'{num // 1000}k'
        return str(num)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    if args.title:
        plt.title(args.title)
    
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument(
        '--start-epoch',
        type=str,
        default='1',
        help='the epoch that you want to start')
    parser_plt.add_argument(
        '--eval-interval',
        type=str,
        default='1',
        help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument(
        '--y-max',
        type=float,
        default=None,
        help='set the maximum value of y-axis')
    parser_plt.add_argument(
        '--y-min',
        type=float,
        default=0,
        help='set the minimum value of y-axis (default: 0)')
    parser_plt.add_argument(
        '--ylabel',
        type=str,
        default=None,
        help='set the label of y-axis (default: use metric name)')


def add_epoch_wise_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_epoch_wise',
        help='parser for plotting epoch-wise curves of metric')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['loss'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument(
        '--y-max',
        type=float,
        default=None,
        help='set the maximum value of y-axis')
    parser_plt.add_argument(
        '--y-min',
        type=float,
        default=0,
        help='set the minimum value of y-axis (default: 0)')
    parser_plt.add_argument(
        '--ylabel',
        type=str,
        default=None,
        help='set the label of y-axis (default: use metric name)')


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    add_epoch_wise_plot_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            epoch = 1
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                val_flag = False
                # skip lines only contains one key
                if not len(log) > 1:
                    continue

                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)

                for k, v in log.items():
                    if '/' in k:
                        log_dict[epoch][k.split('/')[-1]].append(v)
                        val_flag = True
                    elif val_flag:
                        continue
                    else:
                        log_dict[epoch][k].append(v)

                if 'epoch' in log.keys():
                    epoch = log['epoch']

    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
