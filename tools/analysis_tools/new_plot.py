# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
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
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[int(args.eval_interval) - 1]]:
                if 'mAP' in metric:
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

            if 'mAP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    if log_dict[epoch][metric]:
                        xs += [epoch]
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                for epoch in epochs:
                    iters = log_dict[epoch]['step']
                    xs.append(np.array(iters))
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
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
            for line in log_file:
                log = json.loads(line.strip())
                if not len(log) > 1:
                    continue
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    if '/' in k:
                        log_dict[epoch][k.split('/')[-1]].append(v)
                    else:
                        log_dict[epoch][k].append(v)
                if 'epoch' in log.keys():
                    epoch = log['epoch']
    return log_dicts


def main():
    parser = argparse.ArgumentParser(description='Custom Plot Script')
    parser.add_argument('json_logs', type=str, nargs='+', help='Log file paths')
    parser.add_argument('--keys', type=str, nargs='+', default=['bbox_mAP'], help='Metrics to plot')
    parser.add_argument('--legend', type=str, nargs='+', default=None, help='Legend names')
    parser.add_argument('--out', type=str, default='custom_plot.pdf', help='Output file path')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    parser.add_argument('--style', type=str, default='whitegrid', help='Seaborn style (whitegrid, darkgrid, white, dark)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 7], help='Figure size (width, height)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figure')
    parser.add_argument('--ylabel', type=str, default=None, help='Y-axis label')
    args = parser.parse_args()
    
    log_dicts = load_json_logs(args.json_logs)
    metrics = args.keys
    
    # Set plot style and figure size
    sns.set_style(args.style)
    plt.figure(figsize=(args.figsize[0], args.figsize[1]))
    
    # Set grid with better appearance
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set legend
    legend = args.legend
    if legend is None:
        legend = [f'Model {i+1}' for i in range(len(args.json_logs))]
    
    # Define a good color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dicts)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']
    
    # Store data for all models to determine appropriate x-ticks
    all_epochs = []
    
    for i, log_dict in enumerate(log_dicts):
        epochs = sorted(log_dict.keys())
        all_epochs.extend(epochs)
        
        for j, metric in enumerate(metrics):
            # Process and plot data
            data_x = []
            data_y = []
            epoch_boundaries = []
            last_epoch = None
            
            for epoch in epochs:
                if metric in log_dict[epoch]:
                    metric_name = metric
                elif 'coco/'+metric in log_dict[epoch]:
                    metric_name = 'coco/'+metric
                else:
                    continue
                
                # Collect all iterations for this epoch
                if 'step' in log_dict[epoch] and log_dict[epoch][metric_name]:
                    iters = log_dict[epoch]['step']
                    values = log_dict[epoch][metric_name][:len(iters)]
                    
                    # Calculate iteration-based x-coordinate
                    total_iters = max(iters) if iters else 0
                    
                    # Adjust the display to show each iteration while keeping epoch-based x-axis
                    for iter_idx, value in zip(iters, values):
                        # Add a small fraction based on iteration progress within the epoch
                        frac = iter_idx / total_iters if total_iters > 0 else 0
                        x_val = epoch - 1 + frac
                        data_x.append(x_val)
                        data_y.append(value)
                    
                    # Mark epoch boundaries for reference
                    if last_epoch is not None and last_epoch != epoch - 1:
                        epoch_boundaries.append(epoch - 0.5)
                    last_epoch = epoch
            
            if data_x:
                # Sort by x-coordinate
                sorted_indices = np.argsort(data_x)
                data_x = np.array(data_x)[sorted_indices]
                data_y = np.array(data_y)[sorted_indices]
                
                # Plot line connecting the points
                plt.plot(data_x, data_y, label=legend[i], color=colors[i], linewidth=1.5, alpha=0.8)
                
                # Add scatter points for better visibility at each data point
                if len(data_x) < 100:  # Only add markers if not too dense
                    plt.scatter(data_x, data_y, color=colors[i], marker=markers[i % len(markers)], 
                                s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Determine appropriate x-ticks
    unique_epochs = sorted(list(set(all_epochs)))
    if len(unique_epochs) <= 20:  # If not too many epochs, show all
        plt.xticks(unique_epochs)
    else:  # Otherwise show a reasonable number of ticks
        step = max(1, len(unique_epochs) // 10)
        plt.xticks(unique_epochs[::step])
    
    # Add vertical lines at epoch boundaries for better readability
    for boundary in epoch_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    y_label = args.ylabel if args.ylabel else metrics[0]
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    
    if args.title:
        plt.title(args.title, fontsize=14, fontweight='bold')
    
    # Improve legend
    plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.8, 
               edgecolor='gray', loc='best')
    
    # Tight layout for better use of space
    plt.tight_layout()
    
    # Add minor ticks for more precise reading
    plt.minorticks_on()
    
    # Save figure
    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f'Plot saved to {args.out}')
    
    # Show figure
    plt.show()


if __name__ == '__main__':
    main()
