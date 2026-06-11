# 用于寻找最佳的结果
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict

def analyze_training_log(log_file):
    """分析训练日志文件"""
    # 存储各项指标
    data = {
        'epoch': [], 'iter': [], 'lr': [],
        'loss': [], 'loss_cls': [], 'loss_bbox': [], 'loss_dfl': [],
        'grad_norm': [], 'data_time': [], 'time': [],
        'bbox_mAP': [], 'bbox_mAP_50': [], 'bbox_mAP_75': [],
        'bbox_mAP_s': [], 'bbox_mAP_m': [], 'bbox_mAP_l': []
    }

    # 跟踪最佳性能
    best_map = {'value': 0, 'epoch': 0}

    try:
        with open(log_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())

                # 收集训练过程数据
                if 'loss' in record:
                    for key in ['epoch', 'iter', 'lr', 'loss', 'loss_cls',
                                'loss_bbox', 'loss_dfl', 'grad_norm', 'data_time', 'time']:
                        if key in record:
                            data[key].append(record[key])

                # 收集评估结果并跟踪最佳性能
                if 'coco/bbox_mAP' in record:
                    current_map = record['coco/bbox_mAP']
                    current_epoch = record.get('epoch', 0)
                    data['bbox_mAP'].append(current_map)
                    data['bbox_mAP_50'].append(record['coco/bbox_mAP_50'])
                    data['bbox_mAP_75'].append(record['coco/bbox_mAP_75'])
                    data['bbox_mAP_s'].append(record['coco/bbox_mAP_s'])
                    data['bbox_mAP_m'].append(record['coco/bbox_mAP_m'])
                    data['bbox_mAP_l'].append(record['coco/bbox_mAP_l'])

                    # 更新最佳性能
                    if current_map > best_map['value']:
                        best_map['value'] = current_map
                        best_map['epoch'] = current_epoch

        # 打印训练统计信息
        print("\n=== 训练统计 ===")
        print(f"总迭代次数: {len(data['iter'])}")
        print(f"训练轮次: {max(data['epoch'])}")

        # 打印最佳性能
        print("\n=== 最佳性能 ===")
        print(f"最佳mAP: {best_map['value']:.4f} (Epoch {best_map['epoch']})")

        # Loss统计
        print("\n=== Loss 统计 ===")
        print(f"最终总Loss: {data['loss'][-1]:.4f}")
        print(f"最终分类Loss: {data['loss_cls'][-1]:.4f}")
        print(f"最终边界框Loss: {data['loss_bbox'][-1]:.4f}")
        print(f"最终DFL Loss: {data['loss_dfl'][-1]:.4f}")

        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(data['iter'], data['loss'], label='Total Loss')
        plt.plot(data['iter'], data['loss_cls'], label='Cls Loss')
        plt.plot(data['iter'], data['loss_bbox'], label='Bbox Loss')
        plt.plot(data['iter'], data['loss_dfl'], label='DFL Loss')
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('loss_curves.png')
        plt.close()

        # 绘制学习率曲线
        plt.figure(figsize=(12, 6))
        plt.plot(data['iter'], data['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig('lr_curve.png')
        plt.close()

        # 绘制mAP曲线
        if data['bbox_mAP']:
            plt.figure(figsize=(12, 6))
            x = range(len(data['bbox_mAP']))
            plt.plot(x, data['bbox_mAP'], label='mAP', marker='o')
            plt.plot(x, data['bbox_mAP_50'], label='mAP@50', marker='s')
            plt.plot(x, data['bbox_mAP_75'], label='mAP@75', marker='^')
            plt.title('COCO mAP')
            plt.xlabel('Evaluation')
            plt.ylabel('mAP')
            plt.grid(True)
            plt.legend()
            plt.savefig('map_curves.png')
            plt.close()

            # 绘制不同尺度的mAP曲线
            plt.figure(figsize=(12, 6))
            plt.plot(x, data['bbox_mAP_s'], label='small', marker='o')
            plt.plot(x, data['bbox_mAP_m'], label='medium', marker='s')
            plt.plot(x, data['bbox_mAP_l'], label='large', marker='^')
            plt.title('COCO mAP by Object Size')
            plt.xlabel('Evaluation')
            plt.ylabel('mAP')
            plt.grid(True)
            plt.legend()
            plt.savefig('map_size_curves.png')
            plt.close()

        return data, best_map

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return None, None


# 使用示例
log_file = r"D:\project\mmdetection-main\runs\windturbine_ATSS\20241205_110640\20241205_110640.json"  # 替换为您的日志文件路径
results, best_performance = analyze_training_log(log_file)

if results:
    # 计算一些额外的统计信息
    print("\n=== 训练效率统计 ===")
    avg_time = np.mean(results['time'])
    avg_data_time = np.mean(results['data_time'])
    print(f"平均每次迭代时间: {avg_time:.4f}秒")
    print(f"平均数��加载时间: {avg_data_time:.4f}秒")
    print(f"平均计算时间: {(avg_time - avg_data_time):.4f}秒")


def extract_metrics_from_txt(file_path: str) -> Dict:
    """从txt文件中提取指标"""
    metrics = {}

    try:
        # 使用 'mbcs' 编码（Windows ANSI 编码）
        with open(file_path, 'r', encoding='mbcs') as f:
            content = f.read()
            # 继续处理内容...
            # 提取 mAP、类别准确率等指标的逻辑
            # ...
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

    return metrics