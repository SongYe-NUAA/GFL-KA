#用于寻找最佳的结果
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict

def analyze_json_log(log_file: str):
    """分析 JSON 格式的日志文件"""
    # 存储各项指标
    data = {
        'bbox_mAP': [],
        'bbox_mAP_50': [],
        'bbox_mAP_75': [],
        'bbox_mAP_s': [],
        'bbox_mAP_m': [],
        'bbox_mAP_l': [],
        'data_time': [],
        'time': []
    }

    # 跟踪最佳性能
    best_map = {'value': 0, 'epoch': 0}

    try:
        with open(log_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())

                # 收集评估结果并跟踪最佳性能
                if 'coco/bbox_mAP' in record:
                    current_map = record['coco/bbox_mAP']
                    data['bbox_mAP'].append(current_map)
                    data['bbox_mAP_50'].append(record['coco/bbox_mAP_50'])
                    data['bbox_mAP_75'].append(record['coco/bbox_mAP_75'])
                    data['bbox_mAP_s'].append(record['coco/bbox_mAP_s'])
                    data['bbox_mAP_m'].append(record['coco/bbox_mAP_m'])
                    data['bbox_mAP_l'].append(record['coco/bbox_mAP_l'])
                    
                    # 更新最佳性能
                    if current_map > best_map['value']:
                        best_map['value'] = current_map
                        best_map['epoch'] = len(data['bbox_mAP'])  # 使用当前的索引作为 epoch

                # 收集时间数据
                if 'data_time' in record:
                    data['data_time'].append(record['data_time'])
                if 'time' in record:
                    data['time'].append(record['time'])

        # 打印训练统计信息
        print("\n=== 训练统计 ===")
        print(f"总评估次数: {len(data['bbox_mAP'])}")

        # 打印最佳性能
        print("\n=== 最佳性能 ===")
        print(f"最佳mAP: {best_map['value']:.4f} (Epoch {best_map['epoch']})")

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

        # 绘制时间曲线
        if data['data_time']:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(data['data_time'])), data['data_time'], label='Data Time', marker='o')
            plt.plot(range(len(data['time'])), data['time'], label='Total Time', marker='s')
            plt.title('Time Analysis')
            plt.xlabel('Evaluation')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            plt.legend()
            plt.savefig('time_curves.png')
            plt.close()

        return data, best_map

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return None, None

# 使用示例
log_file = r"D:\project\mmdetection-main\runs\windturbine_AutoASSIGN\20241205_162958\20241205_162958.json"  # 替换为您的日志文件路径
results, best_performance = analyze_json_log(log_file)

if results:
    # 计算一些额外的统计信息
    print("\n=== 训练效率统计 ===")
    avg_time = np.mean(results['time']) if results['time'] else 0
    avg_data_time = np.mean(results['data_time']) if results['data_time'] else 0
    print(f"平均每次评估时间: {avg_time:.4f}秒")
    print(f"平均数据加载时间: {avg_data_time:.4f}秒")
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