import torch
import pickle
import os
import copy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.registry import init_default_scope
from mmdet.structures import DetDataSample

class SaveKurtosisGFocalHead:
    """
    为GFocalHead添加保存峰度信息的功能的装饰器类
    """
    
    def __init__(self, original_gfocal_head, save_path):
        """
        初始化
        
        Args:
            original_gfocal_head: 原始的GFocalHead实例
            save_path: 保存峰度信息的路径
        """
        self.original_head = original_gfocal_head
        self.save_path = save_path
        self.kurtosis_values = []
        # 转发所有属性访问到原始头部
        for attr_name in dir(original_gfocal_head):
            if not attr_name.startswith('__'):
                attr = getattr(original_gfocal_head, attr_name)
                if not callable(attr):
                    setattr(self, attr_name, attr)
        
        # 重写前向方法
        self.original_forward_single = original_gfocal_head.forward_single
        self.original_head.forward_single = self.forward_single
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def forward_single(self, x, scale):
        """
        修改后的forward_single方法，用于捕获峰度信息
        
        Args:
            x: 输入特征图
            scale: 尺度因子
            
        Returns:
            tuple: 分类得分和边界框预测
        """
        # 调用原始前向方法
        cls_score, bbox_pred = self.original_forward_single(x, scale)
        
        # 检查是否可以访问峰度数据
        if hasattr(self.original_head, 'kurtosis'):
            kurtosis = self.original_head.kurtosis.detach().cpu().numpy()
            self.kurtosis_values.append(kurtosis)
        
        if hasattr(self.original_head, 'kurtosis_weight'):
            kurtosis_weight = self.original_head.kurtosis_weight.detach().cpu().numpy()
            self.kurtosis_values.append(kurtosis_weight)
        
        return cls_score, bbox_pred
    
    def save_kurtosis_info(self):
        """保存收集到的峰度信息"""
        if self.kurtosis_values:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.kurtosis_values, f)
            print(f"峰度信息已保存到: {self.save_path}")
        else:
            print("没有收集到峰度信息!")

def modify_model_to_save_kurtosis(model, save_path):
    """
    修改模型以保存峰度信息
    
    Args:
        model: 模型实例
        save_path: 保存峰度信息的路径
        
    Returns:
        model: 修改后的模型实例
    """
    # 查找GFocalHead实例
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'GFocalHead':
            # 对GFocalHead应用装饰器
            decorated_head = SaveKurtosisGFocalHead(module, save_path)
            # 更新模型中的头部
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                head_name = name.rsplit('.', 1)[1]
                setattr(parent_module, head_name, decorated_head)
            else:
                setattr(model, name, decorated_head)
            print(f"已为{name}添加峰度信息保存功能")
    
    return model

def inference_and_save_kurtosis(config_file, checkpoint, img_dir, save_path):
    """
    进行推理并保存峰度信息
    
    Args:
        config_file: 模型配置文件路径
        checkpoint: 模型检查点文件路径
        img_dir: 测试图像目录
        save_path: 保存峰度信息的路径
    """
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # 设置工作环境
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # 构建模型
    model = MODELS.build(cfg.model)
    
    # 加载检查点权重
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    # 修改模型以保存峰度信息
    model = modify_model_to_save_kurtosis(model, save_path)
    
    # 设置为评估模式
    model.eval()
    
    # 在这里添加推理代码，使用img_dir中的图像进行推理
    # ...
    
    # 这里需要为模型提供一些数据样本以触发前向传播
    # 示例：创建一个伪数据样本
    batch_size = 1
    input_shape = (3, 224, 224)  # 假设输入为224x224的RGB图像
    
    # 创建伪输入
    pseudo_input = torch.rand(batch_size, *input_shape)
    
    # 创建伪数据样本
    data_samples = [DetDataSample() for _ in range(batch_size)]
    
    # 执行前向传播
    with torch.no_grad():
        model(pseudo_input, data_samples=data_samples, mode='predict')
    
    # 保存峰度信息
    for name, module in model.named_modules():
        if isinstance(module, SaveKurtosisGFocalHead):
            module.save_kurtosis_info()
    
    print("推理完成")

def compare_kurtosis(original_file, improved_file, output_dir):
    """
    比较原始模型和改进模型的峰度分布
    
    Args:
        original_file: 原始模型的峰度信息文件路径
        improved_file: 改进模型的峰度信息文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载峰度信息
    with open(original_file, 'rb') as f:
        original_kurtosis = pickle.load(f)
    
    with open(improved_file, 'rb') as f:
        improved_kurtosis = pickle.load(f)
    
    # 确保数据格式一致
    if isinstance(original_kurtosis, list):
        original_kurtosis = np.concatenate([k.flatten() for k in original_kurtosis])
    else:
        original_kurtosis = original_kurtosis.flatten()
    
    if isinstance(improved_kurtosis, list):
        improved_kurtosis = np.concatenate([k.flatten() for k in improved_kurtosis])
    else:
        improved_kurtosis = improved_kurtosis.flatten()
    
    # 计算统计量
    original_mean = np.mean(original_kurtosis)
    original_std = np.std(original_kurtosis)
    original_kurt = stats.kurtosis(original_kurtosis)
    
    improved_mean = np.mean(improved_kurtosis)
    improved_std = np.std(improved_kurtosis)
    improved_kurt = stats.kurtosis(improved_kurtosis)
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.hist(original_kurtosis, bins=50, alpha=0.5, label='原始模型', color='blue', density=True)
    plt.hist(improved_kurtosis, bins=50, alpha=0.5, label='改进模型', color='green', density=True)
    
    # 添加峰度信息
    plt.title('原始模型 vs 改进模型峰度分布对比', fontsize=16)
    plt.xlabel('峰度值', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    
    info_text = f"""统计信息:
原始模型:
  - 均值: {original_mean:.4f}
  - 标准差: {original_std:.4f}
  - 峰度: {original_kurt:.4f}
  
改进模型:
  - 均值: {improved_mean:.4f}
  - 标准差: {improved_std:.4f}
  - 峰度: {improved_kurt:.4f}
  
峰度变化: {improved_kurt - original_kurt:.4f}
"""
    
    # 添加文本框
    plt.text(0.02, 0.97, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    output_path = os.path.join(output_dir, 'kurtosis_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"峰度对比图已保存到: {output_path}")
    
    # 生成报告
    report_path = os.path.join(output_dir, 'kurtosis_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("峰度分析报告\n")
        f.write("=============\n\n")
        f.write(info_text)
        
        if improved_kurt > original_kurt:
            f.write("\n分析结论: 改进模型的峰度值增加，表明分布更加集中在均值附近，\n")
            f.write("预测更加确定，模型对于关键区域的关注度更高。\n")
        else:
            f.write("\n分析结论: 改进模型的峰度值降低，表明分布更加分散，\n")
            f.write("预测的不确定性增加。\n")
    
    print(f"峰度分析报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='修改GFocalHead并提取峰度信息')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 提取命令
    extract_parser = subparsers.add_parser('extract', help='提取峰度信息')
    extract_parser.add_argument('--config', required=True, help='模型配置文件路径')
    extract_parser.add_argument('--checkpoint', required=True, help='模型检查点文件路径')
    extract_parser.add_argument('--img-dir', required=True, help='测试图像目录')
    extract_parser.add_argument('--save-path', required=True, help='保存峰度信息的路径')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较峰度分布')
    compare_parser.add_argument('--original', required=True, help='原始模型的峰度信息文件路径')
    compare_parser.add_argument('--improved', required=True, help='改进模型的峰度信息文件路径')
    compare_parser.add_argument('--output-dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        inference_and_save_kurtosis(args.config, args.checkpoint, args.img_dir, args.save_path)
    elif args.command == 'compare':
        compare_kurtosis(args.original, args.improved, args.output_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 