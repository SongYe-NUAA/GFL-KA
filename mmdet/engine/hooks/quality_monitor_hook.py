"""
质量监控钩子 - 实时监控样本质量与训练效果

功能：
1. 监控选中样本的IoU和分类分数分布
2. 跟踪质量一致性变化趋势
3. 可视化质量感知训练过程
4. 自动调整训练策略
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet.registry import MODELS
import os


@HOOKS.register_module()
class QualityMonitorHook(Hook):
    """质量监控钩子"""
    
    def __init__(self,
                 interval=100,
                 log_quality_stats=True,
                 save_quality_plots=True,
                 plot_save_dir='quality_plots'):
        self.interval = interval
        self.log_quality_stats = log_quality_stats
        self.save_quality_plots = save_quality_plots
        self.plot_save_dir = plot_save_dir
        
        # 质量统计数据
        self.quality_history = {
            'epoch': [],
            'avg_pos_iou': [],
            'avg_pos_cls_score': [],
            'quality_consistency': [],
            'high_quality_ratio': []
        }
        
        if self.save_quality_plots:
            os.makedirs(self.plot_save_dir, exist_ok=True)
    
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """训练迭代后的监控"""
        
        if (runner.iter + 1) % self.interval != 0:
            return
            
        # 获取当前的损失信息
        if 'loss_cls' in outputs['log_vars']:
            self._collect_quality_stats(runner, outputs)
    
    def after_train_epoch(self, runner):
        """训练轮次后的监控"""
        
        current_epoch = runner.epoch + 1
        self._log_epoch_quality_stats(runner, current_epoch)
        
        if self.save_quality_plots:
            self._save_quality_plots(current_epoch)
    
    def _collect_quality_stats(self, runner, outputs):
        """收集质量统计信息"""
        
        try:
            # 从模型中获取质量统计
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
                
            bbox_head = model.bbox_head
            
            # 获取最近的正样本统计
            if hasattr(bbox_head, '_last_pos_stats'):
                pos_stats = bbox_head._last_pos_stats
                
                if pos_stats is not None and len(pos_stats) > 0:
                    pos_ious = pos_stats.get('pos_ious', [])
                    pos_cls_scores = pos_stats.get('pos_cls_scores', [])
                    
                    if len(pos_ious) > 0 and len(pos_cls_scores) > 0:
                        avg_iou = np.mean(pos_ious)
                        avg_cls_score = np.mean(pos_cls_scores)
                        
                        # 计算质量一致性
                        consistency = 1.0 - np.mean(np.abs(np.array(pos_ious) - np.array(pos_cls_scores)))
                        
                        # 计算高质量样本比例
                        high_quality_mask = (np.array(pos_ious) > 0.6) & (np.array(pos_cls_scores) > 0.5)
                        high_quality_ratio = np.mean(high_quality_mask) if len(high_quality_mask) > 0 else 0.0
                        
                        # 记录统计信息
                        self.quality_history['avg_pos_iou'].append(avg_iou)
                        self.quality_history['avg_pos_cls_score'].append(avg_cls_score)
                        self.quality_history['quality_consistency'].append(consistency)
                        self.quality_history['high_quality_ratio'].append(high_quality_ratio)
                        
                        if self.log_quality_stats:
                            runner.logger.info(
                                f"Quality Stats - IoU: {avg_iou:.3f}, "
                                f"ClsScore: {avg_cls_score:.3f}, "
                                f"Consistency: {consistency:.3f}, "
                                f"HighQualityRatio: {high_quality_ratio:.3f}")
                            
        except Exception as e:
            runner.logger.warning(f"Failed to collect quality stats: {e}")
    
    def _log_epoch_quality_stats(self, runner, epoch):
        """记录轮次质量统计"""
        
        if len(self.quality_history['avg_pos_iou']) == 0:
            return
            
        # 计算本轮次的平均质量指标
        recent_data_size = min(len(self.quality_history['avg_pos_iou']), 
                              self.interval // 10)  # 取最近的数据
        
        if recent_data_size > 0:
            avg_iou = np.mean(self.quality_history['avg_pos_iou'][-recent_data_size:])
            avg_cls_score = np.mean(self.quality_history['avg_pos_cls_score'][-recent_data_size:])
            avg_consistency = np.mean(self.quality_history['quality_consistency'][-recent_data_size:])
            avg_high_quality = np.mean(self.quality_history['high_quality_ratio'][-recent_data_size:])
            
            # 记录到历史
            self.quality_history['epoch'].append(epoch)
            
            runner.logger.info(
                f"Epoch {epoch} Quality Summary - "
                f"AvgIoU: {avg_iou:.3f}, "
                f"AvgClsScore: {avg_cls_score:.3f}, "
                f"Consistency: {avg_consistency:.3f}, "
                f"HighQualityRatio: {avg_high_quality:.3f}")
    
    def _save_quality_plots(self, epoch):
        """保存质量趋势图"""
        
        if len(self.quality_history['epoch']) < 2:
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Quality Monitoring - Epoch {epoch}', fontsize=16)
            
            epochs = self.quality_history['epoch']
            
            # IoU趋势
            if len(self.quality_history['avg_pos_iou']) >= len(epochs):
                recent_ious = self.quality_history['avg_pos_iou'][-len(epochs):]
                axes[0, 0].plot(epochs, recent_ious, 'b-', marker='o')
                axes[0, 0].set_title('Average Positive IoU')
                axes[0, 0].set_ylabel('IoU')
                axes[0, 0].grid(True)
            
            # 分类分数趋势
            if len(self.quality_history['avg_pos_cls_score']) >= len(epochs):
                recent_scores = self.quality_history['avg_pos_cls_score'][-len(epochs):]
                axes[0, 1].plot(epochs, recent_scores, 'g-', marker='s')
                axes[0, 1].set_title('Average Classification Score')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].grid(True)
            
            # 质量一致性趋势
            if len(self.quality_history['quality_consistency']) >= len(epochs):
                recent_consistency = self.quality_history['quality_consistency'][-len(epochs):]
                axes[1, 0].plot(epochs, recent_consistency, 'r-', marker='^')
                axes[1, 0].set_title('Quality Consistency')
                axes[1, 0].set_ylabel('Consistency')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].grid(True)
            
            # 高质量样本比例趋势
            if len(self.quality_history['high_quality_ratio']) >= len(epochs):
                recent_ratio = self.quality_history['high_quality_ratio'][-len(epochs):]
                axes[1, 1].plot(epochs, recent_ratio, 'm-', marker='d')
                axes[1, 1].set_title('High Quality Sample Ratio')
                axes[1, 1].set_ylabel('Ratio')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(self.plot_save_dir, f'quality_trends_epoch_{epoch}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Failed to save quality plots: {e}")


def add_quality_monitoring_to_head(bbox_head):
    """为检测头添加质量监控功能"""
    
    def _store_pos_stats(self, pos_ious, pos_cls_scores):
        """存储正样本统计信息"""
        self._last_pos_stats = {
            'pos_ious': pos_ious.detach().cpu().numpy().tolist() if torch.is_tensor(pos_ious) else pos_ious,
            'pos_cls_scores': pos_cls_scores.detach().cpu().numpy().tolist() if torch.is_tensor(pos_cls_scores) else pos_cls_scores
        }
    
    # 动态添加方法到检测头
    bbox_head._store_pos_stats = _store_pos_stats.__get__(bbox_head)
    bbox_head._last_pos_stats = None
    
    return bbox_head


if __name__ == "__main__":
    print("🎯 质量监控钩子")
    print("="*50)
    print("功能：")
    print("1. 实时监控样本质量分布")
    print("2. 跟踪质量一致性变化")
    print("3. 可视化训练过程")
    print("4. 自动质量评估")
    print()
    print("使用方法：")
    print("在配置文件中添加：")
    print("custom_hooks = [")
    print("    dict(")
    print("        type='QualityMonitorHook',")
    print("        interval=100,")
    print("        log_quality_stats=True,")
    print("        save_quality_plots=True)")
    print("]")