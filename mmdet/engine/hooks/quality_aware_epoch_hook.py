# Copyright (c) OpenMMLab. All rights reserved.
"""
质量感知轮次钩子 - 自动更新分配器的epoch信息

功能：
1. 在每个训练轮次开始时更新分配器的epoch信息
2. 监控质量感知权重的变化
3. 记录样本选择质量统计
"""

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class QualityAwareEpochHook(Hook):
    """质量感知轮次钩子
    
    Args:
        log_interval (int): 日志记录间隔
    """
    
    def __init__(self, log_interval=1):
        self.log_interval = log_interval
    
    def before_train_epoch(self, runner):
        """训练轮次开始前的处理"""
        
        current_epoch = runner.epoch
        
        # 获取模型和分配器
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 更新分配器的epoch信息
        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'assigner'):
            assigner = model.bbox_head.assigner
            if hasattr(assigner, 'set_epoch'):
                assigner.set_epoch(current_epoch)
                
                # 记录当前质量权重
                if hasattr(assigner, 'quality_alpha') and hasattr(assigner, 'warmup_epochs'):
                    if current_epoch < assigner.warmup_epochs:
                        quality_weight = (current_epoch / assigner.warmup_epochs) * assigner.quality_alpha
                    else:
                        quality_weight = assigner.quality_alpha
                    
                    if current_epoch % self.log_interval == 0:
                        runner.logger.info(
                            f"Epoch {current_epoch}: Quality weight = {quality_weight:.3f}, "
                            f"Geometric weight = {1.0 - quality_weight:.3f}")
        
        # 也尝试更新train_cfg中的分配器（如果存在）
        if hasattr(runner.model, 'train_cfg') and runner.model.train_cfg is not None:
            train_cfg = runner.model.train_cfg
            if hasattr(train_cfg, 'assigner') and hasattr(train_cfg.assigner, 'set_epoch'):
                train_cfg.assigner.set_epoch(current_epoch)