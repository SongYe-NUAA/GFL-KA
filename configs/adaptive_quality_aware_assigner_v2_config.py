# 🚀 自适应质量感知分配器 2.0 - 前沿配置示例
# 
# 这是目标检测领域最先进的样本分配策略，融合了多项前沿技术：
# 1. 🎯 Focal Quality机制 - 聚焦困难样本
# 2. 🌊 不确定性感知权重 - 提升泛化能力  
# 3. ⚡ 动态TopK选择 - 自适应候选数量
# 4. 🔄 梯度引导优化 - 端到端学习
# 5. 🛡️ 多尺度一致性 - 跨尺度特征对齐

# === 基础配置 ===
_base_ = [
    '../_base_/models/gfocal_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# === 🚀 前沿分配器配置 ===
model = dict(
    bbox_head=dict(
        assigner=dict(
            type='QualityAwareATSSAssigner',
            topk=9,  # 基础TopK值
            
            # 🎯 Focal Quality参数 - 解决样本不平衡 (方案A优化)
            focal_alpha=0.1,            # Focal权重因子，大幅降低 (0.25→0.1)
            focal_gamma=1.0,            # Focal困难度指数，降低 (2.0→1.0)
            focal_quality_weight=0.08,  # Focal质量权重，大幅降低 (0.3→0.08)
            
            # 🌊 不确定性感知参数 - 提升泛化能力 (方案A优化)
            enable_uncertainty_estimation=True,    # 启用不确定性估计
            uncertainty_weight=0.05,              # 不确定性权重系数，大幅降低 (0.2→0.05)
            uncertainty_temperature=0.5,          # 不确定性温度参数，降低 (1.0→0.5)
            
            # ⚡ 动态TopK参数 - 自适应候选选择 (方案A优化)
            enable_dynamic_topk=True,      # 启用动态TopK
            min_topk=8,                   # 最小TopK值，增加 (4→8)
            max_topk=12,                  # 最大TopK值，降低 (15→12)
            complexity_threshold=0.7,     # 场景复杂度阈值，提高 (0.5→0.7)
            
            # 🔄 梯度优化参数 - 端到端学习 (方案A优化)
            enable_gradient_optimization=True,    # 启用梯度引导优化
            meta_learning_rate=0.0002,           # 元学习率，大幅降低 (0.001→0.0002)
            gradient_accumulation_steps=4,       # 梯度累积步数
            
            # 🛡️ 多尺度一致性参数 - 跨尺度特征对齐 (方案A优化)
            enable_multiscale_consistency=True,   # 启用多尺度一致性
            scale_consistency_weight=0.02,       # 尺度一致性权重，大幅降低 (0.1→0.02)
            num_scale_levels=3,                  # 参与一致性计算的尺度层数
            
            # 🎯 高级自适应参数 (方案A优化)
            adaptive_fusion=True,         # 启用自适应融合
            performance_window=200,       # 性能统计窗口，增大 (50→200)
            adaptation_rate=0.005,        # 自适应调整速率，大幅降低 (0.02→0.005)
            weight_bounds=(0.02, 0.2),    # 权重边界约束，大幅缩小 (0.1,0.7)→(0.02,0.2)
        ),
        
        # 🚀 方案A优化：调整损失权重，增强bbox损失权重
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=3.0,  # 大幅增加bbox损失权重 (2.0→3.0)
            reduction='mean'
        ),
        
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=0.8,  # 稍微降低分类损失权重 (1.0→0.8)
        )
    )
)

# === 🚀 训练策略优化 ===
# 针对前沿分配器的训练参数调整

# 优化器配置 - 方案A优化：回归友好的学习率
optimizer = dict(
    type='SGD',
    lr=0.008,  # 降低学习率 (0.01→0.008)，更稳定的训练
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        # 为分配器的可学习参数设置特殊学习率
        custom_keys={
            'bbox_head.assigner': dict(lr_mult=0.05)  # 分配器参数使用更小学习率 (0.1→0.05)
        }
    )
)

# 学习率调度 - 方案A优化：更早降低学习率
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,    # 更长的预热期，让分配器稳定
    warmup_ratio=0.001,
    step=[14, 20])  # 提前降低学习率 (8,11→14,20)，给回归更多时间学习

# === 🎯 数据增强策略 ===
# 配合前沿分配器的数据增强

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # 🚀 添加更多样化的数据增强，测试分配器的泛化能力
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# === 📊 评估和监控配置 ===
evaluation = dict(
    interval=1,
    metric='bbox',
    # 🚀 添加分配器状态监控
    save_best='auto',
    rule='greater'
)

# 🚀 自定义钩子：监控分配器状态
custom_hooks = [
    dict(
        type='AssignerMonitorHook',  # 需要实现这个钩子类
        interval=100,  # 每100步监控一次
        log_assigner_status=True,
        save_status_history=True
    )
]

# === 🎯 运行时配置 ===
# 针对前沿技术的运行时优化
runner = dict(type='EpochBasedRunner', max_epochs=24)  # 增加训练轮数 (12→24)

# 日志配置 - 记录更多分配器信息
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # 🚀 自定义分配器日志
        dict(type='AssignerLoggerHook')  # 需要实现
    ]
)

# === 💡 使用建议 ===
"""
🚀 前沿分配器使用建议：

1. 🎯 参数调优策略：
   - 从默认参数开始，观察训练稳定性
   - 根据数据集特点调整focal_gamma（困难样本比例）
   - 复杂场景增大max_topk，简单场景减小min_topk

2. 🌊 不确定性控制：
   - 验证集性能不稳定时，降低uncertainty_weight
   - 过拟合严重时，增加uncertainty_temperature

3. ⚡ 动态TopK调优：
   - 小目标多的数据集：降低complexity_threshold
   - 大目标为主的数据集：提高complexity_threshold

4. 🔄 梯度优化：
   - 训练初期可关闭gradient_optimization，后期开启
   - meta_learning_rate需要仔细调整，过大会不稳定

5. 🛡️ 多尺度一致性：
   - FPN层数多时增加num_scale_levels
   - 尺度变化大的数据集增加scale_consistency_weight

6. 📊 监控指标：
   - 关注分配器状态变化趋势
   - 监控各技术模块的权重演化
   - 观察场景复杂度分析结果
"""

# === 🎯 不同场景的推荐配置 ===

# 🔥 高性能配置（所有前沿技术全开）
high_performance_config = dict(
    focal_alpha=0.3, focal_gamma=2.5,
    enable_uncertainty_estimation=True, uncertainty_weight=0.25,
    enable_dynamic_topk=True, min_topk=3, max_topk=18,
    enable_gradient_optimization=True, meta_learning_rate=0.002,
    enable_multiscale_consistency=True, scale_consistency_weight=0.15
)

# ⚡ 高效配置（平衡性能与效率）
balanced_config = dict(
    focal_alpha=0.25, focal_gamma=2.0,
    enable_uncertainty_estimation=True, uncertainty_weight=0.2,
    enable_dynamic_topk=True, min_topk=4, max_topk=12,
    enable_gradient_optimization=False,  # 关闭梯度优化提升效率
    enable_multiscale_consistency=True, scale_consistency_weight=0.1
)

# 🎯 稳定配置（保守但稳定）
stable_config = dict(
    focal_alpha=0.2, focal_gamma=1.5,
    enable_uncertainty_estimation=True, uncertainty_weight=0.15,
    enable_dynamic_topk=False,  # 固定TopK
    enable_gradient_optimization=False,
    enable_multiscale_consistency=False
)

# === 🚀 方案A：回归友好的质量感知分配器配置 ===
"""
🎯 方案A核心优化策略：

1. 🎯 大幅降低质量感知权重：
   - focal_quality_weight: 0.3 → 0.08 (降低73%)
   - uncertainty_weight: 0.2 → 0.05 (降低75%)
   - scale_consistency_weight: 0.1 → 0.02 (降低80%)

2. 📈 增强bbox损失权重：
   - loss_bbox_weight: 2.0 → 3.0 (增加50%)
   - loss_cls_weight: 1.0 → 0.8 (降低20%)

3. ⚡ 更保守的动态调整：
   - adaptation_rate: 0.02 → 0.005 (降低75%)
   - weight_bounds: (0.1,0.7) → (0.02,0.2) (大幅缩小范围)
   - performance_window: 50 → 200 (增大窗口)

4. 🔄 回归友好的训练策略：
   - 学习率: 0.01 → 0.008 (降低20%)
   - 学习率衰减: [8,11] → [14,20] (延后衰减)
   - 训练轮数: 12 → 24 (增加100%)

🎯 预期效果：
- ✅ Bbox损失从上升4个点回到正常水平
- ✅ 验证效果保持良好，不会显著下降
- ✅ 训练过程更稳定，减少震荡
- ✅ 分类和回归学习更平衡
- ✅ 质量感知功能保留，但影响大幅降低

🚀 使用建议：
1. 先用此配置训练，观察bbox损失变化
2. 如果效果不理想，可进一步降低质量权重
3. 验证性能如果下降过多，可适当提高质量权重
4. 根据具体数据集特点微调参数
"""