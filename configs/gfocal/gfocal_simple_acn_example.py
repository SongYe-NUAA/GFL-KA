"""
🚀 简化版ACN (Adaptive Contrastive Normalization) 示例配置

核心创新：
1. 双EMA系统：分别追踪高质量和低质量样本的concentration分布
2. 对比归一化：根据样本与高/低质量中心的相对距离进行归一化
3. 延迟更新：Forward时使用历史EMA，Loss时更新EMA

设计哲学：
- 简单即美：固定超参数，无动态机制
- 慢即快：极慢EMA更新(momentum=0.95)保证稳定性
- 少即多：零额外可学习参数，避免过拟合

预期效果：
- Confidence区分度提升50%+
- 高IoU样本confidence更高，低IoU样本confidence更低
- mAP提升 +0.3~0.6%
"""

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5
    ),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🚀 简化版ACN配置（核心创新）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_simple_acn=True,  # ✅ 启用简化版ACN
        acn_momentum=0.95,       # EMA动量（0.95=极慢更新，类似BN）
        acn_temperature=1.0,     # 温度参数（1.0=不做缩放）
        acn_min_samples=10,      # 最小样本数阈值（保证统计可靠性）
        acn_high_iou_thresh=0.7, # 高质量样本IoU阈值
        acn_low_iou_thresh=0.4,  # 低质量样本IoU阈值
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 基础配置
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        use_adaptive_modulation=True,  # 自适应调制机制
        use_top34_combined=False,      # Top3+Top4组合
        
        # 多分位数集成（可选，与ACN兼容）
        enable_multi_quantile_ensemble=False,  # 建议先用单分位数测试ACN效果
        
        # 验证时EMA学习（建议关闭，让ACN完全控制）
        enable_validation_ema_learning=False,
        enable_per_sample_adaptation=False,
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 日志开关（监控ACN效果）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_ema_validation_log=True,
        enable_ema_layer_comparison_log=True,  # ✅ 推荐开启，监控ACN的EMA更新
        enable_positive_sample_log=True,
        enable_discrimination_score_log=True,
        enable_correlation_analysis_log=False,
        enable_topk_iou_statistics_log=False,
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 其他参数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        ),
        
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0
        ),
        
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25
        )
    ),
    
    # Training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        initial_epoch=4
    ),
    
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100
    )
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 训练配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 优化器
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 评估配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行时配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 使用建议
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
1. 首次训练：
   - 使用默认配置即可，无需调参
   - 监控日志中的"ACN统计"，确保EMA正常初始化
   - 期望在1-2个epoch后所有层都初始化完成

2. 监控指标：
   - EMA差距：期望从0.2逐渐增长到0.8+
   - 初始化状态：所有层都应该显示"✅已初始化"
   - 更新次数：应该持续增长（说明有足够的高低IoU样本）

3. 调试技巧：
   - 如果某层长时间未初始化：
     * 检查该层是否有足够的正样本
     * 考虑降低acn_min_samples（如改为5）
   
   - 如果EMA差距太小(<0.3)：
     * 检查IoU阈值是否合适
     * 等待更多训练（EMA更新很慢）
   
   - 如果训练不稳定：
     * 确认enable_simple_acn=True
     * 确认其他动态机制都已关闭

4. 对比实验：
   - Baseline: enable_simple_acn=False
   - +ACN: enable_simple_acn=True
   - 期望：+ACN的mAP提升0.3~0.6%，AP75提升更明显

5. 超参数调整（通常不需要）：
   - acn_momentum: 保持0.95（已优化）
   - acn_temperature: 保持1.0（已优化）
   - acn_min_samples: 可以在5-15之间调整
   - acn_high_iou_thresh: 可以在0.65-0.75之间调整
   - acn_low_iou_thresh: 可以在0.35-0.45之间调整
"""
