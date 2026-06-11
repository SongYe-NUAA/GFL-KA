"""
🔬 方案2：显式Shape-Scale分离配置示例

特征设计：
- Shape路径（4维）：标准化去中心化TOP4
- Scale路径（4维）：mean + max + std + entropy
- 总维度：8维×4方向 = 32维

预期效果：mAP +0.5~0.8
"""

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py'
]

# ====== 模型配置 ======
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        
        # ====== 🔬 方案2：显式Shape-Scale分离 ======
        use_gflv2_features=False,
        use_shape_scale_separation=True,  # ✅ 开启方案2
        use_adaptive_modulation=False,
        add_mean=True,
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        
        # ====== 损失函数配置 ======
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25),
        
        # ====== 训练配置 ======
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            initial_epoch=4)),
    
    # ====== 测试配置 ======
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# ====== 训练配置 ======
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10)

# ====== 优化器配置 ======
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

# ====== 学习率配置 ======
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=1,
        T_max=299,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]

# ====== 运行时配置 ======
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# ====== 日志配置 ======
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# ====== 工作目录 ======
work_dir = './work_dirs/rtmdet_tiny_gfocal_scheme2'
