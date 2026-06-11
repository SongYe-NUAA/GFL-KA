# 端到端自适应路由系统配置示例
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 模型配置
model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DistancePointBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        # --- 端到端自适应路由配置 ---
        use_adaptive_routing=True,  # 启用自适应路由
        adaptive_routing_weight=0.05,  # 自适应路由损失权重
        use_unified_expert_loss=True,  # 启用统一专家损失
        unified_expert_loss_weight=0.1,  # 统一专家损失权重
        # 专家配置
        expert1_hidden_dim=32,  # 简单专家隐藏维度
        expert3_hidden_dim=128,  # 复杂专家隐藏维度
        # 质量损失配置
        loss_quality_weight=0.02,  # 质量损失权重
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)))

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 数据加载器配置
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# 日志配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 自定义钩子：自适应路由监控
custom_hooks = [
    dict(
        type='AdaptiveRoutingMonitorHook',
        interval=100,
        priority='VERY_LOW'
    )
]

# 评估配置
val_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=_base_.val_dataloader.dataset.ann_file,
    metric='bbox')
test_evaluator = val_evaluator

# 随机种子
randomness = dict(seed=None, deterministic=False)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 日志级别
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 工作目录
work_dir = './work_dirs/adaptive_routing_example'

# 实验名称
exp_name = 'adaptive_routing_example' 