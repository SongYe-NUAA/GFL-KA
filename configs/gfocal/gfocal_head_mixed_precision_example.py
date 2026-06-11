# GFocal Head 混合精度训练配置示例

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 🚀 混合精度训练配置
# 使用AMP优化器包装器启用自动混合精度训练
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 自动混合精度优化器包装器
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    loss_scale='dynamic'  # 动态损失缩放
)

# 模型配置 - 使用GFocal Head替换检测头
model = dict(
    type='RetinaNet',
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
        add_extra_convs='on_input',
        num_outs=5
    ),
    bbox_head=dict(
        type='GFocalHead',  # 🚀 使用支持混合精度的GFocal Head
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        # 🚀 梯度一致性配置
        use_gradient_consistency=True,
        consistency_weight=0.3,
        # 损失函数配置
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25
        ),
        # 训练配置
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            initial_epoch=4  # 梯度一致性激活epoch
        ),
        # 测试配置
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100
        )
    ),
    # 训练和测试配置
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100
    )
)

# 🚀 数据配置 - 适配混合精度训练
train_dataloader = dict(
    batch_size=8,  # 混合精度可以使用更大的batch size
    num_workers=4,
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
    batch_size=8,
    num_workers=4,
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

# 🚀 学习率调度配置 - 考虑混合精度训练特点
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500  # 混合精度训练的warmup
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

# 🚀 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 🚀 默认钩子配置 - 适配混合精度训练
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 🚀 自定义钩子 - 混合精度训练监控
custom_hooks = [
    dict(
        type='ModelMonitorHook',  # 自定义钩子监控模型状态
        monitor_gradient_consistency=True,
        monitor_quality_scores=True,
        log_interval=100
    )
]

# 🚀 运行时配置
env_cfg = dict(
    cudnn_benchmark=True,  # 启用cudnn benchmark加速混合精度训练
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# 加载检查点配置
load_from = None
resume = False

# 🚀 混合精度训练使用说明
"""
使用此配置进行混合精度训练的方法：

1. 确保CUDA和PyTorch支持混合精度:
   - CUDA >= 10.1
   - PyTorch >= 1.6.0
   - Tensor Cores支持的GPU (如V100, RTX 2080 Ti, RTX 30系列等)

2. 启动训练:
   python tools/train.py configs/gfocal/gfocal_head_mixed_precision_example.py --amp

3. 或者直接使用此配置文件（已包含AmpOptimWrapper):
   python tools/train.py configs/gfocal/gfocal_head_mixed_precision_example.py

4. 混合精度训练优势:
   - 训练速度提升 1.5-2x
   - 显存使用减少 ~50%
   - 支持更大的batch size
   - 保持数值稳定性（通过FP32保护关键计算）

5. 监控指标:
   - 梯度一致性分数
   - 质量分数预测精度
   - 损失稳定性
   - 梯度范数

6. 注意事项:
   - 首次运行时可能需要编译CUDA kernels
   - 建议使用dynamic loss scaling
   - 监控loss是否有异常跳跃
   - 验证精度应与FP32训练相当
"""