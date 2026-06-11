_base_ = ['mmdet::reppoints/reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco.py']

class_name =(
    'background',
    'hole',
    'leaf-opex',
    'corrosion',
    'stain',
    'corrosion-pit',
    'lightning-arrester-miss',
    'degumming',
    'repair',
    'lightning-arrester',
    'teeth',
    'demould',
    'painting-peel-off',
    'sign',
    'crack',
    'dirt',
    'swell',
    'oil',
)
metainfo = dict(
    classes=class_name
)

# 数据集设置
dataset_type = 'CocoDataset'
data_root = '../WindBlade-30K/'
train_ann_file = 'annotations/train.json'
train_data_prefix = 'images'
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images'
test_ann_file = 'annotations/test.json'
work_dir = "./runs/windturbine_REPPOINTS"
save_epoch_intervals = 3
max_epochs = 36
# 添加设备设置
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# 添加顶层训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    val_begin=5
)
model = dict(
    type='RepPointsDetector',
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
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=len(class_name),
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'),
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
# 验证和测试配置
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 数据加载器设置
# 修改数据加载器设置
# 修改数据加载器设置
train_dataloader = dict(
    batch_size=8,  # 减小批次大小
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
    )
)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True))

test_dataloader = val_dataloader

# 评估器设置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False,
    classwise=True)

test_evaluator = val_evaluator

# 检查点设置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # 每隔多少个epoch保存一次
        max_keep_ckpts=3,  # 最多保存几个
        save_best='coco/bbox_mAP',  # 保存验证集上最好的模型
        rule='greater',  # 'greater' 表示值越大越好
        save_last=True,  # 保存最后一个epoch的模型
     ),
    sampler_seed = dict(type='DistSamplerSeedHook'),
)

# 添加参数设置
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
