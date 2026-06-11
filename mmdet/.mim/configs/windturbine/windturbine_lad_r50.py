_base_ = ['mmdet::lad/lad_r50-paa-r101_fpn_2xb8_coco_1x.py']

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
work_dir = "./runs/windturbine_LAD"
save_epoch_intervals = 3
max_epochs = 36

# 添加顶层训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 使用基于 epoch 训练循环
    max_epochs=max_epochs,  # 使用之前定义的 max_epochs
    val_interval=save_epoch_intervals,  # 验证间隔
    val_begin=5)  # 从第5个epoch开始验证

# 验证和测试配置
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    type='LAD',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    # student
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
        type='LADHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=18,
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
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    # teacher

    teacher_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    teacher_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    teacher_bbox_head=dict(
        type='LADHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=18,
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
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        score_voting=True,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.01))

# 数据加载器设置
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

# 验证和测试数据加载器配置
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,))

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
     ))

# # SGD 配置
# optim_wrapper = dict(
#     optimizer=dict(lr=0.01),
#     paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
#     clip_grad=None)
#
# # 简化学习率调度
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
# # 添加数据归一化和增强
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='RandomResize',
#         scale=(512, 512),
#         ratio_range=(0.8, 1.2),  # 更温和的尺度增强
#         keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,  # 更温和的光度增强
#         contrast_range=(0.8, 1.2),
#         saturation_range=(0.8, 1.2),
#         hue_delta=10),
#     dict(type='PackDetInputs')
# ]
