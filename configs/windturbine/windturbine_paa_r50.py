auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_name = (
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
)
data_root = '../WindBlade-30K/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/root/autodl-tmp/mmdetection-main/runs/sota/paa/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
metainfo = dict(
    classes=(
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
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            0,
            255,
            255,
        ),
        (
            255,
            0,
            0,
        ),
    ])
model = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.3, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=0.5, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=16,
        reg_decoded_bbox=True,
        score_voting=True,
        stacked_convs=4,
        topk=9,
        type='PAAHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.1,
            pos_iou_thr=0.1,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1),
    type='PAA')
num_classes = 19
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=None,
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            hybrid_linear_weight=dict(decay_mult=0.0, lr_mult=0.05))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
save_epoch_intervals = 1
test_ann_file = 'annotations/nobackground/test.json'
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_data_prefix = 'images'
test_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/nobackground/test.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='../WindBlade-30K/',
        metainfo=dict(
            classes=(
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
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='../WindBlade-30K/annotations/nobackground/test.json',
        backend_args=None,
        classwise=True,
        format_only=False,
        metric='bbox',
        type='CocoMetric'),
    dict(
        ann_file='../WindBlade-30K/annotations/nobackground/test.json',
        type='F1PrecisionRecallMetric'),
]
test_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_ann_file = 'annotations/nobackground/train.json'
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=12,
    type='EpochBasedTrainLoop',
    val_begin=5,
    val_interval=1)
train_data_prefix = 'images'
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/nobackground/train.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='../WindBlade-30K/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
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
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
val_ann_file = 'annotations/nobackground/val.json'
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_data_prefix = 'images'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/nobackground/val.json',
        backend_args=None,
        data_prefix=dict(img='images'),
        data_root='../WindBlade-30K/',
        metainfo=dict(
            classes=(
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
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='../WindBlade-30K/annotations/nobackground/val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './runs/sota/paa'
