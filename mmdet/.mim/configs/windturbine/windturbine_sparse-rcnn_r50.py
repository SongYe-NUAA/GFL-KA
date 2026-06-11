_base_ = ['mmdet::sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py']

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
work_dir = "./runs/windturbine_SPARSE-RCNN"
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

num_stages = 6
num_proposals = 50
model = dict(
    type='SparseRCNN',
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
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=len(class_name)-1,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=1024,
                in_channels=256,
                dropout=0.1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# 修改训练数据加载器配置
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=(800, 1333),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0))

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

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
