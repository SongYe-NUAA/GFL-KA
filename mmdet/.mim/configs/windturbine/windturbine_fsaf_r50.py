_base_ = ['mmdet::fsaf/fsaf_r50_fpn_1x_coco.py']

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
data_root = './WindBlade-30K/'
train_ann_file = 'annotations/train.json'
train_data_prefix = 'images'
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images'
test_ann_file = 'annotations/test.json'
work_dir = "./runs/windturbine_FSAF"
save_epoch_intervals = 5
max_epochs = 500

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
    bbox_head=dict(num_classes=len(class_name)))

# 数据加载器设置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
