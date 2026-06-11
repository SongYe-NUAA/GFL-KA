_base_ = 'windturbine_yolox_s_8xb8-300e.py'

img_scale = (640, 640)  # width, height
class_name = (
    'background',  # 0: 添加背景类
    'leaf-opex',  # 0: 添加背景类
    'corrosion-pit',  # 1
    'corrosion',  # 2
    'degumming',  # 3
    'hole',  # 4
    'lightning-arrester',  # 6
    'teeth',  # 7
    'crack',  # 8
    'painting-peel-off',  # 0: 添加背景类
    'crack',  # 9
    'sign',  # 10
    'dirt',  # 11
    'oil',  # 12
)  # 根据 class_with_id.txt 类别信息，设置 class_name
metainfo = dict(classes = class_name,palette = [(220,20,60),(0,255,255),(255,0,0)])

save_epoch_intervals = 5
max_epochs = 1000
num_classes = len(metainfo['classes'])
train_data_prefix = 'images'  # Prefix of train image path
val_data_prefix = 'images'  # Prefix of val image path
data_root = r'./WTB5K/'  # Root path of data
dataset_type = 'CocoDataset'
backend_args = None

train_ann_file = 'v1/annotations/train.json'
test_ann_file= 'v1/annotations/test.json'
val_ann_file = 'v1/annotations/val.json'
work_dir = "./runs/windturbine_YOLOX"
# model settings
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=10)
    ]),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375,
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
                      prefix='backbone')),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96,num_classes=num_classes))

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # Resize and Pad are for the last 15 epochs when Mosaic and
    # RandomAffine are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo = metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo = metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    backend_args=backend_args,
    format_only = False,
    classwise = True,
)
test_evaluator = val_evaluator

#load_from = './load/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
