_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    # 主干网络(backbone)配置
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,# ResNet分为4个阶段
        # stage 1: conv2_x
        # stage 2: conv3_x
        # stage 3: conv4_x
        # stage 4: conv5_x
        out_indices=(0, 1, 2, 3),# 输出4个阶段的特征图
        # 0: 对应conv2_x输出 (C2)
        # 1: 对应conv3_x输出 (C3)
        # 2: 对应conv4_x输出 (C4)
        # 3: 对应conv5_x输出 (C5)
        frozen_stages=1,# 冻结第一阶段的参数
        # -1: 不冻结任何阶段
        # 0: 仅冻结第一个卷积层
        # 1: 冻结第一阶段（conv2_x）
        # 2: 冻结前两个阶段
        # 3: 冻结前三个阶段
        # 4: 冻结所有阶段
        norm_cfg=dict(type='BN', # 使用BatchNormalization
                      requires_grad=True),# BN层参数可训练

        norm_eval=True, # 评估时固定BN统计量
        # True: 测试时BN层统计量固定
        # False: 测试时BN层统计量更新
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),# 使用PyTorch风格的ResNet实现
    # 'pytorch': 使用PyTorch风格的卷积下采样
    # 'caffe': 使用Caffe风格的卷积下采样
    # 特征金字塔网络(FPN)配置
    neck=dict(
        type='FPN',  # 使用特征金字塔网络作为neck
        # ResNet50的各阶段输出通道数
        in_channels=[256, 512, 1024, 2048],
        # - 256: C2层输出
        # - 512: C3层输出
        # - 1024: C4层输出
        # - 2048: C5层输出
        out_channels=256, # FPN所有层级的输出通道数统一为256
        start_level=1, # 从第2层开始构建FPN
        # - 0 对应 C2
        # - 1 对应 C3（开始层）
        # - 2 对应 C4
        # - 3 对应 C5
        add_extra_convs='on_output',# 额外卷积层的添加方式
        # - 'on_input': 在输入特征上添加
        # - 'on_output': 在输出特征上添加
        # - 'on_lateral': 在侧边特征上添加
        # - False: 不添加额外卷积
        num_outs=5), # 输出5个尺度的特征图,由自己指定
    # 检测头配置
    bbox_head=dict(
        type='GFocalHead',# 检测头类型
        num_classes=80,# 类别数量（COCO数据集80类）
        in_channels=256,# 输入特征图的通道数（与FPN输出一致）
        stacked_convs=4, # 堆叠的卷积层数量
        feat_channels=256,# 中间特征层的通道数
        # 锚框生成器配置
        anchor_generator=dict(
            type='AnchorGenerator',# 锚框生成器类型
            ratios=[1.0],# 锚框的宽高比，这里只用正方形
            octave_base_scale=8,# 基础尺度，决定锚框大小
            scales_per_octave=1,# 每个特征层的尺度数
            strides=[8, 16, 32, 64, 128]),# 各特征层的步长，决定锚框的密度
        # 分类损失配置
        loss_cls=dict(
            type='QualityFocalLoss',# 质量感知的Focal Loss
            use_sigmoid=False,# 是否使用sigmoid激活
            beta=2.0,# 调节难易样本权重的系数
            loss_weight=1.0),# 分类损失的权重
        # 分布式焦点损失配置
        loss_dfl=dict(type='DistributionFocalLoss'# 用于回归的分布式焦点损失
                      , loss_weight=0.25),# DFL损失的权重
        # 回归相关配置
        reg_max=16, # 回归编码的最大值，影响回归精度
        reg_topk=4,# 选择置信度最高的前k个预测
        reg_channels=64,# 回归分支的通道数
        add_mean=True, # 是否在回归预测中添加均值
        # 边界框回归损失
        loss_bbox=dict(type='GIoULoss',# 使用GIoU作为边界框回归损失
                       loss_weight=2.0)) ,
    train_cfg = dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
# 测试配置
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# 优化器配置
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


# learning policy
lr_config = dict(step=[8, 11])
total_epochs = 12
# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='PackDetInputs')
]
data = dict(train=dict(pipeline=train_pipeline))


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
