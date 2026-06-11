"""
🚀 方案3：自适应调制机制配置示例

特征设计：
- 调制后Shape（4维）：shape × modulation
- 置信度信号（1维）：confidence
- 总维度：5维×4方向 = 20维

调制策略：
- 高置信度样本：调制系数接近1.5，放大shape差异
- 低置信度样本：调制系数接近0.5，缩小shape差异

预期效果：mAP +0.8~1.2（如果成功）
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

        # ====== 🚀 方案3：自适应调制机制 ======
        use_gflv2_features=False,
        use_shape_scale_separation=False,
        use_adaptive_modulation=True,  # ✅ 开启方案3
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
        # ====== 🚀 WIoU v3 损失（替代 GIoU）======
        # Wise-IoU v3: 动态非单调聚焦策略，当前最佳 bbox 损失
        # 核心思想：高 IoU 样本降低权重（避免过拟合），低 IoU 样本聚焦（梯度增强）
        # 与峰度注意力配合形成互补
        loss_bbox=dict(type='WiseIoULoss', version='v3', beta=1.0, loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25),

        # ====== 🚀 启用 Wise-IoU ======
        use_wise_iou=True,  # 开启 Wise-IoU 损失
        wise_iou_version='v3',  # 使用 WIoU v3（推荐）
        
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
# 方案3可能需要更保守的学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.05),  # 降低学习率
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
        eta_min=0.0001,  # 更低的最小学习率
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
work_dir = './work_dirs/rtmdet_tiny_gfocal_scheme3'

# ====== 方案3特别说明 ======
"""
⚠️ 注意事项：

1. 学习率调整：
   - 初始学习率从0.004降至0.003
   - 最小学习率从0.0002降至0.0001
   - 原因：自适应调制引入非线性，需要更稳定的训练

2. 监控指标：
   - 重点关注前10个epoch的loss稳定性
   - 如果出现NaN，进一步降低学习率

3. 调制参数：
   - 当前调制范围：[0.5, 1.5]
   - 如需调整，修改 gfocal_head.py 中的调制公式

4. 预期表现：
   - 初期可能略慢于方案1/2
   - 中期开始展现优势
   - 后期应该有明显提升

5. 故障排除：
   - loss爆炸 → 降低学习率至0.002
   - loss不降 → 检查数据加载
   - 过拟合 → 增加dropout或weight_decay
"""
