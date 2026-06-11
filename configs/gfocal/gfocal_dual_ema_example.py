"""
🚀 GFocal Head 双EMA系统配置示例

双EMA系统：快速EMA + 慢速EMA 自适应切换
- 快速EMA (α=0.7): 快速捕捉短期分布变化
- 慢速EMA (α=0.95): 保持长期稳定基准
- 自适应策略: 根据concentration方差自动选择
  - 方差小 (<0.3) → 分布稳定 → 使用慢速EMA
  - 方差大 (≥0.3) → 分布波动 → 混合快慢EMA (70%慢+30%快)

预期效果：
✅ 训练初期：快速适应，加速收敛 (+10%)
✅ 训练后期：保持稳定，避免震荡 (+5%)
✅ 验证阶段：更好的泛化性能
"""

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 双EMA系统配置（实验性功能）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 【核心功能开关】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        use_adaptive_modulation=True,  # ✅ 启用自适应调制机制
        use_top34_combined=False,  # ❌ 不合并Top3+Top4（保持20维特征）
        use_harmonic_lqe_weight=True,  # ✅ 启用调和平均权重融合
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🚀 【双EMA系统配置】- 核心改进
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_dual_ema=True,  # ✅ 启用双EMA系统
        dual_ema_fast_alpha=0.7,  # 快速EMA系数 (0.7 = 30%新值, 70%旧值)
        dual_ema_slow_alpha=0.95,  # 慢速EMA系数 (0.95 = 5%新值, 95%旧值)
        dual_ema_variance_threshold=0.3,  # 方差阈值 (低于0.3使用慢速EMA)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⚙️ 【系统参数】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        concentration_reference_quantile=0.75,  # P75分位数
        
        # 🎓 验证时策略
        enable_per_sample_adaptation=False,  # ❌ 不启用Per-Sample微调
        enable_validation_ema_learning=True,  # ✅ 启用验证时EMA学习
        validation_ema_learning_alpha=0.15,  # 验证学习速度 (15%新值)
        
        # 🌟 训练状态感知
        enable_training_state_feature=True,  # ✅ 添加EMA统计特征到LQE输入
        training_state_source='ema',  # 使用EMA统计
        
        # 🎯 特征纠正
        enable_feature_rectification=True,  # ✅ 启用特征门控纠正
        rectification_sensitivity=0.15,  # 门控敏感度
        rectification_min_gate=0.1,  # 最小门控值
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎛️ 【日志开关】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_ema_validation_log=True,  # ✅ 验证EMA监控日志
        enable_ema_layer_comparison_log=True,  # ✅ 所有层EMA对比日志
        enable_positive_sample_log=True,  # ✅ 正样本监控日志
        enable_discrimination_score_log=True,  # ✅ 区分性评分日志
        enable_correlation_analysis_log=False,  # ❌ 相关性分析（开销大）
        enable_topk_iou_statistics_log=False,  # ❌ Top-K统计记录
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 📦 其他标准配置
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),

        # ====== 🚀 WIoU v3 损失（替代 GIoU）======
        loss_bbox=dict(type='WiseIoULoss', version='v3', beta=1.0, loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25),

        # ====== 🚀 启用 Wise-IoU ======
        use_wise_iou=True,
        use_adaptive_modulation=True,
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎯 训练配置
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    train_cfg=dict(
        initial_epoch=4,  # QFL开始生效的epoch
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 优化器配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 学习率调度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 训练配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 💡 使用建议
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
📈 监控指标：

1. 训练日志关注：
   - 快速EMA vs 慢速EMA的差异（训练初期差异大，后期收敛）
   - 方差变化趋势（是否逐渐稳定）
   - 使用策略分布（slow/mixed的比例）

2. 预期现象：
   Epoch 1-3:  方差大 (0.4-0.6) → 多使用mixed策略 → 快速适应
   Epoch 4-8:  方差中 (0.25-0.35) → slow/mixed混合 → 平衡过渡
   Epoch 9-12: 方差小 (<0.25) → 多使用slow策略 → 稳定优化

3. 对比基线：
   - 单一EMA (α=0.9): 作为对照组
   - 双EMA系统: 预期在训练初期收敛更快，后期更稳定

🔧 参数调优建议：

如果训练震荡：
- 降低fast_alpha (0.7 → 0.75)
- 提高slow_alpha (0.95 → 0.97)
- 提高variance_threshold (0.3 → 0.35)

如果收敛太慢：
- 降低fast_alpha (0.7 → 0.65)
- 降低slow_alpha (0.95 → 0.93)
- 降低variance_threshold (0.3 → 0.25)

🚀 进阶实验：

1. 层级感知双EMA：
   layer_fast_alpha = [0.6, 0.65, 0.7, 0.75, 0.8]  # P3-P7
   layer_slow_alpha = [0.93, 0.94, 0.95, 0.96, 0.97]  # P3-P7

2. 动态阈值：
   variance_threshold = 0.3 × (1 - epoch/max_epoch)  # 随训练降低

3. 三EMA系统：
   ultra_fast (α=0.5) + fast (α=0.7) + slow (α=0.95)
"""

🚀 GFocal Head 双EMA系统配置示例

双EMA系统：快速EMA + 慢速EMA 自适应切换
- 快速EMA (α=0.7): 快速捕捉短期分布变化
- 慢速EMA (α=0.95): 保持长期稳定基准
- 自适应策略: 根据concentration方差自动选择
  - 方差小 (<0.3) → 分布稳定 → 使用慢速EMA
  - 方差大 (≥0.3) → 分布波动 → 混合快慢EMA (70%慢+30%快)

预期效果：
✅ 训练初期：快速适应，加速收敛 (+10%)
✅ 训练后期：保持稳定，避免震荡 (+5%)
✅ 验证阶段：更好的泛化性能
"""

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 双EMA系统配置（实验性功能）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 【核心功能开关】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        use_adaptive_modulation=True,  # ✅ 启用自适应调制机制
        use_top34_combined=False,  # ❌ 不合并Top3+Top4（保持20维特征）
        use_harmonic_lqe_weight=True,  # ✅ 启用调和平均权重融合
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🚀 【双EMA系统配置】- 核心改进
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_dual_ema=True,  # ✅ 启用双EMA系统
        dual_ema_fast_alpha=0.7,  # 快速EMA系数 (0.7 = 30%新值, 70%旧值)
        dual_ema_slow_alpha=0.95,  # 慢速EMA系数 (0.95 = 5%新值, 95%旧值)
        dual_ema_variance_threshold=0.3,  # 方差阈值 (低于0.3使用慢速EMA)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ⚙️ 【系统参数】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        concentration_reference_quantile=0.75,  # P75分位数
        
        # 🎓 验证时策略
        enable_per_sample_adaptation=False,  # ❌ 不启用Per-Sample微调
        enable_validation_ema_learning=True,  # ✅ 启用验证时EMA学习
        validation_ema_learning_alpha=0.15,  # 验证学习速度 (15%新值)
        
        # 🌟 训练状态感知
        enable_training_state_feature=True,  # ✅ 添加EMA统计特征到LQE输入
        training_state_source='ema',  # 使用EMA统计
        
        # 🎯 特征纠正
        enable_feature_rectification=True,  # ✅ 启用特征门控纠正
        rectification_sensitivity=0.15,  # 门控敏感度
        rectification_min_gate=0.1,  # 最小门控值
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎛️ 【日志开关】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        enable_ema_validation_log=True,  # ✅ 验证EMA监控日志
        enable_ema_layer_comparison_log=True,  # ✅ 所有层EMA对比日志
        enable_positive_sample_log=True,  # ✅ 正样本监控日志
        enable_discrimination_score_log=True,  # ✅ 区分性评分日志
        enable_correlation_analysis_log=False,  # ❌ 相关性分析（开销大）
        enable_topk_iou_statistics_log=False,  # ❌ Top-K统计记录
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 📦 其他标准配置
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),

        # ====== 🚀 WIoU v3 损失（替代 GIoU）======
        loss_bbox=dict(type='WiseIoULoss', version='v3', beta=1.0, loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25),

        # ====== 🚀 启用 Wise-IoU ======
        use_wise_iou=True,
        use_adaptive_modulation=True,
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎯 训练配置
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    train_cfg=dict(
        initial_epoch=4,  # QFL开始生效的epoch
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 优化器配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 学习率调度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 训练配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 💡 使用建议
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
📈 监控指标：

1. 训练日志关注：
   - 快速EMA vs 慢速EMA的差异（训练初期差异大，后期收敛）
   - 方差变化趋势（是否逐渐稳定）
   - 使用策略分布（slow/mixed的比例）

2. 预期现象：
   Epoch 1-3:  方差大 (0.4-0.6) → 多使用mixed策略 → 快速适应
   Epoch 4-8:  方差中 (0.25-0.35) → slow/mixed混合 → 平衡过渡
   Epoch 9-12: 方差小 (<0.25) → 多使用slow策略 → 稳定优化

3. 对比基线：
   - 单一EMA (α=0.9): 作为对照组
   - 双EMA系统: 预期在训练初期收敛更快，后期更稳定

🔧 参数调优建议：

如果训练震荡：
- 降低fast_alpha (0.7 → 0.75)
- 提高slow_alpha (0.95 → 0.97)
- 提高variance_threshold (0.3 → 0.35)

如果收敛太慢：
- 降低fast_alpha (0.7 → 0.65)
- 降低slow_alpha (0.95 → 0.93)
- 降低variance_threshold (0.3 → 0.25)

🚀 进阶实验：

1. 层级感知双EMA：
   layer_fast_alpha = [0.6, 0.65, 0.7, 0.75, 0.8]  # P3-P7
   layer_slow_alpha = [0.93, 0.94, 0.95, 0.96, 0.97]  # P3-P7

2. 动态阈值：
   variance_threshold = 0.3 × (1 - epoch/max_epoch)  # 随训练降低

3. 三EMA系统：
   ultra_fast (α=0.5) + fast (α=0.7) + slow (α=0.95)
"""

 