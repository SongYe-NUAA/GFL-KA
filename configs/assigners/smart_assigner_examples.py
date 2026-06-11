"""
🚀 智能分配器配置示例与使用指南

本文件提供了三种智能分配器的完整配置示例：
1. 改进版质量感知ATSS分配器 (推荐用于生产环境)
2. 元学习自适应分配器 (研究用途)
3. 智能ATSS分配器 (轻量级高效版本)

选择建议：
- 🎯 追求稳定性能：使用改进版质量感知ATSS
- 🧠 需要极致自适应：使用元学习自适应分配器  
- ⚡ 注重计算效率：使用智能ATSS分配器
"""

# ================================
# 🚀 配置1：改进版质量感知ATSS分配器
# ================================
improved_quality_aware_config = dict(
    # 基础模型配置
    model=dict(
        type='GFL',
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
            add_extra_convs='on_output',
            num_outs=5
        ),
        bbox_head=dict(
            type='GFocalHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]
            ),
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
            reg_max=16,
            # 🚀 改进版质量感知分配器配置
            train_cfg=dict(
                assigner=dict(
                    type='QualityAwareATSSAssigner',
                    topk=9,
                    # 🎯 保守的质量感知参数
                    quality_weight=0.2,  # 降低质量权重，更稳定
                    quality_thr=0.3,
                    adaptive_fusion=True,
                    # 📊 改进的自适应参数
                    adaptation_rate=0.005,  # 更慢的适应速度
                    adaptive_modulation_factor=0.1,  # 保守的调制因子
                    weight_bounds=(0.05, 0.4),  # 严格的权重边界
                    # 🛡️ 稳定性参数
                    performance_window=100,
                    min_pos_samples=3,
                    iou_calculator=dict(type='BboxOverlaps2D'),
                    ignore_iof_thr=-1
                ),
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
    ),
    
    # 训练配置
    train_cfg=dict(
        initial_epoch=12,  # 质量感知启动epoch
        warmup_epochs=5    # 预热期
    ),
    
    # 优化器配置
    optimizer=dict(
        type='SGD', 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=0.0001
    ),
    
    # 学习率调度
    lr_config=dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[8, 11]
    )
)

# ================================
# 🧠 配置2：元学习自适应分配器
# ================================
meta_learning_config = dict(
    # 基础模型配置 (同上，只替换分配器)
    model=dict(
        # ... (其他配置同上)
        bbox_head=dict(
            # ... (其他配置同上)
            train_cfg=dict(
                assigner=dict(
                    type='MetaAdaptiveAssigner',
                    topk=9,
                    # 🧠 元学习参数
                    enable_meta_learning=True,
                    meta_learning_lr=0.001,
                    experience_buffer_size=1000,
                    update_frequency=10,
                    # 🎯 多策略配置
                    base_strategies=['atss', 'quality_aware', 'distance_based', 'confidence_guided'],
                    strategy_weights=[0.4, 0.3, 0.2, 0.1],
                    # 📊 性能监控
                    performance_metrics=['iou', 'precision', 'recall'],
                    adaptation_threshold=0.05,
                    iou_calculator=dict(type='BboxOverlaps2D'),
                    ignore_iof_thr=-1
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            )
        )
    ),
    
    # 🧠 元学习专用训练配置
    train_cfg=dict(
        meta_learning_warmup=20,  # 元学习预热迭代
        strategy_update_interval=50,  # 策略更新间隔
        experience_replay_batch=32   # 经验回放批次大小
    )
)

# ================================
# ⚡ 配置3：智能ATSS分配器 (推荐)
# ================================
smart_atss_config = dict(
    # 基础模型配置 (同上，只替换分配器)
    model=dict(
        # ... (其他配置同上)
        bbox_head=dict(
            # ... (其他配置同上)
            train_cfg=dict(
                assigner=dict(
                    type='SmartATSSAssigner',
                    topk=9,
                    # 🎯 场景自适应参数
                    enable_scene_adaptation=True,
                    enable_online_learning=True,
                    # 📊 基础权重配置
                    base_iou_weight=0.6,
                    base_quality_weight=0.25,
                    base_distance_weight=0.15,
                    # 🔄 在线学习参数
                    learning_rate=0.01,
                    performance_window=50,
                    adaptation_threshold=0.02,
                    # 🛡️ 稳定性保障
                    min_quality_weight=0.05,
                    max_quality_weight=0.45,
                    warmup_iterations=10,
                    # 基础参数
                    iou_calculator=dict(type='BboxOverlaps2D'),
                    ignore_iof_thr=-1
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            )
        )
    ),
    
    # ⚡ 高效训练配置
    train_cfg=dict(
        scene_analysis_interval=1,  # 场景分析频率
        parameter_update_interval=5,  # 参数更新频率
        performance_logging=True     # 性能日志记录
    )
)

# ================================
# 📊 性能对比与选择建议
# ================================
performance_comparison = {
    'QualityAwareATSSAssigner': {
        'stability': '⭐⭐⭐⭐⭐',  # 非常稳定
        'adaptability': '⭐⭐⭐',     # 中等自适应
        'efficiency': '⭐⭐⭐⭐',     # 高效
        'complexity': '⭐⭐⭐',       # 中等复杂度
        'best_for': '生产环境，追求稳定性能',
        'recommended_scenarios': [
            '工业检测',
            '医疗影像',
            '安防监控',
            '自动驾驶'
        ]
    },
    
    'MetaAdaptiveAssigner': {
        'stability': '⭐⭐⭐',       # 中等稳定
        'adaptability': '⭐⭐⭐⭐⭐', # 极强自适应
        'efficiency': '⭐⭐',        # 较低效率
        'complexity': '⭐⭐⭐⭐⭐',   # 高复杂度
        'best_for': '研究实验，追求极致性能',
        'recommended_scenarios': [
            '学术研究',
            '算法竞赛',
            '复杂多变场景',
            '新领域探索'
        ]
    },
    
    'SmartATSSAssigner': {
        'stability': '⭐⭐⭐⭐',     # 高稳定性
        'adaptability': '⭐⭐⭐⭐',   # 强自适应
        'efficiency': '⭐⭐⭐⭐⭐',   # 极高效率
        'complexity': '⭐⭐',        # 低复杂度
        'best_for': '实际应用，平衡性能与效率',
        'recommended_scenarios': [
            '移动端部署',
            '实时检测',
            '资源受限环境',
            '大规模应用'
        ]
    }
}

# ================================
# 🛠️ 使用指南
# ================================
usage_guide = """
🚀 智能分配器使用指南

1. 📦 安装配置
   - 将分配器文件放入 mmdet/models/task_modules/assigners/
   - 在 __init__.py 中注册新的分配器
   - 修改配置文件使用对应的分配器

2. 🎯 参数调优建议
   
   对于 QualityAwareATSSAssigner：
   - quality_weight: 0.1-0.3 (保守范围)
   - adaptive_modulation_factor: 0.05-0.15
   - adaptation_rate: 0.001-0.01
   
   对于 SmartATSSAssigner：
   - base_quality_weight: 0.2-0.3
   - learning_rate: 0.005-0.02
   - warmup_iterations: 5-20

3. 📊 性能监控
   - 使用 get_current_status() 查看实时状态
   - 监控权重变化趋势
   - 关注性能历史曲线

4. 🐛 调试技巧
   - 开启 debug=True 获取详细日志
   - 使用性能可视化工具
   - 分析不同场景下的参数变化

5. 🔧 常见问题
   Q: 训练初期不稳定？
   A: 增加 warmup_iterations，降低 learning_rate
   
   Q: 自适应效果不明显？
   A: 检查 adaptation_threshold 设置，确保有足够的性能变化
   
   Q: 内存占用过高？
   A: 减小 performance_window 和 experience_buffer_size
"""

# ================================
# 🎯 完整配置示例 (推荐使用)
# ================================
recommended_config = dict(
    # 数据配置
    dataset_type='CocoDataset',
    data_root='data/coco/',
    
    # 模型配置 (使用智能ATSS分配器)
    **smart_atss_config,
    
    # 数据管道
    train_pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ],
    
    # 训练数据
    data=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline
        )
    ),
    
    # 运行时配置
    runner=dict(type='EpochBasedRunner', max_epochs=12),
    evaluation=dict(interval=1, metric='bbox'),
    checkpoint_config=dict(interval=1),
    log_config=dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ]
    ),
    
    # 工作目录
    work_dir='./work_dirs/smart_atss_r50_fpn_1x_coco'
)

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)
"""