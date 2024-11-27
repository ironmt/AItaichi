# from datetime import datetime

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=15, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '/root/AItaichi/newTrain/train.pkl'
# ann_file = '/root/AItaichi/train/train_all/all_annotations.pkl' # for test

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=48, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_train'))

# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 200
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
# log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),  # 输出日志到控制台
        # dict(type='FileLoggerHook', by_epoch=True, file_name='loss_log.json'),  # 保存日志到文件
        # dict(type='TensorboardLoggerHook')  # 使用 TensorBoard 保存日志，方便可视化
    ]
)


# runtime settings
log_level = 'INFO'
# Get the current time in the format YYYYMMDD_HHMMSS
# current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define the work directory with the current time
work_dir = f'./work_dirs/stgcn++/stgcn++_ntu60_xsub_taichi_lr4e-1_clip48_full_vidos/j'