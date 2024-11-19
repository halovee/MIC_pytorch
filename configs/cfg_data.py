from data.pipelines import *

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)

gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2560, 1440)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2560, 1520)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # MultiScaleFlipAug is disabled by not providing img_ratios and setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
cityscapes_data_root = '/media/BD_4t/zjh_all/zjhdata/iss/cityscapes'
data = dict(
    samples_per_gpu=2,    # batch size of each GPU.
    workers_per_gpu=4,    # How many subprocesses to use for data loading for each GPU.

    train=dict(
        type='UDADataset',
        rare_class_sampling = dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        sync_crop_size = None,
        source_gta=dict(
            type='GTADataset',
            data_root='/media/BD_4t/zjh_all/zjhdata/iss/gta',
            img_dir='images',
            ann_dir='labels',
            pipeline=gta_train_pipeline),
        source_syn=dict(
            type='SynthiaDataset',
            data_root='/media/BD_4t/zjh_all/zjhdata/iss/synthia',
            img_dir='RGB',
            ann_dir='GT/LABELS',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root=cityscapes_data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline,
            crop_pseudo_margins=[30, 240, 30, 30])),

    val=dict(
        type='CityscapesDataset',
        data_root=cityscapes_data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),

    test=dict(
        type='CityscapesDataset',
        data_root=cityscapes_data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline)
)


def get_cfg_data(name):
    return {
        'data': data,
        'gta_train_pipeline': gta_train_pipeline,
        'city_train_pipeline': cityscapes_train_pipeline,
        'syn_train_pipeline': synthia_train_pipeline,
        'img_norm_cfg': img_norm_cfg,
        'test_pipeline': test_pipeline,

    }[name]





# gta_train_pipeline = [
#     LoadImageFromFile(),
#     LoadAnnotations(),
#     Resize(img_scale=(2560, 1440)),
#     RandomCrop(crop_size=crop_size, cat_max_ratio=0.75),
#     RandomFlip(prob=0.5),
#     # PhotoMetricDistortion()
#     Normalize(**img_norm_cfg),
#     Pad(size=crop_size, pad_val=0, seg_pad_val=255),
#     DefaultFormatBundle(),
#     Collect(keys=['img', 'gt_semantic_seg']),
# ]
# cityscapes_train_pipeline = [
#     LoadImageFromFile(),
#     LoadAnnotations(),
#     Resize(img_scale=(2048, 1024)),
#     RandomCrop(crop_size=crop_size),
#     RandomFlip(prob=0.5),
#     # PhotoMetricDistortion()
#     Normalize(**img_norm_cfg),
#     Pad(size=crop_size, pad_val=0, seg_pad_val=255),
#     DefaultFormatBundle(),
#     Collect(keys=['img', 'gt_semantic_seg', 'valid_pseudo_mask']),
# ]
# synthia_train_pipeline = [
#     LoadImageFromFile(),
#     LoadAnnotations(),
#     Resize(img_scale=(2560, 1520)),
#     RandomCrop(crop_size=crop_size, cat_max_ratio=0.75),
#     RandomFlip(prob=0.5),
#     # PhotoMetricDistortion()
#     Normalize(**img_norm_cfg),
#     Pad(size=crop_size, pad_val=0, seg_pad_val=255),
#     DefaultFormatBundle(),
#     Collect(keys=['img', 'gt_semantic_seg']),
# ]
# test_pipeline = [
#     LoadImageFromFile(),
#     MultiScaleFlipAug(
#         img_scale=(2048, 1024),
#         flip=False,
#         transforms = [
#             Resize(keep_ratio=True),
#             RandomFlip(),
#             Normalize(**img_norm_cfg),
#             ImageToTensor(keys=['img']),
#             Collect(keys=['img'])
#         ]
#     )
# ]

