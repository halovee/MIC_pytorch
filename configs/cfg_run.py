
max_iters=40000

optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0), pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0))))
optimizer_config = None
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

cfg_run = dict(
    optimizer_config = optimizer_config,
    optimizer = optimizer,
    lr_config = lr_config,
    # Random Seed
    seed = 2,

    n_gpus = 2,
    gpu_model = 'NVIDIATITANRTX',
    runner = dict(type='IterBasedRunner', max_iters=max_iters),

    # Logging Configuration
    checkpoint_config = dict(by_epoch=False, interval=max_iters, max_keep_ckpts=1),
    evaluation = dict(by_epoch=False, interval=4000, metric='mIoU'),
    log_config=dict(by_epoch=False, interval=2),

    # yapf:enable
    dist_params = dict(backend='nccl'),
    log_level = 'INFO',
    load_from = None,
    resume_from = None,
    workflow = [('train', 1)],
    cudnn_benchmark = True
)

def get_cfg_run(name):
    return {
        'optimizer': optimizer,
        'lr_config': lr_config,
        'log_config': cfg_run['log_config'],
        'cfg_run': cfg_run,
        'max_iters': max_iters,
    }[name]

