act_cfg=dict(type='ReLU')
norm_cfg = dict(type='BN', requires_grad=True)

scales = [0.5, 1]
hr_crop_size = (512, 512)
hr_slide_inference = True

model = dict(
    type='HRDAEncoderDecoder',
    scales=scales,
    hr_crop_size=hr_crop_size,
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=hr_slide_inference,

    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch', pretrained='pretrained/mit_b5.pth'),

    decode_head=dict(
        # 原decode_head中的内容
        align_corners=False,
        attention_classwise=True,
        channels=256,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg),),
        dropout_ratio=0.1,
        hr_loss_weight=0.1,
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        norm_cfg=norm_cfg,
        num_classes=19,
        single_scale_head='DAFormerHead',
        type='HRDAHead',

        # hedaencoderdecoder 中 decode_head的添加
        scales=sorted(scales),
        enable_hr_crop= hr_crop_size is not None,
        hr_slide_inference = hr_slide_inference,
        # hedaencoderdecoder 中 kwargs的添加
        init_cfg=None,
        input_transform = 'multiple_select',
        # BaseDecodeHead 中专属
        ignore_index=255,
        sampler=None,
        conv_cfg=None,
        act_cfg=act_cfg,
    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', batched_slide=True, stride=[512, 512], crop_size=[1024, 1024])
)

head_cfg = dict(
    # hedaencoderdecoder 中 decode_head添加
    scales=sorted(scales),
    enable_hr_crop= hr_crop_size is not None,
    hr_slide_inference = hr_slide_inference,

    # hrdahead 中 head_cfg
    align_corners=False,
    channels=256,
    decoder_params=dict(
        embed_dims=256,
        embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        fusion_cfg=dict(
            # _delete_=True,
            type='aspp',
            sep=True,
            dilations=(1, 6, 12, 18),
            pool=False,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)),
    dropout_ratio=0.1,
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    loss_decode=dict(type='CrossEntropyLoss',use_sigmoid=False, loss_weight=1.0),
    norm_cfg=dict(type='BN', requires_grad=True),
    num_classes=19,

    # hrdahead 中 添加kwargs
    input_transform='multiple_select',
    init_cfg=None,
    # BaseDecodeHead 中专属
    ignore_index=255,
    sampler=None,
    conv_cfg=None,
    act_cfg=act_cfg,

)

attention_embed_dim = 256
attn_cfg = dict(
    # hedaencoderdecoder 中 decode_head添加
    scales=sorted(scales),
    enable_hr_crop= hr_crop_size is not None,
    hr_slide_inference = hr_slide_inference,

    # hrdahead 中 head_cfg
    align_corners=False,
    channels=attention_embed_dim,
    decoder_params=dict(
        embed_dims=attention_embed_dim,
        embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
        fusion_cfg=dict(  # 与上面head_cfg的不同
            type='conv',
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)),
    dropout_ratio=0.1,
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    norm_cfg=dict(type='BN', requires_grad=True),
    num_classes=19,

    # hrdahead 中 添加kwargs
    input_transform='multiple_select',
    init_cfg=None,
    # BaseDecodeHead 中专属
    ignore_index=255,
    sampler=None,
    conv_cfg=None,
    act_cfg=act_cfg,
)
# Baseline UDA
uda = dict(
    type='DACS',
    source_only=False,
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(type='block', mask_ratio=0.7, mask_block_size=64),
    # debug_img_interval=1000,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True

all = dict(
    model=model,
    head=head_cfg,
    attn=attn_cfg,
    uda=uda,
    norm_cfg=norm_cfg
)

def get_cfg_model(name):
    return {
        'model': model,
        'backbone': model['backbone'],
        'head': head_cfg,
        'attn': attn_cfg,
        'uda': uda,
        'norm_cfg': norm_cfg,
        'all': all
    }[name]