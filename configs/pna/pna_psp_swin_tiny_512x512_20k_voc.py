_base_ = [ 
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'

model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2, 
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='SwinTransformer',
        pretrain_img_size=224, 
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True, 
        patch_size=4, 
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),  
        qkv_bias=True,
        qk_scale=None, 
        drop_rate=0.,
        attn_drop_rate=0., 
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=[ 
        dict(
            type='PSPHead',
            in_channels=768,
            in_index=-1,
            channels=512,
            pool_scales=(1, 2, 3, 6),  
            dropout_ratio=0.1,
            num_classes=21, 
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='PNAHead',
            in_index=[0, 1, 2, 3],
            in_channels=[96, 192, 384, 768],
            embed_dims=[96, 192, 384, 768],
            window_size=[9, 9, 9, 9], 
            out_index=[0, 1, 2, 3],
            mlp_ratio=1,
            channels=512,
            num_classes=21,
            align_corners=False,
            upsample_mode='nearest',
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1))
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
 
data = dict(samples_per_gpu=4)
 
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'rpb': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
 
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()