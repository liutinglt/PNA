_base_ = [ 
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
swin_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=[ 
         dict(
            type='ASPPHead',
            in_channels=2048,
            in_index=-1,
            channels=512,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='PNAHead',
            in_index=[0, 1, 2, 3],
            in_channels=[256, 512, 1024, 2048],
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
        in_channels=1024,
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
 
data = dict(samples_per_gpu=8)
 
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
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
    power=0.9,
    min_lr=0.0,
    by_epoch=False)


# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()