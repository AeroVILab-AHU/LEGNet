_base_ = ['./roi_trans_kfiou_ln_r50_fpn_1x_dota_le90.py']

pretrained = '/data4/lw/UnravelNet/det/backbone_weights/pkinet_s_pretrain.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='PKINet',
        arch='S',
        drop_path_rate=0.1,
        out_indices=(1, 2, 3, 4),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=pretrained),
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        add_extra_convs='on_input',
        num_outs=5))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00015,
    betas=(0.9, 0.999),
    weight_decay=0.05)

evaluation = dict(interval=1, metric='mAP', save_best='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

#optimizer = dict(
#    _delete_=True,
#    type='AdamW',
#    lr=0.0002,
#    betas=(0.9, 0.999),
#    weight_decay=0.05,
#    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
#)
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=1.0 / 3,
#    step=[16, 22, 27])

# evaluation
# evaluation = dict(interval=1, metric='mAP', save_best='mAP')
# runner = dict(type='EpochBasedRunner', max_epochs=30)
# checkpoint_config = dict(interval=1, max_keep_ckpts=10)

      
