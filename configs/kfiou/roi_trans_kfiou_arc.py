_base_ = ['./roi_trans_kfiou_ln_r50_fpn_1x_dota_le90.py']


model = dict(
    backbone=dict(
        _delete_=True,
        type='ARCResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        replace=[['x'], ['0', '1', '2', '3'], ['0', '1', '2', '3', '4', '5'],
                 ['0', '1', '2']],
        kernel_number=4,
        pretrained="/data4/lw/UnravelNet/det/backbone_weights/ARC_ResNet50_xFFF_n4.pth"),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_input',
        num_outs=5))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

# optimizer
optimizer = dict(_delete_=True, type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

evaluation = dict(interval=1, metric='mAP', save_best='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

      
