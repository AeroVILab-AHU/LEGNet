_base_ = [
    './orcnn_r50fpn1x_ss_dota10.py',
]

model = dict(
    pretrained='pretrained/ARC_ResNet50_xFFF_n4.pth',
    backbone=dict(
        type='ARCResNet',
        replace = [
            ['x'],
            ['0', '1', '2', '3'],
            ['0', '1', '2', '3', '4', '5'],
            ['0', '1', '2']
        ],
        kernel_number = 4,
    )
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5)
        }
    )
)
