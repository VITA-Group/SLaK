# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_SLaK.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)

checkpoint_file = '/path/to/checkpoint-best.pth'
# checkpoint_file = '/home/luyin/Project/SLaK/LoRA_LK/Checkpoints/submit/ConvNeXt/120epochs/checkpoint-best.pth'

model = dict(
    backbone=dict(
        type='SLaK',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        kernel_size=[51,49,47,13,5],
        LoRA=True,
        width_factor=1.3,
        sparse=True,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        in_channels=[124, 249, 499, 998],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=499,
        num_classes=150
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW',
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=8)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
