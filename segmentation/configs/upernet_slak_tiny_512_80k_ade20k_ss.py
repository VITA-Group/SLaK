# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 4 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=2:  320k iters, 2 batchsize per GPU
#   we use 4 GPUs with 80k schedule:           80k iters, 8 batchsize per GPU

_base_ = [
    '../_base_/models/upernet_SLaK.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)

checkpoint_file = '/path/to/checkpoint-best.pth'

model = dict(
    backbone=dict(
        type='SLaK',
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.1,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        kernel_size=[51,49,47,13,5],
        Decom=True,
        width_factor=1.3,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        in_channels=[124, 249, 499, 998], # in_channels should in line with the actual dims of backbone, i.e., dims * width_factor
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=499,
        num_classes=150
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=dict(norm_decay_mult=0))

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=2000, metric='mIoU')


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=800,
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
