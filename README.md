#  SLaK

This is the official PyTorch implementation of **SLaK**, from the NeurIPS submission: 

More ConvNets in the 2020s: Scaling up Kernels Beyond 51 x 51 using Sparsity. 

The code is tested used CUDA 11.3.1, cudnn 8.2.0, PyTorch 1.10.0 with A100 GPUs.

To enable training SLaK, we first need to install the efficient large-kernel convolution with PyTorch in https://github.com/MegEngine/cutlass/tree/master/examples/19_large_depthwise_conv2d_torch_extension by the following steps:

1. Clone ```cutlass``` (https://github.com/MegEngine/cutlass), enter the directory.
2. ```cd examples/19_large_depthwise_conv2d_torch_extension```
3. ```./setup.py install --user```. If you get errors, check your ```CUDA_HOME```.
4. A quick check: ```python depthwise_conv2d_implicit_gemm.py```
5. Add ```WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` into your ```PYTHONPATH``` so that you can ```from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM``` anywhere. Then you may use ```DepthWiseConv2dImplicitGEMM``` as a replacement of ```nn.Conv2d```.
6. ```export LARGE_KERNEL_CONV_IMPL=WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` so that RepLKNet will use the efficient implementation. Or you may simply modify the related code (```get_conv2d```) in ```replknet.py```.

## Training code

### ImageNet-1K SLaK-T
```
python -m torch.distributed.launch --nproc_per_node=1 main.py  \
--width-factor 1.3 -u 4000 --init-density 0.6 --method DST --sparse-init ERK \
--LoRA True --epochs 120 --model SLaK_tiny --drop_path 0.1 --batch_size 128 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 72 \
--kernel-size 51 49 47 13 3 --output_dir /path/to/save_results
```

### ImageNet-1K SLaK-S
```
python -m torch.distributed.launch --nproc_per_node=1 main.py  \
--width-factor 1.3 -u 4000 --init-density 0.6 --method DST --sparse-init ERK \
--LoRA True --epochs 120 --model SLaK_small --drop_path 0.4 --batch_size 128 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 72 \
--kernel-size 51 49 47 13 3 --output_dir /path/to/save_results
```
