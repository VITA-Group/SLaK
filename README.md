#  [More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity](https://arxiv.org/abs/2207.03620)

Official PyTorch implementation of **SLaK**, from the following paper: 

[More ConvNets in the 2020s: Scaling up Kernels Beyond 51 x 51 using Sparsity](https://arxiv.org/abs/2207.03620)

[Shiwei Liu](https://shiweiliuiiiiiii.github.io/), [Tianlong Chen](https://tianlong-chen.github.io/about/), [Xiaohan Chen](http://www.xiaohanchen.com/), [Xuxi Chen](https://xxchen.site/), Qiao Xiao, Boqian Wu, [Mykola Pechenizkiy](https://www.win.tue.nl/~mpechen/), [Decebal Mocanu](https://people.utwente.nl/d.c.mocanu), [Zhangyang Wang](https://vita-group.github.io/)

Eindhoven University of Technology, University of Texas at Austin, University of Twente

 [[`arXiv`](https://arxiv.org/pdf/2207.03620.pdf)] [[`Atlas Wang's talk`](https://drive.google.com/file/d/1_dqzEUARr2WgxGtSeGSRPsh1kufQAa-8/view)]

--- 
<p align="center">
<img src="https://github.com/Shiweiliuiiiiiii/SLaK/blob/main/SLaK.png" width="500" height="300">
</p>

We propose **SLaK**, a pure ConvNet model that for the first time is able to scale the convolutional kernels beyond 51x51.


## Catalog
- [x] ImageNet-1K Training Code   
- [x] ImageNet-1K Fine-tuning Code  
- [x] Downstream Transfer (Detection, Segmentation) Code


<!-- ✅ ⬜️  -->

## Results and ImageNet-1K trained models

### SLaK Models with 51x51 kernels trained on ImageNet-1K for 300 epochs

| name | resolution | kernel size |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| SLaK-T | 224x224 | 51x51 |82.5 | 30M | 5.0G | [Google Drive](https://drive.google.com/file/d/1Iut2f5FMS_77jGPYoUJDQzDIXOsax1u4/view?usp=sharing) |
| SLaK-S | 224x224 | 51x51 | 83.8 | 55M | 9.8G |  [Google Drive](https://drive.google.com/file/d/1etM6KQbnlsgDAZ37adsQJ3UI8Bbv2AVe/view?usp=sharing) |
| SLaK-B | 224x224 | 51x51 | 84.0 | 95M | 17.1G |  [Google Drive](https://drive.google.com/file/d/1duUxUD3RSblQ6eDHd0n-u0aulwGypf1j/view?usp=sharing) |

### SLaK-T Models with 31x31, 51,51, and 61x61 kernels trained on ImageNet-1K for 120 epochs

| name | resolution | kernel size |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| SLaK-T | 224x224 | 31x31 | 81.5 | 30M | 4.8G | [Surf Drive](https://surfdrive.surf.nl/files/index.php/s/VXzBxFXQdlAQ7h8) |
| SLaK-T | 224x224 | 51x51 | 81.6 | 30M | 5.0G |  [Surf Drive](https://surfdrive.surf.nl/files/index.php/s/WiQYWNclJ9bW5XV) |
| SLaK-T | 224x224 | 61x61 | 81.5 | 31M | 5.2G |  [Surf Drive](https://surfdrive.surf.nl/files/index.php/s/VpR1te71NmVImJb) |


## Installation

The code is tested used CUDA 11.3.1, cudnn 8.2.0, PyTorch 1.10.0 with A100 GPUs.

### Dependency Setup
Create an new conda virtual environment
```
conda create -n slak python=3.8 -y
conda activate slak
```

Install [Pytorch](https://pytorch.org/)>=1.10.0. For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Clone this repo and install required packages:
```
git clone https://github.com/Shiweiliuiiiiiii/SLaK.git
pip install timm tensorboardX six
```

To enable training SLaK, we follow [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch#use-our-efficient-large-kernel-convolution-with-pytorch) and install the efficient large-kernel convolution with PyTorch provided by MegEngine:

1. ```cd cutlass/examples/19_large_depthwise_conv2d_torch_extension```
2. ```./setup.py install --user```. If you get errors, (1) check your ```CUDA_HOME```; (2) you might need to change the source code a bit to make tensors contiguous see [here](https://github.com/Shiweiliuiiiiiii/SLaK/blob/3f8b1c46eee34da440afae507df13bc6307c3b2c/depthwise_conv2d_implicit_gemm.py#L25) for example. 
3. A quick check: ```python depthwise_conv2d_implicit_gemm.py```
4. Add ```WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` into your ```PYTHONPATH``` so that you can ```from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM``` anywhere. Then you may use ```DepthWiseConv2dImplicitGEMM``` as a replacement of ```nn.Conv2d```.
5. ```export LARGE_KERNEL_CONV_IMPL=WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` so that RepLKNet will use the efficient implementation. Or you may simply modify the related code (```get_conv2d```) in ```SLaK.py```.

## Training code

We provide ImageNet-1K training, and ImageNet-1K fine-tuning commands here.

### ImageNet-1K SLaK-T on a single machine
```
python -m torch.distributed.launch --nproc_per_node=4 main.py  \
--Decom True --sparse --width_factor 1.3 -u 2000 --sparsity 0.4 --sparse_init snip  --prune_rate 0.5 --growth random \
--epochs 300 --model SLaK_tiny --drop_path 0.1 --batch_size 128 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

- **To enable to train/evaluate SLaK models, make sure that you add `--sparse --Decom True --kernel_size 51 49 47 13 5 --sparse_init snip` in your script.** `--sparse`: enable sparse model; `--sparsity`: model sparsity; `--width_factor`: model width; `-u`: adaptation frequency; `--prune_rate`: adaptation rate, `--kernel_size`: [4 * (kernel size of each stage) + the size of the smaller kernel edge].
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder. To resume the training of sparse models, we need to set `--sparse_init resume` to get the masks.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

### ImageNet-1K SLaK-S on a single machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py  \
--Decom True --sparse --width_factor 1.3 -u 100 --sparsity 0.4 --sparse_init snip  --prune_rate 0.3 --growth random \
--epochs 300 --model SLaK_small --drop_path 0.4 --batch_size 64 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

### ImageNet-1K SLaK-B on a single machine
```
python -m torch.distributed.launch --nproc_per_node=16 main.py  \
--Decom True --sparse --width_factor 1.3 -u 100 --sparsity 0.4 --sparse_init snip  --prune_rate 0.3 --growth random \
--epochs 300 --model SLaK_base --drop_path 0.5 --batch_size 32 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

To run ConvNeXt, simple set the kernel size as --kernel_size 7 7 7 7 100. (Make sure that the last number is larger than the first four numbers)

## Evaluation
We give an example evaluation command for a SLaK_tiny on ImageNet-1K :

Single-GPU
```
python main.py --model SLaK_tiny --eval true \
--Decom True --kernel_size 51 49 47 13 5 --width_factor 1.3 \
--resume path/to/checkpoint \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

Multi-GPUs
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model SLaK_tiny --eval true \
--Decom True --kernel_size 51 49 47 13 5 --width_factor 1.3 \
--resume path/to/checkpoint \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

## Semantic Segmentation and Object Detection

We use MMSegmentation and MMDetection frameworks. Just clone MMSegmentation or MMDetection, and

1. Put ```segmentation/slak.py``` into ```mmsegmentation/mmseg/models/backbones/``` or ```mmdetection/mmdet/models/backbones/```. The only difference between ```segmentation/slak.py``` and ```SLaK.py``` for ImageNet classification is the ```@BACKBONES.register_module```.
2. Add RepLKNet into ```mmsegmentation/mmseg/models/backbones/__init__.py``` or ```mmdetection/mmdet/models/backbones/__init__.py```. That is
  ```
  ...
 from .slak import SLaK
  __all__ = ['ResNet', ..., 'SLaK']
  ```
3. Put ```segmentation/configs/*.py``` into ```mmsegmentation/configs/SLaK/``` or ```detection/configs/*.py``` into ```mmdetection/configs/SLaK/```; put files of ```mmsegmentation/mmseg/core/optimizers/''' into ```mmsegmentation/mmseg/core/optimizers/```.
4. Download and use our weights. For examples, to evaluate SLaK-tiny + UperNet on ADE20K
  ```
  python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/SLaK/upernet_slak_tiny_512_80k_ade20k_ss.py --launcher pytorch --eval mIoU
  ```
5. Or you may finetune our released pretrained weights
  ```
   bash tools/dist_train.sh  configs/SLaK/upernet_slak_tiny_512_80k_ade20k_ss.py 4 --work-dir ADE20_SLaK_51_sparse_1000ite/ --auto-resume  --seed 0 --deterministic
   ```
   The path of pretrained models is 'checkpoint_file' in 'upernet_slak_tiny_512_80k_ade20k_ss'.
   
## Visualizing the Effective Receptive Field

The code is highly based on the libracy of [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch#visualizing-the-effective-receptive-field). We have released our script to visualize and analyze the Effective Receptive Field (ERF). The  For example, to automatically download the ResNet-101 from torchvision and obtain the aggregated contribution score matrix,
```
python erf/visualize_erf.py --model resnet101 --data_path /path/to/imagenet-1k --save_path resnet101_erf_matrix.npy
```
Then calculate the high-contribution area ratio and visualize the ERF by
```
python erf/analyze_erf.py --source resnet101_erf_matrix.npy --heatmap_save resnet101_heatmap.png
```
Note this plotting script works with matplotlib 3.3.

To visualize your own model, first define a model that outputs the last feature map rather than the logits (following [this example](https://github.com/VITA-Group/SLaK/blob/a9da48aff07d35571439524212f90cc75b830f4d/erf/SLaK_for_erf.py#L20)), add the code for building model and loading weights [here](https://github.com/VITA-Group/SLaK/blob/a9da48aff07d35571439524212f90cc75b830f4d/erf/visualize_erf.py#L81), then
```
python erf/visualize_erf.py --model your_model --weights /path/to/your/weights --data_path /path/to/imagenet-1k --save_path your_model_erf_matrix.npy
```

We have provided the saved matrices and source code to help reproduce. To reproduce the results of Figure 3 in our paper, run
```
python erf/erf_slak51_convnext7_convnext31.py
```


## More information will come soon.

## Acknowledgement
The released PyTorch training script is based on the code of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch), which were built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories. 

We thank the MegEngine team at MEGVII Technology and the authors of RepLKNet for releasing the efficient implementation of large-kernel convolution.

## License
This project is released under the MIT license.

## Contact
Shiwei Liu: s.liu3@tue.nl

Homepage: https://shiweiliuiiiiiii.github.io/

My open-sourced papers and repos: 

1. ITOP (ICML 2021) **A concept to train sparse model to dense performance**.\
[Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](https://arxiv.org/abs/2102.02887)\
[code](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization).

2. Selfish-RNN (ICML 2021) **Selfish Sparse RNN Training**. \
[Selfish Sparse RNN Training](https://arxiv.org/abs/2101.09048)\
[code](https://github.com/Shiweiliuiiiiiii/Selfish-RNN).

3. GraNet (NeurIPS 2021) **A State-of-the-art brain-inspired sparse training method**. \
[Sparse Training via Boosting Pruning Plasticity with Neuroregeneration](https://arxiv.org/abs/2106.10404)\
[code](https://github.com/VITA-Group/GraNet).

4. Random_Pruning (ICLR 2022) **The Unreasonable Effectiveness of Random Pruning**\
[The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training](https://arxiv.org/pdf/2202.02643.pdf)\
[code](https://github.com/VITA-Group/Random_Pruning).

5. FreeTickets (ICLR 2022) **Efficient Ensemble**\
[Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity](https://arxiv.org/abs/2106.14568).\
[code](https://github.com/VITA-Group/FreeTickets). 


If you find this repository useful, please consider giving a star star and cite our paper.

```
@article{liu2022more,
  title={More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity},
  author={Liu, Shiwei and Chen, Tianlong and Chen, Xiaohan and Chen, Xuxi and Xiao, Qiao and Wu, Boqian and Pechenizkiy, Mykola and Mocanu, Decebal and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2207.03620},
  year={2022}
}
```
