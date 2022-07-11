# A script to visualize the ERF.
# More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity (https://arxiv.org/pdf/2207.03620.pdf)
# Github source: https://github.com/VITA-Group/SLaK
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/DingXiaoH/RepLKNet-pytorch.
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from erf.resnet_for_erf import resnet101, resnet152
from SLaK_for_erf import SLakForERF
from torch import optim as optim

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='resnet101', type=str, help='model name')
    parser.add_argument('--weights', default=None, type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='path_to_imagenet', type=str, help='dataset path')
    parser.add_argument('--save_path', default='temp.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    parser.add_argument('--Decom', type=str2bool, default=False, help='Enabling kernel decomposition')
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def main(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)
    root = os.path.join(args.data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)


    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

    if args.model == 'resnet101':
        model = resnet101(pretrained=args.weights is None)
    elif args.model == 'resnet152':
        model = resnet152(pretrained=args.weights is None)
    elif args.model == 'SLaK':
        model = SLakForERF(kernel_size=[51, 49, 47, 7, 5], width_factor=1.3, Decom=args.Decom)
    elif args.model == 'ConvNeXt':
        model = SLakForERF(kernel_size=[7, 7, 7, 7, 100])
    else:
        raise ValueError('Unsupported model. Please add it here.')

    if args.weights is not None:
        print('load weights')
        weights = torch.load(args.weights, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']
        if 'state_dict' in weights:
            weights = weights['state_dict']
        model.load_state_dict(weights)
        print('loaded')

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in enumerate(data_loader_val):

        if meter.count == args.num_images:
            np.save(args.save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)



if __name__ == '__main__':
    args = parse_args()
    main(args)