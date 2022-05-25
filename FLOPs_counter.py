import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sparse_core import Masking, CosineDecay, LinearDecay
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([(param!=0).sum() if len(param.size()) == 4 or len(param.size()) == 2 else 0 for name,param in model.named_parameters()])
    print('  + Number of params: %.2f' % (total))


def count_model_param_flops(model=None, input_res=224, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)

        num_weight_params = (self.weight.data != 0).float().sum()
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    layer_norm=[]
    def layernorm_hook(self, input, output):
        input = input[0]

        batch_flops = np.prod(input.shape)
        layer_norm.append(batch_flops)

    def foo(net):
        childrens = list(net.children())

        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            if isinstance(net, torch.nn.LayerNorm):
                net.register_forward_hook(layernorm_hook)

            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input =Variable(torch.rand(1, 3, input_res, input_res), requires_grad = True)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample) + sum(layer_norm))

    print('  + Number of FLOPs: %.2f' % (total_flops))

    return total_flops

if __name__ == '__main__':
    # FPR GraNet-ST 0.9, we need to delete the last 6 iteration to make it for 100 epochs
    # for GraNet-ST 0.8, the first 4000 epochs is 5.84G. We need to change the code to if 'density:' and 'proportion' in
    # for GraNet 0.9, we need to isolate the first 377 iterations and only calculate the rest of the flops + 8.18e9*1281152*(5+1500/10009)*3
    # for Graet 0.8, nothing is wrong


    # VGG-16   dense: 622275520
    # customer_sparsity = []
    # with open('used_files/log_GDP_0.9.out') as file:
    #     for line in file:
    #         if 'density:' in line:
    #             customer_sparsity.append(float(line.split()[-1]))
    #
    # customer_sparsity = np.array(customer_sparsity[377:]).reshape(-1, 54)
    # # customer_sparsity = np.array(customer_sparsity[:54])
    # print(len(customer_sparsity))
    # # for i in range(1):
    # #     PLOPs_Para['FLOPs'].append(5.84e9)
    # #     PLOPs_Para['PARA'].append(25502912)
    # # training flops for the first 5 epochs of dense training
    # total_training_flops = 8.18e9 * 1281152 * (5 + 2000 / 10009) * 3
    # # total_training_flops = 5.84e9*1281152*(4000/10009)*3
    # # for i in range()
    # for i in range(7, len(customer_sparsity) - 6):
    #     print('iter:', i)
    #     models = {}
    #
    #     cls, cls_args = (VGG16, ['C', 10])
    #     model = cls(*(cls_args ))
    #     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    #     decay = CosineDecay(0.5, 1000 * (10))
    #     mask = Masking(optimizer, death_mode='magnitude', death_rate_decay=decay, growth_mode='random',
    #                    redistribution_mode='none')
    #     customer_density = mask.add_module(model, density=0.2, sparse_init='fixed_ERK')
    #     model.eval()
    #     cur_flops = count_model_param_flops(model=model)
    #     cur_para = print_model_param_nums(model=model)
        # total_training_flops = total_training_flops + cur_flops * 1281152 * (4000 / 10009) * 3
        # print('+++Right now Total Number of FLOPs: %.2fe18' % (total_training_flops / 1e18))
    # torch.save(PLOPs_Para, 'PLOPs_Para_GDP-ST_0.9.pt')

    # # ResNet-50
    # customer_sparsity = []
    # with open('used_files/log_GDP_0.9.out') as file:
    #     for line in file:
    #         if 'density:' in line:
    #             customer_sparsity.append(float(line.split()[-1]))
    #
    # customer_sparsity = np.array(customer_sparsity[377:]).reshape(-1,54)
    # # customer_sparsity = np.array(customer_sparsity[:54])
    # print(len(customer_sparsity))
    # # for i in range(1):
    # #     PLOPs_Para['FLOPs'].append(5.84e9)
    # #     PLOPs_Para['PARA'].append(25502912)
    # # training flops for the first 5 epochs of dense training
    # total_training_flops = 8.18e9*1281152*(5+2000/10009)*3
    # # total_training_flops = 5.84e9*1281152*(4000/10009)*3
    # # for i in range()
    # for i in range(7, len(customer_sparsity)-6):
    #     print('iter:', i)
    #     model = resnet.build_resnet('resnet50', 'fanin')
    #     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    #     decay = CosineDecay(0.5, 1000 * (10))
    #     mask = Masking(optimizer, death_mode='magnitude', death_rate_decay=decay, growth_mode='random',
    #                    redistribution_mode='none')
    #     mask.add_module(model, density=0.1, sparse_init='customer', customer_density=customer_sparsity[i])
    #
    #     cur_flops = count_model_param_flops(model=model)
    #     cur_para = print_model_param_nums(model=model)
    #     total_training_flops = total_training_flops + cur_flops*1281152*(4000/10009)*3
    #     print('+++Right now Total Number of FLOPs: %.2fe18' % (total_training_flops / 1e18))
    # # torch.save(PLOPs_Para, 'PLOPs_Para_GDP-ST_0.9.pt')

##########################################################################################
    customer_sparsity = []
    with open('used_files/log_GDP_0.9.out') as file:
        for line in file:
            if 'density:' in line:
                customer_sparsity.append(float(line.split()[-1]))
    customer_sparsity = np.array(customer_sparsity[:13446]).reshape(-1, 54)


    # training flops for the first 5 epochs of dense training
    total_training_flops = 8.2e9*1281152*(5+1500/10009)*3
    # for i in range(len(customer_sparsity)-7):
        # print('iter:', i)
    model = resnet.build_resnet('resnet50', 'fanin')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    decay = CosineDecay(0.5, 1000 * (10))
    mask = Masking(optimizer, death_mode='magnitude', death_rate_decay=decay, growth_mode='random',
                   redistribution_mode='none')
    mask.add_module(model, density=0.1, sparse_init='customer', customer_density=customer_sparsity[-1])

    total_flops = count_model_param_flops(model=model)
    total_training_flops += total_flops*1281152*(4000/10009)*3
    print('+++Right now Total Number of FLOPs: %.2fe18' % (total_training_flops / 1e18))

