#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension

import _depthwise_conv2d_implicit_gemm_C as _extension


__all__ = ["DepthWiseConv2dImplicitGEMM"]


class _DepthWiseConv2dImplicitGEMMFP32(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _extension.forward_fp32(x, w)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad = grad.contiguous()
        x = x.contiguous()
        w = w.contiguous()
        dx = _extension.backward_data_fp32(grad, w)
        dw = _extension.backward_filter_fp32(grad, x, w)
        return dx, dw


class _DepthWiseConv2dImplicitGEMMFP16(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return _extension.forward_fp16(x, w)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad = grad.contiguous()
        x = x.contiguous()
        w = w.contiguous()
        dx = _extension.backward_data_fp16(grad, w)
        dw = _extension.backward_filter_fp16(grad, x, w)
        return dx, dw


class DepthWiseConv2dImplicitGEMM(nn.Conv2d):
    def __init__(self, channels, kernel, bias=False):
        super().__init__(channels, channels, kernel, groups=channels, bias=bias)
        # _load_extension()

    def forward(self, x):
        if x.dtype == torch.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, self.weight)
        elif x.dtype == torch.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, self.weight)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        if self.bias is not None:
            x = x + self.bias.to(x).view(1, -1, 1, 1)
        return x


if __name__ == "__main__":
    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        x = torch.randn(64, 384, 32, 32).cuda()
        m1 = DepthWiseConv2dImplicitGEMM(384, 31, bias=False).cuda()
        m2 = nn.Conv2d(384, 384, 31, padding=31 // 2, bias=False, groups=384).cuda()
        m2.load_state_dict(m1.state_dict())
        with torch.cuda.amp.autocast(True):
            y1 = m1(x)
            y2 = m2(x)
        (y1.mean() * 1024).backward()
        (y2.mean() * 1024).backward()
        print("output difference:", ((y1 - y2) ** 2).mean())
        print("gradient difference:", ((m1.weight.grad - m2.weight.grad) ** 2).mean())
