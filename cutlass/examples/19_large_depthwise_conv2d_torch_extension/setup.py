#!/usr/bin/env python3
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

setup(
    name='depthwise_conv2d_implicit_gemm',
    py_modules=['depthwise_conv2d_implicit_gemm'],
    ext_modules=[
        CUDAExtension(
            name='_depthwise_conv2d_implicit_gemm_C',
            sources=[
                "frontend.cpp",
                "forward_fp32.cu",
                "backward_data_fp32.cu",
                "backward_filter_fp32.cu",
                "forward_fp16.cu",
                "backward_data_fp16.cu",
                "backward_filter_fp16.cu",
            ],
            include_dirs=[
                ".",
                os.path.join(CUTLASS_ROOT, "include"),
                os.path.join(CUTLASS_ROOT, "tools", "library", "include"),
                os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
                os.path.join(CUTLASS_ROOT, "examples", "common"),
            ],
            extra_compile_args=['-g']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
