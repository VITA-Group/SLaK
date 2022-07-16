/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file test/unit/convolution/device/conv2d_implicit_gemm_s8_sm75.cu
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief Tests for device-wide GEMM interface
*/
#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/convolution/kernel/default_conv2d_fprop.h"
#include "cutlass/convolution/kernel/default_conv2d_dgrad.h"
#include "cutlass/convolution/device/implicit_gemm_precomp_convolution.h"

#include "conv2d_bias_testbed_interleaved.h"
#include "conv2d_bias_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

TEST(SM75_Device_Conv2d_Fprop_Nt_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_simt_s32,
     128x128_32x2_32x64x32) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dFpropKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dFprop<
                    ElementA, cutlass::layout::TensorNCxHWx<4>, ElementB,
                    cutlass::layout::TensorCxRSKx<4>, ElementC,
                    cutlass::layout::TensorNCxHWx<4>, ElementAccumulator,
                    cutlass::arch::OpClassSimt, cutlass::arch::Sm61,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<32, 64, 32>,
                    cutlass::gemm::GemmShape<1, 1, 4>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            4,         // The number of elements per vectorized.
                                // memory access. This becomes the vector width
                                // of math instructions in the epilogue too.
                            ElementAccumulator,  // Data type of accumulator
                            ElementBias,         // Data type of bias
                            ElementCompute       // Data type for alpha/beta in
                                                 // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionFpropNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 4, 16>::Kernel;

    using Conv2dFprop = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dFpropKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dFprop,
                                                                  4>()));
}

TEST(SM75_Device_Conv2d_Fprop_Nt_ImplicitGemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32,
     128x128_64x2_64x64x64) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dFpropKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dFprop<
                    ElementA, cutlass::layout::TensorNCxHWx<32>, ElementB,
                    cutlass::layout::TensorCxRSKx<32>, ElementC,
                    cutlass::layout::TensorNCxHWx<32>, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<128, 128, 64>,
                    cutlass::gemm::GemmShape<64, 64, 64>,
                    cutlass::gemm::GemmShape<8, 8, 16>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            64 / cutlass::sizeof_bits<ElementC>::
                                            value,  // The number of elements
                                                    // per vectorized. memory
                                                    // access. This becomes the
                                                    // vector width of math
                                                    // instructions in the
                                                    // epilogue too.
                            ElementAccumulator,     // Data type of accumulator
                            ElementBias,            // Data type of bias
                            ElementCompute  // Data type for alpha/beta in
                                            // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionFpropNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 16, 16>::Kernel;

    using Conv2dFprop = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dFpropKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dFprop,
                                                                  32>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Nt_ImplicitGemm_s8ncxhwx_s8kxrscx_s8ncxhwx_simt_s32,
     128x128_32x2_32x64x32) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNCxHWx<4>, ElementB,
                    cutlass::layout::TensorKxRSCx<4>, ElementC,
                    cutlass::layout::TensorNCxHWx<4>, ElementAccumulator,
                    cutlass::arch::OpClassSimt, cutlass::arch::Sm61,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<32, 64, 32>,
                    cutlass::gemm::GemmShape<1, 1, 4>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            4,         // The number of elements per vectorized.
                                // memory access. This becomes the vector width
                                // of math instructions in the epilogue too.
                            ElementAccumulator,  // Data type of accumulator
                            ElementBias,         // Data type of bias
                            ElementCompute       // Data type for alpha/beta in
                                                 // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 4, 16>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dDgrad,
                                                                  4>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Nt_ImplicitGemm_s8ncxhwx_s8kxrscx_s8ncxhwx_simt_s32,
     16x128_16x2_16x64x16) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNCxHWx<4>, ElementB,
                    cutlass::layout::TensorKxRSCx<4>, ElementC,
                    cutlass::layout::TensorNCxHWx<4>, ElementAccumulator,
                    cutlass::arch::OpClassSimt, cutlass::arch::Sm61,
                    cutlass::gemm::GemmShape<16, 128, 16>,
                    cutlass::gemm::GemmShape<16, 64, 16>,
                    cutlass::gemm::GemmShape<1, 1, 4>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            4,         // The number of elements per vectorized.
                                // memory access. This becomes the vector width
                                // of math instructions in the epilogue too.
                            ElementAccumulator,  // Data type of accumulator
                            ElementBias,         // Data type of bias
                            ElementCompute       // Data type for alpha/beta in
                                                 // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 4, 4>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dDgrad,
                                                                  4>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Nt_ImplicitGemm_s8ncxhwx_s8kxrscx_s8ncxhwx_tensor_op_s32,
     128x128_64x2_64x64x64) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNCxHWx<32>, ElementB,
                    cutlass::layout::TensorKxRSCx<32>, ElementC,
                    cutlass::layout::TensorNCxHWx<32>, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<128, 128, 64>,
                    cutlass::gemm::GemmShape<64, 64, 64>,
                    cutlass::gemm::GemmShape<8, 8, 16>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            64 / cutlass::sizeof_bits<ElementC>::
                                            value,  // The number of elements
                                                    // per vectorized. memory
                                                    // access. This becomes the
                                                    // vector width of math
                                                    // instructions in the
                                                    // epilogue too.
                            ElementAccumulator,     // Data type of accumulator
                            ElementBias,            // Data type of bias
                            ElementCompute  // Data type for alpha/beta in
                                            // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 16, 16>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dDgrad,
                                                                  32>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Nt_ImplicitGemm_s8ncxhwx_s8kxrscx_s8ncxhwx_tensor_op_s32,
     64x128_64x2_32x64x64) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNCxHWx<32>, ElementB,
                    cutlass::layout::TensorKxRSCx<32>, ElementC,
                    cutlass::layout::TensorNCxHWx<32>, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<64, 128, 64>,
                    cutlass::gemm::GemmShape<32, 64, 64>,
                    cutlass::gemm::GemmShape<8, 8, 16>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            64 / cutlass::sizeof_bits<ElementC>::
                                            value,  // The number of elements
                                                    // per vectorized. memory
                                                    // access. This becomes the
                                                    // vector width of math
                                                    // instructions in the
                                                    // epilogue too.
                            ElementAccumulator,     // Data type of accumulator
                            ElementBias,            // Data type of bias
                            ElementCompute  // Data type for alpha/beta in
                                            // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradNCxHWxThreadblockSwizzle,
                    2, cutlass::arch::OpMultiplyAddSaturate, 16, 16>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllInterleavedConv2dBias<Conv2dDgrad,
                                                                  32>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Tn_ImplicitGemm_s8nhwc_s8ckxrsx_s8nhwc_tensor_op_s32,
     128x32_32x2_64x32x32) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNHWC, ElementB,
                    cutlass::layout::TensorCKxRSx<4>, ElementC,
                    cutlass::layout::TensorNHWC, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<128, 32, 32>,
                    cutlass::gemm::GemmShape<64, 32, 32>,
                    cutlass::gemm::GemmShape<8, 8, 16>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            8,
                            ElementAccumulator,  // Data type of accumulator
                            ElementBias,         // Data type of bias
                            ElementCompute       // Data type for alpha/beta in
                                                 // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradTransThreadblockSwizzle,
                    1, cutlass::arch::OpMultiplyAddSaturate, 4, 4,
                    cutlass::conv::SpecialOptimizeDesc::
                            DECONV_DOUBLE_UPSAMPLING,
                    cutlass::conv::ImplicitGemmMode::GEMM_TN>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllConv2dBias<Conv2dDgrad, 16, 2>()));
}

TEST(SM75_Device_Conv2d_Dgrad_Tn_ImplicitGemm_s8nhwc_s8ckxrsx_s8nhwc_tensor_op_s32,
     64x16_32x2_64x16x32) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int8_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;

    using Conv2dDgradKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementA, cutlass::layout::TensorNHWC, ElementB,
                    cutlass::layout::TensorCKxRSx<4>, ElementC,
                    cutlass::layout::TensorNHWC, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<64, 16, 32>,
                    cutlass::gemm::GemmShape<64, 16, 32>,
                    cutlass::gemm::GemmShape<8, 8, 16>,
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                            ElementC,  // Data type of output matrix.
                            4,
                            ElementAccumulator,  // Data type of accumulator
                            ElementBias,         // Data type of bias
                            ElementCompute       // Data type for alpha/beta in
                                                 // linear combination,
                            >,
                    cutlass::conv::threadblock::
                            ConvolutionDgradTransThreadblockSwizzle,
                    1, cutlass::arch::OpMultiplyAddSaturate, 4, 4,
                    cutlass::conv::SpecialOptimizeDesc::
                            DECONV_DOUBLE_UPSAMPLING,
                    cutlass::conv::ImplicitGemmMode::GEMM_TN>::Kernel;

    using Conv2dDgrad = cutlass::conv::device::ImplicitGemmPrecompConvolution<
            Conv2dDgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE((test::conv::device::TestAllConv2dBias<Conv2dDgrad, 16, 2>()));
}

////////////////////////////////////////////////////////////////////////////////
#endif  // CUTLASS_ARCH_MMA_SM75_SUPPORTED
