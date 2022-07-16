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
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file
 * test/unit/convolution/device/convolution_s4nhwc_s4ncxhwx_s4nhwc_tensor_op_s32_sm75_perf.cu
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

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/convolution/device/convolution.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"

#define DEF_CONVOLUTION(AccessSize, Stages, OutputNum, WithoutSharedLoad)     \
    using Convolution_##Stages##_##WithoutSharedLoad =                        \
            cutlass::conv::device::Convolution<                               \
                    cutlass::int4b_t, cutlass::layout::TensorNHWC,            \
                    cutlass::int4b_t,                                         \
                    cutlass::layout::TensorNCxHWx<AccessSize>, ElementOutput, \
                    cutlass::layout::TensorNHWC, ElementBias,                 \
                    cutlass::layout::TensorNHWC, ElementAccumulator,          \
                    cutlass::conv::ConvType::kConvolution,                    \
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,      \
                    ThreadBlockShape, WarpShape, InstructionShape,            \
                    cutlass::epilogue::thread::BiasAddLinearCombinationClamp< \
                            ElementOutput, OutputNum, ElementAccumulator,     \
                            ElementBias, ElementCompute>,                     \
                    cutlass::conv::threadblock::                              \
                            ConvolutionFpropTransThreadblockSwizzle,          \
                    Stages, AccessSize, AccessSize, SpecialOpt,               \
                    cutlass::arch::OpMultiplyAddSaturate,                     \
                    cutlass::conv::ImplicitGemmMode::GEMM_TN,                 \
                    WithoutSharedLoad>

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_perf,
     128x16x64_128x16x64_16) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 16, 64>;
    using WarpShape = cutlass::gemm::GemmShape<128, 16, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    static cutlass::conv::SpecialOptimizeDesc const SpecialOpt =
            cutlass::conv::SpecialOptimizeDesc::NONE;
    DEF_CONVOLUTION(16, 1, 4, false);
    DEF_CONVOLUTION(16, 2, 4, false);
    test::convolution::device::Testbed<Convolution_1_false> testbed_1_false;
    test::convolution::device::Testbed<Convolution_2_false> testbed_2_false;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    args.emplace_back(ConvolutionParameter{16, 368, 640, 16, 16, 3, 3, 368, 640,
                                           1, 1, 1, 1, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed_1_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
        }
    }
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_perf,
     128x32x64_64x32x64_16) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    static cutlass::conv::SpecialOptimizeDesc const SpecialOpt =
            cutlass::conv::SpecialOptimizeDesc::NONE;
    DEF_CONVOLUTION(16, 1, 8, false);
    DEF_CONVOLUTION(16, 2, 8, false);
    test::convolution::device::Testbed<Convolution_1_false> testbed_1_false;
    test::convolution::device::Testbed<Convolution_2_false> testbed_2_false;
    DEF_CONVOLUTION(16, 1, 8, true);
    DEF_CONVOLUTION(16, 2, 8, true);
    test::convolution::device::Testbed<Convolution_1_true> testbed_1_true;
    test::convolution::device::Testbed<Convolution_2_true> testbed_2_true;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    args.emplace_back(ConvolutionParameter{16, 368, 640, 16, 32, 3, 3, 184, 320,
                                           1, 1, 2, 2, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed_1_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_1_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
        }
    }
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_perf_1x1,
     128x32x64_64x32x64_32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    static cutlass::conv::SpecialOptimizeDesc const SpecialOpt =
            cutlass::conv::SpecialOptimizeDesc::CONV_FILTER_UNITY;
    DEF_CONVOLUTION(32, 1, 8, false);
    DEF_CONVOLUTION(32, 2, 8, false);
    test::convolution::device::Testbed<Convolution_1_false> testbed_1_false;
    test::convolution::device::Testbed<Convolution_2_false> testbed_2_false;
    DEF_CONVOLUTION(32, 1, 8, true);
    DEF_CONVOLUTION(32, 2, 8, true);
    test::convolution::device::Testbed<Convolution_1_true, true> testbed_1_true;
    test::convolution::device::Testbed<Convolution_2_true, true> testbed_2_true;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    args.emplace_back(ConvolutionParameter{16, 184, 320, 32, 32, 1, 1, 184, 320,
                                           1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{16, 184, 320, 32, 64, 1, 1, 92, 160,
                                           1, 1, 2, 2, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed_1_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_1_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
        }
    }
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_perf,
     128x32x64_64x32x64_32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    static cutlass::conv::SpecialOptimizeDesc const SpecialOpt =
            cutlass::conv::SpecialOptimizeDesc::NONE;
    DEF_CONVOLUTION(32, 1, 8, false);
    DEF_CONVOLUTION(32, 2, 8, false);
    test::convolution::device::Testbed<Convolution_1_false> testbed_1_false;
    test::convolution::device::Testbed<Convolution_2_false> testbed_2_false;
    DEF_CONVOLUTION(32, 1, 8, true);
    DEF_CONVOLUTION(32, 2, 8, true);
    test::convolution::device::Testbed<Convolution_1_true, true> testbed_1_true;
    test::convolution::device::Testbed<Convolution_2_true, true> testbed_2_true;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{16, 184, 320, 32, 32, 3, 3, 184, 320,
                                           1, 1, 1, 1, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed_1_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_1_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_2_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
        }
    }
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_perf,
     128x64x64_64x64x64_32) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    static cutlass::conv::SpecialOptimizeDesc const SpecialOpt =
            cutlass::conv::SpecialOptimizeDesc::NONE;
    DEF_CONVOLUTION(32, 1, 8, false);
    test::convolution::device::Testbed<Convolution_1_false> testbed_1_false;
    DEF_CONVOLUTION(32, 1, 16, true);
    test::convolution::device::Testbed<Convolution_1_true, true> testbed_1_true;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{16, 184, 320, 32, 64, 3, 3, 92, 160,
                                           1, 1, 2, 2, 1, 1, mode});

    args.emplace_back(ConvolutionParameter{16, 92, 160, 64, 64, 3, 3, 92, 160,
                                           1, 1, 1, 1, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed_1_false.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
            testbed_1_true.perf(
                    arg, cutlass::from_real<ElementCompute>(0.01234567),
                    cutlass::from_real<ElementCompute>(1.07654321),
                    cutlass::from_real<ElementCompute>(gamma), 1000, false);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
