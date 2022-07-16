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
 * \file test/unit/convolution/device/convolution_s4nchw64_s4chwn64_s4nchw64_tensor_op_s32_sm75_perf.cu
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

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Convolution_s4_s4_NC64HW64_tensor_op_mmai8832_perf,
     128x128x128_64x64x128) {
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            cutlass::int4b_t, cutlass::layout::TensorNCxHWx<64>, cutlass::int4b_t,
            cutlass::layout::TensorCxRSKx<64>, ElementOutput,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 128, 128>,
            cutlass::gemm::GemmShape<64, 64, 128>,
            cutlass::gemm::GemmShape<8, 8, 32>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 16, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::
                    ConvolutionFpropNCxHWxThreadblockSwizzle,
            2, 32, 32>;

    EXPECT_TRUE(test::convolution::device::TestConvolutionPerf<Convolution>(
            100, 64, true));
}

TEST(SM75_Device_Convolution_s4_s4_NC64HW64_tensor_op_mmai8832_reorderK_perf,
     128x128x128_64x64x128) {
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using Convolution = cutlass::conv::device::Convolution<
            cutlass::int4b_t, cutlass::layout::TensorNCxHWx<64>,
            cutlass::int4b_t, cutlass::layout::TensorCxRSKx<64>, ElementOutput,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 128, 128>,
            cutlass::gemm::GemmShape<64, 64, 128>,
            cutlass::gemm::GemmShape<8, 8, 32>,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 16, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::ConvolutionFpropTransThreadblockSwizzle,
            2, 32, 32, cutlass::conv::SpecialOptimizeDesc::NONE,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;

    EXPECT_TRUE(
            (test::convolution::device::TestConvolutionPerf<Convolution, true>(
                    100, 256, true, false)));
}

TEST(SM75_Device_Convolution_s4_s4_NC64HW64_tensor_op_mmai8832_reorderK_perf,
     128x64x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using ElementOutput = cutlass::int4b_t;
    using ElementAccumulator = int32_t;
    using ElementBias = int32_t;
    using ElementCompute = float;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    using Convolution = cutlass::conv::device::Convolution<
            cutlass::int4b_t, cutlass::layout::TensorNCxHWx<64>,
            cutlass::int4b_t, cutlass::layout::TensorCxRSKx<64>, ElementOutput,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::layout::TensorNCxHWx<64>, int32_t,
            cutlass::conv::ConvType::kConvolution,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            ThreadBlockShape, WarpShape, InstructionShape,
            cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
                    ElementOutput, 16, ElementAccumulator, int32_t,
                    ElementCompute>,
            cutlass::conv::threadblock::ConvolutionFpropTransThreadblockSwizzle,
            1, 32, 32, cutlass::conv::SpecialOptimizeDesc::NONE,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;
    test::convolution::device::Testbed<Convolution, true> testbed;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{16, 92, 160, 64, 64, 3, 3, 92, 160,
                                           1, 1, 1, 1, 1, 1, mode});

    double problem_gamma[] = {0.0};
    for (auto arg : args) {
        for (auto gamma : problem_gamma) {
            testbed.perf(arg, cutlass::from_real<ElementCompute>(0.01234567),
                         cutlass::from_real<ElementCompute>(1.07654321),
                         cutlass::from_real<ElementCompute>(gamma), 1000,
                         false);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
