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
 * test/unit/convolution/device/convolution_s8nchw32_s8chwn32_s8nchw4_tensor_op_s32_sm75.cu
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

#define RUN_CONVOLUTION                                                        \
    do {                                                                       \
        using ElementOutput = int8_t;                                          \
        using ElementAccumulator = int32_t;                                    \
        using ElementBias = int32_t;                                           \
        using ElementCompute = float;                                          \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;           \
        using Convolution = cutlass::conv::device::Convolution<                \
                int8_t, cutlass::layout::TensorNCxHWx<32>, int8_t,             \
                cutlass::layout::TensorCxRSKx<32>, ElementOutput,              \
                cutlass::layout::TensorNCxHWx<4>, int32_t,                     \
                cutlass::layout::TensorNCxHWx<4>, int32_t,                     \
                cutlass::conv::ConvType::kConvolution,                         \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,           \
                ThreadBlockShape, WarpShape, InstructionShape,                 \
                cutlass::epilogue::thread::BiasAddLinearCombinationClamp<      \
                        ElementOutput, 4, ElementAccumulator, ElementBias,     \
                        ElementCompute>,                                       \
                cutlass::conv::threadblock::                                   \
                        ConvolutionFpropNCxHWxThreadblockSwizzle,              \
                2, 16, 16>;                                                    \
        EXPECT_TRUE(                                                           \
                test::convolution::device::TestConvolutionMma<Convolution>()); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     128x256x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_CONVOLUTION;
}

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     128x128x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_CONVOLUTION;
}

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     64x128x64_32x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
    RUN_CONVOLUTION;
}

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     128x64x64_64x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    RUN_CONVOLUTION;
}

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     64x64x64_32x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
    RUN_CONVOLUTION;
}

TEST(SM75_Device_Convolution_s8NCHW32_s8CHWN32_s8NCHW4_tensor_op_mmai8816,
     32x64x64_32x16x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 64>;
    RUN_CONVOLUTION;
}

////////////////////////////////////////////////////////////////////////////////
