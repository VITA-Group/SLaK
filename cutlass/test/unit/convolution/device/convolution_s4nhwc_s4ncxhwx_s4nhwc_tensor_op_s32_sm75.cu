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
 * test/unit/convolution/device/convolution_s4nhwc_s4nhwcx_s4nhwc_tensor_op_s32_sm75.cu
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

#define RUN_CONVOLUTION(AccessSize, Stages, OutputNum)                       \
    do {                                                                     \
        using ElementOutput = cutlass::int4b_t;                              \
        using ElementAccumulator = int32_t;                                  \
        using ElementBias = int32_t;                                         \
        using ElementCompute = float;                                        \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;         \
        using Convolution = cutlass::conv::device::Convolution<              \
                cutlass::int4b_t, cutlass::layout::TensorNHWC,               \
                cutlass::int4b_t, cutlass::layout::TensorNCxHWx<AccessSize>, \
                ElementOutput, cutlass::layout::TensorNHWC, ElementBias,     \
                cutlass::layout::TensorNHWC, ElementAccumulator,             \
                cutlass::conv::ConvType::kConvolution,                       \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,         \
                ThreadBlockShape, WarpShape, InstructionShape,               \
                cutlass::epilogue::thread::BiasAddLinearCombinationClamp<    \
                        ElementOutput, OutputNum, ElementAccumulator,        \
                        ElementBias, ElementCompute>,                        \
                cutlass::conv::threadblock::                                 \
                        ConvolutionFpropTransThreadblockSwizzle,             \
                Stages, AccessSize, AccessSize,                              \
                cutlass::conv::SpecialOptimizeDesc::NONE,                    \
                cutlass::arch::OpMultiplyAddSaturate,                        \
                cutlass::conv::ImplicitGemmMode::GEMM_TN>;                   \
        EXPECT_TRUE(test::convolution::device::TestConvolutionNHWC<          \
                    Convolution>());                                         \
    } while (0)

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832,
     128x128x128_64x64x128) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    RUN_CONVOLUTION(8, 2, 8);
    RUN_CONVOLUTION(16, 1, 8);
    RUN_CONVOLUTION(32, 1, 8);
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832,
     128x64x128_64x32x128) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 128>;
    RUN_CONVOLUTION(8, 1, 8);
    RUN_CONVOLUTION(16, 2, 8);
    RUN_CONVOLUTION(32, 2, 8);
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832,
     128x32x64_64x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    RUN_CONVOLUTION(8, 1, 8);
    RUN_CONVOLUTION(16, 2, 8);
    RUN_CONVOLUTION(32, 1, 8);
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832,
     128x16x64_128x16x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 16, 64>;
    using WarpShape = cutlass::gemm::GemmShape<128, 16, 64>;
    RUN_CONVOLUTION(16, 1, 4);
}

#define RUN_CONVOLUTION_REORDERK(AccessSize, Stages, OutputNum)              \
    do {                                                                     \
        using ElementOutput = cutlass::int4b_t;                              \
        using ElementAccumulator = int32_t;                                  \
        using ElementBias = int32_t;                                         \
        using ElementCompute = float;                                        \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;         \
        using Convolution = cutlass::conv::device::Convolution<              \
                cutlass::int4b_t, cutlass::layout::TensorNHWC,               \
                cutlass::int4b_t, cutlass::layout::TensorNCxHWx<AccessSize>, \
                ElementOutput, cutlass::layout::TensorNHWC, ElementBias,     \
                cutlass::layout::TensorNHWC, ElementAccumulator,             \
                cutlass::conv::ConvType::kConvolution,                       \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,         \
                ThreadBlockShape, WarpShape, InstructionShape,               \
                cutlass::epilogue::thread::BiasAddLinearCombinationClamp<    \
                        ElementOutput, OutputNum, ElementAccumulator,        \
                        ElementBias, ElementCompute>,                        \
                cutlass::conv::threadblock::                                 \
                        ConvolutionFpropTransThreadblockSwizzle,             \
                Stages, AccessSize, AccessSize,                              \
                cutlass::conv::SpecialOptimizeDesc::NONE,                    \
                cutlass::arch::OpMultiplyAddSaturate,                        \
                cutlass::conv::ImplicitGemmMode::GEMM_TN, true>;             \
        EXPECT_TRUE(test::convolution::device::TestConvolutionNHWC_ReorderK< \
                    Convolution>());                                         \
    } while (0)

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_reorderK,
     128x32x64_64x32x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 32, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
    RUN_CONVOLUTION_REORDERK(8, 1, 8);
    RUN_CONVOLUTION_REORDERK(16, 2, 8);
    RUN_CONVOLUTION_REORDERK(32, 1, 8);
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_tensor_op_mmai8832_reorderK,
     128x64x64_64x64x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    RUN_CONVOLUTION_REORDERK(8, 1, 16);
    RUN_CONVOLUTION_REORDERK(16, 2, 16);
    RUN_CONVOLUTION_REORDERK(32, 1, 16);
}

#define RUN_CONVOLUTION_OUT_NHWC_S8(AccessSize, Stages, OutputNum)           \
    do {                                                                     \
        using ElementOutput = int8_t;                                        \
        using ElementAccumulator = int32_t;                                  \
        using ElementBias = int32_t;                                         \
        using ElementCompute = float;                                        \
        using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;         \
        using Convolution = cutlass::conv::device::Convolution<              \
                cutlass::int4b_t, cutlass::layout::TensorNHWC,               \
                cutlass::int4b_t, cutlass::layout::TensorNCxHWx<AccessSize>, \
                ElementOutput, cutlass::layout::TensorNHWC, ElementBias,     \
                cutlass::layout::TensorNHWC, ElementAccumulator,             \
                cutlass::conv::ConvType::kConvolution,                       \
                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,         \
                ThreadBlockShape, WarpShape, InstructionShape,               \
                cutlass::epilogue::thread::BiasAddLinearCombinationClamp<    \
                        ElementOutput, OutputNum, ElementAccumulator,        \
                        ElementBias, ElementCompute>,                        \
                cutlass::conv::threadblock::                                 \
                        ConvolutionFpropTransThreadblockSwizzle,             \
                Stages, AccessSize, AccessSize,                              \
                cutlass::conv::SpecialOptimizeDesc::NONE,                    \
                cutlass::arch::OpMultiplyAddSaturate,                        \
                cutlass::conv::ImplicitGemmMode::GEMM_TN>;                   \
        EXPECT_TRUE(test::convolution::device::TestConvolutionNHWC<          \
                    Convolution>());                                         \
    } while (0)

TEST(SM75_Device_Convolution_s4_s4_NHWC_s8_NHWC_tensor_op_mmai8832,
     128x16x64_128x16x64) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 16, 64>;
    using WarpShape = cutlass::gemm::GemmShape<128, 16, 64>;
    RUN_CONVOLUTION_OUT_NHWC_S8(32, 1, 4);
}

TEST(SM75_Device_Convolution_s4_s4_NHWC_s8_NHWC_tensor_op_mmai8832,
     128x16x128_64x16x128) {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 16, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 16, 128>;
    RUN_CONVOLUTION_OUT_NHWC_S8(32, 1, 4);
}

////////////////////////////////////////////////////////////////////////////////
