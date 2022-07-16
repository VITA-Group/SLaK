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
 * \file
 * dnn/src/cuda/cutlass/convolution/device/default_convolution_configuration.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/conv/convolution.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish.h"
#include "cutlass/epilogue/thread/linear_combination.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <typename OperatorClass, typename ArchTag, typename ElementSrc,
          typename ElementFilter, typename ElementDst,
          typename ElementAccumulator>
struct DefaultConvolutionConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <typename ArchTag, typename ElementDst>
struct DefaultConvolutionConfiguration<arch::OpClassSimt, ArchTag, int8_t,
                                       int8_t, ElementDst, int32_t> {
    static int const kAlignmentSrc = 4;
    static int const kAlignmentFilter = 4;
    using ThreadblockShape = gemm::GemmShape<128, 128, 32>;
    using WarpShape = gemm::GemmShape<32, 64, 32>;
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    static int const kStages = 2;

    using EpilogueOutputOp = epilogue::thread::BiasAddLinearCombinationClamp<
            ElementDst, 4, int32_t, int32_t, float>;

    using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementDst>
struct DefaultConvolutionConfiguration<arch::OpClassTensorOp, arch::Sm75,
                                       int8_t, int8_t, ElementDst, int32_t> {
    static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
    static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
    using ThreadblockShape = gemm::GemmShape<128, 256, 64>;
    using WarpShape = gemm::GemmShape<64, 64, 64>;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ArchTag = arch::Sm75;
    static int const kStages = 2;

    using EpilogueOutputOp = epilogue::thread::BiasAddLinearCombinationClamp<
            ElementDst, 128 / sizeof_bits<ElementDst>::value, int32_t, int32_t,
            float>;

    using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template <bool Signed, typename ElementDst>
struct DefaultConvolutionConfiguration<arch::OpClassTensorOp, arch::Sm75,
                                       integer_subbyte<4, Signed>, int4b_t,
                                       ElementDst, int32_t> {
    using ElementSrc = integer_subbyte<4, Signed>;
    static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
    static int const kAlignmentB = 128 / sizeof_bits<ElementSrc>::value;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;
    using ArchTag = arch::Sm75;
    static int const kStages = 2;

    using EpilogueOutputOp = epilogue::thread::BiasAddLinearCombinationClamp<
            ElementDst, 128 / sizeof_bits<ElementDst>::value, int32_t, int32_t,
            float>;

    using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ArchTag, typename ElementDst>
struct DefaultConvolutionConfiguration<arch::OpClassSimt, ArchTag, float, float,
                                       ElementDst, float> {
    static int const kAlignmentSrc = 4;
    static int const kAlignmentFilter = 1;
    using ThreadblockShape = gemm::GemmShape<128, 128, 32>;
    using WarpShape = gemm::GemmShape<32, 64, 32>;
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    static int const kStages = 2;

    using EpilogueOutputOp =
            epilogue::thread::BiasAddLinearCombination<ElementDst, 1, float,
                                                       float, float>;

    using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
