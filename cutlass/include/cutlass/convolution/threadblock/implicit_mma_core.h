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
/*! \file
    \brief Defines basic properties needed by CTA-level Convolutions assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp
   instructions.
*/

/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_core.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/cache_operation.h"

#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/convolution/threadblock/conv2d_tile_params.h"
#include "cutlass/convolution/threadblock/implicit_mma_pipelined.h"
#include "cutlass/convolution/threadblock/implicit_mma_singlestage.h"
#include "cutlass/convolution/threadblock/implicit_mma_nt_precomp.h"
#include "cutlass/convolution/threadblock/implicit_mma_tn_precomp.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Shape of threadblock-scoped matrix multiply operator
        typename Shape,
        /// Shape of warp-level matrix multiply operator
        typename WarpShape,
        /// Shape of one matrix production operation (concept: GemmShape)
        typename InstructionShape,
        /// Element data type of Src Tensor operand
        typename ElementSrc,
        /// Layout of operand Src Tensor
        typename LayoutSrc,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Element data type of Filter Tensor operand
        typename ElementFilter,
        /// Layout of operand Filter Tensor
        typename LayoutFilter,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Data type of accumulator
        typename ElementDst,
        /// Layout of accumulator
        typename LayoutDst,
        /// Indicates type of math operator (arch::OpClassSimt or
        /// arch::OpClassTensorOp)
        typename OperatorClass,
        /// Number of stages
        int Stages = 2,
        /// Operation performed by MMA
        typename Operator = typename platform::conditional<
                (platform::is_same<OperatorClass,
                                   cutlass::arch::OpClassTensorOp>::value) &&
                        (platform::is_same<ElementSrc, int8_t>::value ||
                         platform::is_same<ElementSrc, int4b_t>::value ||
                         platform::is_same<ElementSrc, uint8_t>::value ||
                         platform::is_same<ElementSrc, uint4b_t>::value),
                cutlass::arch::OpMultiplyAddSaturate,
                cutlass::arch::OpMultiplyAdd>::type,
        /// Store the accumulators in row major or column major.  Row major is
        /// used when output layout is interleaved.
        bool AccumulatorsInRowMajor = false,
        /// Implicit Gemm Mode
        ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
        /// Cache operation of operand A
        cutlass::arch::CacheOperation::Kind CacheOpSrc =
                cutlass::arch::CacheOperation::Global,
        /// Cache operation of operand B
        cutlass::arch::CacheOperation::Kind CacheOpFilter =
                cutlass::arch::CacheOperation::Global>
struct DefaultMmaCore;

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
