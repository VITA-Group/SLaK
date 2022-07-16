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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/
/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_tn_base.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Policy describing tuning details (concept: MmaPolicy)
        typename Policy_,
        /// Number of stages,
        int Stages,
        /// Used for partial specialization
        typename Enable = bool>
class MmaTnBase {
public:
    ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using Shape = Shape_;

    ///< Policy describing tuning details
    using Policy = Policy_;

    //
    // Dependent types
    //

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Shape describing the overall GEMM computed from shared memory
    /// by each warp.
    using WarpGemm = typename Policy::Operator::Shape;

    /// Shape describing the number of warps filling the CTA
    using WarpCount =
            gemm::GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN,
                            Shape::kK / WarpGemm::kK>;

    /// Number of warp-level GEMM oeprations
    static int const kWarpGemmIterations =
            (WarpGemm::kK / Operator::Policy::MmaShape::kK);

    /// Number of stages
    static int const kStages = Stages;

    /// Tensor reference to the Src Tensor operand
    using TensorRefSrc =
            TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    /// Tensor reference to the Filter operand
    using TensorRefFilter =
            TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

    //
    // Nested structs
    //

    /// Shared storage object needed by threadblock-scoped GEMM
    class SharedStorage {
    public:
        //
        // Type definitions
        //

        /// Shape of the A matrix operand in shared memory
        using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                                   Shape::kK * kStages +
                                           Policy::SmemPaddingA::kColumn>;

        /// Shape of the B matrix operand in shared memory
        using ShapeB =
                MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                            Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:
        //
        // Data members
        //

        /// Buffer for Src Tensor operand
        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_src;

        /// Buffer for Filter Tensor operand
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount>
                operand_filter;

    public:
        //
        // Methods
        //

        /// Returns a layout object for the Src Tensor
        CUTLASS_DEVICE
        static typename Operator::LayoutA LayoutSrc() {
            return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
        }

        /// Returns a layout object for the Filter Tensor
        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutFilter() {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        /// Returns a TensorRef to the Src Tensor operand
        CUTLASS_HOST_DEVICE
        TensorRefSrc operand_src_ref() {
            return TensorRefSrc{operand_src.data(), LayoutSrc()};
        }

        /// Returns a TensorRef to the Filter Tensor operand
        CUTLASS_HOST_DEVICE
        TensorRefFilter operand_filter_ref() {
            return TensorRefFilter{operand_filter.data(), LayoutFilter()};
        }
    };

protected:
    //
    // Data members
    //

    /// Iterator to load a warp-scoped tile of Src Tensor operand from shared
    /// memory
    typename Operator::IteratorA warp_tile_iterator_src_;

    /// Iterator to load a warp-scoped tile of Filter Tensor operand from shared
    /// memory
    typename Operator::IteratorB warp_tile_iterator_filter_;

public:
    /// Construct from tensor references
    CUTLASS_DEVICE
    MmaTnBase(
            ///< Shared storage needed for internal use by threadblock-scoped
            ///< GEMM
            SharedStorage& shared_storage,
            ///< ID within the threadblock
            int thread_idx,
            ///< ID of warp
            int warp_idx,
            ///< ID of each thread within a warp
            int lane_idx)
            : warp_tile_iterator_src_(shared_storage.operand_src_ref(),
                                      lane_idx),
              warp_tile_iterator_filter_(shared_storage.operand_filter_ref(),
                                         lane_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
