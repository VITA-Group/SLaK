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
 * \file include/cutlass/convolution/threadblock/implicit_mma_core_sm70.h
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
#include "cutlass/fast_math.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/layout/tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   A: layout::TensorNCHW
///   B: layout::TensorNCHW
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Element data type of Src Tensor operand
        typename ElementSrc_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Element data type of Filter Tensor operand
        typename ElementFilter_,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Data type of accumulator
        typename ElementDst_,
        /// Layout of accumulator
        typename LayoutDst_,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by Convolution
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 4>, ElementSrc_,
                      layout::TensorNCHW, kAlignmentSrc, ElementFilter_,
                      layout::TensorNCHW, kAlignmentFilter, ElementDst_,
                      LayoutDst_, arch::OpClassTensorOp, Stages, Operator_,
                      true, ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using ElementSrc = ElementSrc_;
    using LayoutSrc = layout::TensorNCHW;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = layout::TensorNCHW;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    // Divisility requirements
    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    /// Number of threads per warp
    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Size of a threadblock-scoped access
    static int const kAccessSizeInBits = 128;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, Shape::kK>;

    // Shared memory layout
    using SmemLayoutFilter =
            layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
                    sizeof_bits<ElementFilter>::value>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<4, 8>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            0, IteratorThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //
    using LogicalLayoutDst = cutlass::layout::RowMajor;

    // Define the warp-level tensor op
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementSrc, cutlass::layout::RowMajor,
                               ElementFilter, cutlass::layout::RowMajor,
                               ElementDst, LogicalLayoutDst,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1>>;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementSrc, SmemLayoutSrc, ElementFilter,
            SmemLayoutFilter, ElementDst, LogicalLayoutDst, Policy>;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   A: layout::TensorNCHW
///   B: layout::TensorNCHW
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Element data type of Src Tensor operand
        typename ElementSrc_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Element data type of Filter Tensor operand
        typename ElementFilter_,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter,
        /// Data type of accumulator
        typename ElementDst_,
        /// Layout of accumulator
        typename LayoutDst_,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by Convolution
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 4>, ElementSrc_,
                      layout::TensorNCHW, kAlignmentSrc, ElementFilter_,
                      layout::TensorNCHW, kAlignmentFilter, ElementDst_,
                      LayoutDst_, arch::OpClassTensorOp, Stages, Operator_,
                      true, ImplicitGemmMode::GEMM_NT> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using ElementSrc = ElementSrc_;
    using LayoutSrc = layout::TensorNCHW;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = layout::TensorNCHW;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    /// Default Operator
    using Operator = Operator_;

    /// Number of warps present
    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    // Divisility requirements
    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    /// Number of threads per warp
    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Size of a threadblock-scoped access
    static int const kAccessSizeInBits = 128;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
            sizeof_bits<ElementSrc>::value>;

    // Shared memory layout
    using SmemLayoutFilter =
            layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
                    sizeof_bits<ElementFilter>::value>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            1, IteratorThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //
    using LogicalLayoutDst = cutlass::layout::RowMajor;

    // Define the warp-level tensor op
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementFilter, cutlass::layout::ColumnMajor,
                               ElementSrc, cutlass::layout::RowMajor,
                               ElementDst, LogicalLayoutDst,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1>>;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementFilter, SmemLayoutFilter, ElementSrc,
            SmemLayoutSrc, ElementDst, LogicalLayoutDst, Policy>;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass
