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
    \brief Defines basic properties needed by CTA-level GEMMs assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting simt
   instructions.
*/

/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_core_sm75.h
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

/// Partial specialization for first partitioning in contiguous dimension, used
/// for Convolution kernel(TensorOp) to reduce register count
template <typename Shape_, int Threads, typename WarpThreadArrangement_,
          int ElementsPerAccess = 1>
struct PitchLinearWarpRakedThreadMapOpt {
    /// Tensor coordinate
    using TensorCoord = layout::PitchLinearCoord;

    /// Tile shape
    using Shape = Shape_;

    /// Number of threads total
    static int const kThreads = Threads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ElementsPerAccess;

    /// Shape of access by each thread
    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    /// Internal details made public to facilitate introspection
    struct Detail {
        /// Fixed arrangement of threads within a warp (units of threads).
        using WarpThreadArrangement = WarpThreadArrangement_;

        /// Number of threads per warp
        static int const kWarpSize = WarpThreadArrangement::kCount;

        /// Number of participating warps
        static int const kWarpCount = kThreads / kWarpSize;

        static_assert(!(Shape::kContiguous % kElementsPerAccess),
                      "Shape must be divisible by vector length.");

        /// Compute the 'shape' of the overall tile in units of vectors
        using ShapeInAccesses = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        // compute number of warp-level accesses total
        using WarpAccessIterations = layout::PitchLinearShape<
                ShapeInAccesses::kContiguous /
                        WarpThreadArrangement::kContiguous,
                ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided>;

        // Divide it into the number of warps, first partitioning the strided
        // dimension then the contiguous.
        static int const kWarpsContiguous =
                (WarpAccessIterations::kContiguous >= kWarpCount
                         ? kWarpCount
                         : WarpAccessIterations::kContiguous);

        static int const kWarpsStrided =
                (kWarpCount > WarpAccessIterations::kContiguous
                         ? kWarpCount / kWarpsContiguous
                         : 1);

        /// Arrangement of warps within a threadblock-scoped tile
        using WarpArrangement =
                layout::PitchLinearShape<kWarpsContiguous, kWarpsStrided>;
    };

    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
            layout::PitchLinearShape<Detail::WarpAccessIterations::kContiguous /
                                             Detail::kWarpsContiguous,
                                     Detail::WarpAccessIterations::kStrided /
                                             Detail::kWarpsStrided>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
    using Delta = layout::PitchLinearShape<
            Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
            Detail::WarpThreadArrangement::kStrided>;

    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space
    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);

        //
        // compute warp-level offset
        //

        // This is the shape of the entire area covered by a warp's memory
        // access (in units of vectors)
        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

        // This is the offset of a specific warp (in units of vectors)
        layout::PitchLinearCoord warp_offset{
                (warp_id % Detail::kWarpsContiguous),
                (warp_id / Detail::kWarpsContiguous)};

        // This is the offset of a specific thread within a warp (units of
        // vectors)
        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        // This is the offset of a thread within a threadblock tile (units of
        // vectors)
        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        // This is the offset of a thread within a threadblock tile (units of
        // elements)
        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   Src Tensor    : layout::TensorNCxHWx<32>
///   Filter Tensor : layout::TensorCxRSKx<32>
///   Operator      : TensorOp class, for mma i8816
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                      LayoutFilter_, kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 32;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

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

    // Warp thread arrangement
    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Src
    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 1,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Filter
    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            0, SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level Tensor Op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            InstructionShape,  /// Instruction-level Gemm shape - concept
                               /// gemm::GemmShape
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst
                               /// Tensor's matrix
                               /// (concept:
                               /// MatrixLayout)
            Operator,          /// Operator describing the tensor operation
            PartitionsK,       /// Number of partitions along K dimension
            AccumulatorsInRowMajor>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/// Mma TN
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                      LayoutFilter_, kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, true,
                      ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 32;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

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

    // Warp thread arrangement
    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    //
    // Shared memory layouts
    //
    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Src
    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Filter
    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;
    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level Tensor Op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            InstructionShape,  /// Instruction-level Gemm shape - concept
                               /// gemm::GemmShape
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst
                               /// Tensor's matrix
                               /// (concept:
                               /// MatrixLayout)
            Operator,          /// Operator describing the tensor operation
            PartitionsK,       /// Number of partitions along K dimension
            AccumulatorsInRowMajor>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   Src Tensor    : layout::TensorNCxHWx<64>
///   Filter Tensor : layout::TensorCxRSKx<64>
///   Operator      : TensorOp class, for mma i8832
///
/// This uses the default warp-level operator given tile sizes
template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNCxHWx<64>,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNCxHWx<64>;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 64;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

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

    // Warp thread arrangement
    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Src
    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 1,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Filter
    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            0, SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level Tensor Op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            InstructionShape,  /// Instruction-level Gemm shape - concept
                               /// gemm::GemmShape
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst
                               /// Tensor's matrix
                               /// (concept:
                               /// MatrixLayout)
            Operator,          /// Operator describing the tensor operation
            PartitionsK,       /// Number of partitions along K dimension
            AccumulatorsInRowMajor>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/// Mma TN
template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNCxHWx<64>,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, true, ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNCxHWx<64>;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 64;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

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

    // Warp thread arrangement
    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    //
    // Shared memory layouts
    //
    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator Src
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Src
    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Src Tensor operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    /// Policy of iterator Filter
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    /// Transpose the ThreadMap of iterator Filter
    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    /// Shared memory iterator to Filter Tensor operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level Tensor Op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,         /// Size of the Gemm problem - concept:
                               /// gemm::GemmShape<> 128, 128, 8
            InstructionShape,  /// Instruction-level Gemm shape - concept
                               /// gemm::GemmShape
            ElementSrc,        /// Data type of Src Tensor elements
            SmemLayoutSrc,     /// Layout of Src Tensor's matrix (concept:
                               /// MatrixLayout)
            ElementFilter,     /// Data type of Filter Tensor elements
            SmemLayoutFilter,  /// Layout of Filter Tensor's Matrix (concept:
                               /// MatrixLayout)
            ElementDst,        /// Element type of Dst Tensor matrix
            layout::RowMajor,  /// Layout of Dst
                               /// Tensor's matrix
                               /// (concept:
                               /// MatrixLayout)
            Operator,          /// Operator describing the tensor operation
            PartitionsK,       /// Number of partitions along K dimension
            AccumulatorsInRowMajor>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///
///   Src Tensor    : layout::TensorNHWC
///   Filter Tensor : layout::TensorNCxHWx<AccessSize>
///   Operator      : TensorOp class, for mma i8832
///
/// This uses the default warp-level operator given tile sizes
template <
        /// ElementSrc is int4b_t or uint4b_t
        bool Signed,
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNHWC,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, false, ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNHWC;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    /// Default Operator
    using Operator = Operator_;

    /// Number of threads per warp
    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Size of a threadblock-scoped access
    static int const kAccessSizeInBits = 128;

    // Warp thread arrangement
    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, Shape::kK>;

    // Shared memory layout
    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, Shape::kK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator A
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    /// Shared memory iterator to A operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    /// ThreadMap of iterator B
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    /// Shared memory iterator to B operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, IteratorThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level tensor op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementSrc, SmemLayoutSrc,
            ElementFilter, SmemLayoutFilter, ElementDst, layout::RowMajor,
            Operator, WarpCount::kK>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/// Partial specialization:
///
///
///   Src Tensor    : layout::TensorNHWC
///   Filter Tensor : layout::TensorNCxHWx<AccessSize>
///   Operator      : TensorOp class, for mma i8816
///
/// This uses the default warp-level operator given tile sizes
template <
        /// Shape of threadblock-scoped matrix multiply operator (concept:
        /// GemmShape)
        typename Shape_,
        /// Shape of warp-level matrix multiply operator (concept: GemmShape)
        typename WarpShape_,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Layout of filter
        typename LayoutFilter_,
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
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNHWC, kAlignmentSrc, int8_t, LayoutFilter_,
                      kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, false,
                      ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNHWC;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    /// Default Operator
    using Operator = Operator_;

    /// Number of threads per warp
    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    /// Number of threads total
    static int const kThreads = WarpCount::kCount * kWarpSize;

    /// Size of a threadblock-scoped access
    static int const kAccessSizeInBits = 128;

    // Warp thread arrangement
    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;

    //
    // Shared memory layouts
    //

    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, Shape::kK>;

    // Shared memory layout
    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, Shape::kK>;

    //
    // Iterators to write to shared memory
    //

    /// ThreadMap of iterator A
    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    /// Shared memory iterator to A operand
    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    /// ThreadMap of iterator B
    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    /// Shared memory iterator to B operand
    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, IteratorThreadMapFilter>;

    //
    // Warp-level matrix multiply operator
    //

    // Define the warp-level tensor op
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementSrc, SmemLayoutSrc,
            ElementFilter, SmemLayoutFilter, ElementDst, layout::RowMajor,
            Operator, WarpCount::kK>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass
