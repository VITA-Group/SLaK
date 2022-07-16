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
  \brief Epilogue for Depthwise convolution

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/dwconv2d_predicated_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <typename Shape_,     /// Threadblock-scoped tile size (concept:
                               /// GemmShape)
          typename Operator_,  /// Warp-scoped epilogue components (concept:
                               /// gemm::warp::Mma)
          typename Element_,   /// Data type of Output tensor
          typename Layout_,    /// Layout of Output Tensor
          typename OutputOp_   /// Function object computing final output
          >
class Dwconv2dWgradDirectEpilogueVoltaTensorOp {
public:
    using Shape = Shape_;
    using Operator = Operator_;
    using Layout = Layout_;
    using Element = Element_;

    /// Number of warps spanning threadblock-scoped tile
    using WarpCount = gemm::GemmShape<Shape::kM / Operator::Shape::kM,
                                      Shape::kN / Operator::Shape::kN,
                                      Shape::kK / Operator::Shape::kK>;

    static_assert(WarpCount::kK == 1,
                  "Depthwise convolution direct epilogue cannot be used with "
                  "when the threadblock "
                  "tile is partitioned along the K dimension.");

    /// Accumulator tile is really the warp-scoped tile
    using AccumulatorTile = typename Operator::FragmentC;

    /// Function operator computing final output
    using OutputOp = OutputOp_;

    /// Reference to Output tensors
    using TensorRef = TensorRef<Element, Layout>;

    /// Logical Layout
    using LogicalLayout = cutlass::layout::RowMajor;

    /// Logical Coordinates
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    /// Tensor Coordinates
    using TensorCoord = typename Layout::TensorCoord;

    /// Policy of operator
    using Policy = typename Operator::Policy;

    /// Underlying matrix multiply operator (concept: arch::Mma)
    using ArchMmaOperator = typename Policy::Operator;

    /// interleaved tile
    using InterleavedTileShape = typename Operator::InterleavedTileShape;

    /// Underlying instruction shape
    using InstructionShape = typename Operator::InstructionShape;

    // Internal constants
    struct Detail {
        static int const kLanesInQuad = 4;
        static int const kRowsPerQuad = 4;
        static int const kColumnsPerQuad = 4;
        static int const kElementsPerAccess = 2;

        static CUTLASS_DEVICE LogicalCoord get_lane_offset(int lane_id) {
            int quad_id = lane_id / Detail::kLanesInQuad;
            int lane_in_quad = (lane_id % Detail::kLanesInQuad);

            int quad_row_offset = (quad_id & 1) * 8 + (quad_id & 4) * 4;
            int quad_col_offset = (quad_id & 2) * 4;

            int thread_row_offset = (lane_in_quad & 1);
            int thread_col_offset = (lane_in_quad & 2);

            int row = quad_row_offset + thread_row_offset;
            int column = quad_col_offset + thread_col_offset;

            return LogicalCoord(row, column);
        }
    };

    /// OutputTileIterator
    using OutputTileIterator =
            Dwconv2dPredicatedAccessTileIterator<Shape, Operator, Element,
                                                 Layout, Detail>;

    /// Number of mma operations performed
    using MmaIterations =
            MatrixShape<InterleavedTileShape::kM / ArchMmaOperator::Shape::kM,
                        InterleavedTileShape::kN / ArchMmaOperator::Shape::kN>;
    using TileIterations =
            MatrixShape<Operator::Shape::kM / InterleavedTileShape::kM,
                        Operator::Shape::kN / InterleavedTileShape::kN>;

    /// Shared storage allocation needed by the epilogue
    struct SharedStorage {};

    /// Constructor
    CUTLASS_DEVICE
    Dwconv2dWgradDirectEpilogueVoltaTensorOp(
            SharedStorage& /* shared_storage */,  ///< Shared storage object
            int /* thread_idx */,  ///< ID of a thread within the threadblock
            int /* warp_idx */,    ///< ID of warp within threadblock
            int /* lane_idx */     ///< Id of thread within warp
    ) {}

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,                ///< Output operator
            OutputTileIterator destination_iterator,  ///< Tile iterator for
                                                      ///< destination
            AccumulatorTile const& accumulators) {    ///< Accumulator tile
        static int const kElementsPerMma =
                ArchMmaOperator::FragmentC::kElements;

        CUTLASS_PRAGMA_UNROLL
        for (int outer_col = 0; outer_col < TileIterations::kColumn;
             ++outer_col) {
            CUTLASS_PRAGMA_UNROLL
            for (int inner_col = 0; inner_col < MmaIterations::kColumn;
                 ++inner_col) {
                CUTLASS_PRAGMA_UNROLL
                for (int outer_row = 0; outer_row < TileIterations::kRow;
                     ++outer_row) {
                    CUTLASS_PRAGMA_UNROLL

                    for (int inner_row = 0; inner_row < MmaIterations::kRow;
                         ++inner_row) {
                        int op_col =
                                inner_col + MmaIterations::kColumn * outer_col;

                        // Column-major serpentine sequence to maximize reuse of
                        // A operand.
                        int inner_row_serp = inner_row;
                        int outer_row_serp = outer_row;
                        if (op_col & 1) {
                            inner_row_serp =
                                    MmaIterations::kRow - inner_row - 1;
                            outer_row_serp =
                                    TileIterations::kRow - outer_row - 1;
                        }
                        int op_idx = inner_row_serp +
                                     MmaIterations::kRow *
                                             (inner_col +
                                              MmaIterations::kColumn *
                                                      (outer_row_serp +
                                                       TileIterations::kRow *
                                                               outer_col));

                        static int const kInnerRowDelta = 4;
                        static int const kInnerColDelta = 4;
                        static int const kOuterRowDelta = 32;
                        static int const kOuterColDelta = 32;
                        LogicalCoord mma_accum_coord(
                                inner_row_serp * kInnerRowDelta +
                                        outer_row_serp * kOuterRowDelta,
                                inner_col * kInnerColDelta +
                                        outer_col * kOuterColDelta);

                        CUTLASS_PRAGMA_UNROLL
                        for (int idx = 0; idx < kElementsPerMma; ++idx) {
                            int idx_row = (idx & 2);
                            int idx_col = 4 * (idx & 4) + (idx & 1);

                            LogicalCoord accum_coord =
                                    mma_accum_coord +
                                    LogicalCoord(idx_row, idx_col);
                            TensorCoord coord;
                            if (destination_iterator.valid(coord,
                                                           accum_coord)) {
                                using FragmentAccumulator =
                                        typename OutputOp::FragmentAccumulator;
                                auto frag_ptr = reinterpret_cast<
                                        FragmentAccumulator const*>(
                                        &accumulators[op_idx * kElementsPerMma +
                                                      idx]);
                                auto output = output_op(*frag_ptr);
                                auto pointer = destination_iterator.get(coord);
                                auto output_ptr =
                                        reinterpret_cast<Element*>(&output[0]);
                                ::atomicAdd(reinterpret_cast<Element*>(pointer),
                                            *output_ptr);
                            }
                        }  // idx
                    }      // inner_row
                }          // outer_row
            }              // inner_col
        }                  // outer_col
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
