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
class Dwconv2dWgradDirectEpilogueSimt {
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

    using Policy = typename Operator::Policy;
    /// Shape of the warp in lanes
    using WarpShape = typename Policy::WarpShape;
    /// Layout function of lanes
    using LaneLayout = typename Policy::LaneLayout;
    /// size of each lane's thread-level matrix product
    using LaneMmaShape = typename Policy::LaneMmaShape;

    /// Logical Layout
    using LogicalLayout = cutlass::layout::RowMajor;

    /// Logical Coordinates
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    /// Tensor Coordinates
    using TensorCoord = typename Layout::TensorCoord;

    struct Detail {
        static CUTLASS_DEVICE LogicalCoord get_lane_offset(int lane_idx) {
            auto lane_layout = Policy::get_lane_layout();
            LogicalCoord lane_offset =
                    lane_layout.inverse(lane_idx) *
                    LogicalCoord(LaneMmaShape::kM, LaneMmaShape::kN);
            return lane_offset;
        }
    };

    /// OutputTileIterator
    using OutputTileIterator =
            Dwconv2dPredicatedAccessTileIterator<Shape, Operator, Element,
                                                 Layout, Detail>;

    /// Accumulator tile shape of each lane
    using AccumulatorTileShape =
            MatrixShape<Operator::Shape::kM / WarpShape::kRow,
                        Operator::Shape::kN / WarpShape::kColumn>;

    /// Number of mma operations performed
    using MmaIterations =
            MatrixShape<AccumulatorTileShape::kRow / LaneMmaShape::kM,
                        AccumulatorTileShape::kColumn / LaneMmaShape::kN>;

public:
    /// Shared storage allocation needed by the epilogue
    struct SharedStorage {};

private:
public:
    /// Constructor
    CUTLASS_DEVICE
    Dwconv2dWgradDirectEpilogueSimt(
            SharedStorage& /* shared_storage */,  ///< Shared storage object
            int /* thread_idx */,  ///< ID of a thread within the threadblock
            int /* warp_idx */,    ///< ID of warp within threadblock
            int /* lane_idx */     ///< Id of thread within warp
    ) {}

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const& accumulators) {  ///< Accumulator tile
        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < MmaIterations::kRow; ++mma_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n = 0; mma_n < MmaIterations::kColumn; ++mma_n) {
                int mma_accum_start = mma_m * LaneMmaShape::kM *
                                              AccumulatorTileShape::kColumn +
                                      mma_n * LaneMmaShape::kN;
                LogicalCoord mma_accum_coord(
                        mma_m * WarpShape::kRow * LaneMmaShape::kM,
                        mma_n * WarpShape::kColumn * LaneMmaShape::kN);

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < LaneMmaShape::kM; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int col = 0; col < LaneMmaShape::kN; ++col) {
                        int idx = mma_accum_start +
                                  row * AccumulatorTileShape::kColumn + col;
                        LogicalCoord accum_coord =
                                mma_accum_coord + MatrixCoord(row, col);

                        TensorCoord coord;
                        if (destination_iterator.valid(coord, accum_coord)) {
                            using FragmentAccumulator =
                                    typename OutputOp::FragmentAccumulator;
                            auto frag_ptr = reinterpret_cast<
                                    FragmentAccumulator const*>(
                                    &accumulators[idx]);
                            auto output = output_op(*frag_ptr);
                            auto pointer = destination_iterator.get(coord);
                            auto output_ptr =
                                    reinterpret_cast<Element*>(&output[0]);
                            ::atomicAdd(reinterpret_cast<Element*>(pointer),
                                        *output_ptr);
                        }
                    }
                }
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
