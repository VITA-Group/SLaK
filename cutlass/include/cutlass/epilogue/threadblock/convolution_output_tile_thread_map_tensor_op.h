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
  \brief Metaprogram for determining the mapping of output elements to threads
  for epilogue tiles.


*/

/**
 * \file
 * include/cutlass/epilogue/threadblock/convolution_output_tile_thread_map.h
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
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <typename Shape_,        ///< Ouput tile shape: concept MatrixShape
          typename Count_,        ///< Output tile count: concept Matrix Shape
          typename WarpCount_,    ///< Warp layout: concept Matrix Shape
          int Interleaved,        ///< Interleaving quantity
          int Threads,            ///< Number of threads
          int ElementsPerAccess,  ///< Elements per access
          int ElementSize         ///< Element size in bits
          >
struct ConvolutionOutputTileOptimalThreadMapTensorOp;

template <typename Shape_, typename Count_, typename WarpCount_,
          int Interleaved, int Threads, int ElementsPerAccess, int ElementSize>
struct ConvolutionOutputTileOptimalThreadMapTensorOp {
    using Shape = Shape_;
    using Count = Count_;
    using WarpCount = WarpCount_;

    static int const kWarpSize = 32;
    static int const kInterleaved = Interleaved;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;

    //
    // Metaprogram computation
    //

    struct Detail {
        static_assert(!(Shape::kRow % kInterleaved),
                      "Warp access rows per iteration must be divided by "
                      "interleaving quantity");

        using OutputTileShape =
                layout::PitchLinearShape<Shape::kColumn * kInterleaved,
                                         Shape::kRow / kInterleaved>;
        using ShapeInAccess =
                layout::PitchLinearShape<OutputTileShape::kContiguous /
                                                 kElementsPerAccess,
                                         OutputTileShape::kStrided>;

        using ThreadArrangement = layout::PitchLinearShape<kWarpSize, 1>;
        static_assert(!(ShapeInAccess::kContiguous %
                        ThreadArrangement::kContiguous) &&
                              !(ShapeInAccess::kContiguous %
                                ThreadArrangement::kStrided),
                      "Divisibility");
    };

    //
    // Output
    //

    using Iterations =
            MatrixShape<Detail::ShapeInAccess::kStrided /
                                Detail::ThreadArrangement::kStrided,
                        Detail::ShapeInAccess::kContiguous /
                                Detail::ThreadArrangement::kContiguous>;
    using Delta =
            MatrixShape<Detail::ThreadArrangement::kStrided * kInterleaved,
                        Detail::ThreadArrangement::kContiguous *
                                kElementsPerAccess / kInterleaved>;

    /// Initial offset function
    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        MatrixCoord warp_footprint{Shape::kRow * Count::kRow,
                                   Shape::kColumn * Count::kColumn};
        MatrixCoord warp_offset{warp_idx % WarpCount::kRow,
                                warp_idx / WarpCount::kRow};
        int row_idx_in_warp = lane_idx / Detail::ThreadArrangement::kContiguous;
        int col_idx_in_warp = lane_idx % Detail::ThreadArrangement::kContiguous;
        int col_offset_in_warp = col_idx_in_warp * kElementsPerAccess;
        MatrixCoord thread_offset_in_warp{
                row_idx_in_warp * kInterleaved +
                        col_offset_in_warp % kInterleaved,
                col_offset_in_warp / kInterleaved};

        return warp_offset * warp_footprint + thread_offset_in_warp;
    }

    /// Compacted thread map in which the 4D region is contiguous
    struct CompactedThreadMap {
        using Shape = Shape_;

        using Iterations =
                MatrixShape<Detail::ShapeInAccess::kStrided /
                                    Detail::ThreadArrangement::kStrided,
                            Detail::ShapeInAccess::kContiguous /
                                    Detail::ThreadArrangement::kContiguous>;
        using Delta =
                MatrixShape<Detail::ThreadArrangement::kStrided * Interleaved,
                            Detail::ThreadArrangement::kContiguous *
                                    kElementsPerAccess / Interleaved>;

        /// Interleaving quantity of fragment loaded by SharedLoadIterator
        static int const kRowsPerIteration = ElementsPerAccess;

        /// Number of elements within each vector access
        static int const kElementsPerAccess = 1;
        static int const kInterleaved = Interleaved;

        /// Number  of threads
        static int const kThreads = Threads;

        /// Function to compute each thread's initial offset
        CUTLASS_HOST_DEVICE
        static MatrixCoord initial_offset(int thread_idx) {
            int warp_idx = thread_idx / kWarpSize;
            int lane_idx = thread_idx % kWarpSize;

            MatrixCoord warp_footprint{Shape::kRow, Shape::kColumn};
            MatrixCoord warp_offset{warp_idx % WarpCount::kRow,
                                    warp_idx / WarpCount::kRow};
            int row_idx_in_warp =
                    lane_idx / Detail::ThreadArrangement::kContiguous;
            int col_idx_in_warp =
                    lane_idx % Detail::ThreadArrangement::kContiguous;
            int col_offset_in_warp = col_idx_in_warp * ElementsPerAccess;
            MatrixCoord thread_offset_in_warp{
                    row_idx_in_warp * kInterleaved +
                            col_offset_in_warp % kInterleaved,
                    col_offset_in_warp / kInterleaved};

            return warp_offset * warp_footprint + thread_offset_in_warp;
        }
    };
};

////////////////////////////////////////////////////////////////////////////////

/// Template metaprogram for partitioning a 4D interleaved layout across warps
/// to achieve several performance objectives:
///
///   - coalesced memory accesses in units of 64 Byte lines
///   - minimal address arithmetic
///   - minimal predicate calculations
///
template <typename Shape_, typename WarpCount_, typename Iterations_,
          int Threads, int ElementsPerAccess, int ElementSize>
struct InterleavedConvolutionOutputTileThreadMap {
    using Shape = Shape_;
    using Count = Iterations_;
    using WarpCount = WarpCount_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;

    //
    // Output
    //

    using Iterations = Iterations_;

    using Type =
            InterleavedConvOutputTileThreadMap<WarpCount, Iterations, Threads,
                                               ElementsPerAccess, ElementSize>;

    using Delta = typename Type::Delta;

    /// Initial offset function
    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        return Type::initial_offset(thread_idx);
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
