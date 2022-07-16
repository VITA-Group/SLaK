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
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// InterleavedRowArrangement determines how one or more warps cover a region of
/// consecutive rows.
template <typename Shape, int Interleaved, int WarpsRemaining,
          int ElementsPerAccess, int ElementSize, bool Is2dTile>
struct InterleavedRowArrangement;

/// TensorRowArrangement in which each warp's access is a 1D tiled arrangement.
template <typename Shape, int Interleaved, int WarpsRemaining,
          int ElementsPerAccess, int ElementSize>
struct InterleavedRowArrangement<Shape, Interleaved, WarpsRemaining,
                                 ElementsPerAccess, ElementSize, false> {
    static int const kWarpSize = 32;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;
    static int const kInterleaved = Interleaved;
    static_assert(Shape::kRow == kInterleaved,
                  "Shape::kRow should be equal to interleave factor");
    static_assert(WarpsRemaining == 1, "WarpRemaining must be 1");
    using InterleavedShape = MatrixShape<Shape::kRow / Interleaved,
                                         Shape::kColumn * Interleaved>;

    static int const kIterationsRow = 1;
    static int const kDeltaRow = Interleaved;
    static int const kIterationsColumn =
            InterleavedShape::kColumn / (kElementsPerAccess * kWarpSize);
    static int const kDeltaColumn =
            kWarpSize * kElementsPerAccess / Interleaved;

    static int const kAccessWidth = kWarpSize;
    static int const kAccessRows = 1;
    static int const kWarpPartitionsRow = 1;
    static int const kWarpPartitionsColumn = 1;
};

/// Specialization for interleaving-quantity = 32
template <typename Shape, int WarpsRemaining, int ElementsPerAccess,
          int ElementSize>
struct InterleavedRowArrangement<Shape, 32, WarpsRemaining, ElementsPerAccess,
                                 ElementSize, false> {
    static int const kWarpSize = 32;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;
    static int const kInterleaved = 32;
    static_assert(!(Shape::kRow % kInterleaved),
                  "Shape::kRow should be a multiple of interleaving quantity");
    using InterleavedShape = MatrixShape<Shape::kRow / kInterleaved,
                                         Shape::kColumn * kInterleaved>;

    static_assert(!(InterleavedShape::kColumn % kElementsPerAccess),
                  "Divisibility");
    using ShapeVec =
            MatrixShape<InterleavedShape::kRow,
                        InterleavedShape::kColumn / kElementsPerAccess>;

    static_assert(!(kWarpSize % ShapeVec::kColumn) ||
                          (ShapeVec::kColumn >= kWarpSize * WarpsRemaining &&
                           !(ShapeVec::kColumn %
                             (kWarpSize * WarpsRemaining))) ||
                          (!(ShapeVec::kColumn % kWarpSize) &&
                           !(WarpsRemaining % (ShapeVec::kColumn / kWarpSize))),
                  "Divisibility");
    static int const kAccessWidth =
            kWarpSize >= ShapeVec::kColumn ? ShapeVec::kColumn : kWarpSize;
    static int const kAccessRows = kWarpSize / kAccessWidth;

    static int const kWarpPartitionsColumn =
            kWarpSize >= ShapeVec::kColumn
                    ? 1
                    : ShapeVec::kColumn >= kWarpSize * WarpsRemaining
                              ? WarpsRemaining
                              : ShapeVec::kColumn / kWarpSize;
    static int const kWarpPartitionsRow =
            WarpsRemaining / kWarpPartitionsColumn;

    static int const kIterationsRow =
            kWarpSize >= ShapeVec::kColumn
                    ? cutlass::const_max(ShapeVec::kRow / ((kWarpSize /
                                                            ShapeVec::kColumn) *
                                                           WarpsRemaining),
                                         1)
                    : cutlass::const_max((ShapeVec::kRow / kWarpPartitionsRow),
                                         1);
    static int const kIterationsColumn =
            kWarpSize >= ShapeVec::kColumn
                    ? 1
                    : (ShapeVec::kColumn / (kWarpSize * kWarpPartitionsColumn));

    static int const kDeltaRow = kAccessRows * kInterleaved;
    static int const kDeltaColumn =
            kAccessWidth * kElementsPerAccess / kInterleaved;
};
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////

template <typename Shape_,        ///< Output tile shape
          typename Count_,        ///< Output tile count
          int Interleaved,        ///< Interleaving quantity
          int Threads,            ///< Number of threads
          int ElementsPerAccess,  ///< Number of elements per memory access
          int ElementSize         ///< Element size in bits
          >
struct ConvolutionOutputTileOptimalThreadMap;

template <typename Shape_, typename Count_, int Interleaved, int Threads,
          int ElementsPerAccess, int ElementSize>
struct ConvolutionOutputTileOptimalThreadMap {
    using Shape = Shape_;
    using Count = Count_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;
    static int const kInterleaved = Interleaved;

    //
    // Metaprogram computation
    //

    struct Detail {
        // Clusters
        static int const kIterationsCluster =
                ((Shape::kCluster > kWarpCount) ? Shape::kCluster / kWarpCount
                                                : 1);

        static int const kDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kRow * Count::kRow * Shape::kGroup *
                                   Count::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kCompactedDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kRow * Shape::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kWarpPartitionsCluster =
                ((Shape::kCluster > kWarpCount) ? kWarpCount
                                                : kWarpCount / Shape::kCluster);

        static int const kWarpsRemainingForGroups =
                ((Shape::kCluster > kWarpCount) ? 1
                                                : kWarpCount / Shape::kCluster);

        // Groups
        static int const kIterationsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kGroup / kWarpsRemainingForGroups
                         : 1);

        static int const kDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kRow * Count::kRow * Shape::kGroup /
                                   kIterationsGroup
                         : 1);

        static int const kCompactedDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kRow * Shape::kGroup / kIterationsGroup
                         : 1);

        static int const kWarpPartitionsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        static int const kWarpsRemainingForRows =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        // Rows
        using RowArrangement = detail::InterleavedRowArrangement<
                Shape, Interleaved, kWarpsRemainingForRows, kElementsPerAccess,
                kElementSize, false>;

        // Warp partitions
        using WarpPartitions =
                OutputTileShape<RowArrangement::kWarpPartitionsColumn,
                                RowArrangement::kWarpPartitionsRow,
                                kWarpPartitionsGroup, kWarpPartitionsCluster,
                                1>;

        static int const kAccessWidth = RowArrangement::kAccessWidth;
        static int const kAccessRows = RowArrangement::kAccessRows;
    };

    //
    // Output
    //

    using Iterations =
            OutputTileShape<Detail::RowArrangement::kIterationsColumn,
                            Detail::RowArrangement::kIterationsRow,
                            Detail::kIterationsGroup,
                            Detail::kIterationsCluster, 1>;

    using Delta =
            OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                            Detail::RowArrangement::kDeltaRow,
                            Detail::kDeltaGroup, Detail::kDeltaCluster, 1>;

    /// Initial offset function
    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
       
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        // Compute warp location
        int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
        int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

        int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
        int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

        int row_idx = residual_group / Detail::WarpPartitions::kRow;
        int col_idx = residual_group % Detail::WarpPartitions::kRow;

        // Compute per-lane offset
        int lane_row_offset = lane_idx / Detail::kAccessWidth;
        int lane_col_offset = lane_idx % Detail::kAccessWidth;

        // Compute coordinate in output space
        int cluster_offset = cluster_idx * Shape::kRow * Count::kRow *
                             Shape::kGroup * Count::kGroup;
        int group_offset = group_idx * Shape::kRow * Count::kRow;
        int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
        int column_offset =
                col_idx * Iterations::kColumn * Detail::kAccessWidth;

        return MatrixCoord(
                cluster_offset + group_offset +
                        (row_offset + lane_row_offset) * kInterleaved,
                (column_offset + lane_col_offset) * kElementsPerAccess /
                        kInterleaved);
    }

    /// Compacted thread map in which the 4D region is contiguous
    struct CompactedThreadMap {
        using Shape = Shape_;

        using Iterations = OutputTileShape<
                Detail::RowArrangement::kIterationsColumn,
                Detail::RowArrangement::kIterationsRow * kInterleaved,
                Detail::kIterationsGroup, Detail::kIterationsCluster, 1>;

        using Delta = OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                                      Detail::RowArrangement::kDeltaRow /
                                              kInterleaved,
                                      Detail::kCompactedDeltaGroup,
                                      Detail::kCompactedDeltaCluster, 1>;

        /// Number of elements within each vector access
        static int const kElementsPerAccess = ElementsPerAccess / Interleaved;

        /// Number  of threads
        static int const kThreads = Threads;

        /// Function to compute each thread's initial offset
        CUTLASS_HOST_DEVICE
        static MatrixCoord initial_offset(int thread_idx) {
            int warp_idx = thread_idx / kWarpSize;
            int lane_idx = thread_idx % kWarpSize;

            // Compute warp location
            int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
            int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

            int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
            int residual_group =
                    residual_cluster % Detail::WarpPartitions::kGroup;

            int row_idx = residual_group / Detail::WarpPartitions::kRow;
            int col_idx = residual_group % Detail::WarpPartitions::kRow;

            // Compute per-lane offset
            int lane_row_offset = lane_idx / Detail::kAccessWidth;
            int lane_col_offset = lane_idx % Detail::kAccessWidth;

            // Compute coordinate in output space
            int cluster_offset = cluster_idx * Shape::kRow * Shape::kGroup;
            int group_offset = group_idx * Shape::kRow;
            int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
            int column_offset =
                    col_idx * Iterations::kColumn * Detail::kAccessWidth;

            MatrixCoord coord(
                    cluster_offset + group_offset +
                            (row_offset + lane_row_offset) * kInterleaved,
                    (column_offset + lane_col_offset) * kElementsPerAccess);

            return coord;
        }
    };
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for interleaving quantity = 32
template <typename Shape_, typename Count_, int Threads, int ElementsPerAccess,
          int ElementSize>
struct ConvolutionOutputTileOptimalThreadMap<Shape_, Count_, 32, Threads,
                                             ElementsPerAccess, ElementSize> {
    using Shape = Shape_;
    using Count = Count_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;
    static int const kInterleaved = 32;
    static_assert(kElementsPerAccess == 4, "Elements per access must be 4");

    //
    // Metaprogram computation
    //

    struct Detail {
        // Clusters
        static int const kIterationsCluster =
                ((Shape::kCluster > kWarpCount) ? Shape::kCluster / kWarpCount
                                                : 1);

        static int const kDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kColumn * Count::kColumn * Shape::kGroup *
                                   Count::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kCompactedDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kColumn * Shape::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kWarpPartitionsCluster =
                ((Shape::kCluster > kWarpCount) ? kWarpCount
                                                : kWarpCount / Shape::kCluster);

        static int const kWarpsRemainingForGroups =
                ((Shape::kCluster > kWarpCount) ? 1
                                                : kWarpCount / Shape::kCluster);

        // Groups
        static int const kIterationsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kGroup / kWarpsRemainingForGroups
                         : 1);

        static int const kDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kColumn * Count::kColumn * Shape::kGroup /
                                   kIterationsGroup
                         : 1);

        static int const kCompactedDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kColumn * Shape::kGroup / kIterationsGroup
                         : 1);

        static int const kWarpPartitionsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        static int const kWarpsRemainingForColumns =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        // Columns
        using RowArrangement = detail::InterleavedRowArrangement<
                Shape, kInterleaved, kWarpsRemainingForColumns,
                kElementsPerAccess, kElementSize, false>;

        // Warp partitions
        using WarpPartitions =
                OutputTileShape<RowArrangement::kWarpPartitionsColumn,
                                RowArrangement::kWarpPartitionsRow,
                                kWarpPartitionsGroup, kWarpPartitionsCluster,
                                1>;

        static int const kAccessWidth = RowArrangement::kAccessWidth;
        static int const kAccessRows = RowArrangement::kAccessRows;
    };

    //
    // Output
    //

    using Iterations =
            OutputTileShape<Detail::RowArrangement::kIterationsColumn,
                            Detail::RowArrangement::kIterationsRow,
                            Detail::kIterationsGroup,
                            Detail::kIterationsCluster, 1>;

    using Delta =
            OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                            Detail::RowArrangement::kDeltaRow,
                            Detail::kDeltaGroup, Detail::kDeltaCluster, 1>;

    /// Initial offset function
    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        // Compute warp location
        int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
        int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

        int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
        int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

        int row_idx = residual_group / Detail::WarpPartitions::kColumn;
        int col_idx = residual_group % Detail::WarpPartitions::kColumn;

        // Compute per-lane offset
        int lane_row_offset = lane_idx / Detail::kAccessWidth;
        int lane_col_offset = lane_idx % Detail::kAccessWidth;

        // Compute coordinate in output space
        int cluster_offset = cluster_idx * Shape::kColumn * Count::kColumn *
                             Shape::kGroup * Count::kGroup;
        int group_offset = group_idx * Shape::kColumn * Count::kColumn;
        int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
        int column_offset =
                col_idx * Iterations::kColumn * Detail::kAccessWidth;

        return MatrixCoord((row_offset + lane_row_offset) * kInterleaved +
                                   ((column_offset + lane_col_offset) *
                                    kElementsPerAccess) %
                                           kInterleaved,
                           (cluster_offset + group_offset) +
                                   (column_offset + lane_col_offset) *
                                           kElementsPerAccess / kInterleaved);
    }

    /// Compacted thread map in which the 4D region is contiguous
    struct CompactedThreadMap {
        using Shape = Shape_;

        using Iterations =
                OutputTileShape<Detail::RowArrangement::kIterationsColumn,
                                Detail::RowArrangement::kIterationsRow,
                                Detail::kIterationsGroup,
                                Detail::kIterationsCluster, 1>;

        using Delta = OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                                      Detail::RowArrangement::kDeltaRow,
                                      Detail::kCompactedDeltaGroup,
                                      Detail::kCompactedDeltaCluster, 1>;

        /// Number of elements within each vector access
        static int const kElementsPerAccess = ElementsPerAccess;

        /// Number  of threads
        static int const kThreads = Threads;

        /// Function to compute each thread's initial offset
        CUTLASS_HOST_DEVICE
        static MatrixCoord initial_offset(int thread_idx) {
            int warp_idx = thread_idx / kWarpSize;
            int lane_idx = thread_idx % kWarpSize;

            // Compute warp location
            int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
            int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

            int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
            int residual_group =
                    residual_cluster % Detail::WarpPartitions::kGroup;

            int row_idx = residual_group / Detail::WarpPartitions::kColumn;
            int col_idx = residual_group % Detail::WarpPartitions::kColumn;

            // Compute per-lane offset
            int lane_row_offset = lane_idx / Detail::kAccessWidth;
            int lane_col_offset = lane_idx % Detail::kAccessWidth;

            // Compute coordinate in output space
            int cluster_offset = cluster_idx * Shape::kColumn * Shape::kGroup;
            int group_offset = group_idx * Shape::kColumn;
            int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
            int column_offset =
                    col_idx * Iterations::kColumn * Detail::kAccessWidth;

            MatrixCoord coord((row_offset + lane_row_offset) * kInterleaved +
                                      ((column_offset + lane_col_offset) *
                                       kElementsPerAccess) %
                                              kInterleaved,
                              cluster_offset + group_offset +
                                      (column_offset + lane_col_offset) *
                                              kElementsPerAccess /
                                              kInterleaved);

            return coord;
        }
    };
};

////////////////////////////////////////////////////////////////////////////////

template <typename Shape_,        ///< Output tile shape
          typename Count_,        ///< Output tile count
          int Threads,            ///< Number of threads
          int ElementsPerAccess,  ///< Number of elements per memory access
          int ElementSize         ///< Element size in bits
          >
struct ConvolutionOutputTileOptimalThreadMapNHWC;

template <typename Shape_, typename Count_, int Threads, int ElementsPerAccess,
          int ElementSize>
struct ConvolutionOutputTileOptimalThreadMapNHWC {
    using Shape = Shape_;
    using Count = Count_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;

    static_assert(!((kElementSize * kElementsPerAccess) % 32),
                  "Elements size in bits per access of one thread must be a "
                  "multiple of 32.");

    using TransposedShape = MatrixShape<Shape::kColumn, Shape::kRow>;
    static_assert(!(TransposedShape::kColumn % kElementsPerAccess),
                  "Divisibility");
    using ShapeVec = MatrixShape<TransposedShape::kRow,
                                 TransposedShape::kColumn / kElementsPerAccess>;
    static_assert(!(kWarpSize % ShapeVec::kColumn), "Divisibility");

    static int const kWarpArrangementColumn = ShapeVec::kColumn;
    static int const kWarpArrangementRow =
            kWarpSize / kWarpArrangementColumn >= ShapeVec::kRow
                    ? ShapeVec::kRow
                    : kWarpSize / kWarpArrangementColumn;

    using WarpPartitionShape =
            MatrixShape<kWarpArrangementRow, kWarpArrangementColumn>;
    static int const kWarpPartitionSize = WarpPartitionShape::kCount;
    static int const kWarpPartitionCount = kThreads / kWarpPartitionSize;
    //
    // Metaprogram computation
    //

    struct Detail {
        // Clusters
        static int const kIterationsCluster =
                ((Shape::kCluster > kWarpPartitionCount)
                         ? Shape::kCluster / kWarpPartitionCount
                         : 1);

        static int const kDeltaCluster =
                ((Shape::kCluster > kWarpPartitionCount)
                         ? Shape::kColumn * Count::kColumn * Shape::kGroup *
                                   Count::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kCompactedDeltaCluster =
                ((Shape::kCluster > kWarpPartitionCount)
                         ? Shape::kColumn * Shape::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kWarpPartitionsCluster =
                ((Shape::kCluster > kWarpPartitionCount)
                         ? kWarpPartitionCount
                         : kWarpPartitionCount / Shape::kCluster);

        static int const kWarpsRemainingForGroups =
                ((Shape::kCluster > kWarpPartitionCount)
                         ? 1
                         : kWarpPartitionCount / Shape::kCluster);

        // Groups
        static int const kIterationsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kGroup / kWarpsRemainingForGroups
                         : 1);

        static int const kDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kColumn * Count::kColumn * Shape::kGroup /
                                   kIterationsGroup
                         : 1);

        static int const kCompactedDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kColumn * Shape::kGroup / kIterationsGroup
                         : 1);

        static int const kWarpPartitionsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        static int const kWarpsRemainingForRows =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        static_assert(kWarpsRemainingForRows == 1, "WarpRemainingForRows must be 1");
        // Warp partitions
        using WarpPartitions =
                OutputTileShape<1, kWarpsRemainingForRows, kWarpPartitionsGroup,
                                kWarpPartitionsCluster, 1>;

        static int const kAccessWidth = ShapeVec::kColumn;
        static int const kAccessRows = kWarpPartitionSize / kAccessWidth;

        static int const kIterationsRow = cutlass::const_max(
                ShapeVec::kRow / (kWarpPartitionSize / ShapeVec::kColumn), 1);
        static int const kIterationsColumn = 1;

        static int const kDeltaRow = kAccessRows;
        static int const kDeltaColumn = kAccessWidth * kElementsPerAccess;
    };

    //
    // Output
    //

    using Iterations =
            OutputTileShape<Detail::kIterationsColumn, Detail::kIterationsRow,
                            Detail::kIterationsGroup,
                            Detail::kIterationsCluster, 1>;

    using Delta =
            OutputTileShape<Detail::kDeltaColumn,
                            Detail::kDeltaRow,
                            Detail::kDeltaGroup, Detail::kDeltaCluster, 1>;

    /// Initial offset function
    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
 
        int warp_idx = thread_idx / kWarpPartitionSize;
        int lane_idx = thread_idx % kWarpPartitionSize;

        // Compute warp location
        int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
        int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

        int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
        int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

        int row_idx = residual_group / Detail::WarpPartitions::kColumn;
        int col_idx = residual_group % Detail::WarpPartitions::kColumn;

        // Compute per-lane offset
        int lane_row_offset = lane_idx / Detail::kAccessWidth;
        int lane_col_offset = lane_idx % Detail::kAccessWidth;

        // Compute coordinate in output space
        int cluster_offset = cluster_idx * Shape::kColumn * Count::kColumn *
                             Shape::kGroup * Count::kGroup;
        int group_offset = group_idx * Shape::kColumn * Count::kColumn;
        int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
        int column_offset =
                col_idx * Iterations::kColumn * Detail::kAccessWidth;

        // row and column must be transposed
        return MatrixCoord(
                (column_offset + lane_col_offset) * kElementsPerAccess,
                cluster_offset + group_offset + row_offset + lane_row_offset);
    }

    /// Compacted thread map in which the 4D region is contiguous
    struct CompactedThreadMap {
        using Shape = Shape_;

        using Iterations = OutputTileShape<
                Detail::kIterationsColumn, Detail::kIterationsRow,
                Detail::kIterationsGroup, Detail::kIterationsCluster, 1>;

        using Delta = OutputTileShape<Detail::kDeltaColumn, Detail::kDeltaRow,
                                      Detail::kCompactedDeltaGroup,
                                      Detail::kCompactedDeltaCluster, 1>;

        /// Number of elements within each vector access
        static int const kElementsPerAccess = ElementsPerAccess;

        /// Number  of threads
        static int const kThreads = Threads;

        /// Function to compute each thread's initial offset
        CUTLASS_HOST_DEVICE
        static MatrixCoord initial_offset(int thread_idx) {
            int warp_idx = thread_idx / kWarpPartitionSize;
            int lane_idx = thread_idx % kWarpPartitionSize;

            // Compute warp location
            int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
            int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

            int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
            int residual_group =
                    residual_cluster % Detail::WarpPartitions::kGroup;

            int row_idx = residual_group / Detail::WarpPartitions::kColumn;
            int col_idx = residual_group % Detail::WarpPartitions::kColumn;

            // Compute per-lane offset
            int lane_row_offset = lane_idx / Detail::kAccessWidth;
            int lane_col_offset = lane_idx % Detail::kAccessWidth;

            // Compute coordinate in output space
            int cluster_offset = cluster_idx * Shape::kColumn * Shape::kGroup;
            int group_offset = group_idx * Shape::kColumn;
            int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
            int column_offset =
                    col_idx * Iterations::kColumn * Detail::kAccessWidth;

            MatrixCoord coord(
                    (column_offset + lane_col_offset) * kElementsPerAccess,
                    cluster_offset + group_offset + row_offset +
                            lane_row_offset);

            return coord;
        }
    };
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
