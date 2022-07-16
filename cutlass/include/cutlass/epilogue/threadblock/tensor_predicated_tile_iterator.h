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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

*/

/**
 * \file include/cutlass/epilogue/threadblock/tensor_predicated_tile_iterator.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/convolution_output_tile_thread_map.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator |
/// ForwardTileIterator
///
template <typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
          typename Layout_,     ///< Tensor Layout
          typename Element_     ///< Element data type
          >
class TensorPredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadMap_, typename Element_, int Interleaved>
class TensorPredicatedTileIterator<
        ThreadMap_, layout::TensorCxRSKx<Interleaved>, Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorCxRSKx<Interleaved>;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;
    static int const kInterleaved = Interleaved;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride;  ///< stride in bytes between rows

        LongIndex increment_row;  ///< increment quantity (in bytes) to advance
                                  ///< when moving between rows
        LongIndex increment_group;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next group
        LongIndex
                increment_cluster;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next cluster

        LongIndex advance_row;    ///< amount to add to move to the next 'row'
                                  ///< position
        LongIndex advance_group;  ///< amount to add to move to the next 'group'
                                  ///< position
        LongIndex advance_cluster;  ///< amount to add to move to the next
                                    ///< 'cluster' position
        LongIndex advance_tile;  ///< amount to add to move to the next 'tile'

        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_) / kInterleaved;

            increment_row = stride * ThreadMap::Delta::kRow;

            increment_group = stride * ThreadMap::Delta::kGroup -
                              stride * ThreadMap::Delta::kRow *
                                      (ThreadMap::Iterations::kRow - 1);

            increment_cluster = stride * ThreadMap::Delta::kCluster -
                                stride * ThreadMap::Delta::kGroup *
                                        (ThreadMap::Iterations::kGroup - 1) -
                                stride * ThreadMap::Delta::kRow *
                                        (ThreadMap::Iterations::kRow - 1);

            advance_row = stride * ThreadMap::Shape::kRow;

            advance_group = stride * (ThreadMap::Shape::kGroup - 1) *
                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            advance_cluster = stride * ThreadMap::Count::kGroup *
                              ThreadMap::Shape::kGroup *
                              ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
            advance_tile = stride * ThreadMap::Shape::kGroup *
                           ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
                           ThreadMap::Shape::kTile;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout) : layout_(layout) {
            initialize(layout.stride()[2] * sizeof_bits<Element>::value / 8);
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// Internal state counter
    int state_[3];

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    TensorPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_row_ = extent.row();
        thread_start_row_ = thread_offset.row();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                    ((thread_offset.column() + ThreadMap::Delta::kColumn * c) <
                     extent.column());
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        params_.layout_(TensorCoord{thread_offset.column(), 0,
                                                    0, thread_offset.row()}) *
                                sizeof_bits<Element>::value / 8;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                            byte_pointer + byte_offset);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)&memory_pointer
                                        [column * ThreadMap::Delta::kColumn *
                                         kInterleaved / kElementsPerAccess],
                                guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                            byte_pointer + byte_offset);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)&memory_pointer
                                        [column * ThreadMap::Delta::kColumn *
                                         kInterleaved / kElementsPerAccess],
                                guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIterator& operator++() {
        ++state_[0];
        byte_pointer_ += params_.advance_row;
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kRow *
                                 ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_row_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }

    CUTLASS_DEVICE
    TensorPredicatedTileIterator& add_coord_offset(
            TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadMap_, typename Element_, int Interleaved>
class TensorPredicatedTileIterator<
        ThreadMap_, layout::TensorNCxHWx<Interleaved>, Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNCxHWx<Interleaved>;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;
    static int const kInterleaved = Interleaved;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride;  ///< stride in bytes between rows

        LongIndex increment_row;  ///< increment quantity (in bytes) to advance
                                  ///< when moving between rows
        LongIndex increment_group;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next group
        LongIndex
                increment_cluster;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next cluster

        LongIndex advance_row;    ///< amount to add to move to the next 'row'
                                  ///< position
        LongIndex advance_group;  ///< amount to add to move to the next 'group'
                                  ///< position
        LongIndex advance_cluster;  ///< amount to add to move to the next
                                    ///< 'cluster' position
        LongIndex advance_tile;  ///< amount to add to move to the next 'tile'

        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        Index nhw_, hw_, w_;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_) / kInterleaved;

            increment_row = stride * ThreadMap::Delta::kRow;

            increment_group = stride * ThreadMap::Delta::kGroup -
                              stride * ThreadMap::Delta::kRow *
                                      (ThreadMap::Iterations::kRow - 1);

            increment_cluster = stride * ThreadMap::Delta::kCluster -
                                stride * ThreadMap::Delta::kGroup *
                                        (ThreadMap::Iterations::kGroup - 1) -
                                stride * ThreadMap::Delta::kRow *
                                        (ThreadMap::Iterations::kRow - 1);

            advance_row = stride * ThreadMap::Shape::kRow;

            advance_group = stride * (ThreadMap::Shape::kGroup - 1) *
                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            advance_cluster = stride * ThreadMap::Count::kGroup *
                              ThreadMap::Shape::kGroup *
                              ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
            advance_tile = stride * ThreadMap::Shape::kGroup *
                           ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
                           ThreadMap::Shape::kTile;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()) { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator conv_operator,
               ConvProblemSize const& problem_size)
                : layout_(layout) {
            w_ = (conv_operator == conv::Operator::kFprop) ? problem_size.Q
                                                           : problem_size.W;
            hw_ = (conv_operator == conv::Operator::kFprop)
                          ? problem_size.P * problem_size.Q
                          : problem_size.H * problem_size.W;
            nhw_ = problem_size.N * hw_;
            initialize(layout.stride()[1] * sizeof_bits<Element>::value / 8);
        }
        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            Index n = coord.column() / hw_;
            Index hw = coord.column() - hw_ * n;
            Index h = hw / w_;
            Index w = hw - w_ * h;
            return TensorCoord(n, h, w, 0);
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params const& params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates
    /// have been computed)
    Index thread_start_col_;

    /// Internal state counter
    int state_[3];

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    TensorPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;
        extent_row_ = extent.row();
        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                    ((thread_offset.column() + ThreadMap::Delta::kColumn * c) <
                     params_.nhw_);
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        thread_start_row_ * params_.stride;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ +
                                           column * ThreadMap::Delta::kColumn);
                        TensorCoord coord = params_(iteration_coord);
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ +
                                           column * ThreadMap::Delta::kColumn);
                        TensorCoord coord = params_(iteration_coord);

                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIterator& operator++() {
        ++state_[0];
        byte_pointer_ += params_.advance_row;
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kRow *
                                 ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_row_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadMap_, typename Element_>
class TensorPredicatedTileIterator<ThreadMap_, layout::TensorNCxHWx<32>,
                                   Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    static int const kInterleaved = 32;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride;  ///< stride in bytes between rows

        LongIndex increment_row;  ///< increment quantity (in bytes) to advance
                                  ///< when moving between rows
        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        Index nhw_, hw_, w_;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_);

            increment_row = stride * ThreadMap::Delta::kRow / kInterleaved;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()) { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator conv_operator,
               ConvProblemSize const& problem_size)
                : layout_(layout) {
            w_ = (conv_operator == conv::Operator::kFprop) ? problem_size.Q
                                                           : problem_size.W;
            hw_ = (conv_operator == conv::Operator::kFprop)
                          ? problem_size.P * problem_size.Q
                          : problem_size.H * problem_size.W;
            nhw_ = problem_size.N * hw_;
            initialize(layout.stride()[1] * sizeof_bits<Element>::value / 8);
        }
        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            Index n = coord.column() / hw_;
            Index hw = coord.column() - hw_ * n;
            Index h = hw / w_;
            Index w = hw - w_ * h;
            return TensorCoord(n, h, w, 0);
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kRow < 8
                                          ? 8
                                          : ThreadMap::Iterations::kRow;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params const& params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates
    /// have been computed)
    Index thread_start_col_;

    /// Internal state counter
    int state_[3];

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    TensorPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;
        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < ThreadMap::Iterations::kRow; ++r) {
            mask_.predicates[r] = ((thread_offset.row() +
                                    ThreadMap::Delta::kRow * r) < extent.row());
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        (thread_start_row_ / kInterleaved) * params_.stride +
                        (thread_start_row_ % kInterleaved) *
                                sizeof_bits<Element>::value / 8;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            CUTLASS_PRAGMA_UNROLL
            for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
                 ++cluster) {
                CUTLASS_PRAGMA_UNROLL
                for (int group = 0; group < ThreadMap::Iterations::kGroup;
                     ++group) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int col_offset = column * ThreadMap::Delta::kColumn +
                                         group * ThreadMap::Delta::kGroup +
                                         cluster * ThreadMap::Delta::kCluster;
                        bool guard = ((thread_start_col_ + col_offset) <
                                      params_.nhw_) &&
                                     mask_.predicates[row];
                        int frag_col_idx =
                                (column +
                                 ThreadMap::Iterations::kColumn *
                                         (group +
                                          ThreadMap::Iterations::kGroup *
                                                  cluster));
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ + col_offset);
                        TensorCoord coord = params_(iteration_coord);
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[row * ThreadMap::Iterations::kCluster *
                                                 ThreadMap::Iterations::kGroup *
                                                 ThreadMap::Iterations::
                                                         kColumn +
                                         frag_col_idx],
                                (void*)(memory_pointer), guard);
                    }
                }
            }

            if (row + 1 < ThreadMap::Iterations::kRow) {
                byte_pointer += params_.increment_row;
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            CUTLASS_PRAGMA_UNROLL
            for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
                 ++cluster) {
                CUTLASS_PRAGMA_UNROLL
                for (int group = 0; group < ThreadMap::Iterations::kGroup;
                     ++group) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int col_offset = column * ThreadMap::Delta::kColumn +
                                         group * ThreadMap::Delta::kGroup +
                                         cluster * ThreadMap::Delta::kCluster;
                        bool guard = ((thread_start_col_ + col_offset) <
                                      params_.nhw_) &&
                                     mask_.predicates[row];
                        int frag_col_idx =
                                (column +
                                 ThreadMap::Iterations::kColumn *
                                         (group +
                                          ThreadMap::Iterations::kGroup *
                                                  cluster));
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ + col_offset);
                        TensorCoord coord = params_(iteration_coord);
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[row * ThreadMap::Iterations::kCluster *
                                                 ThreadMap::Iterations::kGroup *
                                                 ThreadMap::Iterations::
                                                         kColumn +
                                         frag_col_idx],
                                (void*)(memory_pointer), guard);
                    }
                }
            }

            if (row + 1 < ThreadMap::Iterations::kRow) {
                byte_pointer += params_.increment_row;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIterator& operator++() {
        ++state_[0];
        thread_start_col_ += ThreadMap::Shape::kColumn;

        if (state_[0] == ThreadMap::Count::kColumn) {
            state_[0] = 0;
            ++state_[1];

            thread_start_col_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kColumn *
                                 ThreadMap::Count::kColumn;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];

                thread_start_col_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kColumn * ThreadMap::Shape::kColumn;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadMap_, typename Element_>
class TensorPredicatedTileIterator<ThreadMap_, layout::TensorNCHW, Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNCHW;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride;  ///< stride in bytes between rows

        LongIndex increment_row;  ///< increment quantity (in bytes) to advance
                                  ///< when moving between rows
        LongIndex increment_group;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next group
        LongIndex
                increment_cluster;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next cluster

        LongIndex advance_row;    ///< amount to add to move to the next 'row'
                                  ///< position
        LongIndex advance_group;  ///< amount to add to move to the next 'group'
                                  ///< position
        LongIndex advance_cluster;  ///< amount to add to move to the next
                                    ///< 'cluster' position
        LongIndex advance_tile;  ///< amount to add to move to the next 'tile'

        /// Used for converting tensor coordinates into pointer offset
        Layout layout_;

        /// Parameters used for mapping logical coordinates to physical
        /// coordinates
        Index nhw_, hw_, w_;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_);

            increment_row = stride * ThreadMap::Delta::kRow;

            increment_group = stride * ThreadMap::Delta::kGroup -
                              stride * ThreadMap::Delta::kRow *
                                      (ThreadMap::Iterations::kRow - 1);

            increment_cluster = stride * ThreadMap::Delta::kCluster -
                                stride * ThreadMap::Delta::kGroup *
                                        (ThreadMap::Iterations::kGroup - 1) -
                                stride * ThreadMap::Delta::kRow *
                                        (ThreadMap::Iterations::kRow - 1);

            advance_row = stride * ThreadMap::Shape::kRow;

            advance_group = stride * (ThreadMap::Shape::kGroup - 1) *
                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            advance_cluster = stride * ThreadMap::Count::kGroup *
                              ThreadMap::Shape::kGroup *
                              ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
            advance_tile = stride * ThreadMap::Shape::kGroup *
                           ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
                           ThreadMap::Shape::kTile;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()) { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator conv_operator,
               ConvProblemSize const& problem_size)
                : layout_(layout) {
            w_ = (conv_operator == conv::Operator::kFprop) ? problem_size.Q
                                                           : problem_size.W;
            hw_ = (conv_operator == conv::Operator::kFprop)
                          ? problem_size.P * problem_size.Q
                          : problem_size.H * problem_size.W;
            nhw_ = problem_size.N * hw_;
            initialize(layout.stride()[1] * sizeof_bits<Element>::value / 8);
        }
        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            Index n = coord.column() / hw_;
            Index hw = coord.column() - hw_ * n;
            Index h = hw / w_;
            Index w = hw - w_ * h;
            return TensorCoord(n, h, w, 0);
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params const& params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates
    /// have been computed)
    Index thread_start_col_;

    /// Internal state counter
    int state_[3];

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    TensorPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;
        extent_row_ = extent.row();
        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                    ((thread_offset.column() + ThreadMap::Delta::kColumn * c) <
                     params_.nhw_);
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        thread_start_row_ * params_.stride;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ +
                                           column * ThreadMap::Delta::kColumn);
                        TensorCoord coord = params_(iteration_coord);
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        MatrixCoord iteration_coord(
                                0, thread_start_col_ +
                                           column * ThreadMap::Delta::kColumn);
                        TensorCoord coord = params_(iteration_coord);

                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        params_.layout_(coord) *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIterator& operator++() {
        ++state_[0];
        byte_pointer_ += params_.advance_row;
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kRow *
                                 ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_row_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

////////////////////////////////////////////////////////////////////////////////

template <typename ThreadMap_, typename Element_>
class TensorPredicatedTileIterator<ThreadMap_, layout::TensorNHWC,
                                   Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNHWC;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::ColumnMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    struct Params {
        //
        // Data members
        //

        LongIndex stride;  ///< stride in bytes between rows

        LongIndex increment_row;  ///< increment quantity (in bytes) to advance
                                  ///< when moving between rows
        LongIndex increment_group;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next group
        LongIndex
                increment_cluster;  ///< increment quantity (in bytes) to
                                    ///< advance when moving to the next cluster

        LongIndex advance_column;    ///< amount to add to move to the next 'row'
                                  ///< position
        LongIndex advance_group;  ///< amount to add to move to the next 'group'
                                  ///< position
        LongIndex advance_cluster;  ///< amount to add to move to the next
                                    ///< 'cluster' position
        LongIndex advance_tile;  ///< amount to add to move to the next 'tile'


        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_);

            increment_row = stride * ThreadMap::Delta::kRow;

            increment_group = stride * ThreadMap::Delta::kGroup -
                              stride * ThreadMap::Delta::kRow *
                                      (ThreadMap::Iterations::kRow - 1);

            increment_cluster = stride * ThreadMap::Delta::kCluster -
                                stride * ThreadMap::Delta::kGroup *
                                        (ThreadMap::Iterations::kGroup - 1) -
                                stride * ThreadMap::Delta::kRow *
                                        (ThreadMap::Iterations::kRow - 1);

            advance_column = stride * ThreadMap::Shape::kColumn;

            advance_group = stride * (ThreadMap::Shape::kGroup - 1) *
                            ThreadMap::Shape::kColumn *
                            ThreadMap::Count::kColumn;

            advance_cluster = stride * ThreadMap::Count::kGroup *
                              ThreadMap::Shape::kGroup *
                              ThreadMap::Count::kColumn *
                              ThreadMap::Shape::kColumn;
            advance_tile = stride * ThreadMap::Shape::kGroup *
                           ThreadMap::Shape::kColumn *
                           ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator /* operator */,
               ConvProblemSize /* problem_size */) {
            initialize(layout.stride()[0] * sizeof_bits<Element>::value / 8);
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn < 8
                                          ? 8
                                          : ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params const& params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// A thread's starting column position (assuming steady-state predicates
    /// have been computed)
    Index thread_start_col_;

    Index extent_column_;

    /// Internal state counter
    int state_[3];

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    TensorPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;
        thread_start_col_ = thread_offset.column();
        extent_column_ = extent.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                    ((thread_offset.row() + ThreadMap::Delta::kColumn * c) <
                     extent.row());
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        thread_start_col_ * params_.stride +
                        thread_offset.row() * sizeof_bits<Element>::value / 8;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int col_offset = row * ThreadMap::Delta::kRow +
                                         group * ThreadMap::Delta::kGroup +
                                         cluster * ThreadMap::Delta::kCluster;
                        bool guard = ((thread_start_col_ + col_offset) <
                                      extent_column_) &&
                                     mask_.predicates[column];
                        int frag_row_idx =
                                (row + ThreadMap::Iterations::kRow *
                                               (group +
                                                ThreadMap::Iterations::kGroup *
                                                        cluster));
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        column * ThreadMap::Delta::kColumn *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }
                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }
            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int col_offset = row * ThreadMap::Delta::kRow +
                                         group * ThreadMap::Delta::kGroup +
                                         cluster * ThreadMap::Delta::kCluster;
                        bool guard = ((thread_start_col_ + col_offset) <
                                      extent_column_) &&
                                     mask_.predicates[column];
                        int frag_row_idx =
                                (row + ThreadMap::Iterations::kRow *
                                               (group +
                                                ThreadMap::Iterations::kGroup *
                                                        cluster));
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        column * ThreadMap::Delta::kColumn *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }
                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }
            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIterator& operator++() {
        ++state_[0];
        byte_pointer_ += params_.advance_column;
        
        thread_start_col_ += ThreadMap::Shape::kColumn;
        
        if (state_[0] == ThreadMap::Count::kColumn) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_col_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kColumn *
                                 ThreadMap::Count::kColumn;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_col_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kColumn * ThreadMap::Shape::kColumn;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
