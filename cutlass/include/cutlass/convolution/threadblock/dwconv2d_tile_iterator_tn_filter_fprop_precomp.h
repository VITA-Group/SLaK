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
 * include/cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_fprop_precomp.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/conv2d_problem_size.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Strip-mines a pitch-linear tile among a given number of threads, first along
/// the contiguous dimension then along the strided dimension, while each thread
/// access ElementsPerAccess elements.
template <typename Shape_, int Threads, int ElementsPerAccess = 1>
struct PitchLinearStripminedThreadMapStrided {
    /// Tensor coordinate
    using TensorCoord = layout::PitchLinearCoord;

    /// Tile shape
    using Shape = Shape_;

    /// Number of threads total
    static int const kThreads = Threads;

    static_assert(Shape::kStrided <= kThreads,
                  "Stride of shape must be less than thread count");

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ElementsPerAccess;

    /// Shape of access by each thread
    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    /// Internal implementation details
    struct Detail {
        static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

        static_assert(!((Shape::kContiguous * Shape::kStrided) %
                        (kThreads * kElementsPerAccess)),
                      "Shape must be divisible thread count.");

        /// Shape of the tile in units of vectors
        using ShapeVec = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        static_assert(kThreads >= ShapeVec::kStrided &&
                              !(kThreads % ShapeVec::kStrided),
                      "Thread count must be divisible by stride of shape");
    };

    using ThreadArrangement =
            layout::PitchLinearShape<kThreads / Detail::ShapeVec::kStrided,
                                     Detail::ShapeVec::kStrided>;

    /// Number of iterations by each thread
    using Iterations = layout::PitchLinearShape<
            Detail::ShapeVec::kContiguous / ThreadArrangement::kContiguous, 1>;

    /// Interval between accesses along each dimension of the tensor's logical
    /// coordinate space (in units of Elements)
    using Delta = layout::PitchLinearShape<ThreadArrangement::kContiguous *
                                                   kElementsPerAccess,
                                           ThreadArrangement::kStrided>;

    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space (in units of Elements)
    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord((thread_id % ThreadArrangement::kContiguous) *
                                   kElementsPerAccess,
                           thread_id / ThreadArrangement::kContiguous);
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize>
class Dwconv2dTileFilterIteratorFpropPrecomp;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of Dwconv2dTileFilterIteratorFpropPrecomp for
/// TensorNCHW Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class Dwconv2dTileFilterIteratorFpropPrecomp<
        Shape_, Element_, layout::TensorNCHW, ThreadMap_, AccessSize> {
public:
    using Shape = layout::PitchLinearShape<Shape_::kColumn, Shape_::kRow>;
    using Element = Element_;
    using Layout = layout::TensorNCHW;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap<Layout, TileMapType::kRow2IHW_Col2OHW>;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Pointer = Element*;

    /// Type used for internal memory accesses
    using AccessType =
            AlignedArray<Element, AccessSize,
                         (AccessSize * sizeof_bits<Element>::value / 8)>;

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                  "Vectors implied by the thread map must be divisible by the "
                  "access type.");
    static_assert(
            AccessSize == 1,
            "This tile iterator only support access 1 element per access");

    static int const kAccessCount =
            ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// compatible for constructor function
    using ExtraParam = platform::none_type;

    /// compatible for the concept of MaskedTileIteratorConcept
    using Mask = platform::none_type;

    class Params {
    public:
        friend Dwconv2dTileFilterIteratorFpropPrecomp;

    private:
        TileMap tile_map_;
        Index fh_, fw_;

        Layout layout_;

    public:
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, Conv2dProblemSize const& problem_size,
               ExtraParam const extra_param = {})
                : tile_map_(problem_size.W, problem_size.Q,
                            problem_size.stride_h, problem_size.stride_w,
                            problem_size.pad_h, problem_size.pad_w),
                  fh_(problem_size.R),
                  fw_(problem_size.S),
                  layout_(layout) {}

        CUTLASS_HOST_DEVICE
        Params() : tile_map_(TileMap()) {}
    };

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_;

    /// Offset to the first steady-state tile
    Index residue_offset_;

    Index residue_extent_;

    Index filter_r_[kAccessCount];
    Index filter_s_[kAccessCount];

    Index filter_w_[ThreadMap::Iterations::kStrided];

    Index conv_k_iterations_;

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_extent_and_threadblock_offset_(
            LogicalCoord& extent, LogicalCoord& threadblock_offset) {
        auto ranges = make_Coord(
                threadblock_offset.column(),
                threadblock_offset.column() + Shape::kContiguous - 1);
        ranges = params_.tile_map_(ranges, params_.fh_);
        auto row_extent =
                ranges.at(1) < extent.row() ? ranges.at(1) : extent.row();
        extent = LogicalCoord{row_extent, extent.column()};
        threadblock_offset =
                LogicalCoord{ranges.at(0), threadblock_offset.column()};
        conv_k_iterations_ = (extent.row() - threadblock_offset.row() +
                              Shape::kStrided - 1) /
                             Shape::kStrided;
    }

    CUTLASS_DEVICE
    void initialize_filter_coordinates_(LogicalCoord const& thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto src = params_.tile_map_(thread_offset.row() +
                                         s * ThreadMap::Delta::kStrided);
            filter_w_[s] = src.column();
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    Index column = v + c * ThreadMap::Delta::kContiguous +
                                   thread_offset.column();
                    auto coord = params_.tile_map_(column, src);
                    filter_r_[idx] = coord.row();
                    filter_s_[idx] = coord.column();
                }
            }
        }
    }

public:
    /// Constructs a TileIterator
    CUTLASS_HOST_DEVICE
    Dwconv2dTileFilterIteratorFpropPrecomp(
            /// Precomputed parameters object
            Params const& params,
            /// Pointer to start of tensor
            Pointer pointer)
            : params_(params), pointer_(pointer), is_residue_tile_(true) {}

    /// Intialize a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    Dwconv2dTileFilterIteratorFpropPrecomp& initialize(
            /// ID of each participating thread
            int thread_id,
            /// Extent of tensor
            LogicalCoord& extent,
            /// Initial offset of threadblock
            LogicalCoord& threadblock_offset) {
        initialize_extent_and_threadblock_offset_(extent, threadblock_offset);

        residue_offset_ =
                (extent.row() - threadblock_offset.row()) % Shape::kStrided;
        if (!residue_offset_) {
            residue_offset_ = Shape::kStrided;
        }

        residue_extent_ =
                min(threadblock_offset.row() + residue_offset_, extent.row());

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                threadblock_offset + LogicalCoord{thread_offset_.strided(),
                                                  thread_offset_.contiguous()};

        initialize_filter_coordinates_(thread_offset);

        residue_extent_ = residue_extent_ - thread_offset.row();

        return *this;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        pointer_ += pointer_offset;
    }

    /// Advances to the next tile in memory.
    ///
    /// Just update filter's coordinates
    CUTLASS_HOST_DEVICE
    Dwconv2dTileFilterIteratorFpropPrecomp& operator++() {
        int increment = Shape::kStrided;
        if (is_residue_tile_) {
            increment = residue_offset_;
        }
        auto inc_coord_base = params_.tile_map_(increment);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            auto inc_coord = inc_coord_base;
            filter_w_[s] += inc_coord.column();
            if (filter_w_[s] >= params_.tile_map_.wi_) {
                filter_w_[s] -= params_.tile_map_.wi_;
                inc_coord += MatrixCoord{1, -params_.tile_map_.wi_};
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0;
                 i < ThreadMap::Iterations::kContiguous * kAccessesPerVector;
                 ++i) {
                int idx = s * ThreadMap::Iterations::kContiguous *
                                  kAccessesPerVector +
                          i;
                filter_r_[idx] += inc_coord.row();
                filter_s_[idx] += inc_coord.column();
            }
        }
        is_residue_tile_ = false;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    CUTLASS_HOST_DEVICE
    Dwconv2dTileFilterIteratorFpropPrecomp operator++(int) {
        Dwconv2dTileFilterIteratorFpropPrecomp self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() {}

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() {}

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& /* mask */) {}

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& /* mask */) {}

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    Index& filter_r = filter_r_[idx];
                    Index& filter_s = filter_s_[idx];
                    bool guard = filter_r >= 0 && filter_r < params_.fh_ &&
                                 filter_s >= 0 && filter_s < params_.fw_;
                    if (is_residue_tile_)
                        guard &= residue_extent_ > 0;
                    char const* byte_ptr =
                            reinterpret_cast<char const*>(
                                    pointer_ + filter_r * params_.fw_ +
                                    filter_s) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag) {
        load_with_pointer_offset(frag, 0);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        store_with_byte_offset(
                frag, pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kContiguous; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    Index& filter_r = filter_r_[idx];
                    Index& filter_s = filter_s_[idx];
                    bool guard = filter_r >= 0 && filter_r < params_.fh_ &&
                                 filter_s >= 0 && filter_s < params_.fw_;
                    if (is_residue_tile_)
                        guard &= residue_extent_ > 0;
                    char const* byte_ptr =
                            reinterpret_cast<char const*>(
                                    pointer_ + filter_r * params_.fw_ +
                                    filter_s) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
        }
    }

    CUTLASS_DEVICE Dwconv2dTileFilterIteratorFpropPrecomp& add_coord_offset(
            TensorCoord const& coord_offset) {
        auto size = params_.layout_(coord_offset);
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    static Status can_implement(Conv2dProblemSize& problem_size) {
        if (problem_size.mode != Mode::kCrossCorrelation) {
            return Status::kErrorNotSupported;
        }

        if (problem_size.dilation_h != 1 || problem_size.dilation_w != 1) {
            return Status::kErrorNotSupported;
        }

        return Status::kSuccess;
    }

    CUTLASS_DEVICE
    const Index& conv_k_iterations() { return conv_k_iterations_; }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
