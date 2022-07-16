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
    \brief Templates implementing loading of tiles from pitch-linear rank=2
   tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last
   "residue" tile first, with the objective of minimizing predicate mask updates
   during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

/**
 * \file
 * include/cutlass/convolution/threadblock/conv2d_tile_iterator_nt_src_dgrad_precomp.h
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
#include "cutlass/convolution/threadblock/conv2d_tile_params.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

namespace detail {
template <typename Shape_, int Interleaved>
CUTLASS_HOST_DEVICE void compute_offset_dgrad(int* constant_offset_, int fh_,
                                              int fw_, int hi_, int wi_,
                                              int residue_offset_) {
    // hardcoded typedef
    using Shape = Shape_;
    using ShortIndex = int8_t;
    using Index = int;
    static int const kInterleaved = Interleaved;

    Index* offset_ptr = constant_offset_;
    ShortIndex* fhfw_ptr = reinterpret_cast<ShortIndex*>(constant_offset_ + 1);
    Index s = 0;
    Index filter_pixels = fh_ * fw_;
    Index image_pixels = hi_ * wi_;
    // first absolute offset
    CUTLASS_PRAGMA_UNROLL
    for (; s < Shape::kStrided; ++s) {
        Index c = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
    // step residue_offset_
    CUTLASS_PRAGMA_UNROLL
    for (; s < 2 * Shape::kStrided; ++s) {
        Index s_ = s - Shape::kStrided + residue_offset_;
        Index c = s_ / (filter_pixels);
        Index fhfw = s_ - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - residue_offset_;
        c = s_ / (filter_pixels);
        offset_ptr[0] -= (c * image_pixels * kInterleaved);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
    CUTLASS_PRAGMA_UNROLL
    for (; s < (2 + filter_pixels) * Shape::kStrided; ++s) {
        Index s_ = s - Shape::kStrided + residue_offset_;
        Index c = s_ / (filter_pixels);
        Index fhfw = s_ - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        offset_ptr[0] = c * image_pixels * kInterleaved;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - Shape::kStrided;
        c = s_ / (filter_pixels);
        offset_ptr[0] -= (c * image_pixels * kInterleaved);
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
}
}  // namespace detail

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          typename TileMap>
class DgradPrecompParams;

template <typename Shape_, typename Element_, typename ThreadMap_,
          typename TileMap_, int Interleaved>
class DgradPrecompParams<Shape_, Element_, layout::TensorNCxHWx<Interleaved>,
                         ThreadMap_, TileMap_> {
public:
    static int const kInterleaved = Interleaved;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;
    using ExtraParam = platform::none_type;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;
    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    /// Element size in Index
    static int const kElementSize =
            (cutlass::sizeof_bits<Index>::value +
             4 * cutlass::sizeof_bits<ShortIndex>::value) /
            cutlass::sizeof_bits<Index>::value;
    // less than 3.2K
    static int const kPrecomputedOffsetBufferSize = 848;
    static int const kMaxFilterPixels =
            kPrecomputedOffsetBufferSize / (kElementSize * Shape::kStrided) - 2;

    /// Used for converting tensor coordinates into pointer offset
    Layout layout_;

    /// Parameters used for mapping logical coordinates to physical
    /// coordinates
    TileMap tile_map_;
    Index pad_h_, pad_w_;
    FastDivmod stride_h_div_mod_;
    FastDivmod stride_w_div_mod_;
    Index hi_, wi_, n_;
    Index fh_, fw_;
    Index residue_offset_;
    Index constant_offset_max_;
    Index constant_offset_rewind_;
    Index constant_offset_[kPrecomputedOffsetBufferSize];

    CUTLASS_HOST_DEVICE
    DgradPrecompParams() : layout_(Layout()), tile_map_(TileMap()) {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    DgradPrecompParams(Layout const& layout,
                       Conv2dProblemSize const& problem_size,
                       ExtraParam const& /* extra_param */)
            : layout_(layout),
              stride_h_div_mod_(problem_size.stride_h),
              stride_w_div_mod_(problem_size.stride_w),
              pad_h_(problem_size.pad_h),
              pad_w_(problem_size.pad_w),
              fh_(problem_size.R),
              fw_(problem_size.S),
              n_(problem_size.N) {
        tile_map_ = TileMap(problem_size.H * problem_size.W, problem_size.W);
        hi_ = problem_size.P;
        wi_ = problem_size.Q;
        Index conv_iterations =
                problem_size.K * problem_size.R * problem_size.S;

        residue_offset_ = (conv_iterations / kInterleaved) % Shape::kStrided;
        if (!residue_offset_) {
            residue_offset_ = Shape::kStrided;
        }
        detail::compute_offset_dgrad<Shape, kInterleaved>(
                constant_offset_, problem_size.R, problem_size.S, hi_, wi_,
                residue_offset_);
        constant_offset_max_ =
                (1 + problem_size.R * problem_size.S) * Shape::kStrided;
        constant_offset_rewind_ =
                Shape::kStrided * (1 - problem_size.R * problem_size.S);
    }
};

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize, typename TileMap,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE>
class Conv2dTileSrcIteratorDgradPrecomp;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of Conv2dTileSrcIteratorDgradPrecomp for
/// TensorNCxHWx<Interleaved> Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename ThreadMap_,
          int Interleaved, int AccessSize, typename TileMap_,
          SpecialOptimizeDesc SpecialOpt>
class Conv2dTileSrcIteratorDgradPrecomp<
        Shape_, Element_, layout::TensorNCxHWx<Interleaved>, ThreadMap_,
        AccessSize, TileMap_, SpecialOpt> {
public:
    using Shape = layout::PitchLinearShape<Shape_::kColumn * Interleaved,
                                           Shape_::kRow / Interleaved>;
    using Element = Element_;
    static int const kInterleaved = Interleaved;

    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using ShortIndex = int8_t;
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
    static_assert(AccessType::kElements <= kInterleaved,
                  "Access size must equal to interleaving quantity");

    static int const kContiguousCount =
            ThreadMap::Iterations::kContiguous * kAccessesPerVector;

    struct Mask {
        static int const kCount = kContiguousCount < 8 ? 8 : kContiguousCount;

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

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Parameters object is precomputed state and is host-constructible
    using Params = DgradPrecompParams<Shape, Element,
                                      layout::TensorNCxHWx<kInterleaved>,
                                      ThreadMap, TileMap>;

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// predicates
    Mask mask_;

    /// Extent for the first steady-state tile
    Index residue_extent_;

    Index oh[kContiguousCount];
    Index ow[kContiguousCount];

    Index constant_offset_;
    Index strided_[ThreadMap::Iterations::kStrided];

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            Index thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements + thread_offset;

            TensorCoord coord = params_.tile_map_(
                    LogicalCoord{0, col_offset / kInterleaved});

            pointer_[access_idx] =
                    pointer + params_.layout_(TensorCoord{coord.n(), 0, 0, 0}) +
                    col_offset % kInterleaved;
            oh[access_idx] = coord.h();
            ow[access_idx] = coord.w();
            mask_.predicates[access_idx] = coord.n() < params_.n_;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] =
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecomp(
            /// Precomputed parameters object
            Params const& params,
            /// Pointer to start of tensor
            Pointer pointer,
            /// Extent of tensor
            LogicalCoord extent,
            /// ID of each participating thread
            int thread_id,
            /// Initial offset of threadblock
            LogicalCoord const& threadblock_offset)
            : params_(params), is_residue_tile_(true) {
        residue_extent_ = min(threadblock_offset.row() / kInterleaved +
                                      params_.residue_offset_,
                              extent.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset.row() / kInterleaved,
                             threadblock_offset.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize constant offset
        constant_offset_ = thread_offset.row();

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset.column());

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a Conv2dTileSrcIteratorDgradPrecomp with zero threadblock
    /// offset
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecomp(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : Conv2dTileSrcIteratorDgradPrecomp(params, pointer, extent,
                                                thread_id, make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            pointer_[access_idx] +=
                    sizeof_bits<Element>::value * pointer_offset / 8;
        }
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecomp& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kStrided;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] +=
                    params_.constant_offset_[2 *
                                             (constant_offset_ +
                                              s * ThreadMap::Delta::kStrided)];
        }
        is_residue_tile_ = false;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and the
    /// iterator's internal pointer is reverted to the first "steady state"
    /// tile. Subsequent calls are lightweight and must only update the internal
    /// pointer.
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecomp operator++(int) {
        Conv2dTileSrcIteratorDgradPrecomp self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() { mask_.clear(); }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() { mask_.enable(); }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) { mask_ = mask; }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& /* mask */) { /* return mask_; */
    }

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
            auto ptr_ = reinterpret_cast<ShortIndex const*>(
                    params_.constant_offset_ +
                    2 * (constant_offset_ + s * ThreadMap::Delta::kStrided) +
                    1);
            ShortIndex fh = ptr_[0];
            ShortIndex fw = ptr_[1];

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;

                    int mod_h = 0, mod_w = 0;
                    Index ih, iw;

                    if (SpecialOpt ==
                        SpecialOptimizeDesc::DECONV_DOUBLE_UPSAMPLING) {
                        ih = (oh[access_idx] + params_.pad_h_ - fh) >> 1;
                        iw = (ow[access_idx] + params_.pad_w_ - fw) >> 1;
                        mod_h = (oh[access_idx] + params_.pad_h_ - fh) & 0x1;
                        mod_w = (ow[access_idx] + params_.pad_w_ - fw) & 0x1;
                    } else {
                        params_.stride_h_div_mod_(
                                ih, mod_h,
                                oh[access_idx] + params_.pad_h_ - fh);
                        params_.stride_w_div_mod_(
                                iw, mod_w,
                                ow[access_idx] + params_.pad_w_ - fw);
                    }

                    bool guard = mask_.predicates[access_idx] &&
                                 ((ih >= 0) && (ih < params_.hi_) &&
                                  (iw >= 0) && (iw < params_.wi_) &&
                                  (mod_h == 0) && (mod_w == 0));
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    TensorCoord coord{0, ih, iw, 0};

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(
                                    pointer_[access_idx] +
                                    params_.layout_(coord) + strided_[s]) +
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
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    static Status can_implement(Conv2dProblemSize& problem_size) {
        if (problem_size.mode != Mode::kCrossCorrelation) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.R * problem_size.S > Params::kMaxFilterPixels) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.dilation_h != 1 || problem_size.dilation_w != 1) {
            return Status::kErrorInvalidProblem;
        }

        return Status::kSuccess;
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
