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
 * include/cutlass/convolution/threadblock/conv2d_tile_iterator_nt_src_fprop_precomp.h
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
template <typename Shape_, int Interleaved, typename Element>
CUTLASS_HOST_DEVICE void compute_offset_fprop_ncxhwx(int* constant_offset_,
                                                     int fh_, int fw_, int hi_,
                                                     int wi_,
                                                     int residue_offset_) {
    // hardcoded typedef
    using Shape = Shape_;
    using ShortIndex = int8_t;
    using Index = int;
    static int const kInterleaved = Interleaved;
    static int const kElementSizeBits = sizeof_bits<Element>::value;

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
        offset_ptr[0] = (c * image_pixels * kInterleaved +
                         fh * wi_ * kInterleaved + fw * kInterleaved) *
                        kElementSizeBits / 8;
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
        offset_ptr[0] = (c * image_pixels * kInterleaved +
                         fh * wi_ * kInterleaved + fw * kInterleaved) *
                        kElementSizeBits / 8;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - residue_offset_;
        c = s_ / (filter_pixels);
        fhfw = s_ - (filter_pixels)*c;
        fh = fhfw / fw_;
        fw = fhfw - fw_ * fh;
        offset_ptr[0] -= (c * image_pixels * kInterleaved +
                          fh * wi_ * kInterleaved + fw * kInterleaved) *
                         kElementSizeBits / 8;
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
        offset_ptr[0] = (c * image_pixels * kInterleaved +
                         fh * wi_ * kInterleaved + fw * kInterleaved) *
                        kElementSizeBits / 8;
        fhfw_ptr[0] = static_cast<ShortIndex>(fh);
        fhfw_ptr[1] = static_cast<ShortIndex>(fw);
        fhfw_ptr[2] = static_cast<ShortIndex>(-fh);
        fhfw_ptr[3] = static_cast<ShortIndex>(-fw);
        s_ = s_ - Shape::kStrided;
        c = s_ / (filter_pixels);
        fhfw = s_ - (filter_pixels)*c;
        fh = fhfw / fw_;
        fw = fhfw - fw_ * fh;
        offset_ptr[0] -= (c * image_pixels * kInterleaved +
                          fh * wi_ * kInterleaved + fw * kInterleaved) *
                         kElementSizeBits / 8;
        offset_ptr += 2;
        fhfw_ptr += 8;
    }
}
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          typename TileMap,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE>
class FpropPrecompParams;

/// Parameters object is precomputed state and is host-constructible
template <typename Shape_, typename Element_, typename ThreadMap_,
          typename TileMap_, int Interleaved>
class FpropPrecompParams<Shape_, Element_, layout::TensorNCxHWx<Interleaved>,
                         ThreadMap_, TileMap_, SpecialOptimizeDesc::NONE> {
public:
    static int const kInterleaved = Interleaved;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using ExtraParam = typename platform::conditional<
            platform::is_same<Element, uint4b_t>::value, ExtraParamZeroPoint,
            platform::none_type>::type;

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
    // less than 3.5K
    static int const kPrecomputedOffsetBufferSize = 848;
    static int const kMaxFilterPixels =
            kPrecomputedOffsetBufferSize / (kElementSize * Shape::kStrided) - 2;

    /// Used for converting tensor coordinates into pointer offset
    Layout layout_;

    /// Parameters used for mapping logical coordinates to physical
    /// coordinates
    TileMap tile_map_;
    Index stride_h_, stride_w_, pad_h_, pad_w_;
    Index hi_, wi_, n_;
    Index fh_, fw_;
    Index residue_offset_;
    // packed padding for src zero point
    uint32_t pack_pad_;

    Index constant_offset_max_;
    Index constant_offset_rewind_;
    Index constant_offset_[kPrecomputedOffsetBufferSize];

    CUTLASS_HOST_DEVICE
    FpropPrecompParams() : layout_(Layout()), tile_map_(TileMap()) {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    FpropPrecompParams(Layout const& layout,
                       Conv2dProblemSize const& problem_size,
                       ExtraParam const& extra_param = {})
            : layout_(layout),
              tile_map_(
                      TileMap(problem_size.P * problem_size.Q, problem_size.Q)),
              stride_h_(problem_size.stride_h),
              stride_w_(problem_size.stride_w),
              pad_h_(problem_size.pad_h),
              pad_w_(problem_size.pad_w),
              fh_(problem_size.R),
              fw_(problem_size.S),
              n_(problem_size.N) {
        hi_ = problem_size.H;
        wi_ = problem_size.W;
        Index conv_iterations =
                problem_size.C * problem_size.R * problem_size.S;

        residue_offset_ = (conv_iterations / kInterleaved) % Shape::kStrided;
        if (!residue_offset_) {
            residue_offset_ = Shape::kStrided;
        }
        detail::compute_offset_fprop_ncxhwx<Shape, kInterleaved, Element>(
                constant_offset_, problem_size.R, problem_size.S, hi_, wi_,
                residue_offset_);
        constant_offset_max_ =
                (1 + problem_size.R * problem_size.S) * Shape::kStrided;
        constant_offset_rewind_ =
                Shape::kStrided * (1 - problem_size.R * problem_size.S);
        // Host Init
        pack_pad_ = detail::prepare_pack_pad<Element, ExtraParam>(extra_param);
    }

    CUTLASS_DEVICE
    TensorCoord operator()(LogicalCoord const& coord) const {
        TensorCoord tensor_coord = tile_map_(coord);
        tensor_coord.h() = tensor_coord.h() * stride_h_ - pad_h_;
        tensor_coord.w() = tensor_coord.w() * stride_w_ - pad_w_;
        return tensor_coord;
    }
};

template <typename Shape_, typename Element_, typename ThreadMap_,
          typename TileMap_, int Interleaved>
class FpropPrecompParams<Shape_, Element_, layout::TensorNCxHWx<Interleaved>,
                         ThreadMap_, TileMap_,
                         SpecialOptimizeDesc::CONV_FILTER_UNITY> {
public:
    static int const kInterleaved = Interleaved;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;
    using ExtraParam = typename platform::conditional<
            platform::is_same<Element, uint4b_t>::value, ExtraParamZeroPoint,
            platform::none_type>::type;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;
    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    /// Used for converting tensor coordinates into pointer offset
    Layout layout_;
    /// amount (in byte) to increment pointer to move to next access along
    /// strided dimension
    LongIndex inc_strided_;
    /// amount (in byte) to increment pointer from last access to first
    /// access of next tile
    LongIndex inc_next_;
    LongIndex inc_iterations_;

    /// Parameters used for mapping logical coordinates to physical
    /// coordinates
    TileMap tile_map_;
    Index stride_h_, stride_w_, pad_h_, pad_w_;
    Index hi_, wi_, n_;
    uint32_t pack_pad_;

    CUTLASS_HOST_DEVICE
    FpropPrecompParams() : layout_(Layout()), tile_map_(TileMap()) {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    FpropPrecompParams(Layout const& layout,
                       Conv2dProblemSize const& problem_size,
                       ExtraParam const& extra_param = {})
            : layout_(layout),
              tile_map_(
                      TileMap(problem_size.P * problem_size.Q, problem_size.Q)),
              stride_h_(problem_size.stride_h),
              stride_w_(problem_size.stride_w),
              pad_h_(problem_size.pad_h),
              pad_w_(problem_size.pad_w),
              hi_(problem_size.H),
              wi_(problem_size.W),
              n_(problem_size.N) {
        int stride = layout_.stride()[TileMap::kStrideAxis];
        inc_strided_ = (LongIndex(stride) * ThreadMap::Delta::kStrided) *
                       sizeof_bits<Element>::value / 8;

        inc_iterations_ = LongIndex(ThreadMap::Iterations::kStrided - 1) *
                          ThreadMap::Delta::kStrided * LongIndex(stride) *
                          sizeof_bits<Element>::value / 8;

        inc_next_ = Shape::kStrided * LongIndex(stride) *
                            sizeof_bits<Element>::value / 8 -
                    inc_iterations_;

        pack_pad_ = detail::prepare_pack_pad<Element, ExtraParam>(extra_param);
    }

    CUTLASS_DEVICE
    TensorCoord operator()(LogicalCoord const& coord) const {
        TensorCoord tensor_coord = tile_map_(coord);
        tensor_coord.h() = tensor_coord.h() * stride_h_ - pad_h_;
        tensor_coord.w() = tensor_coord.w() * stride_w_ - pad_w_;
        return tensor_coord;
    }
};

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize, typename TileMap,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE,
          ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT>
class Conv2dTileSrcIteratorFpropPrecomp;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of Conv2dTileSrcIteratorFpropPrecomp for
/// TensorNCxHWx<Interleaved> Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename ThreadMap_,
          int Interleaved, int AccessSize, typename TileMap_,
          ImplicitGemmMode GemmMode>
class Conv2dTileSrcIteratorFpropPrecomp<
        Shape_, Element_, layout::TensorNCxHWx<Interleaved>, ThreadMap_,
        AccessSize, TileMap_, SpecialOptimizeDesc::NONE, GemmMode> {
public:
    using Shape = layout::PitchLinearShape<Shape_::kColumn * Interleaved,
                                           Shape_::kRow / Interleaved>;
    using Element = Element_;
    static int const kInterleaved = Interleaved;
    static ImplicitGemmMode const kGemmMode = GemmMode;

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

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Parameters object is precomputed state and is host-constructible
    using Params =
            FpropPrecompParams<Shape, Element,
                               layout::TensorNCxHWx<kInterleaved>, ThreadMap,
                               TileMap, SpecialOptimizeDesc::NONE>;

    using ExtraParam = typename Params::ExtraParam;

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// Extent for the first steady-state tile
    Index residue_extent_;

    Index masks_[kContiguousCount][2];

    Index constant_offset_;
    Index strided_[ThreadMap::Iterations::kStrided];
    uint32_t filter_hw_[ThreadMap::Iterations::kStrided];

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_HOST_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            Index thread_offset) {
        clear_mask();
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements + thread_offset;

            TensorCoord coord =
                    params_(LogicalCoord{0, col_offset / kInterleaved});

            pointer_[access_idx] =
                    pointer +
                    (params_.layout_(coord) + col_offset % kInterleaved) *
                            sizeof_bits<Element>::value / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int kh = 0; kh < params_.fh_; ++kh) {
                bool pred = (coord.n() < params_.n_ && coord.h() >= -kh &&
                             coord.h() < params_.hi_ - kh);
                masks_[access_idx][0] |= (pred << kh);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int kw = 0; kw < params_.fw_; ++kw) {
                bool pred = (coord.w() >= -kw && coord.w() < params_.wi_ - kw);
                masks_[access_idx][1] |= (pred << kw);
            }
        }

        Index offset = 2 * constant_offset_;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] = params_.constant_offset_[offset];
            filter_hw_[s] = *(reinterpret_cast<uint32_t const*>(
                    &params_.constant_offset_[offset + 1]));
            offset += 2 * ThreadMap::Delta::kStrided;
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecomp(
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
        LogicalCoord extent_;
        LogicalCoord threadblock_offset_;
        if (kGemmMode == ImplicitGemmMode::GEMM_NT) {
            extent_ = extent;
            threadblock_offset_ = threadblock_offset;
        } else {
            extent_ = LogicalCoord{extent.column(), extent.row()};
            threadblock_offset_ = LogicalCoord{threadblock_offset.column(),
                                               threadblock_offset.row()};
        }
        residue_extent_ = min(threadblock_offset_.row() / kInterleaved +
                                      params_.residue_offset_,
                              extent_.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset_.row() / kInterleaved,
                             threadblock_offset_.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize constant offset
        constant_offset_ = thread_offset.row();

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset.column());

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a Conv2dTileSrcIteratorFpropPrecomp with zero threadblock
    /// offset
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecomp(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : Conv2dTileSrcIteratorFpropPrecomp(params, pointer, extent,
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
    Conv2dTileSrcIteratorFpropPrecomp& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kStrided;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
        }
        Index offset = 2 * constant_offset_;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            strided_[s] += params_.constant_offset_[offset];
            filter_hw_[s] = *(reinterpret_cast<uint32_t const*>(
                    &params_.constant_offset_[offset + 1]));
            offset += 2 * ThreadMap::Delta::kStrided;
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
    Conv2dTileSrcIteratorFpropPrecomp operator++(int) {
        Conv2dTileSrcIteratorFpropPrecomp self(*this);
        operator++();
        return self;
    }

    /// Clears the predicates
    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            masks_[access_idx][0] = 0;
            masks_[access_idx][1] = 0;
        }
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
        int idx = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            int filter_h = filter_hw_[s] & 0xff;
            int filter_w = (filter_hw_[s] >> 8) & 0xff;
            int access_idx = 0;
            bool guard_ = is_residue_tile_ ? s * ThreadMap::Delta::kStrided <
                                                     residue_extent_
                                           : true;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    bool guard =
                            guard_ &&
                            (masks_[access_idx][0] & (Index(1) << filter_h)) &&
                            (masks_[access_idx][1] & (Index(1) << filter_w));

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard,
                            params_.pack_pad_);
                    idx++;
                    access_idx++;
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

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
        int idx = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            int filter_h = filter_hw_[s] & 0xff;
            int filter_w = (filter_hw_[s] >> 8) & 0xff;
            int access_idx = 0;
            bool guard_ = is_residue_tile_ ? s * ThreadMap::Delta::kStrided <
                                                     residue_extent_
                                           : true;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    bool guard =
                            guard_ &&
                            (masks_[access_idx][0] & (Index(1) << filter_h)) &&
                            (masks_[access_idx][1] & (Index(1) << filter_w));

                    char const* byte_ptr =
                            reinterpret_cast<char const*>(pointer_[access_idx] +
                                                          strided_[s]) +
                            byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                    idx++;
                    access_idx++;
                }
            }
        }
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

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

/// Specialization of Conv2dTileSrcIteratorFpropPrecomp for
/// TensorNCxHWx<Interleaved> Layout. Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename ThreadMap_,
          int Interleaved, int AccessSize, typename TileMap_,
          ImplicitGemmMode GemmMode>
class Conv2dTileSrcIteratorFpropPrecomp<
        Shape_, Element_, layout::TensorNCxHWx<Interleaved>, ThreadMap_,
        AccessSize, TileMap_, SpecialOptimizeDesc::CONV_FILTER_UNITY,
        GemmMode> {
public:
    static int const kInterleaved = Interleaved;
    static ImplicitGemmMode const kGemmMode = GemmMode;
    using Shape = layout::PitchLinearShape<Shape_::kColumn * kInterleaved,
                                           Shape_::kRow / kInterleaved>;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved>;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

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
                  "Access size cannot be greater than interleaving quantity");

    static int const kPredicatesPerByte = 4;
    static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

    static int const kContiguousCount =
            ThreadMap::Iterations::kContiguous * kAccessesPerVector;

    /// Number of 32b words containing predicates
    static int const kPredicateByteCount =
            (kContiguousCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
    static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

    static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

    static_assert(kPredicateWordCount <= 4, "Too many predicates.");

    /// Predicate vector stores mask to guard accesses
    using Mask = Array<uint32_t, kPredicateWordCount>;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Params =
            FpropPrecompParams<Shape, Element,
                               layout::TensorNCxHWx<kInterleaved>, ThreadMap,
                               TileMap, SpecialOptimizeDesc::CONV_FILTER_UNITY>;

    using ExtraParam = typename Params::ExtraParam;

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Internal pointer to first access of tile
    Pointer pointer_[kContiguousCount];

    /// Array of boolean values to contain steady-state predicates
    /// Guard predicates
    uint32_t predicates_[kPredicateWordCount];

    /// Offset to the first steady-state tile
    Index residue_offset_;

    Index residue_extent_;

    /// Used for out-of-order visitation
    bool is_residue_tile_;

private:
    CUTLASS_DEVICE
    void initialize_predicate_and_pointers_(Pointer pointer,
                                            LogicalCoord const& thread_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kContiguousCount; ++access_idx) {
            int c = access_idx / kAccessesPerVector;
            int v = access_idx % kAccessesPerVector;

            Index col_offset = c * ThreadMap::Delta::kContiguous +
                               v * AccessType::kElements +
                               thread_offset.column();
            TensorCoord coord =
                    params_(LogicalCoord{thread_offset.row() * kInterleaved,
                                         col_offset / kInterleaved});

            pointer_[access_idx] =
                    pointer +
                    (params_.layout_(coord) + col_offset % kInterleaved) *
                            sizeof_bits<Element>::value / 8;
            bool guard = coord.n() < params_.n_ && coord.h() >= 0 &&
                         coord.h() < params_.hi_ && coord.w() >= 0 &&
                         coord.w() < params_.wi_;
            int word_idx = access_idx / kPredicatesPerWord;
            int residual = access_idx % kPredicatesPerWord;
            int byte_idx = residual / kPredicatesPerByte;
            int bit_idx = residual % kPredicatesPerByte;
            predicates_[word_idx] |=
                    (unsigned(guard) << (byte_idx * 8 + bit_idx));
        }
    }

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecomp(
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
        LogicalCoord extent_;
        LogicalCoord threadblock_offset_;
        if (kGemmMode == ImplicitGemmMode::GEMM_NT) {
            extent_ = extent;
            threadblock_offset_ = threadblock_offset;
        } else {
            extent_ = LogicalCoord{extent.column(), extent.row()};
            threadblock_offset_ = LogicalCoord{threadblock_offset.column(),
                                               threadblock_offset.row()};
        }
        residue_offset_ = (extent_.row() / kInterleaved -
                           threadblock_offset_.row() / kInterleaved) %
                          Shape::kStrided;
        if (!residue_offset_) {
            residue_offset_ = Shape::kStrided;
        }

        residue_extent_ =
                min(threadblock_offset_.row() / kInterleaved + residue_offset_,
                    extent_.row() / kInterleaved);

        auto thread_offset_ = ThreadMap::initial_offset(thread_id);
        // Per-thread offset in logical coordinates of tensor
        LogicalCoord thread_offset =
                LogicalCoord(threadblock_offset_.row() / kInterleaved,
                             threadblock_offset_.column() * kInterleaved) +
                LogicalCoord(thread_offset_.strided(),
                             thread_offset_.contiguous());

        // Intialize internal pointers
        initialize_predicate_and_pointers_(pointer, thread_offset);

        residue_extent_ = residue_extent_ - thread_offset.row();
    }

    /// Construct a Conv2dTileSrcIteratorFpropPrecomp with zero threadblock
    /// offset
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecomp(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : Conv2dTileSrcIteratorFpropPrecomp(params, pointer, extent,
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
    Conv2dTileSrcIteratorFpropPrecomp& operator++() {
        if (is_residue_tile_) {
            add_pointer_offset(residue_offset_ *
                               params_.layout_.stride()[TileMap::kStrideAxis]);
            CUTLASS_PRAGMA_UNROLL
            for (int access_idx = 0; access_idx < kContiguousCount;
                 ++access_idx) {
                pointer_[access_idx] -= params_.inc_iterations_;
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int access_idx = 0; access_idx < kContiguousCount;
                 ++access_idx) {
                pointer_[access_idx] += params_.inc_next_;
            }
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
    Conv2dTileSrcIteratorFpropPrecomp operator++(int) {
        Conv2dTileSrcIteratorFpropPrecomp self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0xffffffff;
        }
    }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = mask[i];
        }
    }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            mask[i] = predicates_[i];
        }
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
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr = reinterpret_cast<char const*>(
                                                   pointer_[access_idx]) +
                                           byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard,
                            params_.pack_pad_);
                }
            }
            if (s < ThreadMap::Iterations::kStrided - 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < kContiguousCount;
                     ++access_idx) {
                    pointer_[access_idx] += params_.inc_strided_;
                }
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

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
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);
                    int access_idx = v + kAccessesPerVector * c;
                    int word_idx = access_idx / kPredicatesPerWord;
                    int residual = access_idx % kPredicatesPerWord;
                    int byte_idx = residual / kPredicatesPerByte;
                    int bit_idx = residual % kPredicatesPerByte;
                    bool guard = ((predicates_[word_idx] &
                                   (1u << (byte_idx * 8 + bit_idx))) != 0);
                    if (is_residue_tile_) {
                        guard = guard && s * ThreadMap::Delta::kStrided <
                                                 residue_extent_;
                    }

                    char const* byte_ptr = reinterpret_cast<char const*>(
                                                   pointer_[access_idx]) +
                                           byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr, guard);
                }
            }
            if (s < ThreadMap::Iterations::kStrided - 1) {
                CUTLASS_PRAGMA_UNROLL
                for (int access_idx = 0; access_idx < kContiguousCount;
                     ++access_idx) {
                    pointer_[access_idx] += params_.inc_strided_;
                }
            }
        }
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    static Status can_implement(Conv2dProblemSize& problem_size) {
        if (problem_size.mode != Mode::kCrossCorrelation) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.R != 1 || problem_size.S != 1) {
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
