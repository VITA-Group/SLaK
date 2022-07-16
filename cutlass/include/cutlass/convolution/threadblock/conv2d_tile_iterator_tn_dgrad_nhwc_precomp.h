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
    \brief Templates implementing loading of convolution tiles mapped to GEMM A
   (activation tile) matrix from memory.

    This iterator assumes TensorNHWC or TensorNCxHWx<Interleave> layout of
   tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators:
   forward propagation (Fprop), backward data gradient (Dgrad), and backward
   weight gradient (Wgrad).
*/

/**
 * \file
 * include/cutlass/convolution/threadblock/conv2d_tile_iterator_tn_dgrad_nhwc_precomp.h
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
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/convolution/threadblock/conv2d_tile_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

namespace detail {
struct PrecompRSK {
    uint8_t fw, fh;
    uint16_t k;
};

template <typename Shape_, int AccessSize>
CUTLASS_HOST_DEVICE void compute_offset_dgrad_nhwc(int* constant_offset_,
                                                   int fh_, int fw_) {
    // hardcoded typedef
    using Shape = Shape_;
    using Index = int;
    static int const kAccessSize = AccessSize;

    PrecompRSK* precomp_ptr = reinterpret_cast<PrecompRSK*>(constant_offset_);
    Index s = 0;
    Index filter_pixels = fh_ * fw_;
    Index inc_step = Shape::kColumn / kAccessSize;

    // first absolute offset
    CUTLASS_PRAGMA_UNROLL
    for (; s < inc_step; ++s) {
        Index k = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*k;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        PrecompRSK cur;
        cur.fw = (uint8_t)fw;
        cur.fh = (uint8_t)fh;
        cur.k = (uint16_t)(k * kAccessSize);
        *precomp_ptr = cur;
        precomp_ptr++;
    }

    CUTLASS_PRAGMA_UNROLL
    for (; s < (1 + filter_pixels) * inc_step; ++s) {
        Index k = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*k;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        PrecompRSK cur;
        cur.fw = (uint8_t)fw;
        cur.fh = (uint8_t)fh;
        cur.k = (uint16_t)(k * kAccessSize);
        *precomp_ptr = cur;
        precomp_ptr++;
    }
}
}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize>
class DgradPrecompNHWCParams;

/// Parameters object is precomputed state and is host-constructible
template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class DgradPrecompNHWCParams<Shape_, Element_, layout::TensorNHWC, ThreadMap_,
                             AccessSize> {
public:
    static int const kAccessSize = AccessSize;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using ThreadMap = ThreadMap_;

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
            sizeof(detail::PrecompRSK) * 8 / cutlass::sizeof_bits<Index>::value;

    // less than 3.2K
    static int const kPrecomputedOffsetBufferSize = 848;
    static int const kMaxFilterPixels =
            kPrecomputedOffsetBufferSize /
                    (kElementSize * Shape::kColumn / kAccessSize) -
            1;

    static_assert(!(Shape::kColumn % kAccessSize),
                  "Shape::kColumn must be divisible by the AccessSize.");

    /// Used for converting tensor coordinates into pointer offset
    Layout layout_;

    /// Parameters used for mapping logical coordinates to physical
    /// coordinates
    Index constant_offset_max_;
    Index constant_offset_rewind_;
    Index constant_offset_[kPrecomputedOffsetBufferSize];

    FastDivmod hw_divmod;
    FastDivmod w_divmod;
    FastDivmod stride_h_div_mod;
    FastDivmod stride_w_div_mod;

    int N, P, Q, K, R, S;
    int pad_h, pad_w;

    CUTLASS_HOST_DEVICE
    DgradPrecompNHWCParams() : layout_(Layout()) {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    DgradPrecompNHWCParams(Layout const& layout,
                           Conv2dProblemSize const& problem_size,
                           ExtraParam const& extra_param = {})
            : layout_(layout),
              hw_divmod(problem_size.H * problem_size.W),
              w_divmod(problem_size.W),
              stride_h_div_mod(problem_size.stride_h),
              stride_w_div_mod(problem_size.stride_w) {
        detail::compute_offset_dgrad_nhwc<Shape, kAccessSize>(
                constant_offset_, problem_size.R, problem_size.S);
        constant_offset_max_ = (problem_size.R * problem_size.S) *
                               Shape::kColumn / kAccessSize;
        constant_offset_rewind_ = (1 - problem_size.R * problem_size.S) *
                                  Shape::kColumn / kAccessSize;

        N = problem_size.N;
        P = problem_size.P;
        Q = problem_size.Q;
        K = problem_size.K;
        pad_h = problem_size.pad_h;
        pad_w = problem_size.pad_w;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape_, typename Element_, typename Layout_,
          typename ThreadMap_, int AccessSize,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE>
class Conv2dTileSrcIteratorDgradPrecompNHWC;

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize, SpecialOptimizeDesc SpecialOpt>
class Conv2dTileSrcIteratorDgradPrecompNHWC<Shape_, Element_,
                                            layout::TensorNHWC, ThreadMap_,
                                            AccessSize, SpecialOpt> {
public:
    //
    // Types
    //

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using TensorCoord = typename Layout::TensorCoord;
    using ThreadMap = ThreadMap_;
    static int const kAccessSize = AccessSize;
    using AccessType = AlignedArray<Element, kAccessSize>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using ConvProblemSize = typename conv::Conv2dProblemSize;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    //
    // Simplifying assertions
    //
    static_assert(ThreadMap::Iterations::kContiguous == 1,
                  "Require Iterations::kContiguous == 1");

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    //
    // Parameters structure
    //

    using Params = DgradPrecompNHWCParams<Shape, Element, Layout, ThreadMap,
                                          kAccessSize>;

private:
    Params const& params_;

    Index constant_offset_;
    Index k_offset_;

    // One pointer per access
    char const* pointer_[ThreadMap::Iterations::kStrided];

    uint32_t masks_;

    // We map masks into bits packed in this uint32_t container
    static_assert(ThreadMap::Iterations::kStrided < sizeof(uint32_t) * 8,
                  "Currently, the number of loads per iteration is limited by "
                  "the size of the predicates container.");

    int offset_h[ThreadMap::Iterations::kStrided];
    int offset_w[ThreadMap::Iterations::kStrided];

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecompNHWC(Params const& params,
                                          Element const* ptr,
                                          LogicalCoord extent, int thread_idx,
                                          MatrixCoord const& threadblock_offset)
            : params_(params) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        constant_offset_ = thread_coord.contiguous() / kAccessSize;

        k_offset_ = 0;

        masks_ = 0;

        int offset_n[ThreadMap::Iterations::kStrided];

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] = reinterpret_cast<char const*>(ptr);

            int offset_nhw = threadblock_offset.row() + thread_coord.strided() +
                             s * ThreadMap::Delta::kStrided;

            int residual;

            params_.hw_divmod(offset_n[s], residual, offset_nhw);
            params_.w_divmod(offset_h[s], offset_w[s], residual);

            TensorCoord coord = TensorCoord(offset_n[s], 0, 0, 0);

            pointer_[s] +=
                    params_.layout_(coord) * sizeof_bits<Element>::value / 8;

            uint32_t pred = (offset_n[s] < params_.N) ? 1u : 0;
            masks_ |= (pred << s);
        }
    }

private:
    /// Adds a pointer offset in units of element
    CUTLASS_HOST_DEVICE
    void add_byte_offset_(LongIndex byte_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] += byte_offset;
        }
    }

public:
    /// Adds a pointer offset in units of element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Clears the predicates
    CUTLASS_HOST_DEVICE
    void clear_mask() { masks_ = 0; }

    /// Increments to the next memory access
    CUTLASS_HOST_DEVICE Conv2dTileSrcIteratorDgradPrecompNHWC& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kColumn / kAccessSize;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
            k_offset_ += Shape::kColumn;
        }
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and
    /// the iterator's internal pointer is reverted to the first "steady
    /// state" tile. Subsequent calls are lightweight and must only update
    /// the internal pointer.
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorDgradPrecompNHWC operator++(int) {
        Conv2dTileSrcIteratorDgradPrecompNHWC self(*this);
        operator++();
        return self;
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
        for (int v = 0; v < kAccessesPerVector; ++v) {
            detail::PrecompRSK precomp_krs =
                    *(detail::PrecompRSK*)&params_
                             .constant_offset_[constant_offset_ + v];

            uint32_t filter_s = precomp_krs.fw;
            uint32_t filter_r = precomp_krs.fh;
            uint32_t filter_k = precomp_krs.k + k_offset_;

            bool guard_ = (filter_k < params_.K);

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
                bool guard = guard_ && (masks_ & (1u << s));

                int p, q, mod_p, mod_q;

                if (SpecialOpt ==
                    SpecialOptimizeDesc::DECONV_DOUBLE_UPSAMPLING) {
                    p = (offset_h[s] + params_.pad_h - filter_r) >> 1;
                    q = (offset_w[s] + params_.pad_w - filter_s) >> 1;
                    mod_p = (offset_h[s] + params_.pad_h - filter_r) & 0x1;
                    mod_q = (offset_w[s] + params_.pad_w - filter_s) & 0x1;
                } else {
                    params_.stride_h_div_mod(
                            p, mod_p, offset_h[s] + params_.pad_h - filter_r);
                    params_.stride_w_div_mod(
                            q, mod_q, offset_w[s] + params_.pad_w - filter_s);
                }

                guard = guard &&
                        ((p >= 0) && (p < params_.P) && (q >= 0) &&
                         (q < params_.Q) && (mod_p == 0) && (mod_q == 0));

                TensorCoord coord = TensorCoord(0, p, q, filter_k);

                Index stride = params_.layout_(coord) *
                               sizeof_bits<Element>::value / 8;

                char const* byte_ptr =
                        reinterpret_cast<char const*>(pointer_[s] + stride) +
                        byte_offset;

                AccessType const* access_ptr =
                        reinterpret_cast<AccessType const*>(byte_ptr);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[s * kAccessesPerVector + v], access_ptr,
                        guard);
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    static Status can_implement(ConvProblemSize& problem_size) {
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

template <typename Shape_, typename Element_, typename Layout,
          typename ThreadMap_, int AccessSize>
class Conv2dTileFilterIteratorDgradCKxRSx;

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class Conv2dTileFilterIteratorDgradCKxRSx<Shape_, Element_,
                                          layout::TensorCKxRSx<AccessSize>,
                                          ThreadMap_, AccessSize> {
public:
    //
    // Types
    //
    static int const kAccessSize = AccessSize;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorCKxRSx<kAccessSize>;
    using TensorCoord = typename Layout::TensorCoord;
    using ThreadMap = ThreadMap_;
    using AccessType = AlignedArray<Element, kAccessSize>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using ConvProblemSize = typename conv::Conv2dProblemSize;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    /// Logical layout
    using LogicalLayout = layout::RowMajor;

    /// Logical tensor coord
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    //
    // Simplifying assertions
    //
    static_assert(ThreadMap::Iterations::kContiguous == 1,
                  "Require Iterations::kContiguous == 1");

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    //
    // Parameters structure
    //

    class Params {
    public:
        /// stride of pitch-linear layout (units of Element)
        int stride_krs_, C;
        // Default ctor
        CUTLASS_HOST_DEVICE
        Params() : stride_krs_(0), C(0) {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, Conv2dProblemSize const& problem_size)
                : stride_krs_(layout.stride()[2]), C(problem_size.C) {}
    };

private:
    Params const& params_;

    // current filter position (c, r, s)
    Index krs_;

    // One pointer per access
    char const* pointer_[ThreadMap::Iterations::kStrided];

    uint32_t predicates_;

    // We map predicates into bits packed in this uint32_t container
    static_assert(ThreadMap::Iterations::kStrided < sizeof(uint32_t) * 8,
                  "Currently, the number of loads per iteration is limited by "
                  "the size of the predicates container.");

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorDgradCKxRSx(Params const& params,
                                        Element const* ptr, LogicalCoord extent,
                                        int thread_idx,
                                        MatrixCoord const& threadblock_offset)
            : params_(params), predicates_(0) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        krs_ = threadblock_offset.row() + thread_coord.contiguous();

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] = reinterpret_cast<char const*>(ptr);

            Index offset_c = threadblock_offset.column() +
                             thread_coord.strided() +
                             s * ThreadMap::Delta::kStrided;
            uint32_t pred = ((offset_c < params_.C) ? 1u : 0);
            predicates_ |= (pred << s);

            pointer_[s] += offset_c * params_.stride_krs_ *
                           sizeof_bits<Element>::value / 8;
        }
    }

private:
    /// Adds a pointer offset in units of element
    CUTLASS_HOST_DEVICE
    void add_byte_offset_(LongIndex byte_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] += byte_offset;
        }
    }

public:
    /// Adds a pointer offset in units of element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
    }

    /// Clears the predicates
    CUTLASS_HOST_DEVICE
    void clear_mask() { predicates_ = 0; }

    /// Increments to the next memory access
    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorDgradCKxRSx& operator++() {
        krs_ += Shape::kRow;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and
    /// the iterator's internal pointer is reverted to the first "steady
    /// state" tile. Subsequent calls are lightweight and must only update
    /// the internal pointer.
    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorDgradCKxRSx operator++(int) {
        Conv2dTileFilterIteratorDgradCKxRSx self(*this);
        operator++();
        return self;
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
        for (int v = 0; v < kAccessesPerVector; ++v) {
            uint32_t cur_krs = krs_ + v * kAccessSize;

            bool guard_ = (cur_krs < params_.stride_krs_);

            Index stride = cur_krs * sizeof_bits<Element>::value / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
                bool guard = guard_ && (predicates_ & (1u << s));

                char const* byte_ptr =
                        reinterpret_cast<char const*>(pointer_[s] + stride) +
                        byte_offset;

                AccessType const* access_ptr =
                        reinterpret_cast<AccessType const*>(byte_ptr);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[s * kAccessesPerVector + v], access_ptr,
                        guard);
            }
        }
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
