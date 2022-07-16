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
 * \file include/cutlass/convolution/threadblock/conv2d_tile_params.h
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

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {

namespace threadblock {

enum class TileMapType {
    kRow2C_Col2N,      ///< Row map to channel, Column map to batch
    kRow2CHW_Col2N,    ///< Row map to CHW(channel x height x width), Column
                       ///< map to batch
    kRow2C_Col2NHW,    ///< Row map to channel, column map to NHW (batch x
                       ///< height x width)
    kRow2NHW_Col2C,    ///< Row map to NHW(batch x height x width), Column
                       ///< map to channel
    kRow2IHW_Col2OHW,  ///< Row map to IHW (input height x input width), Column
                       ///< map to OHW (output height x output width), used for
                       ///< visiting weight of depthwise conv2d
    kRow2N_Col2HW,     ///< Row map to batch dimension, Column map to HW (input
                    ///< height x input width), used for visiting input tensor
                    ///< of depthwise conv2d
    kRow2OHW_Col2IHW,  ///< Row map to OHW (output height x input width), Column
                       ///< map to IHW (input height x output width), used for
                       ///< visiting weight of depthwise conv2d
};

template <typename Layout, TileMapType tile_map_type_>
struct TileMap;

template <int Interleave>
struct TileMap<layout::TensorCxRSKx<Interleave>, TileMapType::kRow2C_Col2N> {
    using Layout = layout::TensorCxRSKx<Interleave>;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 2;
    CUTLASS_HOST_DEVICE
    TileMap() {}
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        return TensorCoord{coord.column(), 0, 0, coord.row()};
    }
};

template <int Interleave>
struct TileMap<layout::TensorCxRSKx<Interleave>, TileMapType::kRow2CHW_Col2N> {
    using Layout = layout::TensorCxRSKx<Interleave>;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 0;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = coord.column(), h = 0, w = 0, c = 0;
        int tmp = 0;
        fast_divmod(c, tmp, coord.row(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

template <int Interleave>
struct TileMap<layout::TensorKxRSCx<Interleave>, TileMapType::kRow2NHW_Col2C> {
    using Layout = layout::TensorKxRSCx<Interleave>;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 0;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = 0, h = 0, w = 0, c = coord.column();
        int tmp = 0;
        fast_divmod(n, tmp, coord.row(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

template <typename Layout_>
struct TileMap<Layout_, TileMapType::kRow2C_Col2NHW> {
    using Layout = Layout_;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 1;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = 0, h = 0, w = 0, c = coord.row();
        int tmp = 0;
        fast_divmod(n, tmp, coord.column(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

template <>
struct TileMap<layout::TensorNCHW, TileMapType::kRow2IHW_Col2OHW> {
    using Layout = layout::TensorNCHW;
    using TensorCoord = typename Layout::TensorCoord;
    using Index = Layout::Index;
    /// has no trivial strided axis
    static const int kStrideAxis = -1;
    Index wi_, wo_;
    Index sh_, sw_;
    Index ph_, pw_;
    unsigned int wi_mul_, wi_shr_, wo_mul_, wo_shr_;
    CUTLASS_HOST_DEVICE
    TileMap()
            : wi_(0),
              wo_(0),
              sh_(0),
              sw_(0),
              ph_(0),
              pw_(0),
              wi_mul_(0),
              wi_shr_(0),
              wo_mul_(0),
              wo_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index wi, Index wo, Index sh, Index sw, Index ph, Index pw)
            : wi_(wi), wo_(wo), sh_(sh), sw_(sw), ph_(ph), pw_(pw) {
        find_divisor(wi_mul_, wi_shr_, wi);
        find_divisor(wo_mul_, wo_shr_, wo);
    }
    /// convert row of logical coordinates to src height and width
    CUTLASS_HOST_DEVICE
    MatrixCoord operator()(Index const& dividend) const {
        int h, w;
        fast_divmod(h, w, dividend, wi_, wi_mul_, wi_shr_);
        return MatrixCoord{h, w};
    }
    /// convert column of logical coordinates to filter height and width
    CUTLASS_HOST_DEVICE
    MatrixCoord operator()(Index const& dividend,
                           MatrixCoord const& src) const {
        int h = 0, w = 0;
        fast_divmod(h, w, dividend, wo_, wo_mul_, wo_shr_);
        h = src.row() - h * sh_ + ph_;
        w = src.column() - w * sw_ + pw_;
        return MatrixCoord{h, w};
    }
    CUTLASS_HOST_DEVICE
    Coord<2> operator()(Coord<2> const& ranges, int const& offset) const {
        int lo, hi, mod;
        fast_divmod(lo, mod, ranges.at(0), wo_, wo_mul_, wo_shr_);
        fast_divmod(hi, mod, ranges.at(1), wo_, wo_mul_, wo_shr_);
        lo = lo * sh_ - ph_;
        hi = hi * sh_ - ph_ + offset;
        lo = lo >= 0 ? lo : 0;
        return make_Coord(lo * wi_, hi * wi_);
    }
};

template <>
struct TileMap<layout::TensorNCHW, TileMapType::kRow2N_Col2HW> {
    using Layout = layout::TensorNCHW;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 2;
    CUTLASS_HOST_DEVICE
    TileMap() {}
};

template <>
struct TileMap<layout::TensorNCHW, TileMapType::kRow2OHW_Col2IHW> {
    using Layout = layout::TensorNCHW;
    using TensorCoord = typename Layout::TensorCoord;
    using Index = Layout::Index;
    /// has no trivial strided axis
    static const int kStrideAxis = -1;
    Index wi_, wo_;
    Index sh_, sw_;
    Index ph_, pw_;
    unsigned int wi_mul_, wi_shr_, wo_mul_, wo_shr_;
    CUTLASS_HOST_DEVICE
    TileMap()
            : wi_(0),
              wo_(0),
              sh_(0),
              sw_(0),
              ph_(0),
              pw_(0),
              wi_mul_(0),
              wi_shr_(0),
              wo_mul_(0),
              wo_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index wi, Index wo, Index sh, Index sw, Index ph, Index pw)
            : wi_(wi), wo_(wo), sh_(sh), sw_(sw), ph_(ph), pw_(pw) {
        find_divisor(wi_mul_, wi_shr_, wi);
        find_divisor(wo_mul_, wo_shr_, wo);
    }
    /// convert row of logical coordinates to src height and width
    CUTLASS_HOST_DEVICE
    MatrixCoord operator()(Index const& dividend) const {
        int h, w;
        fast_divmod(h, w, dividend, wo_, wo_mul_, wo_shr_);
        return MatrixCoord{h, w};
    }
    /// convert column of logical coordinates to filter height and width
    CUTLASS_HOST_DEVICE
    MatrixCoord operator()(Index const& dividend,
                           MatrixCoord const& grad) const {
        int h = 0, w = 0;
        fast_divmod(h, w, dividend, wi_, wi_mul_, wi_shr_);
        h = h - grad.row() * sh_ + ph_;
        w = w - grad.column() * sw_ + pw_;
        return MatrixCoord{h, w};
    }
    CUTLASS_HOST_DEVICE
    MatrixCoord operator()(MatrixCoord const& coord) const {
        int oh, ow;
        fast_divmod(oh, ow, coord.row(), wo_, wo_mul_, wo_shr_);
        int ih, iw;
        fast_divmod(ih, iw, coord.column(), wi_, wi_mul_, wi_shr_);
        int fh = ih - oh * sh_ + ph_;
        int fw = iw - ow * sw_ + pw_;
        return MatrixCoord{fh, fw};
    }
    CUTLASS_HOST_DEVICE
    Coord<2> operator()(Coord<2> const& ranges, int const& offset) const {
        int lo, hi, mod;
        fast_divmod(lo, mod, ranges.at(0), wi_, wi_mul_, wi_shr_);
        fast_divmod(hi, mod, ranges.at(1), wi_, wi_mul_, wi_shr_);
        lo = (lo + ph_ - offset + 1) / sh_;
        hi = (hi + ph_) / sh_ + 1;
        lo = lo >= 0 ? lo : 0;
        return make_Coord(lo * wo_, hi * wo_);
    }
    CUTLASS_HOST_DEVICE
    Coord<2> operator()(Coord<2> const& row_ranges,
                        Coord<2> const& column_ranges) const {
        int input_lo, input_hi, mod;
        fast_divmod(input_lo, mod, column_ranges.at(0), wi_, wi_mul_, wi_shr_);
        fast_divmod(input_hi, mod, column_ranges.at(1), wi_, wi_mul_, wi_shr_);
        int output_lo, output_hi;
        fast_divmod(output_lo, mod, row_ranges.at(0), wo_, wo_mul_, wo_shr_);
        fast_divmod(output_hi, mod, row_ranges.at(1), wo_, wo_mul_, wo_shr_);
        int filter_lo = input_lo - output_hi * sh_ + ph_;
        int filter_hi = input_hi - output_lo * sh_ + ph_;
        return make_Coord(filter_lo, filter_hi);
    }
};

struct ExtraParamZeroPoint {
    uint8_t src_zero_point;

    CUTLASS_HOST_DEVICE
    ExtraParamZeroPoint() : src_zero_point(0) {}

    CUTLASS_HOST_DEVICE
    ExtraParamZeroPoint(uint8_t src_zero_point_)
            : src_zero_point(src_zero_point_) {}
};

namespace detail {
template <typename Element, typename ExtraParam>
CUTLASS_HOST_DEVICE uint32_t prepare_pack_pad(const ExtraParam& params) {
    return 0;
}

template <>
CUTLASS_HOST_DEVICE uint32_t prepare_pack_pad<uint4b_t, ExtraParamZeroPoint>(
        const ExtraParamZeroPoint& params) {
    uint32_t ret = 0;
    for (size_t i = 0; i < 8; i++) {
        ret |= params.src_zero_point << (4 * i);
    }
    return ret;
}
}  // namespace detail

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
