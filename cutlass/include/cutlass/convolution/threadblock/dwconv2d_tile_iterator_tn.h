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
 * include/cutlass/convolution/threadblock/dwconv2d_tile_iterator.h
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

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize, int AdvanceRank = 0>
class Dwconv2dTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of Dwconv2dTileIterator
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize, int AdvanceRank>
class Dwconv2dTileIterator<Shape_, Element_, layout::TensorNCHW, ThreadMap_,
                           AccessSize, AdvanceRank> {
public:
    using Shape = layout::PitchLinearShape<Shape_::kColumn, Shape_::kRow>;
    using Element = Element_;
    using Layout = layout::TensorNCHW;
    using ThreadMap = ThreadMap_;

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

    using UnderlyingIterator = transform::threadblock::PredicatedTileIterator<
            Shape, Element, layout::PitchLinear, AdvanceRank, ThreadMap,
            AccessSize>;

    /// Type used for internal memory accesses
    using AccessType = typename UnderlyingIterator::AccessType;

    /// Fragment object to be loaded or stored
    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using TileMap = TileMap<Layout, TileMapType::kRow2N_Col2HW>;

    /// Predicate vector stores mask to guard accesses
    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    public:
        /// compatible for constructor function
        using ExtraParam = platform::none_type;

    private:
        friend Dwconv2dTileIterator;

        /// Parameters object
        typename UnderlyingIterator::Params params_;

        Layout layout_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, Conv2dProblemSize const& problem_size,
               ExtraParam const& extra_param = {})
                : params_(layout::PitchLinear(
                          layout.stride()[TileMap::kStrideAxis])),
                  layout_(layout) {}
    };

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const& params_;

    /// Underlying pitch-linear tile iterator
    UnderlyingIterator iterator_;

public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    Dwconv2dTileIterator(
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
            : params_(params),
              iterator_(params.params_, pointer,
                        layout::PitchLinearCoord(extent.column(), extent.row()),
                        thread_id,
                        layout::PitchLinearCoord(threadblock_offset.column(),
                                                 threadblock_offset.row())) {}

    /// Construct a Dwconv2dTileFilterIteratorFpropPrecomp with zero threadblock
    /// offset
    CUTLASS_HOST_DEVICE
    Dwconv2dTileIterator(
            Params const& params,  ///< Precomputed parameters object
            Pointer pointer,       ///< Pointer to start of tensor
            LogicalCoord extent,   ///< Extent of tensor
            int thread_id          ///< ID of each participating thread
            )
            : Dwconv2dTileIterator(params, pointer, extent, thread_id,
                                   make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    /// Advances to the next tile in memory.
    ///
    /// Just update filter's coordinates
    CUTLASS_HOST_DEVICE
    Dwconv2dTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    CUTLASS_HOST_DEVICE
    Dwconv2dTileIterator operator++(int) {
        Dwconv2dTileIterator self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask() { iterator_.clear_mask(); }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void enable_mask() { iterator_.enable_mask(); }

    /// Sets the predicate mask, overriding value stored in predicate iterator
    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) { iterator_.set_mask(mask); }

    /// Gets the mask
    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) { iterator_.get_mask(mask); }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        iterator_.store_with_pointer_offset(frag, pointer_offset);
    }

    /// Store a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    Dwconv2dTileIterator& add_coord_offset(TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
