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
    \brief
*/

/**
 * \file include/cutlass/epilogue/warp/interleaved_tile_iterator_simt.h
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
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/interleaved_simt_policy.h"

#define CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES 1

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <typename WarpShape,      ///< shape of warp-level GEMM (concept:
                                   ///< MatrixShape)
          typename Operator,       ///< matrix multiply operation (concept:
                                   ///< arch::Mma)
          typename Element,        ///< data type of element to be written
          typename Layout,         ///< target shared memory layout
          typename MmaSimtPolicy,  ///< policy defining lane arrangement
                                   ///< (concept: MmaSimtPolicy)
          typename Policy = InterleavedSimtPolicy<WarpShape, Operator, Layout,
                                                  MmaSimtPolicy>  ///< Policy
          >
class InterleavedTileIteratorSimt;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
        typename WarpShape_,  ///< shape of warp-level GEMM (concept: GemmShape)
        typename Operator_,  ///< matrix multiply operation (concept: arch::Mma)
        typename Element_,   ///< data type of element to be written
        typename MmaSimtPolicy_,  ///< policy defining lane arrangement
                                  ///< (concept: MmaSimtPolicy)
        typename Policy_>
class InterleavedTileIteratorSimt<WarpShape_, Operator_, Element_,
                                  layout::RowMajor, MmaSimtPolicy_, Policy_> {
public:
    using WarpShape = WarpShape_;
    using Operator = Operator_;
    using Element = Element_;
    using Layout = layout::RowMajor;

    using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
    using TensorCoord =
            MatrixCoord;  ///< Logical coordinate in referenced tensor
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = Policy_;

    /// Shape of the tile in memory
    using Shape = typename Policy::Shape;

    /// This is the fragment size produced by one access of the iterator.
    using Fragment =
            Array<typename Operator::ElementC, Policy::kElementsPerIteration>;

    /// This is the complete warp-level accumulator tile.
    using AccumulatorTile = Array<typename Operator::ElementC,
                                  Policy::kAccumulatorElementCount>;

    /// Number of times this iterator can be incremented
    static int const kIterations = Policy::kIterations;

    /// Padding quantity
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
    using Padding = MatrixShape<0, 1>;
#endif

private:
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
    /// Storage type for accessing memory
    using AccessType = AlignedArray<Element, 1>;

#else
    /// Storage type for accessing memory
    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;
#endif

    //
    // Data members
    //

    /// Internal pointer to memory
    AccessType* pointer_;

    /// Internal layout object
    Layout layout_;

public:
    /// Default constructor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt() : pointer_(nullptr) {}

    /// Constructor from TensorRef
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0] / AccessType::kElements) {
        auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
        MatrixCoord lane_offset =
                lane_layout.inverse(lane_id) *
                MatrixCoord{Policy::MmaSimtPolicy::LaneMmaShape::kM, 1};

        pointer_ += layout_({lane_offset.row(),
                             lane_offset.column() * Policy::kElementsPerAccess /
                                     int(AccessType::kElements)});
    }

    /// Adds a pointer offset
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset / AccessType::kElements;
        return *this;
    }

    ///< advances in units of whole tiles along the logical coordinate space of
    ///< the tensor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& add_tile_offset(
            TensorCoord const& tile_offset) {
        pointer_ += layout_({tile_offset.row() * Shape::kRow,
                             (tile_offset.column() * Shape::kColumn /
                              int(AccessType::kElements))});

        return *this;
    }

    ///< advances in units of whole tiles along the logical coordinate space of
    ///< the tensor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& operator+=(TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);

        return *this;
    }

    /// Store
    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
        // de-vectorized stores
        using ScalarAccessType = AlignedArray<Element, 1>;
        ScalarAccessType const* scalarFragPtr =
                reinterpret_cast<ScalarAccessType const*>(&frag);
        ScalarAccessType* scalarPointer =
                reinterpret_cast<ScalarAccessType*>(pointer_) + pointer_offset;

        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < Policy::kRowAccessesPerIteration; ++r) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Policy::kColumnAccessesPerIteration; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int s = 0; s < Policy::kElementsPerAccess; s++) {
                    scalarPointer
                            [n * Policy::MmaSimtPolicy::WarpShape::kColumn *
                                     Policy::kElementsPerAccess +
                             s] = scalarFragPtr[n * Policy::kElementsPerAccess +
                                                s];
                }
            }
            scalarPointer += layout_(TensorCoord{1, 0});
            scalarFragPtr += Policy::kColumnAccessesPerIteration *
                             Policy::kElementsPerAccess;
        }
#else
        static_assert(false, "not implemented");
#endif
    }

    /// Store
    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
        typename WarpShape_,  ///< shape of warp-level GEMM (concept: GemmShape)
        typename Operator_,  ///< matrix multiply operation (concept: arch::Mma)
        typename Element_,   ///< data type of element to be written
        typename MmaSimtPolicy_,  ///< policy defining lane arrangement
                                  ///< (concept: MmaSimtPolicy)
        typename Policy_>
class InterleavedTileIteratorSimt<WarpShape_, Operator_, Element_,
                                  layout::ColumnMajor, MmaSimtPolicy_,
                                  Policy_> {
public:
    using WarpShape = WarpShape_;
    using Operator = Operator_;
    using Element = Element_;
    using Layout = layout::ColumnMajor;

    using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
    using TensorCoord =
            MatrixCoord;  ///< Logical coordinate in referenced tensor
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = Policy_;

    /// Shape of the tile in memory
    using Shape = typename Policy::Shape;

    /// This is the fragment size produced by one access of the iterator.
    using Fragment =
            Array<typename Operator::ElementC, Policy::kElementsPerIteration>;

    /// This is the complete warp-level accumulator tile.
    using AccumulatorTile = Array<typename Operator::ElementC,
                                  Policy::kAccumulatorElementCount>;

    /// Number of times this iterator can be incremented
    static int const kIterations = Policy::kIterations;

    /// Padding quantity

#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
    using Padding = MatrixShape<Policy::kElementsPerAccess, 0>;
#endif

private:
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
    /// Storage type for accessing memory
    using AccessType = AlignedArray<Element, 1>;
#endif

    //
    // Data members
    //

    /// Internal pointer to memory
    AccessType* pointer_;

    /// Internal layout object
    Layout layout_;

public:
    /// Default constructor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt() : pointer_(nullptr) {}

    /// Constructor from TensorRef
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0] / AccessType::kElements) {
        auto lane_layout = Policy::MmaSimtPolicy::get_lane_layout();
        MatrixCoord lane_offset =
                lane_layout.inverse(lane_id) *
                MatrixCoord{1, Policy::MmaSimtPolicy::LaneMmaShape::kN};

        pointer_ += layout_({lane_offset.row() * Policy::kElementsPerAccess /
                                     int(AccessType::kElements),
                             lane_offset.column()});
    }

    /// Adds a pointer offset
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset / AccessType::kElements;
        return *this;
    }

    ///< advances in units of whole tiles along the logical coordinate space of
    ///< the tensor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& add_tile_offset(
            TensorCoord const& tile_offset) {
        pointer_ += layout_(
                {(tile_offset.row() * Shape::kRow / int(AccessType::kElements)),
                 tile_offset.column() * Shape::kColumn});

        return *this;
    }

    ///< advances in units of whole tiles along the logical coordinate space of
    ///< the tensor
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorSimt& operator+=(TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);

        return *this;
    }

    /// Store
    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
#if CUTLASS_SIMT_EPILOGUE_USE_SCALAR_STORES
        // de-vectorized stores
        using ScalarAccessType = AlignedArray<Element, 1>;
        ScalarAccessType const* scalarFragPtr =
                reinterpret_cast<ScalarAccessType const*>(&frag);
        ScalarAccessType* scalarPointer =
                reinterpret_cast<ScalarAccessType*>(pointer_) + pointer_offset;

        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < Policy::kColumnAccessesPerIteration; ++c) {
            CUTLASS_PRAGMA_UNROLL
            for (int r = 0; r < Policy::kRowAccessesPerIteration; ++r) {
                CUTLASS_PRAGMA_UNROLL
                for (int s = 0; s < Policy::kElementsPerAccess; s++) {
                    scalarPointer[r * Policy::MmaSimtPolicy::WarpShape::kRow *
                                          Policy::kElementsPerAccess +
                                  s] =
                            scalarFragPtr[r * Policy::kElementsPerAccess + s];
                }
            }
            scalarPointer += layout_(TensorCoord{0, 1});
            scalarFragPtr += Policy::kRowAccessesPerIteration *
                             Policy::kElementsPerAccess;
        }
#else
        static_assert(false, "not implemented");
#endif
    }

    /// Store
    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
