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
 * \file include/cutlass/epilogue/threadblock/interleaved_shared_load_iterator.h
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
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator
///
template <typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
          typename Element_,    ///< Element data type
          int Interleaved,
          int MaxAlignment =
                  ThreadMap_::kElementsPerAccess* sizeof_bits<Element_>::value /
                  8>
class InterleavedSharedLoadIteratorTensorOp {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::RowMajor;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static int const kInterleaved = Interleaved;

    static int const kMinAlignment =
            ThreadMap_::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

    static int const kAlignment =
            (MaxAlignment < kMinAlignment ? MaxAlignment : kMinAlignment);

    static int const kThreads = ThreadMap::kThreads;

    static_assert(
            ThreadMap::Iterations::kRow <= kInterleaved,
            "Iterations::kRow cannot be greater than interleaving quantity");
    static_assert(ThreadMap::Iterations::kRow >= 1,
                  "ThreadMap::Iterations::kRow must be greater than 1");
    static_assert(ThreadMap::Iterations::kColumn >= 1,
                  "ThreadMap::Iterations::kColumn must be greater than 1");
    static_assert(ThreadMap::kRowsPerIteration >= 1,
                  "ThreadMap::kRowPerIteration must be greater than 1");

    /// Fragment object
    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::kRowsPerIteration *
                                            ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType =
            AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    /// Vector type used for SMEM loads
    using LoadType = AlignedArray<Element,
                                  const_min(128 / sizeof_bits<Element>::value,
                                            ThreadMap::kElementsPerAccess),
                                  const_min(16, kAlignment)>;
    static_assert(!(AccessType::kElements % LoadType::kElements),
                  "Divisibility");
    static int const kLoadsPerAccess =
            AccessType::kElements / LoadType::kElements;

private:
    //
    // Data members
    //

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Stride along adjacent rows
    int stride_;

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    InterleavedSharedLoadIteratorTensorOp(TensorRef ref, int thread_idx)
            : byte_pointer_(reinterpret_cast<uint8_t*>(ref.data())),
              stride_((ref.stride(0) * sizeof_bits<Element>::value) / 8) {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

        // Initialize pointer
        byte_pointer_ += thread_offset.row() * stride_ +
                         thread_offset.column() * sizeof(AccessType) /
                                 kElementsPerAccess;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& offset) {
        add_pointer_offset(offset.row() * Shape::kRow * stride_ /
                                   (sizeof_bits<Element>::value / 8) +
                           offset.column() * Shape::kColumn);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int r_outer = 0; r_outer < ThreadMap::Iterations::kRow;
             ++r_outer) {
            CUTLASS_PRAGMA_UNROLL
            for (int r_inner = 0; r_inner < ThreadMap::kRowsPerIteration;
                 ++r_inner) {
                uint8_t const* byte_pointer =
                        byte_pointer_ +
                        r_outer * ThreadMap::Delta::kRow * stride_ +
                        r_inner * stride_ +
                        pointer_offset * sizeof_bits<Element>::value / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int column = 0; column < ThreadMap::Iterations::kColumn;
                     ++column) {
                    int frag_idx =
                            r_inner +
                            (r_outer + ThreadMap::Iterations::kRow * column) *
                                    ThreadMap::kRowsPerIteration;

                    LoadType* frag_ptr = reinterpret_cast<LoadType*>(&frag);
                    LoadType const* memory_pointer =
                            reinterpret_cast<LoadType const*>(byte_pointer);
                    CUTLASS_PRAGMA_UNROLL
                    for (int v = 0; v < kLoadsPerAccess; ++v) {
                        frag_ptr[frag_idx * kLoadsPerAccess + v] =
                                memory_pointer[(column *
                                                ThreadMap::Delta::kColumn /
                                                kElementsPerAccess) *
                                                       kLoadsPerAccess +
                                               v];
                    }
                }
            }
        }
    }

    /// Loads a fragment
    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
