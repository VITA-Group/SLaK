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
 * \file include/cutlass/epilogue/threadblock/convolution_thread_map_simt.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "convolution_output_tile_thread_map_tensor_op.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "tensor_predicated_tile_iterator_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for TensorOp accumulator layouts
template <typename ThreadblockShape_, typename WarpShape_, typename Layout_,
          typename MmaTensorOpPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapTensorOp;

template <typename ThreadblockShape_, typename WarpShape_, int kInterleaved_,
          typename MmaTensorOpPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapTensorOp<
        ThreadblockShape_, WarpShape_, layout::TensorNCxHWx<kInterleaved_>,
        MmaTensorOpPolicy_, Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaTensorOpPolicy = MmaTensorOpPolicy_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kInterleaved_>;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kInterleaved = kInterleaved_;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");
        static_assert(!(WarpShape::kM % kInterleaved), "Divisibility");

        /// Number of warps
        using WarpCount = MatrixShape<ThreadblockShape::kM / WarpShape::kM,
                                      ThreadblockShape::kN / WarpShape::kN>;

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;
        using WarpAccessShape =
                MatrixShape<kInterleaved,
                            MmaTensorOpPolicy::Operator::Shape::kN>;

        using Iterations =
                MatrixShape<WarpShape::kM / kInterleaved,
                            WarpShape::kN /
                                    MmaTensorOpPolicy::Operator::Shape::kN>;

        static int const kIterations = Iterations::kCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMapTensorOp<
            typename Detail::WarpAccessShape, typename Detail::Iterations,
            typename Detail::WarpCount, kInterleaved, Detail::kThreads,
            kElementsPerAccess, sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaTensorOpPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapTensorOp<ThreadblockShape_, WarpShape_,
                                    layout::TensorNCxHWx<4>, MmaTensorOpPolicy_,
                                    Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaTensorOpPolicy = MmaTensorOpPolicy_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<4>;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kInterleaved = 4;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;
        static int const kWarpAccessSize = kWarpSize * ElementsPerAccess;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");

        /// Number of warps
        using WarpCount = MatrixShape<ThreadblockShape::kM / WarpShape::kM,
                                      ThreadblockShape::kN / WarpShape::kN>;

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;
        using WarpAccessShape =
                MatrixShape<MmaTensorOpPolicy::Operator::Shape::kM,
                            kWarpAccessSize / kInterleaved>;

        static_assert(!(WarpShape::kN % WarpAccessShape::kColumn),
                      "Divisibility");

        using Iterations =
                MatrixShape<WarpShape::kM / WarpAccessShape::kRow,
                            WarpShape::kN / WarpAccessShape::kColumn>;

        static int const kIterations = Iterations::kCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMapTensorOp<
            typename Detail::WarpAccessShape, typename Detail::Iterations,
            typename Detail::WarpCount, kInterleaved, Detail::kThreads,
            kElementsPerAccess, sizeof_bits<Element>::value>;
};

/// Defines the optimal thread map for TensorOp accumulator layouts
template <typename ThreadblockShape_, typename WarpShape_, int PartitionsK,
          typename Element_, int ElementsPerAccess, int InterleavedK>
struct InterleavedConvolutionThreadMapTensorOp {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    static int const kPartitionsK = PartitionsK;
    using Element = Element_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kInterleavedK = InterleavedK;

    //
    // Definitions
    //

    struct Detail {
        /// Tensor Operations fundamentally perform operations on 8 rows
        static int const kTensorOpRows = 8;
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kN % WarpShape::kN),
                      "Divisibility");

        /// Number of warps
        using WarpCount = gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                          ThreadblockShape::kN / WarpShape::kN,
                                          kPartitionsK>;

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::MaskedTileIterator satisfying concept
    /// InterleavedOutputTileThreadMap
    using Type = InterleavedConvolutionOutputTileThreadMap<
            MatrixShape<Detail::kTensorOpRows, InterleavedK>,
            MatrixShape<Detail::WarpCount::kM, Detail::WarpCount::kN>,
            MatrixShape<WarpShape::kM / Detail::kTensorOpRows,
                        WarpShape::kN / InterleavedK>,
            Detail::kThreads, kElementsPerAccess, sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
