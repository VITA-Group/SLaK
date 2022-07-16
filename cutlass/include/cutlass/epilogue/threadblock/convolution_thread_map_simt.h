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

#include "convolution_output_tile_thread_map.h"
#include "cutlass/gemm/gemm.h"
#include "tensor_predicated_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the optimal thread map for SIMT accumulator layouts
template <typename ThreadblockShape_, typename WarpShape_, typename Layout_,
          typename MmaSimtPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapSimt;

template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaSimtPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapSimt<ThreadblockShape_, WarpShape_,
                                layout::TensorCxRSKx<4>, MmaSimtPolicy_,
                                Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaSimtPolicy = MmaSimtPolicy_;
    using Element = Element_;
    using Layout = layout::TensorCxRSKx<4>;
    static int const kElementsPerAccess = ElementsPerAccess;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");

        /// Number of warps
        using WarpCount =
                gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                ThreadblockShape::kN / WarpShape::kN, 1>;

        /// Computes number of thread-level matrix multiplies are needed to span
        /// a warp
        static int const kGroupCount =
                WarpShape::kM / (MmaSimtPolicy::WarpShape::kRow *
                                 MmaSimtPolicy::LaneMmaShape::kM);

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;

        /// Number of iterations
        static int const kIterations = kGroupCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMap<
            OutputTileShape<  // Shape
                    ThreadblockShape::kN, MmaSimtPolicy::LaneMmaShape::kM,
                    MmaSimtPolicy::WarpShape::kRow, Detail::WarpCount::kM, 1>,
            OutputTileShape<  // Count
                    1, 1, Detail::kGroupCount, 1, Detail::kIterations>,
            4, Detail::kThreads, kElementsPerAccess,
            sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaSimtPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapSimt<ThreadblockShape_, WarpShape_,
                                layout::TensorNCxHWx<4>, MmaSimtPolicy_,
                                Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaSimtPolicy = MmaSimtPolicy_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<4>;
    static int const kElementsPerAccess = ElementsPerAccess;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");

        /// Number of warps
        using WarpCount =
                gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                ThreadblockShape::kN / WarpShape::kN, 1>;

        /// Computes number of thread-level matrix multiplies are needed to span
        /// a warp
        static int const kGroupCount =
                WarpShape::kM / (MmaSimtPolicy::WarpShape::kRow *
                                 MmaSimtPolicy::LaneMmaShape::kM);

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;

        /// Number of iterations
        static int const kIterations = kGroupCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMap<
            OutputTileShape<  // Shape
                    ThreadblockShape::kN, MmaSimtPolicy::LaneMmaShape::kM,
                    MmaSimtPolicy::WarpShape::kRow, Detail::WarpCount::kM, 1>,
            OutputTileShape<  // Count
                    1, 1, Detail::kGroupCount, 1, Detail::kIterations>,
            4, Detail::kThreads, kElementsPerAccess,
            sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaSimtPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapSimt<ThreadblockShape_, WarpShape_,
                                layout::TensorNCxHWx<32>, MmaSimtPolicy_,
                                Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaSimtPolicy = MmaSimtPolicy_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<32>;
    static int const kElementsPerAccess = ElementsPerAccess;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");

        static_assert(!(ThreadblockShape::kM % 32), "Divisibility");

        /// Number of warps
        using WarpCount =
                gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                ThreadblockShape::kN / WarpShape::kN, 1>;

        /// Computes number of thread-level matrix multiplies are needed to span
        /// a warp
        static int const kGroupCount =
                WarpShape::kN / (MmaSimtPolicy::WarpShape::kColumn *
                                 MmaSimtPolicy::LaneMmaShape::kN);

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;

        /// Number of iterations
        static int const kIterations = kGroupCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMap<
            OutputTileShape<  // Shape
                    MmaSimtPolicy::LaneMmaShape::kN, ThreadblockShape::kM,
                    MmaSimtPolicy::WarpShape::kColumn, Detail::WarpCount::kN,
                    1>,
            OutputTileShape<  // Count
                    1, 1, Detail::kGroupCount, 1, Detail::kIterations>,
            32, Detail::kThreads, kElementsPerAccess,
            sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaSimtPolicy_, typename Element_, int ElementsPerAccess>
struct ConvolutionThreadMapSimt<ThreadblockShape_, WarpShape_,
                                layout::TensorNHWC, MmaSimtPolicy_,
                                Element_, ElementsPerAccess> {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaSimtPolicy = MmaSimtPolicy_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    static int const kElementsPerAccess = ElementsPerAccess;

    //
    // Definitions
    //

    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kM % WarpShape::kM),
                      "Divisibility");

        /// Number of warps
        using WarpCount =
                gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                ThreadblockShape::kN / WarpShape::kN, 1>;

        /// Computes number of thread-level matrix multiplies are needed to span
        /// a warp
        static int const kGroupCount =
                WarpShape::kN / (MmaSimtPolicy::WarpShape::kColumn *
                                 MmaSimtPolicy::LaneMmaShape::kN);

        /// Number of participating threads
        static int const kThreads = WarpCount::kCount * kWarpSize;

        /// Number of iterations
        static int const kIterations = kGroupCount;
    };

    //
    // ThreadMap
    //

    /// ThreadMap to be used by epilogue::PredicatedTileIterator satisfying
    /// concept OutputTileThreadMap
    using Type = ConvolutionOutputTileOptimalThreadMapNHWC<
            OutputTileShape<  // Shape
                    MmaSimtPolicy::LaneMmaShape::kN, ThreadblockShape::kM,
                    MmaSimtPolicy::WarpShape::kColumn, Detail::WarpCount::kN,
                    1>,
            OutputTileShape<  // Count
                    1, 1, Detail::kGroupCount, 1, Detail::kIterations>,
            Detail::kThreads, kElementsPerAccess, sizeof_bits<Element>::value>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
