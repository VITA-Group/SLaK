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
  \brief Epilogue for threadblock scoped GEMMs using SIMT.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

*/

/**
 * \file include/cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h
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
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/threadblock/default_thread_map_simt.h"
#include "cutlass/epilogue/threadblock/convolution_thread_map_simt.h"
#include "cutlass/epilogue/warp/fragment_iterator_simt.h"
#include "cutlass/epilogue/warp/tile_iterator_simt.h"

#include "cutlass/epilogue/threadblock/bias_tile_iterator.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/dwconv2d_predicated_tile_iterator.h"
#include "cutlass/epilogue/warp/interleaved_simt_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape_,        ///< Threadblock-level tile size (concept:
                                  ///< GemmShape)
          typename LayoutDst_,    ///< Layout type for output tensor
          typename LayoutBias_,   ///< Layout type for bias tensor
          typename WarpMmaSimt_,  ///< Warp-level mma operator
          typename OutputOp_,     ///< Thread-level epilogue operator
          int ElementsPerAccess   ///< Elements per access
          >
struct Dwconv2dEpilogueSimt;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape_, typename WarpMmaSimt_, typename OutputOp_,
          int ElementsPerAccess>
struct Dwconv2dEpilogueSimt<Shape_, layout::TensorNCHW, layout::TensorNCHW,
                            WarpMmaSimt_, OutputOp_, ElementsPerAccess> {
    using Shape = Shape_;
    using WarpMmaSimt = WarpMmaSimt_;
    using OutputOp = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

    using ElementOutput = typename OutputOp::ElementOutput;
    using LayoutDst = layout::TensorNCHW;
    using ElementBias = typename OutputOp::ElementBias;
    using LayoutBias = layout::TensorNCHW;
    using ElementAccumulator = typename WarpMmaSimt::ElementC;

    //
    // Thread map
    //

    using OutputTileThreadMap =
            typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
                    Shape, typename WarpMmaSimt::Shape,
                    typename WarpMmaSimt::Policy, kPartitionsK, ElementOutput,
                    kElementsPerAccess>::Type;

    using OutputTileIterator =
            cutlass::epilogue::threadblock::Dwconv2dPredicatedTileIterator<
                    OutputTileThreadMap, LayoutDst, ElementOutput>;

    using AccumulatorFragmentIterator =
            cutlass::epilogue::warp::FragmentIteratorSimt<
                    typename WarpMmaSimt::Shape,
                    typename WarpMmaSimt::ThreadMma,
                    typename WarpMmaSimt::LayoutC,
                    typename WarpMmaSimt::Policy>;

    using WarpTileIterator = cutlass::epilogue::warp::TileIteratorSimt<
            typename WarpMmaSimt::Shape, typename WarpMmaSimt::ThreadMma,
            ElementAccumulator, typename WarpMmaSimt::LayoutC,
            typename WarpMmaSimt::Policy>;

    using SharedLoadIterator =
            cutlass::epilogue::threadblock::SharedLoadIterator<
                    typename OutputTileThreadMap::CompactedThreadMap,
                    ElementAccumulator>;

    using BiasTileIterator =
            cutlass::epilogue::threadblock::Dwconv2dBiasTileIterator<
                    LayoutBias, ElementBias>;

    /// Hard-coded padding elements added
    using Padding = typename WarpTileIterator::Padding;

    //
    // Define the epilogue
    //
    using Epilogue = cutlass::epilogue::threadblock::ConvolutionEpilogue<
            Shape, LayoutDst, kPartitionsK, WarpMmaSimt, OutputTileIterator,
            AccumulatorFragmentIterator, WarpTileIterator, SharedLoadIterator,
            BiasTileIterator, OutputOp, Padding>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
