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
    \brief Template for a double-buffered threadblock-scoped Convolution kernel.
*/
/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_singlestage.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/convolution/threadblock/implicit_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Iterates over tiles of Src Tensor operand in global memory
        ///  (concept: ReadableTileIterator | ForwardTileIterator |
        ///  MaskedTileIterator | RandomAccessTileIterator)
        typename IteratorSrc_,
        /// Iterates over tiles of Src Tensor operand in shared memory
        /// (concept: WriteableTileIterator | RandomAccessTileIterator)
        typename SmemIteratorSrc_,
        /// Iterates over tiles of Filter Tensor operand in global memory
        ///  (concept: ReadableTileIterator | ForwardTileIterator |
        ///  MaskedTileIterator | RandomAccessTileIterator)
        typename IteratorFilter_,
        /// Iterates over tiles of Filter operand in shared memory
        /// (concept: WriteableTileIterator | RandomAccessTileIterator)
        typename SmemIteratorFilter_,
        /// Data type of accumulator Dst Tensor
        typename ElementDst_,
        /// Data type of accumulator Dst Tensor
        typename LayoutDst_,
        /// Policy describing tuning details (concept: MmaPolicy)
        typename Policy_,
        /// Transformation applied to A operand
        typename TransformSrc_ =
                NumericArrayConverter<typename SmemIteratorSrc_::Element,
                                      typename IteratorSrc_::Element,
                                      IteratorSrc_::Fragment::kElements>,
        ///
        /// Transformation applied to B operand
        typename TransformFilter_ =
                NumericArrayConverter<typename SmemIteratorFilter_::Element,
                                      typename IteratorFilter_::Element,
                                      IteratorFilter_::Fragment::kElements>,
        /// Used for partial specialization
        typename Enable = bool>
class MmaNtSingleStage : public MmaBase<Shape_, Policy_, 1> {
public:
    ///< Base class
    using Base = MmaBase<Shape_, Policy_, 1>;

    using Shape =
            Shape_;  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
    using IteratorSrc = IteratorSrc_;  ///< Iterates over tiles of Src Tensor
                                       ///< operand in global memory
    using IteratorFilter =
            IteratorFilter_;         ///< Iterates over tiles of Filter Tensor
                                     ///< operand in global memory
    using ElementDst = ElementDst_;  ///< Data type of accumulator matrix
    using LayoutDst = LayoutDst_;    ///< Layout of accumulator matrix
    using Policy = Policy_;          ///< Policy describing tuning details

    using SmemIteratorSrc = SmemIteratorSrc_;
    using SmemIteratorFilter = SmemIteratorFilter_;

    using TransformSrc = TransformSrc_;
    using TransformFilter = TransformFilter_;

    //
    // Dependent types
    //

    /// Fragment of operand Src Tensor loaded from global memory
    using FragmentSrc = typename IteratorSrc::Fragment;

    /// Fragment of operand B loaded from global memory
    using FragmentFilter = typename IteratorFilter::Fragment;

    /// Fragment of accumulator tile
    using FragmentDst = typename Policy::Operator::FragmentC;

    /// Warp-level Mma
    using Operator = typename Policy::Operator;

    /// Obtain the arch tag from the warp-level operator
    using ArchTag = typename Policy::Operator::ArchTag;

    /// Complex transform on Tensor Src (A operand)
    static ComplexTransform const kTransformSrc = Operator::kTransformB;

    /// Complex transform on Tensor Filter (B operand)
    static ComplexTransform const kTransformFilter = Operator::kTransformA;

    // staticaly assert kStages for MmaSingleStage is 1 (Single-buffered
    // pipeline)
    static_assert((Base::kStages == 1),
                  "MmaSingleStage requires kStages set to value 1");

private:
    using WarpFragmentSrc = typename Operator::FragmentB;
    using WarpFragmentFilter = typename Operator::FragmentA;

protected:
    /// Iterator to write threadblock-scoped tile of Src Tensor operand to
    /// shared memory
    SmemIteratorSrc smem_iterator_src_;

    /// Iterator to write threadblock-scoped tile of Filter Tensor operand to
    /// shared memory
    SmemIteratorFilter smem_iterator_filter_;

public:
    /// Construct from tensor references
    CUTLASS_DEVICE
    MmaNtSingleStage(
            typename Base::SharedStorage&
                    shared_storage,  ///< Shared storage needed for internal
                                     ///< use by threadblock-scoped Convolution
            int thread_idx,          ///< ID within the threadblock
            int warp_idx,            ///< ID of warp
            int lane_idx             ///< ID of each thread within a warp
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_src_(shared_storage.operand_src_ref(), thread_idx),
              smem_iterator_filter_(shared_storage.operand_filter_ref(),
                                    thread_idx) {
        // Compute warp location within threadblock tile by mapping the warp_id
        // to three coordinates:
        //   _m: the warp's position within the threadblock along the M
        //   dimension _n: the warp's position within the threadblock along the
        //   N dimension _k: the warp's position within the threadblock along
        //   the K dimension

        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_src_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
        this->warp_tile_iterator_filter_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    }

    /// Perform a threadblock-scoped matrix multiply-accumulate
    CUTLASS_DEVICE
    void operator()(
            int conv_h_iterations,  ///< iterations of convolution operator over
                                    ///< height dimension
            int conv_w_iterations,  ///< iterations of convolution operator over
                                    ///< width dimension
            int conv_c_iterations,  ///< iterations of convolution operator over
                                    ///< channel dimension
            FragmentDst& accum,     ///< destination accumulator tile
            IteratorSrc iterator_src,  ///< iterator over Src Tensor operand in
                                       ///< global memory
            IteratorFilter iterator_filter,  ///< iterator over Filter Tensor
                                             ///< operand in global memory
            FragmentDst const& src_accum,    ///< source accumulator tile
            TransformSrc transform_src =
                    TransformSrc(),  ///< transformation applied to Src Tensor
                                     ///< fragment
            TransformFilter transform_filter =
                    TransformFilter()) {  ///< transformation applied to Filter
                                          ///< Tensor fragment

        //
        // Prologue
        //

        // Perform accumulation in the 'd' output operand
        accum = src_accum;

        FragmentSrc tb_frag_src;
        FragmentFilter tb_frag_filter;

        tb_frag_src.clear();
        tb_frag_filter.clear();

        // The last kblock is loaded in the prolog
        iterator_src.load(tb_frag_src);
        iterator_filter.load(tb_frag_filter);

        this->smem_iterator_src_.store(transform_src(tb_frag_src));
        this->smem_iterator_filter_.store(transform_filter(tb_frag_filter));

        __syncthreads();

        // Pair of fragments used to overlap shared memory loads and math
        // instructions
        WarpFragmentSrc warp_frag_src[2];
        WarpFragmentFilter warp_frag_filter[2];

        this->warp_tile_iterator_src_.set_kgroup_index(0);
        this->warp_tile_iterator_filter_.set_kgroup_index(0);

        this->warp_tile_iterator_src_.load(warp_frag_src[0]);
        this->warp_tile_iterator_filter_.load(warp_frag_filter[0]);

        ++this->warp_tile_iterator_src_;
        ++this->warp_tile_iterator_filter_;

        Operator warp_mma;

        // Avoid reading out of bounds
        //        if (gemm_k_iterations <= 1) {
        //            iterator_src.clear_mask();
        //            iterator_filter.clear_mask();
        //        }

        CUTLASS_GEMM_LOOP
        for (int h = 0; h <= conv_h_iterations; ++h) {
            CUTLASS_GEMM_LOOP
            for (int w = 0; w <= conv_w_iterations; ++w) {
                CUTLASS_GEMM_LOOP
                for (int c = 0; c < conv_c_iterations; ++c) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int warp_mma_k = 0;
                         warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
                        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                            __syncthreads();
                            // Write fragments to shared memory
                            this->smem_iterator_src_.store(
                                    transform_src(tb_frag_src));

                            this->smem_iterator_filter_.store(
                                    transform_filter(tb_frag_filter));

                            __syncthreads();

                            // Add negative offsets to return smem load
                            // iterators to the 'start' of the shared memory

                            this->warp_tile_iterator_src_.add_tile_offset(
                                    {-Policy::kPartitionsK *
                                             Base::kWarpGemmIterations,
                                     0});
                            this->warp_tile_iterator_filter_.add_tile_offset(
                                    {0, -Policy::kPartitionsK *
                                                Base::kWarpGemmIterations});
                        }

                        if (warp_mma_k == 0) {
                            if (c == conv_c_iterations - 1) {
                                if (h < conv_h_iterations ||
                                    w < conv_w_iterations) {
                                    int inc_w = 1;
                                    int inc_h = 0;
                                    if (w == conv_w_iterations) {
                                        inc_w = -w;
                                        inc_h = 1;
                                    }
                                    iterator_src.add_coord_offset(
                                            typename IteratorSrc::TensorCoord{
                                                    0, inc_h, inc_w,
                                                    -c * Shape::kK});
                                    iterator_filter.add_coord_offset(
                                            typename IteratorFilter::
                                                    TensorCoord{
                                                            0, inc_h, inc_w,
                                                            -c * Shape::kK});
                                }
                            } else {
                                ++iterator_src;
                                ++iterator_filter;
                            }
                            iterator_src.load(tb_frag_src);
                            iterator_filter.load(tb_frag_filter);
                        }

                        this->warp_tile_iterator_src_.set_kgroup_index(
                                (warp_mma_k + 1) % Base::kWarpGemmIterations);
                        this->warp_tile_iterator_filter_.set_kgroup_index(
                                (warp_mma_k + 1) % Base::kWarpGemmIterations);

                        this->warp_tile_iterator_src_.load(
                                warp_frag_src[(warp_mma_k + 1) % 2]);
                        this->warp_tile_iterator_filter_.load(
                                warp_frag_filter[(warp_mma_k + 1) % 2]);

                        ++this->warp_tile_iterator_src_;
                        ++this->warp_tile_iterator_filter_;

                        warp_mma(accum, warp_frag_filter[warp_mma_k % 2],
                                 warp_frag_src[warp_mma_k % 2], accum);
                    }
                }
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
