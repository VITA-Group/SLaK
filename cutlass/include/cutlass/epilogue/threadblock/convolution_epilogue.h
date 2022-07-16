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
 * \file include/cutlass/epilogue/threadblock/convolution_epilogue.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator without splitk
template <typename Shape_,   ///< Shape of threadblock tile (concept: GemmShape)
          typename Layout_,  ///< Output layout
          int PartitionsK,   ///< Tile iterator reading and writing output
                             ///< tensors
          typename WarpMmaOperator_,     ///< Warp-level MMA operator (concept:
                                         ///< gemm::warp::MmaTensorOp)
          typename OutputTileIterator_,  ///< Tile iterator reading and writing
                                         ///< output tensors
          typename AccumulatorFragmentIterator_,  ///< Fragment iterator
                                                  ///< selecting accumulators
          typename WarpTileIterator_,    ///< Warp-scoped tile iterator writing
                                         ///< accumulators to SMEM
          typename SharedLoadIterator_,  ///< Threadblock-scoped tile iterator
                                         ///< loading from SMEM
          typename BiasTileIterator_,    ///< Bias Tile iterator
          typename OutputOp_,            ///< Output operator
          typename Padding_,  ///< Padding added to SMEM allocation to avoid
                              ///< bank conflicts (concept: MatrixShape)
          bool UseSyncWarp = false>
class ConvolutionEpilogue
        : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
                              PartitionsK, AccumulatorFragmentIterator_,
                              WarpTileIterator_, Padding_> {
public:
    using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
                              PartitionsK, AccumulatorFragmentIterator_,
                              WarpTileIterator_, Padding_>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using BiasTileIterator = BiasTileIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;

    /// Output layout is always row-major
    using Layout = Layout_;
    using LongIndex = typename Layout::LongIndex;

    /// The complete warp-level accumulator tile
    using AccumulatorTile = typename Base::AccumulatorTile;

    /// Accumulator element
    using ElementAccumulator = typename WarpTileIterator::Element;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Bias element
    using ElementBias = typename BiasTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess =
            OutputTileIterator::kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef =
            typename cutlass::TensorRef<int,
                                        cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Array type used to output
    using OutputAccessType = Array<typename OutputTileIterator::Element,
                                   OutputTileIterator::kElementsPerAccess>;

    /// Array type used by output functor
    using AccumulatorAccessType = Array<typename WarpTileIterator::Element,
                                        OutputTileIterator::kElementsPerAccess>;

    /// Array type used by bias tensor
    using BiasAccessType = Array<typename BiasTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

    /// Number of warps
    using WarpCount = typename Base::WarpCount;

public:
    static_assert(
            SharedLoadIterator::Fragment::kElements ==
                    OutputTileIterator::Fragment::kElements,
            "Mismatch between shared load iterator and output tile iterator.");

    static_assert((!(OutputTileIterator::kElementsPerAccess % 4) ||
                   OutputTileIterator::kElementsPerAccess == 1),
                  "OutputTileIterator::kElementsPerAccess must be 1 or a "
                  "multiple of 4.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

    static_assert(kPartitionsK == 1,
                  "Split K algorithm for convolution not supported.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    BiasTileIterator::Fragment::kElements),
                  "Divisibility");

private:
    /// Loads fragment from shared memory aligned with output tensor
    SharedLoadIterator shared_load_iterator_;

public:
    /// Constructor
    CUTLASS_DEVICE
    ConvolutionEpilogue(
            typename Base::SharedStorage&
                    shared_storage,  ///< Shared storage object
            int thread_idx,          ///< ID of a thread within the threadblock
            int warp_idx,            ///< ID of warp within threadblock
            int lane_idx             ///< Id of thread within warp
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              shared_load_iterator_(shared_storage.reference(), thread_idx) {}

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            BiasTileIterator bias_iterator,  ///< Tile iterator for bias
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        if (output_op.is_bias_needed() && (!output_op.is_source_needed())) {
            compute_with_bias(output_op, destination_iterator, accumulators,
                              bias_iterator);
        } else if (output_op.is_bias_needed() && output_op.is_source_needed()) {
            compute_with_bias_add_source(output_op, destination_iterator,
                                         accumulators, bias_iterator,
                                         source_iterator);
        } else if ((!output_op.is_bias_needed()) &&
                   (!output_op.is_source_needed())) {
            compute_without_bias(output_op, destination_iterator, accumulators);

        } else {
            compute_add_source(output_op, destination_iterator, accumulators,
                               source_iterator);
        }
    }

private:
    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_with_bias(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const& accumulators,
            BiasTileIterator
                    bias_iterator) {  ///< Complete warp-level accumulator tile
        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //
        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Convert and store fragment
            //

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);

            //
            // Load bias fragment
            //

            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    OutputTileIterator::Fragment::kElements /
                    BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_bias(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations]);
            }

            //
            // Store the final result
            //

            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_with_bias_add_source(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            BiasTileIterator bias_iterator,  ///< Tile iterator for bias tensor
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source_iterator.load(source_fragment);
            ++source_iterator;

            //
            // Convert and store fragment
            //
            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);

            //
            // Load bias fragment
            //

            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    OutputTileIterator::Fragment::kElements /
                    BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_bias_source(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations],
                        source_frag_ptr[i]);
            }

            //
            // Store the final result
            //

            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_add_source(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //

            source_iterator.load(source_fragment);
            ++source_iterator;

            //
            // Convert and store fragment
            //
            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_source(
                        compute_frag_ptr[i], source_frag_ptr[i]);
            }

            //
            // Store the final result
            //

            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_without_bias(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators) {  ///< Complete warp-level accumulator tile
        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //
        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Convert and store fragment
            //

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply(compute_frag_ptr[i]);
            }

            //
            // Store the final result
            //

            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }
};

/// Epilogue operator without splitk
template <typename Shape_,   ///< Shape of threadblock tile (concept: GemmShape)
          typename Layout_,  ///< Output layout
          int PartitionsK,   ///< Tile iterator reading and writing output
                             ///< tensors
          typename WarpMmaOperator_,     ///< Warp-level MMA operator (concept:
                                         ///< gemm::warp::MmaTensorOp)
          typename OutputTileIterator_,  ///< Tile iterator reading and writing
                                         ///< output tensors
          typename AccumulatorFragmentIterator_,  ///< Fragment iterator
                                                  ///< selecting accumulators
          typename BiasTileIterator_,             ///< Bias Tile iterator
          typename OutputOp_                      ///< Output operator
          >
class ConvolutionEpilogueWithoutSharedLoad {
public:
    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using BiasTileIterator = BiasTileIterator_;
    using OutputOp = OutputOp_;

    using AccumulatorTile =
            typename AccumulatorFragmentIterator::AccumulatorTile;

    /// Accumulator element
    using ElementAccumulator = typename AccumulatorTile::Element;

    /// Output layout is always row-major
    using Layout = Layout_;
    using LongIndex = typename Layout::LongIndex;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Bias element
    using ElementBias = typename BiasTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess =
            OutputTileIterator::kElementsPerAccess;

    static int const kOutputElements =
            OutputTileIterator::ThreadMap::Iterations::kColumn *
            OutputTileIterator::ThreadMap::Iterations::kRow *
            kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef =
            typename cutlass::TensorRef<int,
                                        cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Array type used to output
    using OutputAccessType = Array<typename OutputTileIterator::Element,
                                   OutputTileIterator::kElementsPerAccess>;

    /// Array type used by output functor
    using AccumulatorAccessType =
            Array<ElementAccumulator, OutputTileIterator::kElementsPerAccess>;

    /// Array type used by bias tensor
    using BiasAccessType = Array<typename BiasTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

    /// Number of warps
    using WarpCount = gemm::GemmShape<Shape::kM / WarpMmaOperator::Shape::kM,
                                      Shape::kN / WarpMmaOperator::Shape::kN,
                                      kPartitionsK>;

public:
    static_assert((!(OutputTileIterator::kElementsPerAccess % 4) ||
                   OutputTileIterator::kElementsPerAccess == 1),
                  "OutputTileIterator::kElementsPerAccess must be 1 or a "
                  "multiple of 4.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

    static_assert(kPartitionsK == 1,
                  "Split K algorithm for convolution not supported.");

    static_assert(!(kOutputElements % BiasTileIterator::Fragment::kElements),
                  "Divisibility");

    /// Shared storage allocation needed by the epilogue
    struct SharedStorage {};

public:
    /// Constructor
    CUTLASS_DEVICE
    ConvolutionEpilogueWithoutSharedLoad(
            SharedStorage& shared_storage,  ///< Shared storage object
            int thread_idx,  ///< ID of a thread within the threadblock
            int warp_idx,    ///< ID of warp within threadblock
            int lane_idx     ///< Id of thread within warp
    ) {}

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            BiasTileIterator bias_iterator,  ///< Tile iterator for bias
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        if (output_op.is_bias_needed() && (!output_op.is_source_needed())) {
            compute_with_bias(output_op, destination_iterator, accumulators,
                              bias_iterator);
        } else if (output_op.is_bias_needed() && output_op.is_source_needed()) {
            compute_with_bias_add_source(output_op, destination_iterator,
                                         accumulators, bias_iterator,
                                         source_iterator);
        } else if ((!output_op.is_bias_needed()) &&
                   (!output_op.is_source_needed())) {
            compute_without_bias(output_op, destination_iterator, accumulators);

        } else {
            compute_add_source(output_op, destination_iterator, accumulators,
                               source_iterator);
        }
    }

private:
    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_with_bias(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const& accumulators,
            BiasTileIterator
                    bias_iterator) {  ///< Complete warp-level accumulator tile
        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Convert and store fragment
            //

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            //
            // Load bias fragment
            //

            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    kOutputElements / BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_bias(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations]);
            }

            //
            // Store the final result
            //
            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_with_bias_add_source(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            BiasTileIterator bias_iterator,  ///< Tile iterator for bias tensor
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //
            source_iterator.set_iteration_index(iter);
            source_iterator.load(source_fragment);
            ++source_iterator;

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            //
            // Load bias fragment
            //

            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    kOutputElements / BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_bias_source(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations],
                        source_frag_ptr[i]);
            }

            //
            // Store the final result
            //
            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_add_source(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators,  ///< Complete warp-level accumulator tile
            OutputTileIterator
                    source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                        ///< (in units of threadblock tiles)
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Load the source
            //
            source_iterator.set_iteration_index(iter);
            source_iterator.load(source_fragment);
            ++source_iterator;

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply_add_source(
                        compute_frag_ptr[i], source_frag_ptr[i]);
            }

            //
            // Store the final result
            //
            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_without_bias(
            OutputOp const& output_op,  ///< Output operator
            OutputTileIterator
                    destination_iterator,  ///< Tile iterator for destination
            AccumulatorTile const&
                    accumulators) {  ///< Complete warp-level accumulator tile
        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            //
            // Convert and store fragment
            //

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                // Call the output operator
                output_frag_ptr[i] = output_op.apply(compute_frag_ptr[i]);
            }

            //
            // Store the final result
            //
            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
