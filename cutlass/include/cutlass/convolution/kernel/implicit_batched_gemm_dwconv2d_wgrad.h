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
 * include/cutlass/convolution/kernel/implicit_batched_gemm_dwconv2d_wgrad.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief Template for a pipelined Implicit GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename ConvProblemSize_ =
                  Conv2dProblemSize  ///! Convolutional operator on 2D or 3D
                                     /// problem
          >
struct ImplicitBatchedGemmDepthwiseConvolution2dWgrad {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using ConvProblemSize = ConvProblemSize_;

    static Operator const kConvolutionalOperator = Operator::kWgrad;

    using ElementSrc = typename Mma::IteratorSrc::Element;
    using LayoutSrc = typename Mma::IteratorSrc::Layout;
    using ElementDiff = typename Mma::IteratorFilter::Element;
    using LayoutDiff = typename Mma::IteratorFilter::Layout;
    using ElementGrad = typename EpilogueOutputOp::ElementOutput;
    using LayoutGrad = typename Mma::LayoutDst;

    using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
    using ElementCompute = typename EpilogueOutputOp::ElementCompute;

    using WarpMmaOperator = typename Mma::Policy::Operator;

    using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
    using MathOperator = typename ArchMmaOperator::Operator;

    using OperatorClass = typename WarpMmaOperator::OperatorClass;
    using ArchTag = typename WarpMmaOperator::ArchTag;

    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename WarpMmaOperator::Shape;
    using InstructionShape = typename ArchMmaOperator::Shape;

    using TensorRefSrc = typename Mma::IteratorSrc::TensorRef;
    using TensorRefDiff = typename Mma::IteratorFilter::TensorRef;
    using TensorRefGrad = cutlass::TensorRef<ElementGrad, LayoutGrad>;

    static int const kStages = Mma::kStages;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    /// Argument structure
    struct Arguments {
        ConvProblemSize problem_size;
        TensorRefSrc ref_src;
        TensorRefDiff ref_diff;
        TensorRefGrad ref_grad;
        typename EpilogueOutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_diff;

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments() {}

        CUTLASS_HOST_DEVICE
        Arguments(ConvProblemSize const& problem_size_)
                : problem_size(problem_size_) {}

        /// Constructs an Arguments structure
        CUTLASS_HOST_DEVICE
        Arguments(ConvProblemSize const& problem_size_,
                  TensorRefSrc const& ref_src_, TensorRefDiff const& ref_diff_,
                  TensorRefGrad const& ref_grad_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  typename Mma::TransformSrc::Params transform_src_ =
                          typename Mma::TransformSrc::Params(),
                  typename Mma::TransformFilter::Params transform_diff_ =
                          typename Mma::TransformFilter::Params())
                : problem_size(problem_size_),
                  ref_src(ref_src_),
                  ref_diff(ref_diff_),
                  ref_grad(ref_grad_),
                  output_op(epilogue_),
                  transform_src(transform_src_),
                  transform_diff(transform_diff_) {}
    };

    /// Parameters structure
    struct Params {
        ConvProblemSize problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        typename Mma::IteratorSrc::Params params_src;
        TensorRefSrc ref_src;
        typename Mma::IteratorFilter::Params params_diff;
        TensorRefDiff ref_diff;
        typename Epilogue::OutputTileIterator::Params params_grad;
        TensorRefGrad ref_grad;
        typename EpilogueOutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_diff;

        cutlass::gemm::GemmCoord gemm_problem_size;
        int* workspace;
        int conv_k_iterations;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params() : conv_k_iterations(0) {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args,
               cutlass::gemm::GemmCoord const& grid_tiled_shape_,
               int* workspace_ = nullptr)
                : problem_size(args.problem_size),
                  grid_tiled_shape(grid_tiled_shape_),
                  params_src(args.ref_src.layout(), args.problem_size),
                  ref_src(args.ref_src),
                  params_diff(args.ref_diff.layout(), args.problem_size),
                  ref_diff(args.ref_diff),
                  params_grad(args.ref_grad.layout(), args.problem_size),
                  ref_grad(args.ref_grad),
                  output_op(args.output_op),
                  transform_src(args.transform_src),
                  transform_diff(args.transform_diff),
                  workspace(workspace_) {
            gemm_problem_size = cutlass::gemm::GemmCoord(
                    problem_size.P * problem_size.Q,
                    problem_size.H * problem_size.W, problem_size.N);
            conv_k_iterations = (gemm_problem_size.k() + Mma::Shape::kK - 1) /
                                Mma::Shape::kK;
        }
    };

    /// Shared memory storage structure
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    ImplicitBatchedGemmDepthwiseConvolution2dWgrad() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(
            ConvProblemSize problem_size,
            typename Mma::IteratorSrc::TensorRef ref_src,
            typename Mma::IteratorFilter::TensorRef ref_diff,
            typename Epilogue::OutputTileIterator::TensorRef ref_grad) {
        static int const kAlignmentSrc =
                Mma::IteratorSrc::AccessType::kElements;
        static int const kAlignmentDiff =
                Mma::IteratorFilter::AccessType::kElements;
        static int const kAlignmentGrad =
                Epilogue::OutputTileIterator::kElementsPerAccess;

        if (!TensorRef_aligned(ref_src, kAlignmentSrc)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_diff, kAlignmentDiff)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_grad, kAlignmentGrad)) {
            return Status::kErrorMisalignedOperand;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(ConvProblemSize problem_size) { return 0; }

    /// Executes one Convolution
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
                threadblock_swizzle.template get_tile_offset<Mma::Shape>();

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_src{0, threadblock_tile_offset.n()};

        cutlass::MatrixCoord tb_offset_diff{0, threadblock_tile_offset.m()};

        cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m(),
                                                threadblock_tile_offset.n());

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Broadcast the warp_id computed by lane 0 to ensure dependent
        // code is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        // Construct tile iterator writing to gradient tensor.
        typename Epilogue::OutputTileIterator iterator_grad(
                params.params_grad, params.ref_grad.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, warp_idx, lane_idx, threadblock_offset);

        // Early stop for total threadblock to reduce the compution
        if (iterator_grad.early_stop(threadblock_offset))
            return;

        // Construct iterator to Src Tensor operand
        typename Mma::IteratorSrc iterator_src(
                params.params_src, params.ref_src.data(),
                {params.gemm_problem_size.k(), params.gemm_problem_size.n()},
                thread_idx, tb_offset_src);

        // Construct iterator to Filter Tensor operand
        typename Mma::IteratorFilter iterator_diff(
                params.params_diff, params.ref_diff.data(),
                {params.gemm_problem_size.k(), params.gemm_problem_size.m()},
                thread_idx, tb_offset_diff);

        iterator_src.add_coord_offset({0, 0, 0, threadblock_tile_offset.k()});
        iterator_diff.add_coord_offset({0, 0, 0, threadblock_tile_offset.k()});
        iterator_grad.add_coord_offset({threadblock_tile_offset.k(), 0, 0, 0});

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentDst accumulators;
        accumulators.clear();

        mma(params.conv_k_iterations, accumulators, iterator_src, iterator_diff,
            accumulators, params.transform_src, params.transform_diff);

        //
        // Epilogue
        //
        EpilogueOutputOp output_op(params.output_op);

        /// Construct threadblock-scoped epilogue to write back to output tensor
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        // Execute the epilogue operator to update the gradient
        // tensor.
        epilogue(output_op, iterator_grad, accumulators);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
