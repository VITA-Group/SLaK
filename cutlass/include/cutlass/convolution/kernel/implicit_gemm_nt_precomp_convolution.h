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
 * include/cutlass/convolution/kernel/implicit_gemm_nt_precomp_convolution.h
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
          conv::Operator ConvOperator,   ///! Convolutional operator (Fprop,
                                         /// Dgrad, Wgrad)
          typename ConvProblemSize_ =
                  Conv2dProblemSize  ///! Convolutional operator on 2D or 3D
                                     /// problem
          >
struct ImplicitGemmNtPrecompConvolution {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using ConvProblemSize = ConvProblemSize_;

    static Operator const kConvolutionalOperator = ConvOperator;

    using ElementSrc = typename Mma::IteratorSrc::Element;
    using LayoutSrc = typename Mma::IteratorSrc::Layout;
    using ElementFilter = typename Mma::IteratorFilter::Element;
    using LayoutFilter = typename Mma::IteratorFilter::Layout;
    using ElementDst = typename EpilogueOutputOp::ElementOutput;
    using LayoutDst = typename Mma::LayoutDst;
    using ElementBias = typename EpilogueOutputOp::ElementBias;
    using LayoutBias = LayoutDst;

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
    using TensorRefFilter = typename Mma::IteratorFilter::TensorRef;
    using TensorRefBias = cutlass::TensorRef<ElementBias, LayoutBias>;
    using TensorRefDst = cutlass::TensorRef<ElementDst, LayoutDst>;

    static int const kStages = Mma::kStages;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    struct ExtraParam {
        typename Mma::IteratorSrc::Params::ExtraParam extra_param_src;
    };

    /// Argument structure
    struct Arguments {
        ConvProblemSize problem_size;
        TensorRefSrc ref_src;
        TensorRefFilter ref_filter;
        TensorRefBias ref_bias;
        TensorRefDst ref_z;
        TensorRefDst ref_dst;
        typename EpilogueOutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_filter;
        ExtraParam extra_param;

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments() {}

        CUTLASS_HOST_DEVICE
        Arguments(ConvProblemSize const& problem_size_)
                : problem_size(problem_size_) {}

        /// Constructs an Arguments structure
        CUTLASS_HOST_DEVICE
        Arguments(ConvProblemSize const& problem_size_,
                  TensorRefSrc const& ref_src_,
                  TensorRefFilter const& ref_filter_,
                  TensorRefBias const& ref_bias_, TensorRefDst const& ref_z_,
                  TensorRefDst const& ref_dst_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  typename Mma::TransformSrc::Params transform_src_ =
                          typename Mma::TransformSrc::Params(),
                  typename Mma::TransformFilter::Params transform_filter_ =
                          typename Mma::TransformFilter::Params(),
                  ExtraParam extra_param_ = {})
                : problem_size(problem_size_),
                  ref_src(ref_src_),
                  ref_filter(ref_filter_),
                  ref_bias(ref_bias_),
                  ref_z(ref_z_),
                  ref_dst(ref_dst_),
                  output_op(epilogue_),
                  transform_src(transform_src_),
                  transform_filter(transform_filter_),
                  extra_param(extra_param_) {}
    };

    /// Parameters structure
    struct Params {
        ConvProblemSize problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        typename Mma::IteratorSrc::Params params_src;
        TensorRefSrc ref_src;
        typename Mma::IteratorFilter::Params params_filter;
        TensorRefFilter ref_filter;
        typename Epilogue::BiasTileIterator::Params params_bias;
        TensorRefBias ref_bias;
        typename Epilogue::OutputTileIterator::Params params_dst;
        TensorRefDst ref_dst;
        typename Epilogue::OutputTileIterator::Params params_z;
        TensorRefDst ref_z;
        typename EpilogueOutputOp::Params output_op;
        typename Mma::TransformSrc::Params transform_src;
        typename Mma::TransformFilter::Params transform_filter;

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
                  params_src(args.ref_src.layout(), args.problem_size,
                             args.extra_param.extra_param_src),
                  ref_src(args.ref_src),
                  params_filter(args.ref_filter.layout()),
                  ref_filter(args.ref_filter),
                  params_bias(args.ref_bias.layout()),
                  ref_bias(args.ref_bias),
                  params_dst(args.ref_dst.layout(), kConvolutionalOperator,
                             args.problem_size),
                  ref_dst(args.ref_dst),
                  params_z(args.ref_z.layout(), kConvolutionalOperator,
                           args.problem_size),
                  ref_z(args.ref_z),
                  output_op(args.output_op),
                  transform_src(args.transform_src),
                  transform_filter(args.transform_filter),
                  workspace(workspace_) {
            if (kConvolutionalOperator == conv::Operator::kDgrad) {
                gemm_problem_size = cutlass::gemm::GemmCoord(
                        problem_size.C,
                        problem_size.N * problem_size.H * problem_size.W,
                        problem_size.K * problem_size.R * problem_size.S);
            } else {  // Fprop
                gemm_problem_size = cutlass::gemm::GemmCoord(
                        problem_size.K,
                        problem_size.N * problem_size.P * problem_size.Q,
                        problem_size.C * problem_size.R * problem_size.S);
            }
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
    ImplicitGemmNtPrecompConvolution() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(
            ConvProblemSize problem_size,
            typename Mma::IteratorSrc::TensorRef ref_src,
            typename Mma::IteratorFilter::TensorRef ref_filter,
            typename Epilogue::BiasTileIterator::TensorRef ref_bias,
            typename Epilogue::OutputTileIterator::TensorRef ref_z,
            typename Epilogue::OutputTileIterator::TensorRef ref_dst) {
        static int const kAlignmentSrc =
                Mma::IteratorSrc::AccessType::kElements;
        static int const kAlignmentFilter =
                Mma::IteratorFilter::AccessType::kElements;
        static int const kAlignmentDst =
                Epilogue::OutputTileIterator::kElementsPerAccess;

        if (!TensorRef_aligned(ref_src, kAlignmentSrc)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_filter, kAlignmentFilter)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_dst, kAlignmentDst)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_z, kAlignmentDst)) {
            return Status::kErrorMisalignedOperand;
        }

        Status status = Mma::IteratorSrc::can_implement(problem_size);

        return status;
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

        cutlass::MatrixCoord tb_offset_filter{threadblock_tile_offset.m(), 0};

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to Src and Filter Tensor operands
        typename Mma::IteratorSrc iterator_src(
                params.params_src, params.ref_src.data(),
                {params.gemm_problem_size.k(), params.gemm_problem_size.n()},
                thread_idx, tb_offset_src);

        typename Mma::IteratorFilter iterator_filter(
                params.params_filter, params.ref_filter.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.k()},
                thread_idx, tb_offset_filter);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentDst accumulators;

        accumulators.clear();

        mma(params.conv_k_iterations, accumulators, iterator_src,
            iterator_filter, accumulators, params.transform_src,
            params.transform_filter);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        //
        // Masked tile iterators constructed from members
        //
        // assume identity swizzle
        cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m(),
                                                threadblock_tile_offset.n());

        // Tile iterator load bias tensor
        typename Epilogue::BiasTileIterator iterator_bias(
                params.params_bias, params.ref_bias.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_dst(
                params.params_dst, params.ref_dst.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_z(
                params.params_z, params.ref_z.data(),
                {params.gemm_problem_size.m(), params.gemm_problem_size.n()},
                thread_idx, threadblock_offset);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_dst, accumulators, iterator_bias,
                 iterator_z);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
