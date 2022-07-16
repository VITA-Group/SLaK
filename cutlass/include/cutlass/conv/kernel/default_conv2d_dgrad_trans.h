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
    Default kernel-level implicit GEMM convolution definitions combine
   threadblock-scoped matrix multiply-add with the appropriate
   threadblock-scoped epilogue.
*/

/**
 * \file include/cutlass/conv/kernel/default_conv2d_dgrad_trans.h
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
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"

#include "cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_trans_analytic.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/// Defines a kernel for Conv2dDgrad specialzation for Analytic
/// IteratorAlgorithm Dgrad Strided
// and 2 stage pipeline.
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator, typename ArchTag,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, typename MathOperatorTag>
struct DefaultConv2dDgrad<
        ElementA, layout::TensorNHWC, ElementB, layout::TensorCHWN, ElementC,
        layout::TensorNHWC, ElementAccumulator, arch::OpClassTensorOp, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
        ThreadblockSwizzle, 2, MathOperatorTag, IteratorAlgorithm::kAnalytic,
        StrideSupport::kStrided> {
    using LayoutA = layout::TensorNHWC;
    using LayoutB = layout::TensorCHWN;
    using LayoutC = layout::TensorNHWC;

    // Define the core components from GEMM
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA,
            layout::RowMajor, ElementB, layout::ColumnMajor, ElementAccumulator,
            layout::RowMajor, arch::OpClassTensorOp, 2, MathOperatorTag>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kM,
                                                 ThreadblockShape::kK>,
                            ElementA, LayoutA, ThreadMapA,
                            StrideSupport::kStrided> >;

    using SmemIteratorA = typename MmaCore::SmemIteratorA;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradFilterTileAccessIteratorTransAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kK,
                                                 ThreadblockShape::kN>,
                            ElementB, LayoutB, ThreadMapB> >;

    using SmemIteratorB = typename MmaCore::SmemIteratorB;

    // Warp-level GEMM components
    using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
    using MmaPolicy = typename MmaCore::MmaPolicy;

    // Define the Mma
    using Mma = threadblock::ImplicitGemmPipelined<
            ThreadblockShape, IteratorA, SmemIteratorA, IteratorB,
            SmemIteratorB, ElementC, LayoutC, MmaPolicy>;

    // Define the epilogue
    using Epilogue =
            typename detail::DefaultConvEpilogue<ArchTag, ThreadblockShape,
                                                 WarpMmaTensorOp, 1,
                                                 EpilogueOutputOp>::Epilogue;

    // Define the kernel
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator, typename ArchTag,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, typename MathOperatorTag,
          int InterleavedK>
struct DefaultConv2dDgrad<ElementA, layout::TensorNCxHWx<InterleavedK>,
                          ElementB, layout::TensorKxRSCx<InterleavedK>,
                          ElementC, layout::TensorNCxHWx<InterleavedK>,
                          ElementAccumulator, arch::OpClassTensorOp, ArchTag,
                          ThreadblockShape, WarpShape, InstructionShape,
                          EpilogueOutputOp, ThreadblockSwizzle, 2,
                          MathOperatorTag, IteratorAlgorithm::kAnalytic,
                          StrideSupport::kStrided> {
    using LayoutA = layout::TensorNCxHWx<InterleavedK>;
    using LayoutB = layout::TensorKxRSCx<InterleavedK>;
    using LayoutC = layout::TensorNCxHWx<InterleavedK>;

    // Define the core components from GEMM
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA,
            layout::ColumnMajorInterleaved<InterleavedK>, ElementB,
            layout::RowMajorInterleaved<InterleavedK>, ElementAccumulator,
            LayoutC, arch::OpClassTensorOp, 2, MathOperatorTag, true>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::SmemThreadMapA;
    using IteratorA = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kM,
                                                 ThreadblockShape::kK>,
                            ElementA, LayoutA, ThreadMapA,
                            StrideSupport::kStrided> >;

    using SmemIteratorA = typename MmaCore::SmemIteratorA;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::SmemThreadMapB;
    using IteratorB = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradFilterTileAccessIteratorTransAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kK,
                                                 ThreadblockShape::kN>,
                            ElementB, LayoutB, ThreadMapB> >;

    using SmemIteratorB = typename MmaCore::SmemIteratorB;

    // Warp-level GEMM components
    using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
    using MmaPolicy = typename MmaCore::MmaPolicy;

    // Define the Mma
    using Mma = threadblock::ImplicitGemmPipelined<
            ThreadblockShape, IteratorA, SmemIteratorA, IteratorB,
            SmemIteratorB, ElementC, LayoutC, MmaPolicy>;

    // Define the epilogue
    using Epilogue =
            typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
                    ThreadblockShape, WarpMmaTensorOp, 1, EpilogueOutputOp,
                    EpilogueOutputOp::kCount, InterleavedK>::Epilogue;

    // Define the kernel
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator, typename ArchTag,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, typename MathOperatorTag,
          int InterleavedK>
struct DefaultConv2dDgrad<
        ElementA, layout::TensorNCxHWx<InterleavedK>, ElementB,
        layout::TensorKxRSCx<InterleavedK>, ElementC,
        layout::TensorNCxHWx<InterleavedK>, ElementAccumulator,
        arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2,
        MathOperatorTag, IteratorAlgorithm::kAnalytic, StrideSupport::kUnity> {
    using LayoutA = layout::TensorNCxHWx<InterleavedK>;
    using LayoutB = layout::TensorKxRSCx<InterleavedK>;
    using LayoutC = layout::TensorNCxHWx<InterleavedK>;

    // Define the core components from GEMM
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA,
            layout::ColumnMajorInterleaved<InterleavedK>, ElementB,
            layout::RowMajorInterleaved<InterleavedK>, ElementAccumulator,
            LayoutC, arch::OpClassTensorOp, 2, MathOperatorTag, true>;

    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::SmemThreadMapA;
    using IteratorA = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradOutputGradientTileAccessIteratorAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kM,
                                                 ThreadblockShape::kK>,
                            ElementA, LayoutA, ThreadMapA,
                            StrideSupport::kUnity> >;

    using SmemIteratorA = typename MmaCore::SmemIteratorA;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::SmemThreadMapB;
    using IteratorB = cutlass::conv::threadblock::TileIterator<
            cutlass::conv::threadblock::
                    Conv2dDgradFilterTileAccessIteratorTransAnalytic<
                            cutlass::MatrixShape<ThreadblockShape::kK,
                                                 ThreadblockShape::kN>,
                            ElementB, LayoutB, ThreadMapB> >;

    using SmemIteratorB = typename MmaCore::SmemIteratorB;

    // Warp-level GEMM components
    using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
    using MmaPolicy = typename MmaCore::MmaPolicy;

    // Define the Mma
    using Mma = threadblock::ImplicitGemmPipelined<
            ThreadblockShape, IteratorA, SmemIteratorA, IteratorB,
            SmemIteratorB, ElementC, LayoutC, MmaPolicy>;

    // Define the epilogue
    using Epilogue =
            typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
                    ThreadblockShape, WarpMmaTensorOp, 1, EpilogueOutputOp,
                    EpilogueOutputOp::kCount, InterleavedK>::Epilogue;

    // Define the kernel
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
