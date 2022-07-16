/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
modification, are permitted
 * provided that the following conditions are met:
namespace conv {
 *     * Redistributions of source code must retain the above copyright notice,
this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
notice, this list of
 *       conditions and the following disclaimer in the documentation and/or
other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
contributors may be used
 *       to endorse or promote products derived from this software without
specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA
CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file include/cutlass/convolution/kernel/default_conv2d_wgrad.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief
    Default kernel-level implicit GEMM conv definitions combine
   threadblock-scoped matrix multiply-add with the appropriate
   threadblock-scoped epilogue.
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/convolution/kernel/implicit_batched_gemm_dwconv2d_wgrad.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm70.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"

#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_volta_tensor_op.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Element type for Src Tensor operand
        typename ElementSrc,
        /// Layout type for Src Tensor operand
        typename LayoutSrc,
        /// Element type for Diff Tensor operand
        typename ElementDiff,
        /// Layout type for Diff Tensor operand
        typename LayoutDiff,
        /// Element type for Grad Tensor operands
        typename ElementGrad,
        /// Layout type for Grad Tensor operands
        typename LayoutGrad,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// MathOperatorTag class tag
        typename OperatorClass,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by conv
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc,
        /// Access granularity of Diff Tensor in units of elements
        int AlignmentDiff,
        /// Implicit Gemm Mode
        ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kConvolution>
struct DefaultConvolution2dWgrad;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Depthwise SIMT Convolution
template <
        /// Element type for Grad Tensor operands
        typename ElementGrad,
        /// Layout type for Grad Tensor operands
        typename LayoutGrad,
        /// Tag indicating architecture to tune for
        typename ArchTag,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by conv
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Diff Tensor in units of elements
        int kAlignmentDiff>
struct DefaultConvolution2dWgrad<
        float, layout::TensorNCHW, float, layout::TensorNCHW, ElementGrad,
        LayoutGrad, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentDiff, ImplicitGemmMode::GEMM_NT,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = float;
    using ElementDiff = float;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutDiff = layout::TensorNCHW;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_NT;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementDiff, LayoutDiff, kAlignmentDiff,
            ElementAccumulator, LayoutGrad, OperatorClass, Stages,
            MathOperatorTag, true, kGemmMode>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess, 1>;

    // Define iterators over tiles from the Diff Tensor operand
    using IteratorDiff = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementDiff, LayoutDiff, typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess, 1>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// Define the epilogue
    using Epilogue =
            cutlass::epilogue::threadblock::Dwconv2dWgradDirectEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, ElementGrad,
                    LayoutGrad, EpilogueOutputOp>;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::
            ImplicitBatchedGemmDepthwiseConvolution2dWgrad<Mma, Epilogue,
                                                           ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Depthwise Tensor Op Convolution
template <
        /// Element type for Src Tensor operand
        typename ElementSrc,
        /// Element type for Diff Tensor operand
        typename ElementDiff,
        /// Element type for Grad Tensor operands
        typename ElementGrad,
        /// Layout type for Grad Tensor operands
        typename LayoutGrad,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by conv
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Diff Tensor in units of elements
        int kAlignmentDiff>
struct DefaultConvolution2dWgrad<
        ElementSrc, layout::TensorNCHW, ElementDiff, layout::TensorNCHW,
        ElementGrad, LayoutGrad, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm70, ThreadblockShape, WarpShape, gemm::GemmShape<8, 8, 4>,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, MathOperatorTag,
        kAlignmentSrc, kAlignmentDiff, ImplicitGemmMode::GEMM_NT,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutDiff = layout::TensorNCHW;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_NT;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementDiff, LayoutDiff, kAlignmentDiff,
            ElementAccumulator, LayoutGrad, OperatorClass, Stages,
            MathOperatorTag, true, kGemmMode>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            kAlignmentSrc, 1>;

    // Define iterators over tiles from the Diff Tensor operand
    using IteratorDiff = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementDiff, LayoutDiff, typename MmaCore::IteratorThreadMapFilter,
            kAlignmentDiff, 1>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// Define the epilogue
    using Epilogue = cutlass::epilogue::threadblock::
            Dwconv2dWgradDirectEpilogueVoltaTensorOp<
                    ThreadblockShape, typename Mma::Operator, ElementGrad,
                    LayoutGrad, EpilogueOutputOp>;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::
            ImplicitBatchedGemmDepthwiseConvolution2dWgrad<Mma, Epilogue,
                                                           ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
