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
 * \file include/cutlass/convolution/kernel/default_conv2d_dgrad.h
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

#include "cutlass/convolution/kernel/implicit_gemm_nt_precomp_convolution.h"
#include "cutlass/convolution/kernel/implicit_gemm_tn_precomp_convolution.h"
#include "cutlass/convolution/kernel/implicit_batched_gemm_tn_dwconv2d.h"

#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt_src_dgrad_precomp.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_tn_dgrad_nhwc_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_fprop_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_dgrad_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm70.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/convolution_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_volta_tensor_op.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementSrc,
          /// Layout type for Src Tensor operand
          typename LayoutSrc,
          /// Element type for Filter Tensor operand
          typename ElementFilter,
          /// Layout type for Filter Tensor operand
          typename LayoutFilter,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
          /// Layout type for Dst and Z Tensor operands
          typename LayoutDst,
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
          /// Access granularity of Filter Tensor in units of elements
          int AlignmentFilter,
          /// whether use special optimization for stride 2
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE,
          /// Implicit Gemm Mode
          ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
          /// Convolution Type
          ConvType ConvolutionType = ConvType::kConvolution>
struct DefaultConvolution2dDgrad;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW4 layout

template <typename LayoutDst,
          /// Element type for Dst and Z Tensor operands
          typename ElementDst,
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
          /// Operation performed by GEMM
          typename MathOperatorTag,
          /// Access granularity of Src Tensor in units of elements
          int kAlignmentSrc,
          /// Access granularity of Filter Tensor in units of elements
          int kAlignmentFilter,
          /// whether use special optimization for stride 2
          SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<4>, int8_t, layout::TensorKxRSCx<4>,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, SpecialOpt> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using LayoutFilter = layout::TensorKxRSCx<4>;
    static const int kStages = Stages;

    using OperatorClass = arch::OpClassSimt;
    /// Define the threadblock-scoped matrix multiply-accumulate
    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 4,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess =
            cutlass::platform::is_same<LayoutDst, layout::TensorNCHW>::value
                    ? 1
                    : 4;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <  /// Layout type for Dst and Z Tensor operand
        typename LayoutDst,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorKxRSCx<Interleaved>, ElementDst, LayoutDst,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorKxRSCx<Interleaved>;

    static int const kInterleavedK = Interleaved;

    using OperatorClass = arch::OpClassTensorOp;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, kInterleavedK,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using Mma = cutlass::conv::threadblock::MmaNtPrecompPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    /// 64 bit store
    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for NCHW32 layout

template <  /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Interleaving quantity
        int Interleaved,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorKxRSCx<Interleaved>, ElementDst, layout::TensorNCxHWx<4>,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorKxRSCx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<4>;

    using OperatorClass = arch::OpClassTensorOp;

    static int const kInterleavedK = Interleaved;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            MathOperatorTag, true>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, kInterleavedK,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    // Define the threadblock-scoped pipelined matrix multiply
    using Mma = cutlass::conv::threadblock::MmaNtPrecompPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    /// 32 bit store
    static int const kEpilogueElementsPerAccess =
            32 / sizeof_bits<ElementDst>::value;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

template <
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Element type for internal accumulation
        typename ElementAccumulator,
        /// Threadblock-level tile size (concept: gemm::GemmShape)
        typename ThreadblockShape,
        /// Warp-level tile size (concept: gemm::GemmShape)
        typename WarpShape,
        /// Instruction-level tile size (concept: gemm::GemmShape)
        typename InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages,
        /// Operation performed by GEMM
        typename MathOperatorTag,
        /// Access granularity of Src Tensor in units of elements
        int kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNHWC, int8_t,
        layout::TensorCKxRSx<kAlignmentFilter>, ElementDst, layout::TensorNHWC,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt,
        ImplicitGemmMode::GEMM_TN> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNHWC;
    using LayoutFilter = layout::TensorCKxRSx<kAlignmentFilter>;
    using LayoutDst = layout::TensorNHWC;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == kAlignmentFilter,
                  "kAlignmentSrc and kAlignmentFilter must be the same");

    static_assert(kAlignmentSrc % 4 == 0,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter % 4 == 0,
                  "Alignment must match thread data map's vector length");

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, false, ImplicitGemmMode::GEMM_TN>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecompNHWC<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc, kAlignmentSrc,
                    SpecialOpt>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::conv::threadblock::Conv2dTileFilterIteratorDgradCKxRSx<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    kAlignmentFilter>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        EpilogueOutputOp::kCount>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel = cutlass::conv::kernel::ImplicitGemmTnPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Depthwise SIMT Convolution
template <
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst,
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
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter>
struct DefaultConvolution2dDgrad<
        float, layout::TensorNCHW, float, layout::TensorNCHW, ElementDst,
        LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, SpecialOptimizeDesc::NONE, ImplicitGemmMode::GEMM_TN,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = float;
    using ElementFilter = float;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutFilter = layout::TensorNCHW;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_TN;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, kGemmMode>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::conv::threadblock::Dwconv2dTileFilterIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess = 1;

    /// Define the epilogue
    using Epilogue =
            typename cutlass::epilogue::threadblock::Dwconv2dEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel =
            cutlass::conv::kernel::ImplicitBatchedGemmTnDepthwiseConvolution<
                    Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Depthwise TensorOp Convolution
template <
        /// Element type for Src Tensor operand
        typename ElementSrc,
        /// Element type for Filter Tensor operand
        typename ElementFilter,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst,
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
        /// Access granularity of Filter Tensor in units of elements
        int kAlignmentFilter>
struct DefaultConvolution2dDgrad<
        ElementSrc, layout::TensorNCHW, ElementFilter, layout::TensorNCHW,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm70, ThreadblockShape, WarpShape, gemm::GemmShape<8, 8, 4>,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, MathOperatorTag,
        kAlignmentSrc, kAlignmentFilter, SpecialOptimizeDesc::NONE,
        ImplicitGemmMode::GEMM_TN, ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutFilter = layout::TensorNCHW;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_TN;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    // Define the MmaCore components
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, kGemmMode>;

    // Define iterators over tiles from the Src Tensor operand
    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            kAlignmentSrc>;

    // Define iterators over tiles from the Filter Tensor operand
    using IteratorFilter =
            cutlass::conv::threadblock::Dwconv2dTileFilterIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    kAlignmentFilter>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    /// Define the epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
            Dwconv2dEpilogueVoltaTensorOp<ThreadblockShape, LayoutDst,
                                          LayoutDst, typename Mma::Operator,
                                          EpilogueOutputOp,
                                          EpilogueOutputOp::kCount>::Epilogue;

    /// Define the kernel-level conv operator.
    using Kernel =
            cutlass::conv::kernel::ImplicitBatchedGemmTnDepthwiseConvolution<
                    Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
