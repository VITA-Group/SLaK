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
 * \file include/cutlass/convolution/device/convolution.h
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
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/conv/convolution.h"
#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/convolution/device/default_convolution_configuration.h"
#include "cutlass/convolution/kernel/default_conv2d_wgrad.h"
#include "cutlass/convolution/kernel/default_conv2d_dgrad.h"
#include "cutlass/convolution/kernel/default_conv2d_fprop.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Convolution device-level operator.
 */
template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Element type for Filter Tensor operand
        typename ElementFilter_,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Element type for Bias Tensor operands
        typename ElementBias_,
        /// Layout type for Bias Tensor operands
        typename LayoutBias_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kConvolution,
        /// Operator class tag
        typename OperatorClass_ = arch::OpClassSimt,
        /// Tag indicating architecture to tune for
        typename ArchTag_ = arch::Sm61,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle_ =
                typename threadblock::ConvolutionFpropCxRSKxThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kStages,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentFilter = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentFilter,
        /// whether use special optimization for convolution 1x1
        cutlass::conv::SpecialOptimizeDesc SpecialOpt =
                cutlass::conv::SpecialOptimizeDesc::NONE,
        /// Operation performed by Convolution
        typename Operator_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::Operator,
        /// Implicit Gemm Mode
        cutlass::conv::ImplicitGemmMode GemmMode =
                cutlass::conv::ImplicitGemmMode::GEMM_NT,
        /// use reorder filter K to avoid shared load
        bool WithoutSharedLoad = false>
class Convolution {
public:
    using ElementSrc = ElementSrc_;
    using LayoutSrc = LayoutSrc_;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = LayoutFilter_;
    using ElementBias = ElementBias_;
    using LayoutBias = LayoutBias_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static const ConvType kConvolutionType = ConvolutionType;
    static int const kStages = Stages;
    static int const kAlignmentSrc = AlignmentSrc;
    static int const kAlignmentFilter = AlignmentFilter;
    static int const kAlignmentDst = EpilogueOutputOp::kCount;
    static cutlass::conv::SpecialOptimizeDesc const kSpecialOpt = SpecialOpt;
    static cutlass::conv::ImplicitGemmMode const kGemmMode = GemmMode;
    static bool const kWithoutSharedLoad = WithoutSharedLoad;

    using ConvolutionKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dFprop<
                    ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                    ElementDst, LayoutDst, ElementAccumulator, OperatorClass,
                    ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                    EpilogueOutputOp, ThreadblockSwizzle, kStages, Operator,
                    kAlignmentSrc, kAlignmentFilter, kSpecialOpt, kGemmMode,
                    WithoutSharedLoad, kConvolutionType>::Kernel;
    using TensorRefSrc = typename ConvolutionKernel::TensorRefSrc;
    using TensorRefFilter = typename ConvolutionKernel::TensorRefFilter;
    using TensorRefBias = typename ConvolutionKernel::TensorRefBias;
    using TensorRefDst = typename ConvolutionKernel::TensorRefDst;

    using Arguments = typename ConvolutionKernel::Arguments;
    using ExtraParam = typename ConvolutionKernel::ExtraParam;

    using ConvolutionParameter = typename ConvolutionKernel::ConvProblemSize;

    static cutlass::conv::Operator const kConvolutionalOperator =
            ConvolutionKernel::kConvolutionalOperator;

private:
    /// Kernel parameters object
    typename ConvolutionKernel::Params params_;

public:
    /// Constructs the GEMM.
    Convolution() {}

    /// Determines whether the GEMM can execute the given problem.
    /// Determines whether the Implicit GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        Status status = ConvolutionKernel::can_implement(
                args.problem_size, args.ref_src, args.ref_filter, args.ref_bias,
                args.ref_z, args.ref_dst);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        return ConvolutionKernel::get_workspace_size(args.problem_size);
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        // Determine grid shape
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord grid_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.problem_size,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK});

        // Initialize the Params structure
        params_ = typename ConvolutionKernel::Params{
                args, grid_shape, static_cast<int*>(workspace)};

        return Status::kSuccess;
    }

    /// Initializes GEMM state from arguments.
    Status update(Arguments const& args, void* workspace = nullptr) {
        // update the params structure from the arguments
        params_.ref_src.reset(args.ref_src.data());
        params_.ref_filter.reset(args.ref_filter.data());
        params_.ref_bias.reset(args.ref_bias.data());
        params_.ref_z.reset(args.ref_z.data());
        params_.ref_dst.reset(args.ref_dst.data());
        params_.output_op = args.output_op;
        params_.transform_src = args.transform_src;
        params_.transform_filter = args.transform_filter;
        params_.extra_param = args.extra_param;
        params_.workspace = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(ConvolutionKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvolutionKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<ConvolutionKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        result = cudaGetLastError();

        return result == cudaSuccess ? Status::kSuccess
                                     : Status::kErrorInternal;
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};

////////////////////////////////////////////////////////////////////////////////
/*! Deconvolution device-level operator.
 */
template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Element type for Filter Tensor operand
        typename ElementFilter_,
        /// Layout type for Filter Tensor operand
        typename LayoutFilter_,
        /// Element type for Dst and Z Tensor operands
        typename ElementDst_,
        /// Layout type for Dst and Z Tensor operands
        typename LayoutDst_,
        /// Element type for Bias Tensor operands
        typename ElementBias_,
        /// Layout type for Bias Tensor operands
        typename LayoutBias_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kConvolution,
        /// Operator class tag
        typename OperatorClass_ = arch::OpClassSimt,
        /// Tag indicating architecture to tune for
        typename ArchTag_ = arch::Sm61,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle_ =
                typename threadblock::ConvolutionDgradCxRSKxThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kStages,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentFilter = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::kAlignmentFilter,
        /// whether use special optimization for deconv stride 2
        cutlass::conv::SpecialOptimizeDesc SpecialOpt =
                cutlass::conv::SpecialOptimizeDesc::NONE,
        /// Operation performed by Convolution
        typename Operator_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementFilter_,
                ElementDst_, ElementAccumulator_>::Operator,
        /// Implicit Gemm Mode
        cutlass::conv::ImplicitGemmMode GemmMode =
                cutlass::conv::ImplicitGemmMode::GEMM_NT>
class Deconvolution {
public:
    using ElementSrc = ElementSrc_;
    using LayoutSrc = LayoutSrc_;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = LayoutFilter_;
    using ElementBias = ElementBias_;
    using LayoutBias = LayoutBias_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using TensorRefZ = TensorRef<ElementDst, LayoutDst>;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static const ConvType kConvolutionType = ConvolutionType;
    static int const kStages = Stages;
    static int const kAlignmentSrc = AlignmentSrc;
    static int const kAlignmentFilter = AlignmentFilter;
    static int const kAlignmentDst = EpilogueOutputOp::kCount;
    static cutlass::conv::SpecialOptimizeDesc const kSpecialOpt = SpecialOpt;
    static cutlass::conv::ImplicitGemmMode const kGemmMode = GemmMode;
    static bool const kWithoutSharedLoad = false;

    using ConvolutionKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                    ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                    ElementDst, LayoutDst, ElementAccumulator, OperatorClass,
                    ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                    EpilogueOutputOp, ThreadblockSwizzle, kStages, Operator,
                    kAlignmentSrc, kAlignmentFilter, kSpecialOpt, kGemmMode,
                    kConvolutionType>::Kernel;

    using TensorRefSrc = typename ConvolutionKernel::TensorRefSrc;
    using TensorRefFilter = typename ConvolutionKernel::TensorRefFilter;
    using TensorRefBias = typename ConvolutionKernel::TensorRefBias;
    using TensorRefDst = typename ConvolutionKernel::TensorRefDst;

    using Arguments = typename ConvolutionKernel::Arguments;
    using ExtraParam = typename ConvolutionKernel::ExtraParam;

    using ConvolutionParameter = typename ConvolutionKernel::ConvProblemSize;

    static cutlass::conv::Operator const kConvolutionalOperator =
            ConvolutionKernel::kConvolutionalOperator;

private:
    /// Kernel parameters object
    typename ConvolutionKernel::Params params_;

public:
    /// Constructs the GEMM.
    Deconvolution() {}

    /// Determines whether the GEMM can execute the given problem.
    /// Determines whether the Implicit GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        Status status = ConvolutionKernel::can_implement(
                args.problem_size, args.ref_src, args.ref_filter, args.ref_bias,
                args.ref_z, args.ref_dst);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        return ConvolutionKernel::get_workspace_size(args.problem_size);
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        // Determine grid shape
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord grid_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.problem_size,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK});

        // Initialize the Params structure
        params_ = typename ConvolutionKernel::Params{
                args, grid_shape, static_cast<int*>(workspace)};

        return Status::kSuccess;
    }

    /// Initializes GEMM state from arguments.
    Status update(Arguments const& args, void* workspace = nullptr) {
        // update the params structure from the arguments
        params_.ref_src.reset(args.ref_src.data());
        params_.ref_filter.reset(args.ref_filter.data());
        params_.ref_bias.reset(args.ref_bias.data());
        params_.ref_z.reset(args.ref_z.data());
        params_.ref_dst.reset(args.ref_dst.data());
        params_.output_op = args.output_op;
        params_.transform_src = args.transform_src;
        params_.transform_filter = args.transform_filter;
        params_.extra_param = args.extra_param;
        params_.workspace = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(ConvolutionKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvolutionKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<ConvolutionKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        result = cudaGetLastError();

        return result == cudaSuccess ? Status::kSuccess
                                     : Status::kErrorInternal;
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};

////////////////////////////////////////////////////////////////////////////////
/*! ConvolutionBackwardFilter device-level operator.
 *  This operator is only used for depthwise convolution now
 */
template <
        /// Element type for Src Tensor operand
        typename ElementSrc_,
        /// Layout type for Src Tensor operand
        typename LayoutSrc_,
        /// Element type for Diff Tensor operand
        typename ElementDiff_,
        /// Layout type for Diff Tensor operand
        typename LayoutDiff_,
        /// Element type for Grad Tensor operands
        typename ElementGrad_,
        /// Layout type for Grad Tensor operands
        typename LayoutGrad_,
        /// Element type for internal accumulation
        typename ElementAccumulator_,
        /// Convolution Type
        ConvType ConvolutionType = ConvType::kDepthwiseConvolution,
        /// Operator class tag
        typename OperatorClass_ = arch::OpClassSimt,
        /// Tag indicating architecture to tune for
        typename ArchTag_ = arch::Sm61,
        /// Threadblock-level tile size (concept: GemmShape)
        typename ThreadblockShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::ThreadblockShape,
        /// Warp-level tile size (concept: GemmShape)
        typename WarpShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::WarpShape,
        /// Instruction-level tile size (concept: GemmShape)
        typename InstructionShape_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::InstructionShape,
        /// Epilogue output operator
        typename EpilogueOutputOp_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::EpilogueOutputOp,
        /// Threadblock-level swizzling operator
        typename ThreadblockSwizzle_ = typename threadblock::
                DepthwiseConvolutionWgradThreadblockSwizzle,
        /// Number of stages used in the pipelined mainloop
        int Stages = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::kStages,
        /// Access granularity of Src Tensor in units of elements
        int AlignmentSrc = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::kAlignmentSrc,
        /// Access granularity of Filter Tensor in units of elements
        int AlignmentDiff = DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::kAlignmentFilter,
        /// whether use special optimization for deconv stride 2
        cutlass::conv::SpecialOptimizeDesc SpecialOpt =
                cutlass::conv::SpecialOptimizeDesc::NONE,
        /// Operation performed by Convolution
        typename Operator_ = typename DefaultConvolutionConfiguration<
                OperatorClass_, ArchTag_, ElementSrc_, ElementDiff_,
                ElementGrad_, ElementAccumulator_>::Operator,
        /// Implicit Gemm Mode
        cutlass::conv::ImplicitGemmMode GemmMode =
                cutlass::conv::ImplicitGemmMode::GEMM_NT>
class ConvolutionBackwardFilter {
public:
    using ElementSrc = ElementSrc_;
    using LayoutSrc = LayoutSrc_;
    using ElementDiff = ElementDiff_;
    using LayoutDiff = LayoutDiff_;
    using ElementGrad = ElementGrad_;
    using LayoutGrad = LayoutGrad_;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static const ConvType kConvolutionType = ConvolutionType;
    static int const kStages = Stages;
    static int const kAlignmentSrc = AlignmentSrc;
    static int const kAlignmentDiff = AlignmentDiff;
    static int const kAlignmentGrad = EpilogueOutputOp::kCount;
    static cutlass::conv::SpecialOptimizeDesc const kSpecialOpt = SpecialOpt;
    static cutlass::conv::ImplicitGemmMode const kGemmMode = GemmMode;

    /// SpecialOptimizeDesc is not used in the backward filter kernel now
    using ConvolutionKernel =
            typename cutlass::conv::kernel::DefaultConvolution2dWgrad<
                    ElementSrc, LayoutSrc, ElementDiff, LayoutDiff, ElementGrad,
                    LayoutGrad, ElementAccumulator, OperatorClass, ArchTag,
                    ThreadblockShape, WarpShape, InstructionShape,
                    EpilogueOutputOp, ThreadblockSwizzle, kStages, Operator,
                    kAlignmentSrc, kAlignmentDiff, kGemmMode,
                    kConvolutionType>::Kernel;

    using TensorRefSrc = typename ConvolutionKernel::TensorRefSrc;
    using TensorRefDiff = typename ConvolutionKernel::TensorRefDiff;
    using TensorRefGrad = typename ConvolutionKernel::TensorRefGrad;

    using Arguments = typename ConvolutionKernel::Arguments;

    using ConvolutionParameter = typename ConvolutionKernel::ConvProblemSize;

    static cutlass::conv::Operator const kConvolutionalOperator =
            ConvolutionKernel::kConvolutionalOperator;

private:
    /// Kernel parameters object
    typename ConvolutionKernel::Params params_;

public:
    /// Constructs the GEMM.
    ConvolutionBackwardFilter() {}

    /// Determines whether the GEMM can execute the given problem.
    /// Determines whether the Implicit GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        Status status = ConvolutionKernel::can_implement(
                args.problem_size, args.ref_src, args.ref_diff, args.ref_grad);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        return ConvolutionKernel::get_workspace_size(args.problem_size);
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        // Determine grid shape
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord grid_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.problem_size,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK});

        // Initialize the Params structure
        params_ = typename ConvolutionKernel::Params{
                args, grid_shape, static_cast<int*>(workspace)};

        return Status::kSuccess;
    }

    /// Initializes GEMM state from arguments.
    Status update(Arguments const& args, void* workspace = nullptr) {
        // update the params structure from the arguments
        params_.ref_src.reset(args.ref_src.data());
        params_.ref_diff.reset(args.ref_diff.data());
        params_.ref_grad.reset(args.ref_grad.data());
        params_.output_op = args.output_op;
        params_.transform_src = args.transform_src;
        params_.transform_filter = args.transform_filter;
        params_.workspace = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(ConvolutionKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ConvolutionKernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<ConvolutionKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        size_t count = params_.ref_grad.layout().capacity(
                               params_.problem_size.filter_extent()) *
                       sizeof_bits<ElementGrad>::value / 8;
        result = cudaMemsetAsync(params_.ref_grad.data(), 0, count, stream);
        if (result != cudaSuccess) {
            return Status::kErrorInternal;
        }

        cutlass::Kernel<ConvolutionKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        result = cudaGetLastError();

        return result == cudaSuccess ? Status::kSuccess
                                     : Status::kErrorInternal;
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};

}  // namespace device
}  // namespace conv
}  // namespace cutlass

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
