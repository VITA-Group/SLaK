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
 * \file include/cutlass/convolution/device/implicit_gemm_precomp_convolution.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/* \file
   \brief Template for device-level Implicit GEMM Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ImplicitGemmKernel_>
class ImplicitGemmPrecompConvolution {
public:
    using ImplicitGemmKernel = ImplicitGemmKernel_;

    using ElementSrc = typename ImplicitGemmKernel::ElementSrc;
    using LayoutSrc = typename ImplicitGemmKernel::LayoutSrc;
    using ElementFilter = typename ImplicitGemmKernel::ElementFilter;
    using LayoutFilter = typename ImplicitGemmKernel::LayoutFilter;
    using ElementBias = typename ImplicitGemmKernel::ElementBias;
    using LayoutBias = typename ImplicitGemmKernel::LayoutBias;
    using ElementDst = typename ImplicitGemmKernel::ElementDst;
    using LayoutDst = typename ImplicitGemmKernel::LayoutDst;
    using ElementAccumulator = typename ImplicitGemmKernel::ElementAccumulator;
    using ElementCompute = typename ImplicitGemmKernel::ElementCompute;
    using OperatorClass = typename ImplicitGemmKernel::OperatorClass;
    using ArchTag = typename ImplicitGemmKernel::ArchTag;
    using ThreadblockShape = typename ImplicitGemmKernel::ThreadblockShape;
    using WarpShape = typename ImplicitGemmKernel::WarpShape;
    using InstructionShape = typename ImplicitGemmKernel::InstructionShape;
    using EpilogueOutputOp = typename ImplicitGemmKernel::EpilogueOutputOp;
    using ThreadblockSwizzle = typename ImplicitGemmKernel::ThreadblockSwizzle;
    using WarpMmaOperator = typename ImplicitGemmKernel::WarpMmaOperator;
    using ArchMmaOperator = typename ImplicitGemmKernel::ArchMmaOperator;
    using MathOperator = typename ImplicitGemmKernel::MathOperator;

    static cutlass::conv::Operator const kConvolutionalOperator =
            ImplicitGemmKernel::kConvolutionalOperator;

    static int const kStages = ImplicitGemmKernel::kStages;
    static int const kAlignmentSrc = ImplicitGemmKernel::kAlignmentSrc;
    static int const kAlignmentFilter = ImplicitGemmKernel::kAlignmentFilter;
    static int const kAlignmentDst = EpilogueOutputOp::kCount;
    static cutlass::conv::SpecialOptimizeDesc const kSpecialOpt =
            ImplicitGemmKernel::kSpecialOpt;

    /// Argument structure
    using Arguments = typename ImplicitGemmKernel::Arguments;

private:
    /// Kernel parameters object
    typename ImplicitGemmKernel::Params params_;

public:
    /// Constructs Implicit GEMM
    ImplicitGemmPrecompConvolution() {}

    /// Determines whether the Implicit GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        Status status = ImplicitGemmKernel::can_implement(
                args.problem_size, args.ref_src, args.ref_filter, args.ref_bias,
                args.ref_z, args.ref_dst);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        return ImplicitGemmKernel::get_workspace_size(args.problem_size);
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
        params_ = typename ImplicitGemmKernel::Params{
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
        params_.workspace = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(ImplicitGemmKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<ImplicitGemmKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<ImplicitGemmKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<ImplicitGemmKernel>
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

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
