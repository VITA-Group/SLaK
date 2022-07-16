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
 * \file test/unit/convolution/device/conv2d_bias_testbed_interleaved.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/convolution/device/implicit_gemm_precomp_convolution.h"

#include "../../conv/device/conv2d_problems.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/host_reorder.h"

#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/device/convolution.h"

#include "cutlass/core_io.h"
#include "cutlass/util/tensor_view_io.h"

namespace test {
namespace conv {
namespace device {

template <typename Conv2d, int InterleavedK>
class InterleavedTestbedConv2dBias {
public:
    using ElementA = typename Conv2d::ElementSrc;
    using LayoutA = typename Conv2d::LayoutSrc;
    using ElementB = typename Conv2d::ElementFilter;
    using LayoutB = typename Conv2d::LayoutFilter;
    using ElementC = typename Conv2d::ElementDst;
    using ElementBias = typename Conv2d::ElementBias;
    using LayoutC = typename Conv2d::LayoutDst;
    using ElementAccumulator = typename Conv2d::ElementAccumulator;
    using ElementCompute = typename Conv2d::ElementCompute;
    using EpilogueOutputOp = typename Conv2d::EpilogueOutputOp;

    static cutlass::conv::Operator const kConvolutionalOperator =
            Conv2d::kConvolutionalOperator;

public:
    /// Initialization
    cutlass::Distribution::Kind init_A;
    cutlass::Distribution::Kind init_B;
    cutlass::Distribution::Kind init_C;
    uint64_t seed;

    cutlass::HostTensor<ElementA, LayoutA> tensor_A;
    cutlass::HostTensor<ElementB, LayoutB> tensor_B;
    cutlass::HostTensor<ElementBias, LayoutC> tensor_Bias;
    cutlass::HostTensor<ElementC, LayoutC> tensor_C;
    cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
    cutlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

public:
    InterleavedTestbedConv2dBias(cutlass::Distribution::Kind init_A_ =
                                         cutlass::Distribution::Uniform,
                                 cutlass::Distribution::Kind init_B_ =
                                         cutlass::Distribution::Uniform,
                                 cutlass::Distribution::Kind init_C_ =
                                         cutlass::Distribution::Uniform,
                                 uint64_t seed_ = 2080)
            : init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    void initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            int scope;
            int bits = cutlass::sizeof_bits<Element>::value;

            if (bits <= 8) {
                scope = 2;
            } else if (bits == 16) {
                scope = 3;
            } else {
                scope = 8;
            }
            cutlass::reference::host::TensorFillRandomUniform(view, seed, scope,
                                                              -scope, 0);
        } else if (dist_kind == cutlass::Distribution::Identity) {
            cutlass::reference::host::TensorFillIdentity(view);
        } else if (dist_kind == cutlass::Distribution::Gaussian) {
            cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0,
                                                               0.5);
        } else if (dist_kind == cutlass::Distribution::Sequential) {
            cutlass::reference::host::BlockFillSequential(view.data(),
                                                          view.capacity());
        } else {
        }
    }

    void initialize(cutlass::conv::Conv2dProblemSize const& problem_size,
                    uint64_t seed = 2019) {
        tensor_A.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator,
                                                      problem_size));
        tensor_B.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator,
                                                      problem_size));
        tensor_C.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator,
                                                      problem_size));
        tensor_Bias.resize(implicit_gemm_tensor_bias_extent(
                kConvolutionalOperator, problem_size));
        tensor_D_computed.resize(implicit_gemm_tensor_c_extent(
                kConvolutionalOperator, problem_size));
        tensor_D_reference.resize(implicit_gemm_tensor_c_extent(
                kConvolutionalOperator, problem_size));

        initialize_tensor(tensor_A.host_view(), init_A, seed);
        initialize_tensor(tensor_B.host_view(), init_B, seed * 17);
        initialize_tensor(tensor_C.host_view(), init_C, seed * 39);
        initialize_tensor(tensor_Bias.host_view(), init_C, seed * 50);
        tensor_A.sync_device();
        tensor_B.sync_device();
        tensor_Bias.sync_device();
        tensor_C.sync_device();
        tensor_D_computed.sync_device();
        tensor_D_reference.sync_device();
    }

    bool sufficient() const {
        //
        // Determine SMEM requirements and waive if not satisfied
        //

        int smem_size =
                int(sizeof(typename Conv2d::ImplicitGemmKernel::SharedStorage));

        cudaDeviceProp properties;
        int device_idx;
        cudaError_t result = cudaGetDevice(&device_idx);

        if (result != cudaSuccess) {
            throw std::runtime_error("cudaGetDevice() API call failed.");
        }

        result = cudaGetDeviceProperties(&properties, device_idx);

        if (result != cudaSuccess) {
            throw std::runtime_error("cudaGetDeviceProperties() failed");
        }

        if (properties.sharedMemPerMultiprocessor < smem_size) {
            return false;
        }

        return true;
    }

    /// Executes one test
    bool run(cutlass::conv::Conv2dProblemSize const& problem_size,
             cutlass::conv::SplitKMode const& split_k_mode =
                     cutlass::conv::SplitKMode::kSerial,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(1),
             ElementCompute gamma = ElementCompute(1)) {
        // Waive test if CUDA device is insufficient
        if (!sufficient()) {
            return true;
        }

#if 0  // display conv2d problem size for debugging
    std::cout << problem_size << std::endl
              << "alpha, beta: (" << float(alpha) << ", " << float(beta) << ")" << std::endl
              << "split_k_mode: " << ((split_k_mode == cutlass::conv::SplitKMode::kSerial) ? "(serial)" : "(parallel)") << std::endl
              << std::endl;
#endif

        initialize(problem_size);

        // configure the operator
        Conv2d conv2d_op;

        typename EpilogueOutputOp::Params epilogue_param{alpha, beta, gamma};

        typename Conv2d::Arguments conv2d_args(
                problem_size, tensor_A.device_ref(), tensor_B.device_ref(),
                tensor_Bias.device_ref(), tensor_C.device_ref(),
                tensor_D_computed.device_ref(), epilogue_param);

        // find workspace requirement for parallel split-k reduction
        size_t workspace_size = Conv2d::get_workspace_size(conv2d_args);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status =
                conv2d_op.initialize(conv2d_args, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess);
        if (status != cutlass::Status::kSuccess) {
            return false;
        }

        // run conv2d operator
        status = conv2d_op();

        EXPECT_TRUE(status == cutlass::Status::kSuccess);
        if (status != cutlass::Status::kSuccess) {
            return false;
        }

        bool passed = false;

        tensor_D_computed.sync_host();

        if (kConvolutionalOperator == cutlass::conv::Operator::kFprop) {
            cutlass::reference::host::Convolution<
                    cutlass::conv::ConvType::kConvolution,
                    typename Conv2d::ElementSrc, typename Conv2d::LayoutSrc,
                    typename Conv2d::ElementFilter,
                    typename Conv2d::LayoutFilter, typename Conv2d::ElementDst,
                    typename Conv2d::LayoutDst, typename Conv2d::ElementBias,
                    typename Conv2d::LayoutBias, ElementCompute,
                    ElementAccumulator, cutlass::arch::OpMultiplyAddSaturate>
                    reference_convolution;

            reference_convolution(
                    problem_size, alpha, tensor_A.host_ref(),
                    tensor_B.host_ref(), beta, tensor_Bias.host_ref(), gamma,
                    tensor_C.host_ref(), tensor_D_reference.host_ref(),
                    ElementAccumulator(0));
        } else {
            cutlass::reference::host::Deconvolution<
                    typename Conv2d::ElementSrc, typename Conv2d::LayoutSrc,
                    typename Conv2d::ElementFilter,
                    typename Conv2d::LayoutFilter, typename Conv2d::ElementDst,
                    typename Conv2d::LayoutDst, typename Conv2d::ElementBias,
                    typename Conv2d::LayoutBias, ElementCompute,
                    ElementAccumulator>
                    reference_deconvolution;

            reference_deconvolution(
                    problem_size, alpha, tensor_A.host_ref(),
                    tensor_B.host_ref(), beta, tensor_Bias.host_ref(), gamma,
                    tensor_C.host_ref(), tensor_D_reference.host_ref(),
                    ElementAccumulator(0));
        }

        passed = cutlass::reference::host::TensorEquals(
                tensor_D_computed.host_view(), tensor_D_reference.host_view());

        EXPECT_TRUE(passed);

        if (!passed) {
            std::stringstream fname;

            fname << "error_Conv2d_ImplicitGemm_device_"
                  << (split_k_mode == cutlass::conv::SplitKMode::kSerial
                              ? "serial_reduction_"
                              : "parallel_reduction_")
                  << (Conv2d::kConvolutionalOperator ==
                                      cutlass::conv::Operator::kFprop
                              ? "fprop_"
                              : (Conv2d::kConvolutionalOperator ==
                                                 cutlass::conv::Operator::kDgrad
                                         ? "dgrad_"
                                         : "wgrad_"))
                  << "nhwc_" << problem_size.N << "x" << problem_size.H << "x"
                  << problem_size.W << "x" << problem_size.C << "_krsc_"
                  << problem_size.K << "x" << problem_size.R << "x"
                  << problem_size.S << "x" << problem_size.C << "_padding_"
                  << problem_size.pad_h << "x" << problem_size.pad_w
                  << "_stride_" << problem_size.stride_h << "x"
                  << problem_size.stride_w << "_dilation_"
                  << problem_size.dilation_h << "x" << problem_size.dilation_w
                  << "_"
                  << (problem_size.mode ==
                                      cutlass::conv::Mode::kCrossCorrelation
                              ? "xcorr_"
                              : "conv_")
                  << Conv2d::ThreadblockShape::kM << "x"
                  << Conv2d::ThreadblockShape::kN << "x"
                  << Conv2d::ThreadblockShape::kK << "_"
                  << Conv2d::WarpShape::kM << "x" << Conv2d::WarpShape::kN
                  << "x" << Conv2d::WarpShape::kK << ".txt";

            std::cout << fname.str() << std::endl;

            std::ofstream results(fname.str());

            results << problem_size << std::endl;

            results << "\nA:\n"
                    << tensor_A.host_view() << "\n"
                    << "\nB:\n"
                    << tensor_B.host_view() << "\n"
                    << "\nBias:\n"
                    << tensor_Bias.host_view() << "\n"
                    << "\nC:\n"
                    << tensor_C.host_view() << "\n"
                    << "\nD reference:\n"
                    << tensor_D_reference.host_view() << "\n"
                    << "\nD computed:\n"
                    << tensor_D_computed.host_view() << "\n";
        }

        return passed;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// TestAllConv: Runs cutlass::conv::device::ImplicitGemmConvolution operator and
// compares it with reference TestAllConv runs conv operator on default conv
// problem sizes from test::conv::device::TestbedConv2dProblemSizes Additionaly,
// each conv2d test can provide conv problem sizes (conv_test_sizes) and
// blacklist of sizes (conv_blacklist_sizes)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ImplicitGemm, int InterleavedK>
bool TestAllInterleavedConv2dBias(
        const Conv2dProblemVector& conv_test_sizes = Conv2dProblemVector(),
        const Conv2dProblemVector& conv_blacklist_sizes =
                Conv2dProblemVector()) {
    static cutlass::conv::Operator const kConvolutionalOperator =
            ImplicitGemm::kConvolutionalOperator;

    bool passed = true;

    //
    // Testbed object
    //

    InterleavedTestbedConv2dBias<ImplicitGemm, InterleavedK> testbed;

    //
    // Get conv problem sizes to run conv operator
    //
    TestbedConv2dProblemSizes conv_problems(
            InterleavedK);  // minimum channel size must be multiple of
                            // InterleavedK for interleaved layout

    // Vector of conv2d problem sizes to avoid duplicate runs
    Conv2dProblemVector conv_tested_sizes;

    Conv2dProblemVector const* problem_vectors[] = {
        &conv_test_sizes,                     // run user specified sizes
        &conv_problems.conv2d_default_sizes,  // run default and cudnn bug sizes
        &conv_problems.conv2d_resnet50_sizes,  // run resnet50 sizes
#if CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
        &conv_problems.conv2d_rigorous_sizes,  // run large and rigorous sizes
                                               // if enabled
#endif
    };

    // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1,
    // alpha=1.0, beta=0.0)
    for (Conv2dProblemVector const* problem_vector : problem_vectors) {
        ChannelDivisibilitySpecification channel_spec(
                InterleavedK);  // input and output channels must be multiple of
                                // InterleavedK
        auto pruned_problem_vector = prune(*problem_vector, channel_spec);

        //  Run conv testbed on default convolution sizes
        for (auto conv_problem : pruned_problem_vector) {
            // Skip blacklist and avoid duplicate problem sizes
            if (std::find(conv_blacklist_sizes.begin(),
                          conv_blacklist_sizes.end(),
                          conv_problem) != conv_blacklist_sizes.end() ||
                std::find(conv_tested_sizes.begin(), conv_tested_sizes.end(),
                          conv_problem) != conv_tested_sizes.end()) {
                continue;
            }

            //
            // Test
            //
            // push back tested problem size to avoid re-running duplicates
            conv_tested_sizes.push_back(conv_problem);
            if (kConvolutionalOperator == cutlass::conv::Operator::kFprop) {
                // test mode = xcross
                passed = testbed.run(conv_problem,
                                     cutlass::conv::SplitKMode::kSerial, 1.f,
                                     1.f, 1.f);
            } else {
                passed = testbed.run(conv_problem,
                                     cutlass::conv::SplitKMode::kSerial, 1.f,
                                     0.f, 0.f);
            }

            if (!passed) {
                return false;
            }
        }
    }

    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace conv
}  // namespace test
