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
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

/**
 * \file test/unit/convolution/device/testbed.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_reorder.h"

#include "testbed.h"

namespace test {
namespace convolution {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
struct Conv2dWgradTestbed {
    using ElementAccumulator = typename Convolution::ElementAccumulator;
    using ElementCompute = typename Convolution::ConvolutionKernel::Epilogue::
            OutputOp::ElementCompute;

    /// Initialization
    cutlass::Distribution::Kind init_src;
    cutlass::Distribution::Kind init_diff;
    uint64_t seed;

    cutlass::HostTensor<typename Convolution::ElementSrc,
                        typename Convolution::LayoutSrc>
            tensor_src;
    cutlass::HostTensor<typename Convolution::ElementDiff,
                        typename Convolution::LayoutDiff>
            tensor_diff;
    cutlass::HostTensor<typename Convolution::ElementGrad,
                        typename Convolution::LayoutGrad>
            tensor_grad;
    cutlass::HostTensor<typename Convolution::ElementGrad,
                        typename Convolution::LayoutGrad>
            reference_grad;

    //
    // Methods
    //

    Conv2dWgradTestbed(cutlass::Distribution::Kind init_src_ =
                               cutlass::Distribution::Uniform,
                       cutlass::Distribution::Kind init_diff_ =
                               cutlass::Distribution::Uniform,
                       uint64_t seed_ = 2080)
            : init_src(init_src_), init_diff(init_diff_), seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            double scope_max, scope_min;
            int bits_input = cutlass::sizeof_bits<Element>::value;
            int bits_output = cutlass::sizeof_bits<
                    typename Convolution::ElementGrad>::value;

            if (bits_input == 1) {
                scope_max = 2;
                scope_min = 0;
            } else if (bits_input <= 8) {
                scope_max = 8;
                scope_min = -8;
            } else if (bits_output == 16) {
                scope_max = 5;
                scope_min = -5;
            } else {
                scope_max = 8;
                scope_min = -8;
            }

            cutlass::reference::host::TensorFillRandomUniform(
                    view, seed, scope_max, scope_min, 0);
        } else if (dist_kind == cutlass::Distribution::Identity) {
            cutlass::reference::host::TensorFillIdentity(view);
        } else if (dist_kind == cutlass::Distribution::Gaussian) {
            cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0,
                                                               0.5);
        } else if (dist_kind == cutlass::Distribution::Sequential) {
            cutlass::reference::host::BlockFillSequential(view.data(),
                                                          view.capacity());
        } else if (dist_kind == cutlass::Distribution::Constant) {
            cutlass::reference::host::TensorFill(view, Element(1));
        } else {
            // TODO: Implement the rest
            EXPECT_TRUE(false) << "Not implemented";
            return false;
        }

        return true;
    }

    /// Initializes data structures
    void initialize(cutlass::conv::Conv2dProblemSize conv_param) {
        if_constexpr<is_depthwise_convolution<Convolution>()>([&](auto _) {
            auto&& conv_param_ = _(conv_param);
            ASSERT_EQ(conv_param_.K, conv_param_.C);
        });

        //
        // Allocate the CONVOLUTION workspace
        //

        tensor_src.resize(typename Convolution::LayoutSrc::TensorCoord{
                conv_param.N, conv_param.H, conv_param.W, conv_param.C});
        tensor_diff.resize(typename Convolution::LayoutSrc::TensorCoord{
                conv_param.N, conv_param.P, conv_param.Q, conv_param.K});
        if_constexpr<is_depthwise_convolution<Convolution>()>(
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    ASSERT_EQ(conv_param_.K, conv_param_.C);
                    tensor_grad.resize(
                            typename Convolution::LayoutGrad::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    1});
                    reference_grad.resize(
                            typename Convolution::LayoutGrad::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    1},
                            false);
                },
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    tensor_grad.resize(
                            typename Convolution::LayoutGrad::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    conv_param_.C});
                    reference_grad.resize(
                            typename Convolution::LayoutGrad::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    conv_param_.C},
                            false);
                });

        EXPECT_TRUE(initialize_tensor(tensor_src.host_view(), init_src,
                                      seed + 2019));
        EXPECT_TRUE(initialize_tensor(tensor_diff.host_view(), init_diff,
                                      seed + 2018));

        tensor_src.sync_device();
        tensor_diff.sync_device();
    }

    /// Compares computed reference with device reference and outputs to a file
    /// if incorrect
    bool compare_reference() {
        tensor_grad.sync_host();

        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_src.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_diff.host_view()),
                  0);

        if (tensor_grad.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              tensor_grad.host_view()),
                      0);

        if (reference_grad.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              reference_grad.host_view()),
                      0);

        bool passed = cutlass::reference::host::TensorEquals(
                reference_grad.host_view(), tensor_grad.host_view());

        if (!passed) {
            std::stringstream fname_ref;

            fname_ref << "error_Conv2d_Wgrad_device_reference_"
                      << Convolution::ThreadblockShape::kM << "x"
                      << Convolution::ThreadblockShape::kN << "x"
                      << Convolution::ThreadblockShape::kK << "_"
                      << Convolution::WarpShape::kM << "x"
                      << Convolution::WarpShape::kN << "x"
                      << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_ref(fname_ref.str());

            file_ref << "Reference =\n" << reference_grad.host_view();

            std::stringstream fname_comp;

            fname_comp << "error_Conv2d_Wgrad_device_computed_"
                       << Convolution::ThreadblockShape::kM << "x"
                       << Convolution::ThreadblockShape::kN << "x"
                       << Convolution::ThreadblockShape::kK << "_"
                       << Convolution::WarpShape::kM << "x"
                       << Convolution::WarpShape::kN << "x"
                       << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_comp(fname_comp.str());

            file_comp << "Computed =\n" << tensor_grad.host_view();
        }

        EXPECT_TRUE(passed);

        return passed;
    }

    /// Verifies the result is a GEMM
    bool verify(cutlass::conv::Conv2dProblemSize conv_param,
                ElementCompute alpha) {
        //
        // Verify
        //

        cutlass::reference::host::Convolution2dBackwardFilter<
                Convolution::kConvolutionType, typename Convolution::ElementSrc,
                typename Convolution::LayoutSrc,
                typename Convolution::ElementDiff,
                typename Convolution::LayoutDiff,
                typename Convolution::ElementGrad,
                typename Convolution::LayoutGrad, ElementCompute,
                ElementAccumulator, typename Convolution::Operator>
                reference_convolution;

        reference_convolution(conv_param, alpha, tensor_src.host_ref(),
                              tensor_diff.host_ref(), reference_grad.host_ref(),
                              ElementAccumulator(0));

        return compare_reference();
    }

    /// Executes one test
    bool run(cutlass::conv::Conv2dProblemSize conv_param,
             ElementCompute alpha = ElementCompute(1)) {
        this->initialize(conv_param);

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  tensor_diff.device_ref(),
                                                  tensor_grad.device_ref(),
                                                  {alpha}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Verify
        //

        bool passed = this->verify(conv_param, alpha);

        if (!passed) {
            std::cout << "Error with alpha = " << alpha << "\n"
                      << conv_param << std::endl;
        }

        return passed;
    }

    bool perf(cutlass::conv::Conv2dProblemSize conv_param,
              ElementCompute alpha = ElementCompute(1), int iterations = 1,
              bool verify = false) {
        this->initialize(conv_param);

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  tensor_diff.device_ref(),
                                                  tensor_grad.device_ref(),
                                                  {alpha}};

        Convolution conv_op;

        size_t workspace_size = Convolution::get_workspace_size(arguments);

        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = conv_op.initialize(arguments, workspace.get());

        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

        //
        // Run the CONVOLUTION
        //

        status = conv_op();
        status = conv_op();

        TimerGPU timer;
        for (int i = 0; i < iterations; ++i) {
            status = conv_op();
        }
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        float time_ms = timer.read() / static_cast<float>(iterations);
        float ops = 2.f * static_cast<float>(
                                  static_cast<int64_t>(conv_param.N) *
                                  conv_param.K * conv_param.P * conv_param.Q *
                                  conv_param.R * conv_param.S * conv_param.C);
        if_constexpr<is_depthwise_convolution<Convolution>()>([&](auto _) {
            auto&& conv_param_ = _(conv_param);
            ops /= static_cast<float>(conv_param_.C);
        });

        std::cout << conv_param << "Time = " << time_ms << "ms"
                  << "\n"
                  << "Performance = " << ops / (time_ms * 1e9) << "Tops"
                  << std::endl;

        bool passed = true;
        if (verify) {
            //
            // Verify
            //

            passed = this->verify(conv_param, alpha);

            if (!passed) {
                std::cout << "Error with alpha = " << alpha << "\n"
                          << std::endl;
            }
        }
        return passed;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestDepthwiseConv2dWgrad() {
    bool passed = true;

    double problem_alpha[] = {1.0};

    Conv2dWgradTestbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {160, 48, 33}) {
        for (int g : {3, 7}) {
            for (int ih : {16}) {
                for (int iw : {16}) {
                    for (int fh : {15, 7, 5, 3}) {
                        for (int ph : {static_cast<int>(fh / 2), 0}) {
                            for (int sh : {1, 2}) {
                                int oh = (ih + 2 * ph - fh) / sh + 1;
                                int ow = (iw + 2 * ph - fh) / sh + 1;
                                args.emplace_back(ConvolutionParameter{
                                        n, ih, iw, g, g, fh, fh, oh, ow, ph, ph,
                                        sh, sh, 1, 1, mode, 1, g});
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            passed =
                    testbed.run(arg, cutlass::from_real<ElementCompute>(alpha));
            if (!passed) {
                return false;
            }
        }
    }
    return passed;
}

template <typename Convolution>
bool BenchDepthwiseConv2dWgrad(int batch = 64, int iterations = 1,
                               bool do_verify = true) {
    bool passed = true;

    double problem_alpha[] = {1.0};

    Conv2dWgradTestbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int fh : {31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3}) {
        int ph = fh / 2;
        int sh = 1;
        int oh = (32 + 2 * ph - fh) / sh + 1;
        int ow = (32 + 2 * ph - fh) / sh + 1;
        args.emplace_back(ConvolutionParameter{batch, 32, 32, 384, 384, fh, fh,
                                               oh, ow, ph, ph, sh, sh, 1, 1,
                                               mode, 1, 384});
    }

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            passed =
                    testbed.perf(arg, cutlass::from_real<ElementCompute>(alpha),
                                 iterations, verify);

            cnt++;
            if (cnt >= 5)
                verify = false;
            if (!passed) {
                return false;
            }
        }
    }

    return passed;
}

}  // namespace device
}  // namespace convolution
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
