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

namespace {
template <typename Convolution>
constexpr bool is_depthwise_convolution() {
    return Convolution::kConvolutionType ==
           cutlass::conv::ConvType::kDepthwiseConvolution;
}

struct if_constexpr_identity {
    template <typename T>
    decltype(auto) operator()(T&& x) {
        return std::forward<T>(x);
    }
};

template <bool cond>
struct if_constexpr_impl;

template <>
struct if_constexpr_impl<true> {
    template <class Then, class Else>
    static decltype(auto) run(Then&& then, Else&&) {
        return then(if_constexpr_identity{});
    }
};

template <>
struct if_constexpr_impl<false> {
    template <class Then, class Else>
    static decltype(auto) run(Then&&, Else&& else_) {
        return else_(if_constexpr_identity{});
    }
};

//! functionally equivalent to "if constexpr" in C++17.
//! then/else callbacks receives an Identity functor as its sole argument.
//! The functor is useful for masking eager type check on generic lambda,
//! preventing syntax error on not taken branches.
template <bool Cond, class Then, class Else>
decltype(auto) if_constexpr(Then&& then, Else&& else_) {
    return if_constexpr_impl<Cond>::run(then, else_);
}

template <bool Cond, class Then>
decltype(auto) if_constexpr(Then&& then) {
    return if_constexpr<Cond>(std::forward<Then>(then), [](auto) {});
}
}  // namespace

namespace cutlass {
template <typename Element, typename Layout, int Interleaved>
void reorder_row(TensorRef<Element, Layout> dest,
                 TensorRef<Element, Layout> src, int rows, int cols) {
    TensorRef<Element, layout::RowMajor> mappedDest(dest.data(), cols);
    TensorRef<Element, layout::RowMajor> mappedSrc(src.data(), cols);

    const int InstructionShapeCol = 8;
    // 4 threads per Quad
    const int ElementsPerThread = InstructionShapeCol / 4;
    // 4 threads per Quad
    const int ReorderedElementsPerThread = Interleaved / 4;

    for (int k = 0; k < rows; k++) {
        for (int n = 0; n < cols; n++) {
            mappedDest.at(
                    {(k / Interleaved) * Interleaved +
                             ((k % ReorderedElementsPerThread) /
                              ElementsPerThread) *
                                     InstructionShapeCol +
                             ((k % Interleaved) / ReorderedElementsPerThread) *
                                     ElementsPerThread +
                             (k % ElementsPerThread),
                     n}) = mappedSrc.at({k, n});
        }
    }
}
};  // namespace cutlass

namespace test {
namespace convolution {
namespace device {
namespace {
/////////////////////////////////////////////////////////////////////////////////////////////////

inline char const* to_string(cutlass::Status status) {
    switch (status) {
        case cutlass::Status::kSuccess:
            return "kSuccess";
        case cutlass::Status::kErrorMisalignedOperand:
            return "kErrorMisalignedOperand";
        case cutlass::Status::kErrorInvalidLayout:
            return "kErrorInvalidLayout";
        case cutlass::Status::kErrorInvalidProblem:
            return "kErrorInvalidProblem";
        case cutlass::Status::kErrorNotSupported:
            return "kErrorNotSupported";
        case cutlass::Status::kErrorWorkspaceNull:
            return "kErrorWorkspaceNull";
        case cutlass::Status::kErrorInternal:
            return "kErrorInternal";
        case cutlass::Status::kInvalid:
            return "kInvalid";
        default:
            break;
    }
    return "invalid";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

class TimerGPU {
public:
    cudaEvent_t start, stop;
    cudaStream_t stream;
    TimerGPU(cudaStream_t stream_ = 0) : stream{stream_} {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    ~TimerGPU() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float read() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution, bool ReorderK = false>
struct Testbed {
    using ElementAccumulator = typename Convolution::ElementAccumulator;
    using ElementCompute = typename Convolution::ConvolutionKernel::Epilogue::
            OutputOp::ElementCompute;

    static bool const kReorderK = ReorderK;

    /// Initialization
    cutlass::Distribution::Kind init_src;
    cutlass::Distribution::Kind init_filter;
    cutlass::Distribution::Kind init_bias;
    cutlass::Distribution::Kind init_z;
    uint64_t seed;

    cutlass::HostTensor<typename Convolution::ElementSrc,
                        typename Convolution::LayoutSrc>
            tensor_src;
    cutlass::HostTensor<typename Convolution::ElementFilter,
                        typename Convolution::LayoutFilter>
            tensor_filter;
    cutlass::HostTensor<typename Convolution::ElementBias,
                        typename Convolution::LayoutBias>
            tensor_bias;
    cutlass::HostTensor<typename Convolution::ElementFilter,
                        typename Convolution::LayoutFilter>
            tensor_filter_reordered;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_z;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_dst;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            reference_dst;

    //
    // Methods
    //

    Testbed(cutlass::Distribution::Kind init_src_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_filter_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_bias_ =
                    cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_z_ =
                    cutlass::Distribution::Uniform,
            uint64_t seed_ = 2080)
            : init_src(init_src_),
              init_filter(init_filter_),
              init_bias(init_bias_),
              init_z(init_z_),
              seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            double scope_max, scope_min;
            int bits_input = cutlass::sizeof_bits<Element>::value;
            int bits_output = cutlass::sizeof_bits<
                    typename Convolution::ElementDst>::value;

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
        if_constexpr<is_depthwise_convolution<Convolution>()>(
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    ASSERT_EQ(conv_param_.K, conv_param_.C);
                    tensor_filter.resize(
                            typename Convolution::LayoutFilter::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    1});
                },
                [&](auto _) {
                    auto&& conv_param_ = _(conv_param);
                    tensor_filter.resize(
                            typename Convolution::LayoutFilter::TensorCoord{
                                    conv_param_.K, conv_param_.R, conv_param_.S,
                                    conv_param_.C});
                });
        tensor_bias.resize(typename Convolution::LayoutBias::TensorCoord{
                1, 1, 1, conv_param.K});
        tensor_z.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.P, conv_param.Q, conv_param.K});
        tensor_dst.resize(typename Convolution::LayoutDst::TensorCoord{
                conv_param.N, conv_param.P, conv_param.Q, conv_param.K});
        reference_dst.resize(
                typename Convolution::LayoutDst::TensorCoord{
                        conv_param.N, conv_param.P, conv_param.Q, conv_param.K},
                false);

        EXPECT_TRUE(initialize_tensor(tensor_src.host_view(), init_src,
                                      seed + 2019));
        EXPECT_TRUE(initialize_tensor(tensor_filter.host_view(), init_filter,
                                      seed + 2018));
        EXPECT_TRUE(initialize_tensor(tensor_bias.host_view(), init_bias,
                                      seed + 2017));
        EXPECT_TRUE(
                initialize_tensor(tensor_z.host_view(), init_z, seed + 2016));

        cutlass::reference::host::TensorCopy(reference_dst.host_view(),
                                             tensor_z.host_view());

        tensor_src.sync_device();
        tensor_filter.sync_device();
        tensor_bias.sync_device();
        tensor_z.sync_device();
        tensor_dst.sync_device();

        if (kReorderK) {
            tensor_filter_reordered.resize(
                    typename Convolution::LayoutFilter::TensorCoord{
                            conv_param.K, conv_param.R, conv_param.S,
                            conv_param.C});

            if (cutlass::platform::is_same<
                        cutlass::layout::TensorNHWC,
                        typename Convolution::LayoutSrc>::value &&
                conv_param.K % 16 == 0) {
                const int kN = Convolution::ThreadblockShape::kN;
                EXPECT_TRUE(kN == 64 || kN == 32 || kN == 16);
                if (kN == 64) {
                    cutlass::reorder_row<typename Convolution::ElementFilter,
                                         typename Convolution::LayoutFilter,
                                         64>(
                            tensor_filter_reordered.host_ref(),
                            tensor_filter.host_ref(), conv_param.K,
                            conv_param.R * conv_param.S * conv_param.C);
                } else if (kN == 32) {
                    cutlass::reorder_row<typename Convolution::ElementFilter,
                                         typename Convolution::LayoutFilter,
                                         32>(
                            tensor_filter_reordered.host_ref(),
                            tensor_filter.host_ref(), conv_param.K,
                            conv_param.R * conv_param.S * conv_param.C);
                } else {
                    cutlass::reorder_row<typename Convolution::ElementFilter,
                                         typename Convolution::LayoutFilter,
                                         16>(
                            tensor_filter_reordered.host_ref(),
                            tensor_filter.host_ref(), conv_param.K,
                            conv_param.R * conv_param.S * conv_param.C);
                }
            } else if (cutlass::platform::is_same<
                               cutlass::layout::TensorNCxHWx<32>,
                               typename Convolution::LayoutSrc>::value) {
                cutlass::reorder_convK<32>(
                        tensor_filter_reordered.host_ref(),
                        tensor_filter.host_ref(),
                        implicit_gemm_problem_size(
                                cutlass::conv::Operator::kFprop, conv_param));
            } else if (cutlass::platform::is_same<
                               cutlass::layout::TensorNCxHWx<64>,
                               typename Convolution::LayoutSrc>::value) {
                cutlass::reorder_convK<64>(
                        tensor_filter_reordered.host_ref(),
                        tensor_filter.host_ref(),
                        implicit_gemm_problem_size(
                                cutlass::conv::Operator::kFprop, conv_param));
            } else {
                throw std::runtime_error("unsupport reorderK layout");
            }

            tensor_filter_reordered.sync_device();
        }
    }

    /// Compares computed reference with device reference and outputs to a file
    /// if incorrect
    bool compare_reference() {
        tensor_dst.sync_host();

        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_src.host_view()),
                  0);
        EXPECT_GT(
                cutlass::reference::host::TensorNorm(tensor_filter.host_view()),
                0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_bias.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_z.host_view()),
                  0);

        if (tensor_dst.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              tensor_dst.host_view()),
                      0);

        if (reference_dst.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              reference_dst.host_view()),
                      0);

        bool passed = cutlass::reference::host::TensorEquals(
                reference_dst.host_view(), tensor_dst.host_view());

        if (!passed) {
            std::stringstream fname_ref;

            fname_ref << "error_Convolution_device_reference_"
                      << Convolution::ThreadblockShape::kM << "x"
                      << Convolution::ThreadblockShape::kN << "x"
                      << Convolution::ThreadblockShape::kK << "_"
                      << Convolution::WarpShape::kM << "x"
                      << Convolution::WarpShape::kN << "x"
                      << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_ref(fname_ref.str());

            file_ref << "Reference =\n" << reference_dst.host_view();

            std::stringstream fname_comp;

            fname_comp << "error_Convolution_device_computed_"
                       << Convolution::ThreadblockShape::kM << "x"
                       << Convolution::ThreadblockShape::kN << "x"
                       << Convolution::ThreadblockShape::kK << "_"
                       << Convolution::WarpShape::kM << "x"
                       << Convolution::WarpShape::kN << "x"
                       << Convolution::WarpShape::kK << ".txt";

            std::ofstream file_comp(fname_comp.str());

            file_comp << "Computed =\n" << tensor_dst.host_view();
        }

        EXPECT_TRUE(passed);

        return passed;
    }

    /// Verifies the result is a GEMM
    bool verify(cutlass::conv::Conv2dProblemSize conv_param,
                ElementCompute alpha, ElementCompute beta,
                ElementCompute gamma) {
        //
        // Verify
        //

        cutlass::reference::host::Convolution<
                Convolution::kConvolutionType, typename Convolution::ElementSrc,
                typename Convolution::LayoutSrc,
                typename Convolution::ElementFilter,
                typename Convolution::LayoutFilter,
                typename Convolution::ElementDst,
                typename Convolution::LayoutDst,
                typename Convolution::ElementBias,
                typename Convolution::LayoutBias, ElementCompute,
                ElementAccumulator, typename Convolution::Operator>
                reference_convolution;

        reference_convolution(conv_param, alpha, tensor_src.host_ref(),
                              tensor_filter.host_ref(), beta,
                              tensor_bias.host_ref(), gamma,
                              tensor_z.host_ref(), reference_dst.host_ref(),
                              ElementAccumulator(0));

        return compare_reference();
    }

    /// Executes one test
    bool run(cutlass::conv::Conv2dProblemSize conv_param,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(1),
             ElementCompute gamma = ElementCompute(0)) {
        this->initialize(conv_param);

        cutlass::TensorRef<typename Convolution::ElementFilter,
                           typename Convolution::LayoutFilter>
                filter_dev_ref =
                        kReorderK ? tensor_filter_reordered.device_ref()
                                  : tensor_filter.device_ref();

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  filter_dev_ref,
                                                  tensor_bias.device_ref(),
                                                  tensor_z.device_ref(),
                                                  tensor_dst.device_ref(),
                                                  {alpha, beta, gamma}};

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

        bool passed = this->verify(conv_param, alpha, beta, gamma);

        if (!passed) {
            std::cout << "Error with alpha = " << alpha << ", beta = " << beta
                      << ", gamma = " << gamma << "\n"
                      << conv_param << std::endl;
        }

        return passed;
    }

    bool perf(cutlass::conv::Conv2dProblemSize conv_param,
              ElementCompute alpha = ElementCompute(1),
              ElementCompute beta = ElementCompute(1),
              ElementCompute gamma = ElementCompute(0), int iterations = 1,
              bool verify = false) {
        this->initialize(conv_param);

        cutlass::TensorRef<typename Convolution::ElementFilter,
                           typename Convolution::LayoutFilter>
                filter_dev_ref =
                        kReorderK ? tensor_filter_reordered.device_ref()
                                  : tensor_filter.device_ref();

        //
        // Initialize the CONVOLUTION operator
        //

        typename Convolution::Arguments arguments{conv_param,
                                                  tensor_src.device_ref(),
                                                  filter_dev_ref,
                                                  tensor_bias.device_ref(),
                                                  tensor_z.device_ref(),
                                                  tensor_dst.device_ref(),
                                                  {alpha, beta, gamma}};

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

            passed = this->verify(conv_param, alpha, beta, gamma);

            if (!passed) {
                std::cout << "Error with alpha = " << alpha
                          << ", beta = " << beta << ", gamma = " << gamma
                          << "\n"
                          << std::endl;
            }
        }
        return passed;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestAllConvolution() {
    bool passed = true;

    double problem_alpha[] = {0.019980327};

    double problem_beta[] = {-1.001234567};

    double problem_gamma[] = {0.019990229};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {128, 48, 33}) {
        for (int ic : {32, 24, 64}) {
            for (int oc : {128, 32, 24}) {
                for (int ih : {8}) {
                    for (int iw : {8}) {
                        for (int fh : {3, 5, 7}) {
                            for (int ph : {static_cast<int>(fh / 2), 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ih, iw, ic, oc, fh, fh, oh, ow,
                                            ph, ph, sh, sh, 1, 1, mode});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (cutlass::platform::is_same<
                                cutlass::layout::TensorNCxHWx<32>,
                                typename Convolution::LayoutDst>::value &&
                        arg.K % 32 != 0)
                        continue;
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

template <typename Convolution>
bool TestConvolutionNHWC() {
    bool passed = true;

    double problem_alpha[] = {0.019980327};

    double problem_beta[] = {0.02};

    double problem_gamma[] = {0.019990229};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {8, 24, 33}) {
        for (int ic : {8, 16, 32}) {
            for (int oc : {16, 24, 32}) {
                for (int ih : {7}) {
                    for (int iw : {9}) {
                        for (int fh : {1, 3, 5}) {
                            for (int ph : {static_cast<int>(fh / 2), 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ih, iw, ic, oc, fh, fh, oh, ow,
                                            ph, ph, sh, sh, 1, 1, mode});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (cutlass::platform::is_same<
                                cutlass::layout::TensorNCxHWx<8>,
                                typename Convolution::LayoutFilter>::value &&
                        (arg.K % 8 != 0 || arg.C % 8 != 0))
                        continue;
                    else if (cutlass::platform::is_same<
                                     cutlass::layout::TensorNCxHWx<16>,
                                     typename Convolution::LayoutFilter>::
                                     value &&
                             (arg.K % 16 != 0 || arg.C % 16 != 0))
                        continue;
                    else if (cutlass::platform::is_same<
                                     cutlass::layout::TensorNCxHWx<32>,
                                     typename Convolution::LayoutFilter>::
                                     value &&
                             (arg.K % 32 != 0 || arg.C % 32 != 0))
                        continue;
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

template <typename Convolution>
bool TestConvolutionNHWC_ReorderK() {
    bool passed = true;

    double problem_alpha[] = {0.019980327};

    double problem_beta[] = {-1.001234567};

    double problem_gamma[] = {0.019990229};

    Testbed<Convolution, true> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {8, 24, 33}) {
        for (int ic : {8, 16, 32}) {
            for (int oc : {32, 64}) {
                for (int ih : {7}) {
                    for (int iw : {9}) {
                        for (int fh : {1, 3, 5}) {
                            for (int ph : {static_cast<int>(fh / 2), 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ih, iw, ic, oc, fh, fh, oh, ow,
                                            ph, ph, sh, sh, 1, 1, mode});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (arg.K % Convolution::ThreadblockShape::kN != 0)
                        continue;
                    if (cutlass::platform::is_same<
                                cutlass::layout::TensorNCxHWx<4>,
                                typename Convolution::LayoutFilter>::value &&
                        (arg.C % 4 != 0))
                        continue;
                    if (cutlass::platform::is_same<
                                cutlass::layout::TensorNCxHWx<8>,
                                typename Convolution::LayoutFilter>::value &&
                        (arg.C % 8 != 0))
                        continue;
                    else if (cutlass::platform::is_same<
                                     cutlass::layout::TensorNCxHWx<16>,
                                     typename Convolution::LayoutFilter>::
                                     value &&
                             (arg.C % 16 != 0))
                        continue;
                    else if (cutlass::platform::is_same<
                                     cutlass::layout::TensorNCxHWx<32>,
                                     typename Convolution::LayoutFilter>::
                                     value &&
                             (arg.C % 32 != 0))
                        continue;
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution>
bool TestConvolutionMma(int interleaved = 32) {
    bool passed = true;

    double problem_alpha[] = {1.0};

    double problem_beta[] = {-1.3141413421};

    double problem_gamma[] = {1.0};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {128, 48, 33}) {
        for (int ic : {2, 3}) {      // times interleaved
            for (int oc : {4, 3}) {  // times interleaved
                for (int ih : {8}) {
                    for (int iw : {8}) {
                        for (int fh : {3, 5, 7}) {
                            for (int ph : {fh / 2, 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ih, iw, ic * interleaved,
                                            oc * interleaved, fh, fh, oh, ow,
                                            ph, ph, sh, sh, 1, 1, mode});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

template <typename Convolution>
bool TestConvolutionMmaReorderK(int interleaved = 32) {
    bool passed = true;

    double problem_alpha[] = {1.0};

    double problem_beta[] = {-1.3141413421};

    double problem_gamma[] = {1.0};

    Testbed<Convolution, true> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    for (int n : {128, 48, 33}) {
        for (int ic : {2, 3}) {      // times interleaved
            for (int oc : {4, 3}) {  // times interleaved
                for (int ih : {8}) {
                    for (int iw : {8}) {
                        for (int fh : {3, 5, 7}) {
                            for (int ph : {fh / 2, 0}) {
                                for (int sh : {1, 2}) {
                                    int oh = (ih + 2 * ph - fh) / sh + 1;
                                    int ow = (iw + 2 * ph - fh) / sh + 1;
                                    args.emplace_back(ConvolutionParameter{
                                            n, ih, iw, ic * interleaved,
                                            oc * interleaved, fh, fh, oh, ow,
                                            ph, ph, sh, sh, 1, 1, mode});
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Convolution, bool ReorderK = false>
bool TestConvolutionPerf(int iterations = 1, int batch = 64,
                         bool tensor_op = false, bool do_verify = true) {
    bool passed = true;

    double problem_alpha[] = {0.01234567};
    double problem_beta[] = {-1.07654321};
    double problem_gamma[] = {0.0};

    Testbed<Convolution, ReorderK> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    /// resnet-50
    args.emplace_back(ConvolutionParameter{batch, 224, 224, 4, 64, 7, 7, 112,
                                           112, 3, 3, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 64, 256, 1, 1, 56, 56,
                                           0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 256, 512, 1, 1, 28,
                                           28, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 256, 128, 1, 1, 28,
                                           28, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 128, 1, 1, 28,
                                           28, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 128, 512, 1, 1, 28,
                                           28, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 128, 128, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1, mode});

    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 1024, 1, 1, 14,
                                           14, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 256, 1, 1, 14,
                                           14, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 256, 1, 1, 14,
                                           14, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 256, 256, 3, 3, 14,
                                           14, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 256, 1024, 1, 1, 14,
                                           14, 0, 0, 1, 1, 1, 1, mode});

    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 2048, 1, 1, 7,
                                           7, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 512, 1, 1, 7, 7,
                                           0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 2048, 512, 1, 1, 7, 7,
                                           0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 512, 512, 3, 3, 7, 7, 1,
                                           1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 512, 2048, 1, 1, 7, 7,
                                           0, 0, 1, 1, 1, 1, mode});

    /// VGG-16
    args.emplace_back(ConvolutionParameter{batch, 224, 224, 64, 64, 3, 3, 224,
                                           224, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 112, 112, 64, 128, 3, 3, 112,
                                           112, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 112, 112, 128, 128, 3, 3, 112,
                                           112, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 128, 256, 3, 3, 56,
                                           56, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 256, 256, 3, 3, 56,
                                           56, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 256, 512, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 512, 3, 3, 28,
                                           28, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 512, 512, 3, 3, 14,
                                           14, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 512, 512, 3, 3, 7, 7, 1,
                                           1, 1, 1, 1, 1, mode});

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (tensor_op && (arg.C % 32 != 0 || arg.K % 32 != 0))
                        continue;
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

template <typename Convolution, bool ReorderK = false>
bool TestConvolution1x1Perf(int iterations = 1, int batch = 64,
                            bool tensor_op = false, bool do_verify = true) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {-1.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution, ReorderK> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    /// resnet-50
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 64, 256, 1, 1, 56, 56,
                                           0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 256, 512, 1, 1, 28,
                                           28, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 56, 56, 256, 128, 1, 1, 28,
                                           28, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 128, 1, 1, 28,
                                           28, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 128, 512, 1, 1, 28,
                                           28, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 1024, 1, 1, 14,
                                           14, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 28, 28, 512, 256, 1, 1, 14,
                                           14, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 256, 1, 1, 14,
                                           14, 0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 256, 1024, 1, 1, 14,
                                           14, 0, 0, 1, 1, 1, 1, mode});

    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 2048, 1, 1, 7,
                                           7, 0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 14, 14, 1024, 512, 1, 1, 7, 7,
                                           0, 0, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 2048, 512, 1, 1, 7, 7,
                                           0, 0, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 7, 7, 512, 2048, 1, 1, 7, 7,
                                           0, 0, 1, 1, 1, 1, mode});

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (tensor_op && (arg.C % 32 != 0 || arg.K % 32 != 0))
                        continue;
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

template <typename Convolution, bool ReorderK = false>
bool TestDetectionPerf(int iterations = 1, int batch = 16,
                       bool tensor_op = false, bool do_verify = true,
                       int alignment = 32) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {-1.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution, ReorderK> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{batch, 92, 160, 16, 16, 3, 3, 92,
                                           160, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 92, 160, 4, 16, 3, 3, 92, 160,
                                           1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 46, 80, 16, 16, 3, 3, 46, 80,
                                           1, 1, 1, 1, 1, 1, mode});

    args.emplace_back(ConvolutionParameter{batch, 184, 320, 32, 32, 1, 1, 184,
                                           320, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 184, 320, 32, 64, 1, 1, 92,
                                           160, 1, 1, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 184, 320, 32, 32, 3, 3, 184,
                                           320, 1, 1, 1, 1, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 184, 320, 32, 64, 3, 3, 92,
                                           160, 1, 1, 2, 2, 1, 1, mode});

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    if (tensor_op &&
                        (arg.C % alignment != 0 || arg.K % alignment != 0))
                        continue;
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

template <typename Convolution, bool ReorderK = false>
bool BenchFirstConvolution(int iterations = 1, int batch = 16,
                           bool tensor_op = false, bool do_verify = true) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {-1.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution, ReorderK> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{batch, 224, 224, 4, 64, 7, 7, 112,
                                           112, 3, 3, 2, 2, 1, 1, mode});
    args.emplace_back(ConvolutionParameter{batch, 768, 1280, 4, 16, 3, 3, 384,
                                           640, 1, 1, 2, 2, 1, 1, mode});

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

template <typename Convolution>
bool TestDepthwiseConvolution() {
    bool passed = true;

    double problem_alpha[] = {1.0};

    double problem_beta[] = {1.0};

    double problem_gamma[] = {1.0};

    Testbed<Convolution> testbed;

    using ElementCompute =
            typename Convolution::EpilogueOutputOp::ElementCompute;

    using ConvolutionParameter = cutlass::conv::Conv2dProblemSize;
    std::vector<ConvolutionParameter> args;
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    args.emplace_back(ConvolutionParameter{128, 16, 16, 1, 1, 7, 7, 10, 10, 0,
                                           0, 1, 1, 1, 1, mode});

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
                                        sh, sh, 1, 1, mode});
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.run(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma));
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }
    return passed;
}

template <typename Convolution>
bool BenchDepthwiseConvolution(int batch = 64, int iterations = 1,
                               bool do_verify = true) {
    bool passed = true;

    double problem_alpha[] = {1.0};
    double problem_beta[] = {0.0};
    double problem_gamma[] = {0.0};

    Testbed<Convolution> testbed;

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
                                               mode});
    }

    bool verify = do_verify;
    int cnt = 0;
    for (auto arg : args) {
        for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
                for (auto gamma : problem_gamma) {
                    passed = testbed.perf(
                            arg, cutlass::from_real<ElementCompute>(alpha),
                            cutlass::from_real<ElementCompute>(beta),
                            cutlass::from_real<ElementCompute>(gamma),
                            iterations, verify);

                    cnt++;
                    if (cnt >= 5)
                        verify = false;
                    if (!passed) {
                        return false;
                    }
                }
            }
        }
    }

    return passed;
}

}  // namespace device
}  // namespace convolution
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
