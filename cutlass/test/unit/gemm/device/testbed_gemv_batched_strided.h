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
 * \file test/unit/gemm/device/testbed_gemv_batched_strided.cu
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

#include "cutlass/gemm/kernel/gemv_batched_strided.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel>
struct TestbedGemvBatchedStrided {
    using ElementAccumulator = typename GemvKernel::ElementAccumulator;
    using ElementCompute = ElementAccumulator;

    /// Initialization
    cutlass::Distribution::Kind init_A;
    cutlass::Distribution::Kind init_B;
    cutlass::Distribution::Kind init_C;
    uint64_t seed;

    cutlass::HostTensor<typename GemvKernel::ElementA,
                        typename GemvKernel::LayoutA>
            tensor_A;
    cutlass::HostTensor<typename GemvKernel::ElementB,
                        typename GemvKernel::LayoutB>
            tensor_B;
    cutlass::HostTensor<typename GemvKernel::ElementCD,
                        typename GemvKernel::LayoutCD>
            tensor_C;
    cutlass::HostTensor<typename GemvKernel::ElementCD,
                        typename GemvKernel::LayoutCD>
            tensor_D;
    cutlass::HostTensor<typename GemvKernel::ElementCD,
                        typename GemvKernel::LayoutCD>
            reference_D;

    //
    // Methods
    //

    TestbedGemvBatchedStrided(cutlass::Distribution::Kind init_A_ =
                                      cutlass::Distribution::Uniform,
                              cutlass::Distribution::Kind init_B_ =
                                      cutlass::Distribution::Uniform,
                              cutlass::Distribution::Kind init_C_ =
                                      cutlass::Distribution::Uniform,
                              uint64_t seed_ = 2080)
            : init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {}

    /// Helper to initialize a tensor view
    template <typename Element, typename Layout>
    bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                           cutlass::Distribution::Kind dist_kind,
                           uint64_t seed) {
        if (dist_kind == cutlass::Distribution::Uniform) {
            double scope_max, scope_min;
            int bits_input = cutlass::sizeof_bits<Element>::value;
            int bits_output =
                    cutlass::sizeof_bits<typename GemvKernel::ElementCD>::value;

            if (bits_input == 1) {
                scope_max = 2;
                scope_min = 0;
            } else if (bits_input <= 8) {
                scope_max = 2;
                scope_min = -2;
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
    void initialize(cutlass::gemm::BatchedGemmCoord problem_size) {
        int m = problem_size.batch();
        int n = problem_size.n();
        int k = problem_size.k();

        //
        // Allocate the GEMM workspace
        //

        tensor_A.resize({m, k});
        tensor_B.resize({k, n});
        tensor_C.resize({m, n});
        tensor_D.resize({m, n});
        reference_D.resize({m, n}, false);

        EXPECT_TRUE(
                initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
        EXPECT_TRUE(
                initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
        EXPECT_TRUE(
                initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

        // It is possible to randomly initialize to all zeros, so override this
        // with non-zeros in the upper left corner of each operand.
        tensor_A.host_view().at({0, 0}) = typename GemvKernel::ElementA(1);
        tensor_B.host_view().at({0, 0}) = typename GemvKernel::ElementB(1);
        tensor_C.host_view().at({0, 0}) = typename GemvKernel::ElementCD(1);

        cutlass::reference::host::TensorCopy(reference_D.host_view(),
                                             tensor_C.host_view());

        tensor_A.sync_device();
        tensor_B.sync_device();
        tensor_C.sync_device();
        tensor_D.sync_device();
    }

    /// Compares computed reference with device reference and outputs to a file
    /// if incorrect
    bool compare_reference(cutlass::gemm::BatchedGemmCoord problem_size,
                           ElementCompute alpha, ElementCompute beta) {
        tensor_D.sync_host();

        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()),
                  0);
        EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()),
                  0);

        if (tensor_D.size() > 1)
            EXPECT_GT(
                    cutlass::reference::host::TensorNorm(tensor_D.host_view()),
                    0);

        if (reference_D.size() > 1)
            EXPECT_GT(cutlass::reference::host::TensorNorm(
                              reference_D.host_view()),
                      0);

        bool passed = cutlass::reference::host::TensorEquals(
                reference_D.host_view(), tensor_D.host_view());

        EXPECT_TRUE(passed);

        if (!passed) {
            std::stringstream fname;

            fname << "error_Gemv_device_" << problem_size.batch() << "x"
                  << problem_size.m() << "x" << problem_size.n() << "x"
                  << problem_size.k() << "_" << GemvKernel::ThreadBlockShape::kM
                  << "x" << GemvKernel::ThreadBlockShape::kN << "x"
                  << GemvKernel::ThreadBlockShape::kK << "_"
                  << GemvKernel::ThreadShape::kM << "x"
                  << GemvKernel::ThreadShape::kN << "x"
                  << GemvKernel::ThreadShape::kK << ".txt";

            std::ofstream file(fname.str());

            file << "problem: " << problem_size << ", alpha: " << alpha
                 << ", beta: " << beta << "\n\n";

            file << "A =\n"
                 << tensor_A.host_view() << "\nB =\n"
                 << tensor_B.host_view() << "\nC =\n"
                 << tensor_C.host_view() << "\n\nReference =\n"
                 << reference_D.host_view() << "\nComputed =\n"
                 << tensor_D.host_view();
        }

        return passed;
    }

    /// Verifies the result is a GEMM
    bool verify(cutlass::gemm::BatchedGemmCoord problem_size,
                ElementCompute alpha, ElementCompute beta) {
        int m = problem_size.batch();
        int n = problem_size.n();
        int k = problem_size.k();

        //
        // Verify
        //

        cutlass::reference::host::Gemm<
                typename GemvKernel::ElementA, typename GemvKernel::LayoutA,
                typename GemvKernel::ElementB, typename GemvKernel::LayoutB,
                typename GemvKernel::ElementCD, typename GemvKernel::LayoutCD,
                ElementCompute, ElementAccumulator>
                reference_gemm;

        reference_gemm({m, n, k}, alpha, tensor_A.host_ref(),
                       tensor_B.host_ref(), beta, reference_D.host_ref(),
                       ElementAccumulator(0));

        return compare_reference(problem_size, alpha, beta);
    }

    /// Executes one test
    bool run(cutlass::gemm::BatchedGemmCoord problem_size,
             ElementCompute alpha = ElementCompute(1),
             ElementCompute beta = ElementCompute(0)) {
        this->initialize(problem_size);

        void (*kern)(cutlass::gemm::BatchedGemmCoord, ElementCompute,
                     ElementCompute, typename GemvKernel::IteratorA::TensorRef,
                     typename GemvKernel::IteratorA::TensorRef::LongIndex,
                     typename GemvKernel::IteratorB::TensorRef,
                     typename GemvKernel::IteratorB::TensorRef::LongIndex,
                     typename GemvKernel::IteratorCD::TensorRef,
                     typename GemvKernel::IteratorCD::TensorRef::LongIndex,
                     typename GemvKernel::IteratorCD::TensorRef,
                     typename GemvKernel::IteratorCD::TensorRef::LongIndex);

        static int constexpr kThreadsPerN = GemvKernel::Core::kThreadsPerN;
        static int constexpr kThreadsPerK = GemvKernel::Core::kThreadsPerK;

        kern = cutlass::gemm::kernel::GemvBatchedStrided<GemvKernel,
                                                         ElementCompute, false>;

        auto tile_size = cutlass::gemm::BatchedGemmCoord(
                GemvKernel::ThreadBlockShape::kM,
                GemvKernel::ThreadBlockShape::kN,
                GemvKernel::ThreadBlockShape::kK, 1);
        typename GemvKernel::ThreadBlockSwizzle swizzler;
        auto tiled_shape = swizzler.get_tiled_shape(problem_size, tile_size);
        dim3 grid = swizzler.get_grid_shape(tiled_shape);
        dim3 block(kThreadsPerN, kThreadsPerK, 1);
        int smem_size = int(
                sizeof(typename GemvKernel::ThreadBlockGemv::SharedStorage));
        int batch_stride_a = problem_size.k(), batch_stride_b = 0,
            batch_stride_c = problem_size.n();

        typename GemvKernel::IteratorA::TensorRef tensor_a{
                tensor_A.device_ref().data(),
                typename GemvKernel::LayoutA{problem_size.k()}};
        typename GemvKernel::IteratorB::TensorRef tensor_b{
                tensor_B.device_ref().data(),
                typename GemvKernel::LayoutB{problem_size.n()}};
        typename GemvKernel::IteratorCD::TensorRef tensor_c{
                tensor_C.device_ref().data(),
                typename GemvKernel::LayoutCD{problem_size.n()}};
        typename GemvKernel::IteratorCD::TensorRef tensor_d{
                tensor_D.device_ref().data(),
                typename GemvKernel::LayoutCD{problem_size.n()}};

        kern<<<grid, block, smem_size>>>(
                problem_size, alpha, beta, tensor_a, batch_stride_a, tensor_b,
                batch_stride_b, tensor_c, batch_stride_c, tensor_d,
                batch_stride_c);

        auto result = cudaGetLastError();

        bool passed = result == cudaSuccess;

        if (!passed)
            return passed;

        //
        // Verify
        //

        passed = this->verify(problem_size, alpha, beta);

        if (!passed) {
            std::cout << "Error with alpha = " << alpha << std::endl;
        }

        return passed;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel>
bool TestAllGemvKernel() {
    bool passed = true;

    int problem_size_m[] = {1, 4, 8, 16, 17};

    int problem_size_n[] = {128, 130, 256, 2304, 9};

    int problem_size_k[] = {64, 66, 128, 130, 256, 1024};

    double problem_alpha[] = {1.0};

    double problem_beta[] = {2.0};

    TestbedGemvBatchedStrided<GemvKernel> testbed;

    for (int m : problem_size_m) {
        for (int n : problem_size_n) {
            for (int k : problem_size_k) {
                for (auto alpha : problem_alpha) {
                    for (auto beta : problem_beta) {
                        using ElementAccumulator =
                                typename GemvKernel::ElementAccumulator;
                        using ElementCompute = ElementAccumulator;

                        cutlass::gemm::BatchedGemmCoord problem_size(1, n, k,
                                                                     m);

                        static int const LDG_K = GemvKernel::ThreadShape::kK;
                        static int const LDG_N = GemvKernel::ThreadShape::kN;
                        if (n % LDG_N != 0 || k % LDG_K != 0)
                            continue;

                        bool passed = testbed.run(
                                problem_size,
                                cutlass::from_real<ElementCompute>(alpha),
                                cutlass::from_real<ElementCompute>(beta));

                        if (!passed) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return passed;
}

}  // namespace device
}  // namespace gemm
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
