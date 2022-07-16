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
    \brief Unit tests for thread-level GEMM
*/

#include <fstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"

#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue_simt.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "bias_add_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued Integer  tests
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_32x64_32x64x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<32, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<4, 8>,
                    cutlass::layout::RowMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_128x128_32x64x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<4, 8>,
                    cutlass::layout::RowMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_32x128_32x64x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<32, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<4, 8>,
                    cutlass::layout::RowMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_64x128_32x64x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<64, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<4, 8>,
                    cutlass::layout::RowMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_128x64_32x64x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<128, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<4, 8>,
                    cutlass::layout::RowMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_128x32_64x32x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<128, 32, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<8, 4>,
                    cutlass::layout::ColumnMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

TEST(SM61_Bias_add_epilogue_threadblock_C4RSK4, simt_i8_64x32_64x32x8) {
    //
    // Define the warp-level matrix multiply
    //

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int;
    using ElementOutput = int8_t;
    using ElementAccumulator = int;
    using ElementBias = int;
    using ElementCompute = float;

    int const kElementsPerAccess = 4;

    using Shape = cutlass::gemm::GemmShape<64, 32, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
    using ElementC = ElementAccumulator;
    using LayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
    using LayoutB = cutlass::layout::RowMajorInterleaved<4>;
    using LayoutC = cutlass::layout::RowMajor;

    using WarpMmaSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            cutlass::gemm::warp::MmaSimtPolicy<
                    cutlass::MatrixShape<8, 4>,
                    cutlass::layout::ColumnMajorInterleaved<2>,
                    cutlass::gemm::GemmShape<4, 4, 4> > >;

    //
    // Output operator
    //

    using OutputOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
            ElementOutput, kElementsPerAccess, ElementAccumulator, ElementBias,
            ElementCompute>;

    //
    // Define the epilogue
    //

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    Shape, cutlass::layout::TensorCxRSKx<4>,
                    cutlass::layout::TensorCxRSKx<4>, WarpMmaSimt, OutputOp,
                    kElementsPerAccess>::Epilogue;

    //
    // Instantiate epilogue
    //

    EpilogueTestbed<Epilogue> testbed;

    bool passed = testbed.run_all();

    EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
