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

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/convolution/device/convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and
// computation between elements
using ElementAccumulator = float;  // Data type of accumulator
using ElementComputeEpilogue =
        float;                // Data type of epilogue computation (alpha, beta)
using ElementSrc = float;     // Data type of elements in src tensor
using ElementFilter = float;  // Data type of elements in filter tensor
using ElementDst = float;     // Data type of elements in output tensor

using LayoutSrc = cutlass::layout::TensorNCHW;
using LayoutFilter = cutlass::layout::TensorNCHW;
using LayoutDst = cutlass::layout::TensorNCHW;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassSimt;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm50;

// This code section describes the tile size a thread block will compute
using ThreadblockShape =
        cutlass::gemm::GemmShape<64, 128, 8>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;  // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape =
        cutlass::gemm::GemmShape<1, 1, 1>;  // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
        cutlass::conv::threadblock::DepthwiseConvolutionFpropThreadblockSwizzle;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// This code section describes the epilogue part of the kernel, we use default
// value
using EpilogueOp = cutlass::epilogue::thread::BiasAddLinearCombination<
        ElementDst,               // Data type of output matrix.
        1, ElementAccumulator,    // Data type of accumulator
        ElementDst,               // Data type of bias
        ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                  // combination

using Convolution = cutlass::conv::device::Convolution<
        ElementSrc, LayoutSrc, ElementFilter, LayoutFilter, ElementDst,
        LayoutDst, ElementDst, LayoutDst, ElementDst,
        cutlass::conv::ConvType::kDepthwiseConvolution, MMAOp, SmArch,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
        SwizzleThreadBlock, NumStages, 1, 1,
        cutlass::conv::SpecialOptimizeDesc::NONE, cutlass::arch::OpMultiplyAdd,
        cutlass::conv::ImplicitGemmMode::GEMM_TN>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {
    bool help;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    bool reference_check;
    bool measure_performance;
    int iterations;
    bool save_workspace;
    ElementComputeEpilogue alpha;
    ElementComputeEpilogue beta;
    bool benchmark;
    std::string tag;

    Options()
            : help(false),
              input_size(1, 32, 32, 32),
              filter_size(32, 3, 3, 1),
              padding(1, 1, 1, 1),
              conv_stride(1, 1),
              dilation(1, 1),
              reference_check(false),
              measure_performance(true),
              iterations(1000),
              save_workspace(false),
              alpha(1),
              beta(0),
              benchmark(false) {}

    // Verify the problem size is compatible with the CUTLASS Convolution
    // implementation.
    bool valid() {
        int const kAlignment = 1;

        if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
            // misaligned tensors
            return false;
        }

        // Invalid padding
        if ((padding.h() != filter_size.h() / 2) ||
            (padding.w() != filter_size.w() / 2)) {
            return false;
        }

        return true;
    }

    /// Updates input and filter sizes
    void update(cutlass::Tensor4DCoord input_size,
                cutlass::Tensor4DCoord filter_size) {
        this->input_size = input_size;
        this->filter_size = filter_size;

        padding.n() = filter_size.h() / 2;
        padding.h() = filter_size.h() / 2;
        padding.w() = filter_size.w() / 2;
        padding.c() = filter_size.w() / 2;
    }

    // Parses the command line
    void parse(int argc, char const** args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
        }

        if (cmd.check_cmd_line_flag("ref-check")) {
            reference_check = true;
        }

        if (cmd.check_cmd_line_flag("perf-check")) {
            measure_performance = true;
        }

        if (cmd.check_cmd_line_flag("save-workspace")) {
            save_workspace = true;
        }

        if (cmd.check_cmd_line_flag("benchmark")) {
            benchmark = true;
        }

        cmd.get_cmd_line_argument("n", input_size.n());
        cmd.get_cmd_line_argument("h", input_size.h());
        cmd.get_cmd_line_argument("w", input_size.w());
        cmd.get_cmd_line_argument("g", input_size.c());

        filter_size.n() = input_size.c();
        cmd.get_cmd_line_argument("r", filter_size.h());
        cmd.get_cmd_line_argument("s", filter_size.w());
        filter_size.c() = input_size.c();

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);

        cmd.get_cmd_line_argument("iterations", iterations);
        cmd.get_cmd_line_argument("tag", tag);

        padding.n() = filter_size.h() / 2;
        padding.h() = filter_size.h() / 2;
        padding.w() = filter_size.w() / 2;
        padding.c() = filter_size.w() / 2;
    }

    /// Prints the usage statement.
    std::ostream& print_usage(std::ostream& out) const {
        out << "16_large_depthwise_conv2dfprop example\n\n"
            << "  This example uses Large Kernel Depthwise Convolution on FP32 "
               "data"
               "types to compute\n"
            << "  forward convolution on tensors of layout NCHW.\n\n"
            << "Options:\n\n"
            << "  --help               If specified, displays this usage "
               "statement.\n\n"
            << "  --n <int>            Input tensor extent N\n"
            << "  --h <int>            Input tensor extent H\n"
            << "  --w <int>            Input tensor extent W\n"
            << "  --g <int>            Group of depthwise conv2d, a.k.a input "
               "tensor extent C, a.k.a output tensor extent K\n"
            << "  --r <int>            Filter extent R\n"
            << "  --s <int>            Filter extent S\n\n"
            << "  --alpha <float>      Epilogue scalar alpha\n"
            << "  --beta <float>       Epilogue scalar beta\n\n"
            << "  --ref-check          If set (true), reference check on the "
               "host is computed\n"
            << "  --perf-check         If set (true), performance is "
               "measured.\n"
            << "  --benchmark          If set (true), performance benchmarking "
               "on several layers and batch-size.\n"
            << "  --iterations <int>   Number of profiling iterations to "
               "perform.\n"
            << "  --save-workspace     If set, workspace is written to a text "
               "file.\n"
            << "  --tag <string>       String to replicate across the first "
               "column in the results table\n";

        out << "\n\nExamples:\n\n"
            << "$ "
               "./examples/16_large_depthwise_conv2dfprop/"
               "16_large_depthwise_conv2dfprop  --n=64 --h=32 --w=32 --g=384 "
               "--r=1 --s=1\n\n"
            << "$ "
               "./examples/16_large_depthwise_conv2dfprop/"
               "16_large_depthwise_conv2dfprop  --n=64 --h=32 --w=32 --g=384 "
               "--r=31 --s=31 --ref-check\n\n";

        return out;
    }

    /// Computes the output tensor size (NPQK)
    cutlass::Tensor4DCoord output_size() const {
        return cutlass::Tensor4DCoord(
                input_size.n(),
                (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                conv_stride.row() +
                        1,
                (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                conv_stride.column() +
                        1,
                filter_size.n());
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const {
        // Number of multiply-adds = NPQK * CRS / K
        int64_t fmas =
                output_size().product() *
                int64_t(filter_size.h() * filter_size.w() * filter_size.c()) /
                output_size().c();

        // Two flops per multiply-add
        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
    double runtime_ms;
    double gflops;
    cutlass::Status status;
    cutlass::Status reference_check;
    cudaError_t error;

    Result()
            : runtime_ms(0),
              gflops(0),
              status(cutlass::Status::kSuccess),
              reference_check(cutlass::Status::kInvalid),
              error(cudaSuccess) {}

    static std::ostream& print_header(std::ostream& out,
                                      Options const& options) {
        if (!options.tag.empty()) {
            out << "Name,";
        }

        out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

        return out;
    }

    std::ostream& print(std::ostream& out, int idx, Options const& options) {
        if (!options.tag.empty()) {
            out << options.tag << ",";
        }

        out << "conv_" << idx << "," << options.input_size.n() << ","
            << options.input_size.h() << "," << options.input_size.w() << ","
            << options.input_size.c() << "," << options.filter_size.n() << ","
            << options.filter_size.h() << "," << options.filter_size.w() << ","
            << runtime_ms << "," << gflops;

        return out;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const& options) {
    Result result;

    //
    // Allocate host-device tensors using the CUTLASS Utilities.
    //

    cutlass::HostTensor<ElementSrc, LayoutSrc> tensor_src(options.input_size);
    cutlass::HostTensor<ElementFilter, LayoutFilter> tensor_filter(
            options.filter_size);
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_bias;
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_z(options.output_size());
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_dst(options.output_size());
    cutlass::HostTensor<typename Convolution::ElementDst,
                        typename Convolution::LayoutDst>
            tensor_ref_dst(options.output_size());

    //
    // Initialize tensors
    //

    // Fill tensor src on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_src.host_view(), 1, ElementSrc(7), ElementSrc(-8), 0);

    // Fill tensor filter on host with uniform-distribution random data
    cutlass::reference::host::TensorFillRandomUniform(tensor_filter.host_view(),
                                                      1, ElementFilter(7),
                                                      ElementFilter(-8), 0);

    // Fill tensor dst on host with zeros
    cutlass::reference::host::TensorFill(tensor_dst.host_view());

    // Fill tensor C for reference on host with zeros
    cutlass::reference::host::TensorFill(tensor_ref_dst.host_view());

    // Copy data from host to GPU
    tensor_src.sync_device();
    tensor_filter.sync_device();
    tensor_dst.sync_device();
    tensor_ref_dst.sync_device();

    //
    // Define arguments for CUTLASS Convolution
    //

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    typename Convolution::Arguments arguments{
            {options.input_size, options.filter_size, options.padding,
             options.conv_stride, options.dilation, options.output_size(), mode,
             split_k_slices, options.filter_size.n()},
            tensor_src.device_ref(),
            tensor_filter.device_ref(),
            tensor_bias.device_ref(),
            tensor_z.device_ref(),
            tensor_dst.device_ref(),
            {options.alpha, 0, options.beta}};

    //
    // Initialize CUTLASS Convolution
    //

    Convolution conv_op;

    size_t workspace_size = conv_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    result.status = conv_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(result.status);

    //
    // Launch initialized CUTLASS kernel
    //
    result.status = conv_op();

    CUTLASS_CHECK(result.status);

    //
    // Optional reference check
    //

    if (options.reference_check) {
        std::cout << "Verification on host...\n";

        cutlass::conv::Conv2dProblemSize problem_size(
                options.input_size, options.filter_size, options.padding,
                options.conv_stride, options.dilation, mode, 1,
                options.filter_size.n());

        // Compute with reference implementation
        cutlass::reference::host::Convolution<
                Convolution::kConvolutionType, typename Convolution::ElementSrc,
                typename Convolution::LayoutSrc,
                typename Convolution::ElementFilter,
                typename Convolution::LayoutFilter,
                typename Convolution::ElementDst,
                typename Convolution::LayoutDst,
                typename Convolution::ElementDst,
                typename Convolution::LayoutDst, ElementComputeEpilogue,
                ElementAccumulator, typename Convolution::Operator>
                reference_convolution;

        reference_convolution(problem_size, options.alpha,
                              tensor_src.host_ref(), tensor_filter.host_ref(),
                              0, tensor_bias.host_ref(), options.beta,
                              tensor_z.host_ref(), tensor_ref_dst.host_ref(),
                              ElementAccumulator(0));

        // Check if output from CUTLASS kernel and reference kernel are equal or
        // not
        tensor_ref_dst.sync_host();

        bool passed = cutlass::reference::host::TensorEquals(
                tensor_dst.host_view(), tensor_ref_dst.host_view());

        if (!passed) {
            result.reference_check = cutlass::Status::kErrorInternal;
            std::cout << "ERROR - results miscompared.\n";
        } else {
            result.reference_check = cutlass::Status::kSuccess;
            std::cout << "Passed.\n";
        }
    } else {
        result.reference_check = cutlass::Status::kInvalid;
    }

    if (options.save_workspace) {
        std::stringstream ss;

        ss << "16_workspace_large_depthwise_conv2dfprop_"
           << options.input_size.n() << "x" << options.input_size.h() << "x"
           << options.input_size.w() << "x" << options.input_size.c() << "_"
           << options.filter_size.n() << "x" << options.filter_size.h() << "x"
           << options.filter_size.w() << "x" << options.filter_size.c()
           << ".dat";

        std::ofstream output_workspace(ss.str());

        output_workspace << "Input = \n"
                         << tensor_src.host_view() << "\n\n"
                         << "Filters = \n"
                         << tensor_filter.host_view() << "\n\n";

        if (options.reference_check) {
            output_workspace << "Reference = \n"
                             << tensor_ref_dst.host_view() << "\n\n";
        }

        output_workspace << "Computed = \n"
                         << tensor_dst.host_view() << std::endl;

        std::cout << "Results written to '" << ss.str() << "'." << std::endl;
    }

    //
    // Performance measurement
    //

    if (options.measure_performance) {
        cudaEvent_t events[2];

        for (auto& event : events) {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess) {
                std::cerr << "cudaEventCreate() failed: "
                          << cudaGetErrorString(result.error) << std::endl;
                return result;
            }
        }

        // Record an event at the start of a series of convolution operations.
        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Launch a sequence of implicit GEMM operations on the device
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            result.status = conv_op();
            CUTLASS_CHECK(result.status);
        }

        // Record an event when the convolutions have been launched.
        result.error = cudaEventRecord(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Wait for work on the device to complete.
        result.error = cudaEventSynchronize(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventSynchronize() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Measure elapsed runtime
        float runtime_ms = 0;
        result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventElapsed() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Print average runtime and GFLOPs.
        result.runtime_ms = double(runtime_ms) / double(options.iterations);
        result.gflops = options.gflops(result.runtime_ms / 1000.0);

        // Cleanup
        for (auto event : events) {
            (void)cudaEventDestroy(event);
        }
    }

    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** args) {
    bool notSupported = false;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    if (!(props.major > 5 || (props.major == 5 && props.minor >= 0))) {
        std::cerr << "This example (16_large_depthwise_conv2dfprop) must be run "
                     "on a machine with compute "
                     "capability at least 50."
                  << std::endl;
        notSupported = true;
    }

    if (notSupported) {
        return 0;
    }

    Options options;

    options.parse(argc, args);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    if (options.benchmark) {
        // Benchmark several layers

        int batch_sizes[] = {1, 64};

        struct Benchmark {
            int h, w, c, k, r, s;
        } layers[] = {
                {32, 32, 512, 512, 31, 31}, {24, 24, 512, 512, 23, 23},
                {24, 24, 512, 512, 27, 27}, {56, 56, 512, 512, 31, 31},
                {96, 96, 512, 512, 31, 31}, {32, 32, 384, 384, 21, 21},
                {32, 32, 384, 384, 17, 17},
        };

        Result::print_header(std::cout, options) << std::endl;

        int idx = 1;

        for (auto const& layer : layers) {
            for (auto N : batch_sizes) {
                options.update({N, layer.h, layer.w, layer.c},
                               {layer.k, layer.r, layer.s, layer.c});

                Result result = profile_convolution(options);
                result.print(std::cout, idx, options) << std::endl;
            }

            ++idx;
        }
    } else {
        // Execute one problem size
        if (!options.valid()) {
            std::cerr << "Invalid problem." << std::endl;
            return -1;
        }

        Result result = profile_convolution(options);

        Result::print_header(std::cout, options) << std::endl;
        result.print(std::cout, 1, options) << std::endl;
    }

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
