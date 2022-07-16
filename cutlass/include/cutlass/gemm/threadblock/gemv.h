/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
/*! \file
    \brief Template for a threadblock-scoped GEMV kernel.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

namespace {
template <typename Op, typename T, int Threads>
struct Reduce;

template <typename T, int N, int Threads>
struct Reduce<plus<T>, Array<T, N>, Threads> {
    using AccessType = Array<T, N>;
    CUTLASS_DEVICE void operator()(int t, T* x, int stride = 1) const {
        int DIM_X = blockDim.x;
        AccessType* pointer = reinterpret_cast<AccessType*>(x);
        plus<Array<T, N>> _op;
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int i = (Threads >> 1); i >= 1; i >>= 1) {
            if (t < i) {
                pointer[t * stride] =
                        _op(pointer[t * stride], pointer[(t + i) * stride]);
            }
            if (i * DIM_X > 16)
                __syncthreads();
            else {
#if __CUDACC_VER_MAJOR__ >= 9
                __syncwarp();
#endif
            }
        }
    }
};
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix-vector product using SIMT math
/// instructions.
template <class Core_  //< GemvCore
          >
class Gemv {
public:
    using Shape = typename Core_::Shape;

    /// The MMA operator that computes GEMV
    using Operator = typename Core_::Operator;

    /// Iterates over A in global memory
    using IteratorA = typename Core_::IteratorA;

    /// Iterates over B in global memory
    using IteratorB = typename Core_::IteratorB;

    /// Fragment of operand C loaded from global memory
    using IteratorC = typename Core_::IteratorC;

    /// Fragment of operand A loaded from global memory
    using FragmentA = typename IteratorA::Fragment;

    /// Fragment of operand B loaded from global memory
    using FragmentB = typename IteratorB::Fragment;

    /// Fragment of operand accumulator loaded/stored to global memory
    using FragmentC = typename Operator::FragmentC;

    /// Shape of the per-thread GEMV operation
    using ThreadShape = typename Core_::ThreadShape;

public:
    CUTLASS_DEVICE
    Gemv() {}

    CUTLASS_DEVICE
    void operator()(
            GemmCoord const& problem_size,  ///< problem size of batched GEMV
            FragmentC& accum,               ///< destination accumulator tile
            IteratorA iterator_A,           ///< iterator over A operand in
                                            ///< global memory
            IteratorB iterator_B,           ///< iterator over B operand in
                                            ///< global memory
            FragmentC const& src_accum) {   ///< source accumualtor tile

        //
        // Prologue
        //

        FragmentA frag_A;
        FragmentB frag_B;
        frag_A.clear();
        frag_B.clear();

        iterator_A.load(frag_A);
        iterator_B.load(frag_B);
        ++iterator_A;
        ++iterator_B;

        //
        // Mainloop
        //
        Operator thread_mma;
        int gemm_k = problem_size.k();

        // iterate over K to accumulate result
        CUTLASS_GEMM_LOOP
        for (; gemm_k > 0; gemm_k -= Shape::kK) {
            thread_mma(accum, frag_A, frag_B, accum);

            if (gemm_k > Shape::kK) {
                iterator_A.load(frag_A);
                iterator_B.load(frag_B);
                ++iterator_A;
                ++iterator_B;
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix-vector product using batched
/// reduction .
template <class Core_  //< GemvCore
          >
class GemvBatchedReduction {
public:
    using Shape = typename Core_::Shape;

    /// The MMA operator that computes GEMV
    using Operator = typename Core_::Operator;

    /// Iterates over A in global memory
    using IteratorA = typename Core_::IteratorA;

    /// Iterates over B in global memory
    using IteratorB = typename Core_::IteratorB;

    /// Fragment of operand C loaded from global memory
    using IteratorC = typename Core_::IteratorC;

    /// Fragment of operand A loaded from global memory
    using FragmentA = typename IteratorA::Fragment;

    /// Fragment of operand B loaded from global memory
    using FragmentB = typename IteratorB::Fragment;

    /// Fragment of operand accumulator loaded/stored to global memory
    using FragmentC = typename Operator::FragmentC;

    /// Shape of the per-thread GEMV operation
    using ThreadShape = typename Core_::ThreadShape;

    /// Tensor ref for accumulator
    using TensorRefAccumulator =
            TensorRef<typename Core_::ElementC, typename Core_::LayoutC>;

    /// ThreadMap used by threadblock-scoped reduction
    using ReductionThreadMap = cutlass::transform::
            PitchLinear2DTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<Shape::kN,
                                             Shape::kK / ThreadShape::kK>,
                    typename Core_::ThreadArrangement, ThreadShape::kN>;

    /// Tile iterator for storing accumulator into shared memory
    using SmemTileIteratorAccumulator =
            cutlass::transform::threadblock::RegularTileIterator<
                    MatrixShape<Shape::kK / ThreadShape::kK, Shape::kN>,
                    typename Core_::ElementC, layout::RowMajor, 0,
                    ReductionThreadMap>;

    //
    // Nested structs
    //

    /// Shared storage object needed by threadblock-scoped GEMV Batched
    /// Reduction

    class SharedStorage {
    public:
        //
        // Type definitions
        //

        /// Shape of the accumulator in shared memory
        using ShapeAccumulator =
                MatrixShape<Shape::kK / ThreadShape::kK, Shape::kN>;

    public:
        //
        // Data members
        //

        /// Buffer for accumulator
        AlignedBuffer<typename Core_::ElementC, ShapeAccumulator::kCount>
                buffer_accumulator;

    public:
        //
        // Methods
        //

        /// Returns a layout object for the accumulator matrix
        CUTLASS_HOST_DEVICE
        static typename Core_::LayoutC LayoutAccumulator() {
            return Core_::LayoutC::packed(
                    {ShapeAccumulator::kRow, ShapeAccumulator::kColumn});
        }

        /// Returns a TensorRef to the accumulator
        CUTLASS_HOST_DEVICE
        TensorRefAccumulator buffer_ref() {
            return TensorRefAccumulator{buffer_accumulator.data(),
                                        LayoutAccumulator()};
        }
    };

public:
    CUTLASS_DEVICE
    GemvBatchedReduction() {}

    CUTLASS_DEVICE
    void operator()(
            GemmCoord const& problem_size,  ///< problem size of batched GEMV
            FragmentC& accum,               ///< destination accumulator tile
            IteratorA iterator_A,           ///< iterator over A operand in
                                            ///< global memory
            IteratorB iterator_B,           ///< iterator over B operand in
                                            ///< global memory
            FragmentC const& src_accum) {   ///< source accumualtor tile

        //
        // Prologue
        //

        FragmentA frag_A;
        FragmentB frag_B;
        frag_A.clear();
        frag_B.clear();

        iterator_A.load(frag_A);
        iterator_B.load(frag_B);
        ++iterator_A;
        ++iterator_B;

        //
        // Mainloop
        //
        Operator thread_mma;
        int gemm_k = problem_size.k();

        // iterate over K to accumulate result
        CUTLASS_GEMM_LOOP
        for (; gemm_k > 0; gemm_k -= Shape::kK) {
            thread_mma(accum, frag_A, frag_B, accum);

            if (gemm_k > Shape::kK) {
                iterator_A.load(frag_A);
                iterator_B.load(frag_B);
                ++iterator_A;
                ++iterator_B;
            }
        }

        // threadblock-scoped reduction
        if (Shape::kK > ThreadShape::kK) {
            // Dynamic shared memory base pointer
            extern __shared__ int SharedStorageBase[];

            // Declare pointer to dynamic shared memory.
            SharedStorage& shared_storage =
                    *(reinterpret_cast<SharedStorage*>(SharedStorageBase));
            SmemTileIteratorAccumulator smem_tile_iterator(
                    shared_storage.buffer_ref(), threadIdx.x);
            smem_tile_iterator.store(accum);

            using Reduction = Reduce<plus<typename Core_::ElementC>, FragmentC,
                                     Core_::kThreadsPerK>;
            Reduction reduce_op;
            auto stride = SharedStorage::LayoutAccumulator().stride(0) /
                          FragmentC::kElements;
            typename Core_::ElementC* pointer =
                    shared_storage.buffer_ref().data();
            reduce_op(threadIdx.y, &pointer[threadIdx.x * FragmentC::kElements],
                      stride);
            if (threadIdx.y == 0)
                smem_tile_iterator.load(accum);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
