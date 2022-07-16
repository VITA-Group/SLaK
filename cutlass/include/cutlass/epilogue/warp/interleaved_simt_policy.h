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
/*! \file
    \brief Defines basic structures needed for implementing the warp-scoped
   phase of the epilogue. These quantities assume a 'column-major' arrangement
   of SimtOp instructions, of which a row-oriented slice is visible per
   iteration.
*/

/**
 * \file include/cutlass/epilogue/warp/interleaved_simt_policy.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        typename WarpShape,  ///< shape of warp-level GEMM (concept: GemmShape)
        typename Operator,   ///< matrix multiply operation (concept: arch::Mma)
        typename Layout,     ///< destination layout in shared memory
        typename MmaSimtPolicy  ///< policy defining lane arrangement (concept:
                                ///< MmaSimtPolicy)
        >
struct InterleavedSimtPolicy;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major in shared memory and CxRSKx layout in
/// global memory
template <typename WarpShape_,     ///< shape of warp-level GEMM (concept:
                                   ///< MatrixShape)
          typename Operator_,      ///< matrix multiply operation (concept:
                                   ///< arch::Mma)
          typename MmaSimtPolicy_  ///< policy defining lane arrangement
                                   ///< (concept: MmaSimtPolicy)
          >
struct InterleavedSimtPolicy<WarpShape_, Operator_, layout::RowMajor,
                             MmaSimtPolicy_> {
    using WarpShape = WarpShape_;
    using Operator = Operator_;
    using MmaSimtPolicy = MmaSimtPolicy_;

    static_assert(!(WarpShape::kM % (MmaSimtPolicy::WarpShape::kRow *
                                     MmaSimtPolicy::LaneMmaShape::kM)),
                  "Divisibility");
    static_assert(!(WarpShape::kN % (MmaSimtPolicy::WarpShape::kColumn *
                                     MmaSimtPolicy::LaneMmaShape::kN)),
                  "Divisibility");

    using Shape = MatrixShape<MmaSimtPolicy::WarpShape::kRow *
                                      MmaSimtPolicy::LaneMmaShape::kM,
                              WarpShape::kN>;

    using Iterations = MatrixShape<WarpShape::kM / Shape::kRow,
                                   WarpShape::kN / Shape::kColumn>;

    /// Number of iterations
    static int const kIterations = Iterations::kCount;

    /// Number of accumulators written per iteration
    static int const kElementsPerIteration =
            (MmaSimtPolicy::LaneMmaShape::kM * WarpShape::kN /
             MmaSimtPolicy::WarpShape::kColumn);

    /// Total number of accumulators
    static int const kAccumulatorElementCount =
            kElementsPerIteration * kIterations;

    /// Number of consecutive elements
    static int const kElementsPerAccess = MmaSimtPolicy::LaneMmaShape::kN;

    /// Number of rows per epilogue iteration
    static int const kRowsPerIteration =
            MmaSimtPolicy::WarpShape::kRow * MmaSimtPolicy::LaneMmaShape::kM;

    /// Number of accesses made in one iteration
    static int const kAccessesPerIteration =
            kElementsPerIteration / kElementsPerAccess;

    static int const kRowAccessesPerIteration = MmaSimtPolicy::LaneMmaShape::kM;
    static int const kColumnAccessesPerIteration =
            kAccessesPerIteration / kRowAccessesPerIteration;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major in shared memory
template <typename WarpShape_,     ///< shape of warp-level GEMM (concept:
                                   ///< MatrixShape)
          typename Operator_,      ///< matrix multiply operation (concept:
                                   ///< arch::Mma)
          typename MmaSimtPolicy_  ///< policy defining lane arrangement
                                   ///< (concept: MmaSimtPolicy)
          >
struct InterleavedSimtPolicy<WarpShape_, Operator_, layout::ColumnMajor,
                             MmaSimtPolicy_> {
    using WarpShape = WarpShape_;
    using Operator = Operator_;
    using MmaSimtPolicy = MmaSimtPolicy_;

    static_assert(!(WarpShape::kM % (MmaSimtPolicy::WarpShape::kRow *
                                     MmaSimtPolicy::LaneMmaShape::kM)),
                  "Divisibility");
    static_assert(!(WarpShape::kN % (MmaSimtPolicy::WarpShape::kColumn *
                                     MmaSimtPolicy::LaneMmaShape::kN)),
                  "Divisibility");

    using Shape =
            MatrixShape<WarpShape::kM, MmaSimtPolicy::WarpShape::kColumn *
                                               MmaSimtPolicy::LaneMmaShape::kN>;

    using Iterations = MatrixShape<WarpShape::kM / Shape::kRow,
                                   WarpShape::kN / Shape::kColumn>;

    /// Number of iterations
    static int const kIterations = Iterations::kCount;

    /// Number of accumulators written per iteration
    static int const kElementsPerIteration =
            (MmaSimtPolicy::LaneMmaShape::kN * WarpShape::kM /
             MmaSimtPolicy::WarpShape::kRow);

    /// Total number of accumulators
    static int const kAccumulatorElementCount =
            kElementsPerIteration * kIterations;

    /// Number of consecutive elements
    static int const kElementsPerAccess = MmaSimtPolicy::LaneMmaShape::kM;

    /// Number of accesses made in one iteration
    static int const kAccessesPerIteration =
            kElementsPerIteration / kElementsPerAccess;

    static int const kColumnAccessesPerIteration =
            MmaSimtPolicy::LaneMmaShape::kN;

    static int const kRowAccessesPerIteration =
            kAccessesPerIteration / kColumnAccessesPerIteration;

    /// Number of columns per epilogue iteration
    static int const kColumnsPerIteration = MmaSimtPolicy::WarpShape::kColumn *
                                            MmaSimtPolicy::LaneMmaShape::kN;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
