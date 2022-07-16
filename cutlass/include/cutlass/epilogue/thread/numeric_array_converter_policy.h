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
  \brief Functor performing linear scaling operations used by epilogues. Values
  are clamped before converting to the output element type.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

////////////////////////////////////////////////////////////////////////////////
template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementBias_, typename ElementCompute_,
          FloatRoundStyle Round>
struct NumericArrayConverterPolicy {
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementBias = ElementBias_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    using SourceConverter =
            NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>;
    using AccumulatorConverter =
            NumericArrayConverter<ElementCompute, ElementAccumulator, kCount,
                                  Round>;
    using BiasConverter =
            NumericArrayConverter<ElementCompute, ElementBias, kCount, Round>;
    using OutputConverter =
            NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementBias_, typename ElementCompute_,
          FloatRoundStyle Round>
struct FastNumericArrayConverterPolicy {
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementBias = ElementBias_;
    using ElementCompute = ElementCompute_;

    static_assert(platform::is_same<ElementAccumulator, int>::value &&
                          platform::is_same<ElementCompute, float>::value,
                  "Fast conversion only support accululator type(int) and "
                  "compute type(float)");

    static int const kCount = Count;

    using SourceConverter =
            FastNumericArrayConverter<ElementCompute, ElementOutput, kCount,
                                      Round>;
    using AccumulatorConverter =
            FastNumericArrayConverter<ElementCompute, ElementAccumulator,
                                      kCount, Round>;
    using BiasConverter = FastNumericArrayConverter<ElementCompute, ElementBias,
                                                    kCount, Round>;
    using OutputConverter =
            FastNumericArrayConverter<ElementOutput, ElementCompute, kCount,
                                      Round>;
};
////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
