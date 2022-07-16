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
    \brief Unit tests for fast math: fast_divmod.
*/
#include <cstdlib>
#include <ctime>

#include "../common/cutlass_unit_test.h"

#include "cutlass/fast_math.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion template
template <typename data_t>
__global__ void divmod_kernel(data_t* src, int* div, int* quo, data_t* rem,
                              int N) {
    unsigned int mul, shr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
        cutlass::find_divisor(mul, shr, div[i]);
        cutlass::fast_divmod(quo[i], rem[i], src[i], div[i], mul, shr);
    }
}

/// Conversion template
template <typename data_t>
void divmod_host(data_t* src, int* div, int* quo, data_t* rem, int N) {
    for (int i = 0; i < N; i += 1) {
        unsigned int mul, shr;
        cutlass::find_divisor(mul, shr, div[i]);
        cutlass::fast_divmod(quo[i], rem[i], src[i], div[i], mul, shr);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace core
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
void CalculateDivMod_device() {
    using ITensor = cutlass::HostTensor<int, cutlass::layout::RowMajor>;
    using ETensor = cutlass::HostTensor<Element, cutlass::layout::RowMajor>;
    static const int Length = (1 << 20);
    static const int Extent = (1 << 20);

    srand((unsigned)time(NULL));

    ETensor Src({1, Length});
    ITensor Div({1, Length});
    ITensor Quo({1, Length});
    ETensor Rem({1, Length});
    ITensor Quo_gold({1, Length});
    ETensor Rem_gold({1, Length});

    for (int i = 0; i < Length; ++i) {
        Src.host_data()[i] = (Element)(rand() % Extent);
        Div.host_data()[i] = (rand() % Extent + 1);
        Quo.host_data()[i] = 0;
        Rem.host_data()[i] = (Element)0;
        Quo_gold.host_data()[i] = int(Src.host_data()[i] / Div.host_data()[i]);
        Rem_gold.host_data()[i] = Src.host_data()[i] -
                                  Quo_gold.host_data()[i] * Div.host_data()[i];
    }

    Rem.sync_device();
    Quo.sync_device();
    Div.sync_device();
    Src.sync_device();

    int block = 256;
    int grid = (Length + block - 1) / block > 256
                       ? 256
                       : (Length + block - 1) / block;

    test::core::kernel::divmod_kernel<Element><<<grid, block>>>(
            reinterpret_cast<Element*>(Src.device_data()),
            reinterpret_cast<int*>(Div.device_data()),
            reinterpret_cast<int*>(Quo.device_data()),
            reinterpret_cast<Element*>(Rem.device_data()), Length);

    Quo.sync_host();
    Rem.sync_host();

    for (int i = 0; i < Length; ++i) {
        int quo_gold = Quo_gold.host_data()[i];
        Element rem_gold = Rem_gold.host_data()[i];
        int quo = Quo.host_data()[i];
        Element rem = Rem.host_data()[i];

        EXPECT_TRUE(quo_gold == quo);
        EXPECT_TRUE(rem_gold == rem);
    }
}

TEST(CalculateDivMod_device, int) {
    CalculateDivMod_device<int>();
}

TEST(CalculateDivMod_device, int64_t) {
    CalculateDivMod_device<int64_t>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
void CalculateDivMod_host() {
    using ITensor = cutlass::HostTensor<int, cutlass::layout::RowMajor>;
    using ETensor = cutlass::HostTensor<Element, cutlass::layout::RowMajor>;
    static const int Length = (1 << 10);
    static const int Extent = (1 << 20);

    srand((unsigned)time(NULL));

    ETensor Src({1, Length});
    ITensor Div({1, Length});
    ITensor Quo({1, Length});
    ETensor Rem({1, Length});
    ITensor Quo_gold({1, Length});
    ETensor Rem_gold({1, Length});

    for (int i = 0; i < Length; ++i) {
        Src.host_data()[i] = (Element)(rand() % Extent);
        Div.host_data()[i] = (rand() % Extent + 1);
        Quo.host_data()[i] = 0;
        Rem.host_data()[i] = (Element)0;
        Quo_gold.host_data()[i] = int(Src.host_data()[i] / Div.host_data()[i]);
        Rem_gold.host_data()[i] = Src.host_data()[i] -
                                  Quo_gold.host_data()[i] * Div.host_data()[i];
    }

    test::core::kernel::divmod_host<Element>(
            reinterpret_cast<Element*>(Src.host_data()),
            reinterpret_cast<int*>(Div.host_data()),
            reinterpret_cast<int*>(Quo.host_data()),
            reinterpret_cast<Element*>(Rem.host_data()), Length);

    for (int i = 0; i < Length; ++i) {
        int quo_gold = Quo_gold.host_data()[i];
        Element rem_gold = Rem_gold.host_data()[i];
        int quo = Quo.host_data()[i];
        Element rem = Rem.host_data()[i];

        EXPECT_TRUE(quo_gold == quo);
        EXPECT_TRUE(rem_gold == rem);
    }
}

TEST(CalculateDivMod_host, int) {
    CalculateDivMod_host<int>();
}

TEST(CalculateDivMod_host, int64_t) {
    CalculateDivMod_host<int64_t>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
