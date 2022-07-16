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
/**
 * \file
 * test/unit/arch/device/memory.cu
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
/*! \file
    \brief Tests for global load/store
*/

#include "../../common/cutlass_unit_test.h"

#include "cutlass/arch/memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int LoadBytes>
__global__ void check_global_load_kernel(void* src, void* dst, bool guard,
                                         uint32_t pack_pad) {
    using AccessType = cutlass::Array<uint8_t, LoadBytes>;
    AccessType frag;  // registers
    cutlass::arch::global_load<AccessType, LoadBytes>(frag, src, guard,
                                                      pack_pad);
    // guard is always true here because this is test is for global_load
    cutlass::arch::global_store<AccessType, LoadBytes>(frag, dst, true);
}

uint32_t gen_uint32() {
    uint32_t res = 0;
    for (int i = 0; i < 32; i += 8) {
        uint8_t byte = uint8_t(rand() % 256);
        res |= (byte << i);
    }
    return res;
}

template <int LoadBytes>
void check_global_load(bool guard, uint32_t pack_pad) {
    uint32_t* host_src = reinterpret_cast<uint32_t*>(malloc(LoadBytes));
    uint32_t* host_dst = reinterpret_cast<uint32_t*>(malloc(LoadBytes));

    size_t nr_elems = LoadBytes / sizeof(uint32_t);
    if (nr_elems == 0)
        nr_elems = 1;

    for (size_t i = 0; i < nr_elems; i++) {
        host_src[i] = gen_uint32();
        host_dst[i] = gen_uint32();  // initiate dev_dst with random bits too
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    void *dev_src, *dev_dst;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dev_src, LoadBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dev_dst, LoadBytes));

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(dev_src, host_src, LoadBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(dev_dst, host_dst, LoadBytes, cudaMemcpyHostToDevice));

    check_global_load_kernel<LoadBytes>
            <<<1, 1>>>(dev_src, dev_dst, guard, pack_pad);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(host_dst, dev_dst, LoadBytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nr_elems; i++) {
        uint32_t result = host_dst[i];
        uint32_t expect = host_src[i];
        if (!guard) {
            expect = pack_pad;
        }
        if (LoadBytes == 1) {
            // only compare the first byte
            ASSERT_EQ(*reinterpret_cast<uint8_t*>(&result),
                      *reinterpret_cast<uint8_t*>(&expect));
        } else if (LoadBytes == 2) {
            // only compare the first two bytes
            ASSERT_EQ(*reinterpret_cast<uint16_t*>(&result),
                      *reinterpret_cast<uint16_t*>(&expect));
        } else {
            // compare four bytes
            ASSERT_TRUE(LoadBytes >= 4 and LoadBytes % 4 == 0);
            ASSERT_EQ(expect, result);
        }
    }
}

TEST(Memory, global_load) {
    std::vector<bool> guards = {true, false};
    std::vector<uint32_t> pack_pads;

    for (int i = 0; i < 20; i++) {
        pack_pads.push_back(gen_uint32());
    }

    for (bool guard : guards) {
        for (uint32_t pack_pad : pack_pads) {
            check_global_load<1>(guard, pack_pad);
            check_global_load<2>(guard, pack_pad);
            check_global_load<4>(guard, pack_pad);
            check_global_load<8>(guard, pack_pad);
            check_global_load<16>(guard, pack_pad);
            check_global_load<32>(guard, pack_pad);
            check_global_load<64>(guard, pack_pad);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
