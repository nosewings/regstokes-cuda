#pragma once

#include "regstokes/detail/util.cuh"

namespace regstokes {
namespace soa {
namespace detail {

template<unsigned tile_dim, unsigned block_h, unsigned warp_size,
        bool specialize>
struct read {
    static
    __device__
    void _(
            float const * in1,
            float const * in2,
            float const * in3,
            float * out1,
            float *out2,
            float *out3)
    {
        auto thread_idx = threadIdx.y * tile_dim + threadIdx.x;
        constexpr auto block_size = block_h * tile_dim;
        constexpr auto num_iters = regstokes::detail::util::saturating_div(
                3 * tile_dim,
                block_size);
#pragma unroll
        for (auto t = 0; t < num_iters; t++) {
            float const * in;
            float * out;
            auto my_idx = t * block_size + thread_idx;
            switch (my_idx / tile_dim) {
            case 0:
                in = in1;
                out = out1;
                break;
            case 1:
                in = in2;
                out = out2;
                break;
            case 2:
                in = in3;
                out = out3;
                break;
            default:
                return;
            }
            auto idx = my_idx % tile_dim;
            out[idx] = in[idx];
        }
    }
};

template<>
struct read<64, 4, 32, true> {
    static
    __device__
    void _(
            float const * in1,
            float const * in2,
            float const * in3,
            float * out1,
            float * out2,
            float * out3)
    {
        switch (threadIdx.y) {
        case 0:
            out1[threadIdx.x] = in1[blockIdx.y + threadIdx.x];
            break;
        case 1:
            out2[threadIdx.x] = in2[blockIdx.y + threadIdx.x];
            break;
        case 2:
            out3[threadIdx.x] = in3[blockIdx.y + threadIdx.x];
            break;
        }
    }
};

template<>
struct read<128, 4, 32, true> {
    static
    __device__
    void _(
            float const * in1,
            float const * in2,
            float const * in3,
            float * out1,
            float * out2,
            float * out3)
    {
        switch (threadIdx.y) {
        case 0:
            out1[threadIdx.x] = in1[blockIdx.y + threadIdx.x];
            break;
        case 1:
            out2[threadIdx.x] = in2[blockIdx.y + threadIdx.x];
            break;
        case 2:
            out3[threadIdx.x] = in3[blockIdx.y + threadIdx.x];
            break;
        }
    }
};

} // namespace detail
} // namespace soa
} // namespace regstokes
