#pragma once

#include "regstokes/soa/detail.cuh"

namespace regstokes {
namespace soa {

///
/// Kernel for the method of regularized Stokeslets (structure-of-arrays
/// version).
///
template<unsigned tile_dim, unsigned block_h, unsigned warp_size = 32,
        bool specialize = true>
__global__
void kernel(
        float const * src_x,
        float const * src_y,
        float const * src_z,
        float const * fld_x,
        float const * fld_y,
        float const * fld_z,
        float * out,
        size_t pitch,
        float fac,
        float eps_sqr,
        float eps_sqr_2)
{
    constexpr auto block_size = tile_dim * block_h;

    static_assert(
            tile_dim % block_h == 0,
            "tile dimension must be a multiple of block height"
    );
    static_assert(block_size > 0, "block size must be nonzero");
    static_assert(
            block_size % warp_size == 0,
            "block size must be a multiple of the warp size"
    );

    cuda_assume(blockDim.x == tile_dim);
    cuda_assume(blockDim.y == block_h);
    cuda_assume(threadIdx.x < tile_dim);
    cuda_assume(threadIdx.y < block_h);

    __shared__ float fld_x_tile[tile_dim];
    __shared__ float fld_y_tile[tile_dim];
    __shared__ float fld_z_tile[tile_dim];
    __shared__ float out_tile[3 * block_h][3 * tile_dim];

    regstokes::soa::detail::read<tile_dim, block_h, warp_size, specialize>::_(
            fld_x,
            fld_y,
            fld_z,
            fld_x_tile,
            fld_y_tile,
            fld_z_tile);

    if constexpr (block_size > warp_size) {
        __syncthreads();
    }

    auto sx = src_x[blockIdx.x + threadIdx.x];
    auto sy = src_y[blockIdx.x + threadIdx.x];
    auto sz = src_z[blockIdx.x + threadIdx.x];

    constexpr auto num_iters = tile_dim / block_h;
#pragma unroll
    for (auto t = 0; t < num_iters; t++) {
        auto fx = fld_x_tile[t * block_h + threadIdx.y];
        auto fy = fld_y_tile[t * block_h + threadIdx.y];
        auto fz = fld_z_tile[t * block_h + threadIdx.y];
        float xx, xy, xz, yy, yz, zz;
        regstokes::core(
                sx,
                sy,
                sz,
                fx,
                fy,
                fz,
                out,
                pitch,
                fac,
                eps_sqr,
                eps_sqr_2,
                xx,
                xy,
                xz,
                yy,
                yz,
                zz);
        out_tile[0][3 * threadIdx.x + 0] = xx;
        out_tile[0][3 * threadIdx.x + 1] = xy;
        out_tile[0][3 * threadIdx.x + 2] = xz;
        out_tile[1][3 * threadIdx.x + 0] = xy;
        out_tile[1][3 * threadIdx.x + 1] = yy;
        out_tile[1][3 * threadIdx.x + 2] = yz;
        out_tile[2][3 * threadIdx.x + 0] = xz;
        out_tile[2][3 * threadIdx.x + 1] = yz;
        out_tile[2][3 * threadIdx.x + 2] = zz;

        if constexpr (block_size > warp_size) {
            __syncthreads();
        }

#pragma unroll
        for (auto j = 0; j < 3; j++) {
            auto in_row = j;
            auto out_row = 3 * num_iters * blockIdx.y + 3 * t + in_row;
#pragma unroll
            for (auto i = 0; i < 3; i++) {
                auto in_col = i * tile_dim + threadIdx.x;
                auto out_col = 3 * tile_dim * blockIdx.x + in_row;
                auto out_ptr = pitched(out, pitch, out_row, out_col);
                *out_ptr = out_tile[in_row][in_col];
            }
        }
    }
}

} // namespace soa
} // namespace regstokes
