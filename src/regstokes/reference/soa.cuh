#pragma once

#include "regstokes/reference/detail.cuh"

namespace regstokes {
namespace reference {
namespace soa {

///
/// Slow-but-simple reference implementation (structure-of-arrays version).
///
__global__
void kernel(
        float const * fld_x,
        float const * fld_y,
        float const * fld_z,
        float const * src_x,
        float const * src_y,
        float const * src_z,
        float fac,
        float eps_sqr,
        float eps_sqr_2,
        float * out,
        size_t pitch)
{
    auto i = blockIdx.y + threadIdx.y;
    auto j = blockIdx.x + threadIdx.x;

    auto fx = fld_x[i];
    auto fy = fld_y[i];
    auto fz = fld_z[i];

    auto sx = src_x[j];
    auto sy = src_y[j];
    auto sz = src_y[j];

    detail::common(
            3 * i,
            3 * j,
            fx,
            fy,
            fz,
            sx,
            sy,
            sz,
            out,
            pitch,
            fac,
            eps_sqr,
            eps_sqr_2);
}

void regstokes(
        float const * fld_x,
        float const * fld_y,
        float const * fld_z,
        float const * src_x,
        float const * src_y,
        float const * src_z,
        float mu,
        float eps,
        float * out,
        size_t pitch,
        dim3 blocks,
        dim3 threads)
{
    float fac, eps_sqr, eps_sqr_2;
    regstokes::detail::precore(mu, eps, fac, eps_sqr, eps_sqr_2);
    kernel<<<blocks, threads>>>(
            fld_x,
            fld_y,
            fld_z,
            src_x,
            src_y,
            src_z,
            fac,
            eps_sqr,
            eps_sqr_2,
            out,
            pitch);
}

} // namespace soa
} // namespace reference
} // namespace regstokes
