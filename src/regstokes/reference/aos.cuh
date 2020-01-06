#pragma once

#include "regstokes/detail.cuh"
#include "regstokes/reference/detail.cuh"

namespace regstokes {
namespace reference {
namespace aos {

///
/// Slow-but-simple reference implementation (array-of-structures version).
///
__global__
void kernel(
        float const * fld,
        float const * src,
        float fac,
        float eps_sqr,
        float eps_sqr_2,
        float * out,
        size_t pitch)
{
    auto i = 3 * (blockIdx.y + threadIdx.y);
    auto j = 3 * (blockIdx.x + threadIdx.x);

    auto fx = fld[i + 0];
    auto fy = fld[i + 1];
    auto fz = fld[i + 2];

    auto sx = src[j + 0];
    auto sy = src[j + 1];
    auto sz = src[j + 2];

    detail::common(
            i,
            j,
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
        float const * fld,
        float const * src,
        float mu,
        float eps,
        float * out,
        size_t pitch,
        dim3 blocks,
        dim3 threads)
{
    float fac, eps_sqr, eps_sqr_2;
    regstokes::detail::precore(mu, eps, fac, eps_sqr, eps_sqr_2);
    kernel<<<blocks, threads>>>(fld, src, fac, eps_sqr, eps_sqr_2, out, pitch);
}

} // namespace aos
} // namespace reference
} // namespace regstokes
