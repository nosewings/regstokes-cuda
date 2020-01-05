#pragma once

#include "detail.cuh"

namespace regstokes {
namespace reference {
namespace aos {

///
/// Slow-but-simple reference implementation (array-of-structures version).
///
__global__
void kernel(
        float const * src,
        float const * fld,
        float * out,
        size_t pitch,
        float fac,
        float eps_sqr,
        float eps_sqr_2)
{
    auto i = 3 * (blockIdx.y + threadIdx.y);
    auto j = 3 * (blockIdx.x + threadIdx.x);

    auto sx = src[i + 0];
    auto sy = src[i + 1];
    auto sz = src[i + 2];

    auto fx = fld[j + 0];
    auto fy = fld[j + 1];
    auto fz = fld[j + 2];

    detail::common(
            i,
            j,
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
            eps_sqr_2);
}

} // namespace aos
} // namespace reference
} // namespace regstokes
