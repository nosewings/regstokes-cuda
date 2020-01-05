#pragma once

namespace regstokes {
namespace reference {
namespace soa {

///
/// Slow-but-simple reference implementation (structure-of-arrays version).
///
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
    auto i = blockIdx.y + threadIdx.y;
    auto j = blockIdx.x + threadIdx.x;

    auto sx = src_x[i];
    auto sy = src_y[i];
    auto sz = src_y[i];

    auto fx = fld_x[j];
    auto fy = fld_y[j];
    auto fz = fld_z[j];

    detail::common(
            3 * i,
            3 * j,
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

} // namespace soa
} // namespace reference
} // namespace regstokes
