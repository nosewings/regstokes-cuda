#pragma once

#include "regstokes/detail.cuh"

namespace regstokes {
namespace reference {
namespace detail {

///
/// Code common to the reference implementations.
///
__device__
void common(
        unsigned i,
        unsigned j,
        float fx,
        float fy,
        float fz,
        float sx,
        float sy,
        float sz,
        float * out,
        size_t pitch,
        float fac,
        float eps_sqr,
        float eps_sqr_2)
{
    float xx, xy, xz, yy, yz, zz;
    regstokes::detail::core(
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
            eps_sqr_2,
            xx,
            xy,
            xz,
            yy,
            yz,
            zz);

    auto out0 = regstokes::detail::util::pitched(out, pitch, i + 0, j);
    auto out1 = regstokes::detail::util::pitched(out, pitch, i + 1, j);
    auto out2 = regstokes::detail::util::pitched(out, pitch, i + 2, j);

    out0[0] = xx;
    out0[1] = xy;
    out0[2] = xz;
    out1[0] = xy;
    out1[1] = yy;
    out1[2] = yz;
    out2[0] = xz;
    out2[1] = yz;
    out2[2] = zz;
}

} // namespace detail
} // namespace reference
} // namespace regstokes
