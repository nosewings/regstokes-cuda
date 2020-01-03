#pragma once

#include "../core.cuh"
#include "../util.cuh"

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
    float sx,
    float sy,
    float sz,
    float fx,
    float fy,
    float fz,
    float * out,
    size_t pitch,
    float fac,
    float eps_sqr,
    float eps_sqr_2)
{
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

    auto out0 = pitched(out, pitch, i, j);
    auto out1 = pitched(out0, pitch, 1, 0);
    auto out2 = pitched(out1, pitch, 1, 0);

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
