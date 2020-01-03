#pragma once

#include "util.cuh"

namespace regstokes {

/// The numerical core of the method of regularized Stokeslets.
__device__
void core(
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
    float eps_sqr_2,
    float & xx,
    float & xy,
    float & xz,
    float & yy,
    float & yz,
    float & zz)
{
    auto dx = sx - fx;
    auto dy = sy - fy;
    auto dz = sz - fz;

    auto r_sqr = dx * dx + dy * dy + dz * dz;
    auto h2 = rpowf_3_2(r_sqr + eps_sqr);
    auto h1 = r_sqr + eps_sqr_2;
    auto the_fac = fac * h2;

    xx = the_fac * (dx * dx + h1);
    xy = the_fac * (dx * dy);
    xz = the_fac * (dx * dz);
    yy = the_fac * (dy * dy + h1);
    yz = the_fac * (dy * dz);
    zz = the_fac * (dz * dz + h1);
}

} // namespace regstokes
