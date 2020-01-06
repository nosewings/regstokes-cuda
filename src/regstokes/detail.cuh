#pragma once

#include "math_constants.h"
#include "regstokes/detail/util.cuh"

namespace regstokes {
namespace detail {

void precore(
        float mu,
        float eps,
        float & fac,
        float & eps_sqr,
        float & eps_sqr_2)
{
    fac = 1.0f / (8.0f * CUDART_PI_F * mu);
    eps_sqr = eps * eps;
    eps_sqr_2 = eps_sqr + eps_sqr;
}

/// The numerical core of the method of regularized Stokeslets.
__device__
void core(
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
        float eps_sqr_2,
        float & xx,
        float & xy,
        float & xz,
        float & yy,
        float & yz,
        float & zz)
{
    auto dx = fx - sx;
    auto dy = fy - sy;
    auto dz = fz - sz;

    auto r_sqr = dx * dx + dy * dy + dz * dz;
    auto h2 = regstokes::detail::util::rpowf_3_2(r_sqr + eps_sqr);
    auto h1 = r_sqr + eps_sqr_2;
    auto the_fac = fac * h2;

    xx = the_fac * (dx * dx + h1);
    xy = the_fac * (dx * dy);
    xz = the_fac * (dx * dz);
    yy = the_fac * (dy * dy + h1);
    yz = the_fac * (dy * dz);
    zz = the_fac * (dz * dz + h1);
}

} // namespace detail
} // namespace regstokes
