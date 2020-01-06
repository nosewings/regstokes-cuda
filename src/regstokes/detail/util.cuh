#pragma once

#include <iostream>

namespace regstokes {
namespace detail {
namespace util {

///////////////////////////////////////////////////////////////////////////////
// Declarations
///////////////////////////////////////////////////////////////////////////////

constexpr
float constexpr_ceil(float);

__device__
void cuda_assume(bool);

void cuda_check_error_with(cudaError_t, char const *, int);

[[noreturn]]
__device__
void cuda_unreachable();

template<typename T, typename I>
constexpr
T * pitched(T *, size_t, I, I);

__device__
float rpowf_3_2(float);

template<typename T>
constexpr
T saturating_div(T, T);

///////////////////////////////////////////////////////////////////////////////
// Definitions
///////////////////////////////////////////////////////////////////////////////

constexpr
float constexpr_ceil(float x)
{
    auto i = static_cast<int64_t>(x);
    auto r = static_cast<float>(x);
    return i == x ? r : x > 0.0f ? r + 1 : r;
}

__device__
void cuda_assume(bool cond)
{
    if (!cond) {
        cuda_unreachable();
    }
}

void cuda_check_error_with(cudaError_t err, char const * file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << "(" << file << ", line " << line
                << "): " << cudaGetErrorString(err) << std::endl;
        std::abort();
    }
}

[[noreturn]]
__device__
void cuda_unreachable()
{
    asm("trap;");
}

///
/// Compute the address of an element in a pitched array.
///
template<typename T, typename I>
constexpr
T * pitched(T * ptr, size_t pitch, I row, I col)
{
    return reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + row * pitch)
            + col;
}

///
/// Computes \f$x^{-\frac{3}{2}}\f$.
///
__device__
float rpowf_3_2(float x)
{
    return (1 / x) * rsqrtf(x);
}

template<typename T>
constexpr
T saturating_div(T x, T y)
{
    return static_cast<T>(constexpr_ceil(
            static_cast<double>(x) / static_cast<double>(y)));
}

} // namespace util
} // namespace detail
} // namespace regstokes

#define REGSTOKES_DETAIL_UTIL_CUDA_CHECK_ERROR(err) \
    regstokes::detail::util::cuda_check_error_with((err), __FILE__, __LINE__)
#define REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR() \
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_ERROR(cudaGetLastError())
