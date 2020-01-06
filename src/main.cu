#include <iostream>
#include <cuda_profiler_api.h>

#include "regstokes/reference.cuh"
#include "regstokes/soa.cuh"

using namespace regstokes;

void test_reference_soa(
        dim3 blocks,
        dim3 threads,
        int n,
        float fac,
        float eps_sqr,
        float eps_sqr_2,
        int num_runs)
{
    float * x;
    float * y;
    float * z;
    float * out;
    cudaMalloc(&x, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&y, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&z, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&out, 9 * n * n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaProfilerStart();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    for (auto i = 0; i < num_runs; i++) {
        reference::soa::kernel<<<blocks, threads>>>(
                x,
                y,
                z,
                x,
                y,
                z,
                fac,
                eps_sqr,
                eps_sqr_2,
                out,
                3 * n * sizeof(float));
        cudaDeviceSynchronize();
        REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    }

    cudaProfilerStop();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaFree(x);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(y);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(z);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(out);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
}

void test_reference_aos(
        dim3 blocks,
        dim3 threads,
        int n,
        float fac,
        float eps_sqr,
        float eps_sqr_2,
        int num_runs)
{
    float * arr;
    float * out;
    cudaMalloc(&arr, 3 * n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&out, 9 * n * n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaProfilerStart();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    for (auto i = 0; i < num_runs; i++) {
        reference::aos::kernel<<<blocks, threads>>>(
                arr,
                arr,
                fac,
                eps_sqr,
                eps_sqr_2,
                out,
                3 * n * sizeof(float));
        cudaDeviceSynchronize();
        REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    }

    cudaProfilerStop();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaFree(arr);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(out);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
}

template<unsigned tile_dim, unsigned block_h, bool specialize>
void test_soa(int n, float fac, float eps_sqr, float eps_sqr_2, int num_runs)
{
    dim3 blocks(n / tile_dim, n / tile_dim);
    dim3 threads(tile_dim, block_h);

    float * x;
    float * y;
    float * z;
    float * out;
    cudaMalloc(&x, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&y, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&z, n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&out, 9 * n * n * sizeof(float));
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaProfilerStart();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    for (auto i = 0; i < num_runs; i++) {
        soa::kernel<tile_dim, block_h><<<blocks, threads>>>(
                x,
                y,
                z,
                x,
                y,
                z,
                out,
                3 * n * sizeof(float),
                fac,
                eps_sqr,
                eps_sqr_2);
        cudaDeviceSynchronize();
        REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    }
    cudaDeviceSynchronize();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaProfilerStop();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    cudaFree(x);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(y);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(z);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
    cudaFree(out);
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();
}

int main()
{
    cudaProfilerStop();
    REGSTOKES_DETAIL_UTIL_CUDA_CHECK_LAST_ERROR();

    constexpr unsigned n = 1 << 12;

    constexpr auto fac = 0.1;
    constexpr auto eps_sqr = 0.001;
    constexpr auto eps_sqr_2 = 0.002;
    constexpr auto num_runs = 1000;

    constexpr dim3 blocks(n / 32, n / 32);
    constexpr dim3 threads(32, 32);
    test_reference_soa(blocks, threads, n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_reference_aos(blocks, threads, n, fac, eps_sqr, eps_sqr_2, num_runs);

    constexpr auto specialize = true;

    test_soa<8, 4, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<8, 8, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<16, 2, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<16, 4, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<16, 8, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 1, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 2, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 4, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 8, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 16, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<32, 32, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<64, 1, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<64, 2, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<64, 4, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<64, 8, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<64, 16, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<128, 1, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<128, 2, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<128, 4, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
    test_soa<128, 8, specialize>(n, fac, eps_sqr, eps_sqr_2, num_runs);
}
