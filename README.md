# Overview

`regstokes/reference/` contains two reference implementations: an
array-of-structures implementation (`aos.cuh`) and a structure-of-arrays
implementation (`soa.cuh`). (We generally expect a structure-of-arrays
formulation to perform better on GPUs, but the difference here is minimal.)
The reference implementations are unoptimized; but they are written in the
simplest possible way, so we measure the correctness of other implementations
against them.

`regstokes/soa/` contains an optimized structure-of-arrays implementation. This
kernel uses a tiling approach to promote coalsecing and optimal latency/compute
balance. The kernel is templated over tile size and threads-per-block (with
each block responsible for a specific tile), and the optimal configuration is
likely to depend on problem size, compute capability, and the particular card
in use. On my setup (NVIDIA GTX 1660 Ti, compute capability 7.5), I find that a
tile size of 128x128 or 64x64 with (respectively) 512 or 256 threads per block
tends to perform best. 

# Performance

AMD Ryzen 9 3900x, NVIDIA GTX 1660 Ti, 4096 source and field elements, `-O3`
where applicable.

| Implementation        | Time   |
|-----------------------|--------|
| Python                | 500 ms |
| C++ (single-threaded) | 250 ms |
| C++ (multi-threaded)  | 50 ms  |
| CUDA (reference)      | 5 ms   |
| CUDA (optimized)      | 0.5 ms |

These are only approximate, but they get across the order-of-magnitude
differences.

# Caveats

In my use-cases, building the kernel matrix tends to account for at most 10% of
the runtime, so having a highly-optimized implementation is not especially
important. However, I believe that there are other use-cases where building the
kernel matrix will account for a significantly larger proportion of the
runtime. In any case, linear algebra will almost always be faster on the GPU at
scale; and if we're going to move to the GPU in order to increase performance,
we might as well take the time to do some basic optimizations.