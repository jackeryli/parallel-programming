# Parallel Programming

This repo covers various parallel programming techniques such as pthread, OpenMP, MPI, vectorization, and CUDA.

## [Odd-even sort](/1-odd-even-sort/README.md)

- Achieved a 2x speedup on a single machine using pthread (1 process vs. 8 processes).
- Improved performance by 1.12x on multiple machines using MPI (1 machine vs. 4 machines).

## [Mandelbrot set](/2-mandelbrot-set/README.md)

- Enhanced performance by 6x using multi-threading with pthread (1 core vs. 12 cores).
- Achieved a 6x speedup with MPI (2 machines vs. 12 machines).

## [All-pairs shortest path on CPUs](/3-all-pairs-shortest-path/README.md)

- Utilized vectorization and OpenMP to achieve an 8x performance improvement compared to a single-core CPU.

## [Blocked all-pairs shortest path on GPU](/4-blocked-all-pairs-shortest-path/README.md)

- Implemented with CUDA and achieved a 160x performance improvement over the CPU version.

## [Multi-GPU blocked all-pairs shortest path](/5-multi-gpu-blocked-all-pairs-shortest-path/README.md)

- Reduced execution time by 50% by running on 2 GPUs with OpenMP.