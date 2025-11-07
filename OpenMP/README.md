1. **Installation and Running of OpenMP in Operating Systems for C/C++**

Linux: 

sudo apt install build-essentials

It’s a package that pulls essential tools needed to run C/C++ on linux

- **`gcc`** → the GNU C compiler
- **`g++`** → the GNU C++ compiler
- **`make`** → a build automation tool
- **`libc6-dev`** (or `libc-dev`) → C standard library headers

- **`dpkg-dev`** → development tools for Debian packaging

#include<omp.h> This is an OpenMP header which defines functions, constant and enables pragmas

1.#pragma omp parallel

Compiler tells openMP to run the following block with multiple threads in parallel

openMP’s parallel region spawns a team of threads

Default number of threads = no. of CPU cores(8 Octacore)

2.omp_set_num_threads(no.ofthreads)

Sets the number of threads to be spawned

3.omp_get_thread_num(): 

Each thread gets a unique ID in range [0, num_threads-1]

print(thread_id) 

Order is unpredictable because threads run concurrently

4.omp_get_wtime() 

Measures execution time precisely

Scalability: program tests performance at multiple thread counts

Amdahl’s law: Speedup is limited by the fraction of the program that can’t be parallelized

5.#pragma omp parallel for reduction(:+total_sum)

Each thread gets its own copy of sum, at the end OpenMP sums them all up safely into the shared total_sum preventing race conditions

Application: Approximating integrals with midpoint method

6.#pragma omp parallel for collapse(2) 

Merges two loops into one improving load balancing across threads

Speedups may flatten due to overheads and memory bandwidth limits

7.#pragma omp parallel for schedule(runtime)

Policy used by openMP defined by the environment variable

Can be an overhead when CPU cycles are long

8.setenv(”OMP_SCHEDULE”, variable_name, 1)

Scheduling types:
Static→ Iterations are divided into equal chunks beforehand

Dynamic→Threads grab new chunks as they finish(Useful when varying iteration cost)

Chunk Size → Controls how many iterations each thread gets at once

9.#pragma opm for schedule(static, 1) nowait

Normally, after each loop omp inserts an implicit barrier all threads wait

with nowait → threads that finish one loop don’t wait for others, immediately start next loop

Remove synchronization barriers for efficiency

10.#pragma omp parallel sections

Spawns a team of threads, each block is executed by one thread

11.#pragma omp critical ensures only one thread at a time avoiding race conditions on particular variables

Alternatives are: omp_lock_t, #pragma omp atomic

Busy waiting→ while(!placed) etc, 

Data parallelism→ Same computation across many data elements

Compute bound problems scale well

Memory bound problems hit a bottleneck quickly


12#pragma omp barrier → Ensures all threads finish step1 before computing
