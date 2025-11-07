#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world from thread %d (block %d, thread %d)\n",
           global_id, blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv) {
    int blocks = 4;   // default number of blocks
    int threads = 64; // default threads per block

    if (argc > 1) {
        int b = atoi(argv[1]);
        if (b > 0) blocks = b;
    }
    if (argc > 2) {
        int t = atoi(argv[2]);
        if (t > 0) threads = t;
    }

    // Launch multiple blocks with multiple threads
    hello_kernel<<<blocks, threads>>>();

    // Check for launch errors
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish and flush device-side printf output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
