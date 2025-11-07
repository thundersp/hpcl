#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello world from thread %d\n", threadIdx.x);
}

int main(int argc, char** argv) {
    int threads = 64; // default threads per block
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threads = t;
    }

    // Launch one block with 'threads' threads
    hello_kernel<<<1, threads>>>();

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
