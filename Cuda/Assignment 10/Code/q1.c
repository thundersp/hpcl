#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, dev);
        if (err != cudaSuccess) {
            printf("cudaGetDeviceProperties failed for device %d: %s\n",
                   dev, cudaGetErrorString(err));
            continue;
        }

        if (dev == 0) {
            if (deviceProp.major < 1) {
                printf("There is no device supporting CUDA.\n");
            } else if (deviceCount == 1) {
                printf("There is 1 device supporting CUDA\n");
            } else {
                printf("There are %d devices supporting CUDA\n", deviceCount);
            }
        }

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n", deviceProp.major);
        printf("  Minor revision number:                         %d\n", deviceProp.minor);
        printf("  Total amount of global memory:                 %zu bytes\n", deviceProp.totalGlobalMem);
        printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Multiprocessor count:                          %d\n", deviceProp.multiProcessorCount);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %d kilohertz\n", deviceProp.clockRate);
    }

    return 0;
}
