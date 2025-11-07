#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

static void fill_random(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = (float)rand() / RAND_MAX;
}

static double dot_cpu(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)a[i] * (double)b[i];
    return sum;
}

__global__ void dot_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ result,
                           int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float thread_sum = 0.0f;

    // Grid-stride loop
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        thread_sum += a[i] * b[i];
    }

    // Write to shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block reduction in shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Accumulate block result to global sum
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

static void run_case(int N) {
    printf("==== N = %d ====\n", N);

    // Host allocations
    size_t bytes = (size_t)N * sizeof(float);
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    if (!hA || !hB) {
        fprintf(stderr, "Host allocation failed\n");
        exit(1);
    }

    srand(1234);
    fill_random(hA, N);
    fill_random(hB, N);

    // CPU timing
    auto t0 = std::chrono::high_resolution_clock::now();
    double cpu_result = dot_cpu(hA, hB, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Device allocations
    float *dA = nullptr, *dB = nullptr, *dOut = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dA, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dOut, sizeof(float)));

    // Events
    cudaEvent_t evStart, evAfterH2D, evAfterKernel, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evAfterH2D));
    CHECK_CUDA(cudaEventCreate(&evAfterKernel));
    CHECK_CUDA(cudaEventCreate(&evStop));

    // Copy in + init output
    CHECK_CUDA(cudaEventRecord(evStart));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));
    CHECK_CUDA(cudaEventRecord(evAfterH2D));

    // Launch configuration
    int blockSize = 256;
    int maxBlocks = 65535;
    int gridSize = (N + blockSize - 1) / blockSize;
    if (gridSize > maxBlocks) gridSize = maxBlocks;

    // Kernel launch (shared memory = blockSize * sizeof(float))
    dot_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(dA, dB, dOut, N);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(evAfterKernel));
    CHECK_CUDA(cudaEventSynchronize(evAfterKernel));

    // Copy result back
    float gpu_result_f = 0.0f;
    CHECK_CUDA(cudaMemcpy(&gpu_result_f, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, evStart, evAfterH2D));
    CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, evAfterH2D, evAfterKernel));
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, evStart, evStop));

    double gpu_result = (double)gpu_result_f;

    // Accuracy
    double abs_err = fabs(cpu_result - gpu_result);
    double rel_err = fabs(cpu_result) > 0 ? abs_err / fabs(cpu_result) : abs_err;

    // Report
    printf("CPU result:         %.6f\n", cpu_result);
    printf("GPU result:         %.6f\n", gpu_result);
    printf("Abs error:          %.6e\n", abs_err);
    printf("Rel error:          %.6e\n", rel_err);
    printf("CPU time:           %.3f ms\n", cpu_ms);
    printf("GPU copy-in:        %.3f ms\n", ms_h2d);
    printf("GPU kernel:         %.3f ms\n", ms_kernel);
    printf("GPU end-to-end:     %.3f ms (H2D + kernel + D2H)\n", ms_total);
    if (ms_kernel > 0.0f) printf("Speedup (CPU/kernel):     %.2fx\n", cpu_ms / ms_kernel);
    if (ms_total > 0.0f)  printf("Speedup (CPU/end-to-end): %.2fx\n", cpu_ms / ms_total);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(evStart));
    CHECK_CUDA(cudaEventDestroy(evAfterH2D));
    CHECK_CUDA(cudaEventDestroy(evAfterKernel));
    CHECK_CUDA(cudaEventDestroy(evStop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dOut));
    free(hA);
    free(hB);
}

int main(int argc, char** argv) {
    // Optional custom N via CLI
    if (argc == 2) {
        int N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid N\n");
            return 1;
        }
        run_case(N);
        return 0;
    }

    // Default test sizes
    const int Ns[] = {100000, 1000000, 10000000};
    const int num = sizeof(Ns) / sizeof(Ns[0]);
    for (int i = 0; i < num; ++i) {
        run_case(Ns[i]);
        puts("");
    }
    return 0;
}
