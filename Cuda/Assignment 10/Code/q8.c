#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define TILE 16

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while (0)

static void initMatrix(float* m, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        m[i] = (float)rand() / RAND_MAX; 
    }
}

static void matmulCPU(const float* A, const float* B, float* C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            const int aRow = i * N;
            for (int k = 0; k < N; ++k) {
                sum += A[aRow + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int P) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..P)

    float acc = 0.0f;
    int tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        int aCol = t * TILE + threadIdx.x; // column in A
        int bRow = t * TILE + threadIdx.y; // row in B

        // Load tile from A and B with bounds checks
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < P) ? B[bRow * P + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < TILE; ++e) {
            acc += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = acc;
    }
}

static float maxAbsDiff(const float* A, const float* B, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(A[i] - B[i]);
        if (d > m) m = d;
    }
    return m;
}

static void run_case(int M, int N, int P) {
    size_t bytesA = (size_t)M * N * sizeof(float);
    size_t bytesB = (size_t)N * P * sizeof(float);
    size_t bytesC = (size_t)M * P * sizeof(float);

    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC_cpu = (float*)malloc(bytesC);
    float* hC_gpu = (float*)malloc(bytesC);

    if (!hA || !hB || !hC_cpu || !hC_gpu) {
        fprintf(stderr, "Host allocation failed\n");
        exit(1);
    }

    srand(1234);
    initMatrix(hA, M, N);
    initMatrix(hB, N, P);

    // CPU multiply and time
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(hA, hB, hC_cpu, M, N, P);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Device memory
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dA, bytesA));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytesB));
    CHECK_CUDA(cudaMalloc((void**)&dC, bytesC));

    // Events for GPU timing
    cudaEvent_t evStart, evAfterCopyIn, evAfterKernel, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evAfterCopyIn));
    CHECK_CUDA(cudaEventCreate(&evAfterKernel));
    CHECK_CUDA(cudaEventCreate(&evStop));

    // Copy to device (measure end-to-end time too)
    CHECK_CUDA(cudaEventRecord(evStart));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(evAfterCopyIn));

    dim3 block(TILE, TILE);
    dim3 grid((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Launch kernel and measure kernel-only time
    matmulTiled<<<grid, block>>>(dA, dB, dC, M, N, P);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(evAfterKernel));
    CHECK_CUDA(cudaEventSynchronize(evAfterKernel));

    // Copy back
    CHECK_CUDA(cudaMemcpy(hC_gpu, dC, bytesC, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float ms_copyin = 0.0f, ms_kernel = 0.0f, ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_copyin, evStart, evAfterCopyIn));
    CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, evAfterCopyIn, evAfterKernel));
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, evStart, evStop));

    // Verify
    float err_max = maxAbsDiff(hC_cpu, hC_gpu, M * P);
    const float tol = 1e-3f;

    printf("---- %dx%dx%d (A %dx%d, B %dx%d, C %dx%d) ----\n", M, N, P, M, N, N, P, M, P);
    printf("CPU time:           %.3f ms\n", cpu_ms);
    printf("GPU copy-in:        %.3f ms\n", ms_copyin);
    printf("GPU kernel:         %.3f ms\n", ms_kernel);
    printf("GPU end-to-end:     %.3f ms (H2D + kernel + D2H)\n", ms_total);
    printf("Max abs error:      %.6f %s\n", err_max, (err_max < tol ? "(OK)" : "(LARGE)"));
    if (ms_kernel > 0.0f)
        printf("Speedup (CPU / kernel):     %.2fx\n", cpu_ms / ms_kernel);
    if (ms_total > 0.0f)
        printf("Speedup (CPU / end-to-end): %.2fx\n", cpu_ms / ms_total);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(evStart));
    CHECK_CUDA(cudaEventDestroy(evAfterCopyIn));
    CHECK_CUDA(cudaEventDestroy(evAfterKernel));
    CHECK_CUDA(cudaEventDestroy(evStop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC_cpu);
    free(hC_gpu);
}

int main(int argc, char** argv) {
    // Optional single-case CLI: M N P
    if (argc == 4) {
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int P = atoi(argv[3]);
        if (M <= 0 || N <= 0 || P <= 0) {
            fprintf(stderr, "Invalid dimensions.\n");
            return 1;
        }
        run_case(M, N, P);
        return 0;
    }

    // Default sizes (square)
    int sizes[] = {100, 500, 1000};
    int num = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < num; ++i) {
        int n = sizes[i];
        run_case(n, n, n);
        puts("");
    }
    return 0;
}
