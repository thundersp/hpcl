#include <stdio.h>
#include <omp.h>

long long fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    long long x, y;

    #pragma omp task shared(x)
    {
        x = fibonacci(n - 1);
    }

    #pragma omp task shared(y)
    {
        y = fibonacci(n - 2);
    }

    #pragma omp taskwait

    return x + y;
}

int main() {
    int n = 28;
    int max_threads = omp_get_max_threads();
    long long result = 0;

    FILE *fp = fopen("results.csv", "a");
    if (!fp) {
        perror("File open failed");
        return 1;
    }

    // Run with 1 thread
    omp_set_num_threads(1);
    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fibonacci(n);
        }
    }
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    printf("[Fibonacci] Threads=1 -> %.4f sec (Result=%lld)\n", elapsed, result);
    fprintf(fp, "%d,%f,Fibonacci\n", 1, elapsed);

    // Run with 2..max_threads (step = 2)
    for (int threads = 2; threads <= max_threads; threads += 2) {
        omp_set_num_threads(threads);
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                result = fibonacci(n);
            }
        }
        end_time = omp_get_wtime();
        elapsed = end_time - start_time;
        printf("[Fibonacci] Threads=%d -> %.4f sec (Result=%lld)\n", threads, elapsed, result);
        fprintf(fp, "%d,%f,Fibonacci\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
