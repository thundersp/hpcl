#include <stdio.h>
#include <omp.h>

#define N 100000000

int main() {
    static double A[N], prefix[N];
    int i;

    for (i = 0; i < N; i++) A[i] = 1.0;

    FILE *fp = fopen("results.csv", "a");
    if (!fp) { perror("file open"); return 1; }

    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        double start = omp_get_wtime();

        double sum = 0;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int nth = omp_get_num_threads();

            int chunk = N / nth;
            int start_idx = id * chunk;
            int end_idx = (id == nth-1) ? N : start_idx + chunk;

            double local_sum = 0;
            for (int k = start_idx; k < end_idx; k++) {
                local_sum += A[k];
                prefix[k] = local_sum;
            }

            #pragma omp barrier

            double offset = 0;
            for (int t = 0; t < id; t++) {
                int s = t * chunk;
                int e = (t == nth-1) ? N : s + chunk;
                for (int k = s; k < e; k++) offset += A[k];
            }

            for (int k = start_idx; k < end_idx; k++)
                prefix[k] += offset;
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("[Prefix-Sum] Threads=%d -> %f sec\n", threads, elapsed);
        fprintf(fp, "%d,%f,PrefixSum\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
