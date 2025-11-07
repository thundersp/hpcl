#include <stdio.h>
#include <omp.h>

#define N 10000

int main() {
    static double A[N][N], x[N], y[N];
    int i, j;

    for (i = 0; i < N; i++) {
        x[i] = 1.0;
        for (j = 0; j < N; j++)
            A[i][j] = (i == j) ? 2.0 : 1.0;
    }

    FILE *fp = fopen("results.csv", "a");
    if (!fp) { perror("file open"); return 1; }

    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        double start = omp_get_wtime();

        #pragma omp parallel for private(i,j) shared(A,x,y)
        for (i = 0; i < N; i++) {
            double sum = 0;
            for (j = 0; j < N; j++)
                sum += A[i][j] * x[j];
            y[i] = sum;
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("[Matrix-Vector] Threads=%d -> %f sec\n", threads, elapsed);
        fprintf(fp, "%d,%f,MatrixVector\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
