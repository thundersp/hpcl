#include <stdio.h>
#include <omp.h>

#define N 10000
#define SCALAR 5.0

int main() {
    static double A[N][N];
    int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            A[i][j] = i + j;

    FILE *fp = fopen("results.csv", "a");
    if (!fp) { perror("file open"); return 1; }

    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        double start = omp_get_wtime();

        #pragma omp parallel for private(i,j) shared(A)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                A[i][j] *= SCALAR;

        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("[Matrix-Scalar] Threads=%d -> %f sec\n", threads, elapsed);
        fprintf(fp, "%d,%f,MatrixScalar\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
