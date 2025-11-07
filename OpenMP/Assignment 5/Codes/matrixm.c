#include <stdio.h>
#include <omp.h>

#define N 500

int main()
{
    static double A[N][N], B[N][N], C[N][N];
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }

    FILE *fp = fopen("results.csv", "a");
    if (!fp)
    {
        perror("file open");
        return 1;
    }

    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
        omp_set_num_threads(threads);

        double start = omp_get_wtime();

#pragma omp parallel for private(i, j, k) shared(A, B, C)
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                double sum = 0;
                for (k = 0; k < N; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }

        double end = omp_get_wtime();
        double elapsed = end - start;

        printf("[Matrix-Matrix] Threads=%d -> %f sec\n", threads, elapsed);
        fprintf(fp, "%d,%f,MatrixMatrix\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
