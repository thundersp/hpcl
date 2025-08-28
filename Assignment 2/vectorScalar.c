#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define VECTOR_SIZE 541006540

void add_serial(int *vector, int scalar, int *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = vector[i] + scalar;
    }
}

void add_parallel(int *vector, int scalar, int *result, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        result[i] = vector[i] + scalar;
    }
}

int main()
{
    int *vector, *result_serial, *result_parallel;
    int scalar = 5;
    int max_th = omp_get_max_threads();
    int size = VECTOR_SIZE;
    double start_time, end_time;
    int optimal_th = 0;
    double min_time = 1000;

    vector = (int *)malloc(size * sizeof(int));
    result_serial = (int *)malloc(size * sizeof(int));
    result_parallel = (int *)malloc(size * sizeof(int));

    if (vector == NULL || result_serial == NULL || result_parallel == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    srand(27);
    for (int i = 0; i < size; i++)
    {
        vector[i] = rand() % 100;
    }
    printf("Vector size: %d\n", size);
    printf("Scalar value: %d\n", scalar);
    printf("Maximum number of threads: %d\n", omp_get_max_threads());
    start_time = omp_get_wtime();
    add_serial(vector, scalar, result_serial, size);
    end_time = omp_get_wtime();
    double serial_time = end_time - start_time;
    printf("Serial execution time: %f seconds\n", serial_time);

    for (int threads = 2; threads <= max_th; threads += 2)
    {
        start_time = omp_get_wtime();
        omp_set_num_threads(threads);
        add_parallel(vector, scalar, result_parallel, size);
        end_time = omp_get_wtime();
        double parallel_time = end_time - start_time;
        printf("Parallel execution time: %f seconds with %d threads\n", parallel_time, threads);
        if (parallel_time < min_time)
        {
            min_time = parallel_time;
            optimal_th = threads;
        }
        if (parallel_time > 0)
        {
            double speedup = serial_time / parallel_time;
            printf("\nSpeedup: %.2fx\n", speedup);
        }
    }

    free(vector);
    free(result_serial);
    free(result_parallel);
    
    printf("\nOptimal thread count (%d):\n", optimal_th);
    
    return 0;
}
