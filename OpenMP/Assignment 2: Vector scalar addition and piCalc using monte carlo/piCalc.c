#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define SIZE 541006540

int main(){
    long long int i, count = 0;
    double x, y, z, pi;
    double start_time, end_time, serial_time, parallel_time;
    int threads;

    printf("Enter number of threads: ");
    scanf("%d", &threads);

    start_time = omp_get_wtime();
    count = 0;
    for (i = 0; i < SIZE; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        z = x * x + y * y;
        if (z <= 1) count++;
    }
    pi = (double)count / SIZE * 4;
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    
    printf("Serial Execution:\n");
    printf("Calculated PI = %lf\n", pi);
    printf("Time taken = %lf seconds\n\n", serial_time);

    start_time = omp_get_wtime();
    count = 0;
    
    #pragma omp parallel num_threads(threads) private(x, y, z) reduction(+:count)
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < SIZE; i++) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            z = x * x + y * y;
            if (z <= 1) count++;
        }
    }
    
    pi = (double)count / SIZE * 4;
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    
    printf("Parallel Execution with %d threads:\n", threads);
    printf("Calculated PI = %lf\n", pi);
    printf("Time taken = %lf seconds\n", parallel_time);
    printf("Speedup = %lf\n", serial_time / parallel_time);
    return 0;
}