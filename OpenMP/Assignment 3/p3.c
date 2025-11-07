#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h> 

#define VECTOR_SIZE 200
#define SCALAR 5.0

void vector_add_and_time(const char* schedule_type, int chunk_size) {
    double vector[VECTOR_SIZE];
    double result[VECTOR_SIZE];

    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = (double)i;
    }

    printf("--- Schedule: %-8s | Chunk Size: %-4d ---\n", schedule_type, chunk_size);
    printf("Threads |   Time (s)   |   Speedup\n");
    printf("-------------------------------------------\n");

    
    double start_serial = omp_get_wtime();
    for (int i = 0; i < VECTOR_SIZE; i++) {
        result[i] = vector[i] + SCALAR;
    }
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;
    printf("%5d   |  %10.8f  |   1.0000\n", 1, time_serial);

    int thread_counts[] = {2, 4, 8};
    for (int i = 0; i < 3; i++) {
        int threads = thread_counts[i];
        omp_set_num_threads(threads);

        double start_parallel = omp_get_wtime();
        #pragma omp parallel for schedule(runtime)
        for (int j = 0; j < VECTOR_SIZE; j++) {
            result[j] = vector[j] + SCALAR;
        }

        double end_parallel = omp_get_wtime();
        double time_parallel = end_parallel - start_parallel;
        double speedup = time_serial / time_parallel;
        printf("%5d   |  %10.8f  |   %.4f\n", threads, time_parallel, speedup);
    }
    printf("\n");
}

int main() {
    printf("Vector Size: %d\n\n", VECTOR_SIZE);

    int static_chunks[] = {1, 10, 50, 100};
    for (int i = 0; i < 4; i++) {
        
        char schedule_str[32];
        sprintf(schedule_str, "static,%d", static_chunks[i]);
        setenv("OMP_SCHEDULE", schedule_str, 1);
        vector_add_and_time("STATIC", static_chunks[i]);
    }

    int dynamic_chunks[] = {1, 10, 50, 100};
    for (int i = 0; i < 4; i++) {
        char schedule_str[32];
        sprintf(schedule_str, "dynamic,%d", dynamic_chunks[i]);
        setenv("OMP_SCHEDULE", schedule_str, 1);
        vector_add_and_time("DYNAMIC", dynamic_chunks[i]);
    }
    printf("--- Demonstration of 'nowait' clause ---\n");
    printf("With 'nowait', threads from loop 1 can start loop 2 immediately.\n\n");
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static, 1) nowait
        for (int i = 0; i < 4; i++) {
            printf("Thread %d is executing loop 1, iteration %d\n", thread_id, i);
            if (thread_id == 0) sleep(1);
        }

        #pragma omp for schedule(static, 1)
        for (int i = 0; i < 4; i++) {
            printf("Thread %d is executing loop 2, iteration %d\n", thread_id, i);
        }
    }

    return 0;
}