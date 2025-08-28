#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double** allocate_matrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    return matrix;
}

void initialize_matrix(int size, double** matrix) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)(i + j);
        }
    }
}

void free_matrix(int size, double** matrix) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


void add_matrices(int size, double** a, double** b, double** c) {
   #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

int main() {
    int sizes[] = {250, 500, 750, 1000, 2000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {2, 4, 8, 12}; 
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("2D Matrix Addition Performance Analysis\n\n");

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        printf("====================================================\n");
        printf("Matrix Size: %d x %d\n", size, size);
        printf("----------------------------------------------------\n");
        printf("Threads |   Time (s)   |   Speedup\n");
        printf("----------------------------------------------------\n");

    
        double** matrix_a = allocate_matrix(size);
        double** matrix_b = allocate_matrix(size);
        double** matrix_c = allocate_matrix(size);
        initialize_matrix(size, matrix_a);
        initialize_matrix(size, matrix_b);

        omp_set_num_threads(1);
        double start_time_serial = omp_get_wtime();
        add_matrices(size, matrix_a, matrix_b, matrix_c);
        double end_time_serial = omp_get_wtime();
        double time_serial = end_time_serial - start_time_serial;
        printf("%5d   |  %10.6f  |   1.0000\n", 1, time_serial);

        for (int j = 0; j < num_thread_counts; j++) {
            int threads = thread_counts[j];
            omp_set_num_threads(threads);

            double start_time_parallel = omp_get_wtime();
            add_matrices(size, matrix_a, matrix_b, matrix_c);
            double end_time_parallel = omp_get_wtime();
            double time_parallel = end_time_parallel - start_time_parallel;

            double speedup = time_serial / time_parallel;
            printf("%5d   |  %10.6f  |   %.4f\n", threads, time_parallel, speedup);
        }

        
        free_matrix(size, matrix_a);
        free_matrix(size, matrix_b);
        free_matrix(size, matrix_c);
    }

    return 0;
}