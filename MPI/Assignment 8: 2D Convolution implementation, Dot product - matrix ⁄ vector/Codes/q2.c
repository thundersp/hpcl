#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MASTER 0
#define DIM 840

void fill_matrix(double *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = 1.0;  
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    int rows_per_proc;
    double start, end, local_start, local_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (DIM % size != 0) {
        if (rank == MASTER)
            printf("Error: DIM (%d) not divisible by number of processes (%d)\n", DIM, size);
        MPI_Finalize();
        return 0;
    }

    rows_per_proc = DIM / size;

    B = (double *)malloc(DIM * DIM * sizeof(double));
    local_A = (double *)malloc(rows_per_proc * DIM * sizeof(double));
    local_C = (double *)calloc(rows_per_proc * DIM, sizeof(double));

    if (rank == MASTER) {
        A = (double *)malloc(DIM * DIM * sizeof(double));
        C = (double *)calloc(DIM * DIM, sizeof(double));

        fill_matrix(A, DIM, DIM);
        fill_matrix(B, DIM, DIM);
    }

    MPI_Scatter(A, rows_per_proc * DIM, MPI_DOUBLE, local_A,
                rows_per_proc * DIM, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Bcast(B, DIM * DIM, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < DIM; j++) {
            double sum = 0.0;
            for (int k = 0; k < DIM; k++)
                sum += local_A[i * DIM + k] * B[k * DIM + j];
            local_C[i * DIM + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Gather(local_C, rows_per_proc * DIM, MPI_DOUBLE,
               C, rows_per_proc * DIM, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        double elapsed = end - start;
        printf("%d,%f\n", size, elapsed);  
    }

    free(local_A);
    free(local_C);
    free(B);
    if (rank == MASTER) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
