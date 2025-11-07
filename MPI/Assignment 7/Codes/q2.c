#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 1000;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) N = 1000;

    int base_rows = N / world_size;
    int rem = N % world_size;
    int local_rows = base_rows + (world_rank < rem ? 1 : 0);

    int *sendcounts = NULL, *displs = NULL;
    if (world_rank == 0) {
        sendcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            int rows = base_rows + (r < rem ? 1 : 0);
            sendcounts[r] = rows * N; 
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    double *local_A = (double*)malloc((size_t)local_rows * (size_t)N * sizeof(double));
    double *B = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
    double *local_C = (double*)malloc((size_t)local_rows * (size_t)N * sizeof(double));
    if (!local_A || !B || !local_C) {
        fprintf(stderr, "Rank %d: allocation failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *fullA = NULL;
    if (world_rank == 0) {
        fullA = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
        if (!fullA) { fprintf(stderr, "Root: failed to alloc fullA\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        srand(123);
        for (int i = 0; i < N * N; ++i) fullA[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < N * N; ++i) B[i] = (double)rand() / RAND_MAX;
    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(fullA, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        double* arow = local_A + (size_t)i * (size_t)N;
        double* crow = local_C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) sum += arow[k] * B[k * N + j];
            crow[j] = sum;
        }
    }

    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *fullC = NULL;
    int *recvcounts = NULL, *rdispls = NULL;
    if (world_rank == 0) {
        fullC = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
        recvcounts = (int*)malloc(world_size * sizeof(int));
        rdispls = (int*)malloc(world_size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            int rows = base_rows + (r < rem ? 1 : 0);
            recvcounts[r] = rows * N;
            rdispls[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(local_C, local_rows * N, MPI_DOUBLE,
                fullC, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("%d,%.8f,%d\n", world_size, max_elapsed, N);
        fflush(stdout);
    }

    if (fullA) free(fullA);
    if (fullC) free(fullC);
    if (sendcounts) free(sendcounts);
    if (displs) free(displs);
    if (recvcounts) free(recvcounts);
    if (rdispls) free(rdispls);
    free(local_A);
    free(B);
    free(local_C);

    MPI_Finalize();
    return 0;
}

