#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 2000;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) N = 2000;

    int base_rows = N / world_size;
    int rem = N % world_size;
    int local_rows = base_rows + (world_rank < rem ? 1 : 0);

    int* sendcounts = NULL;
    int* displs = NULL;
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

    double* local_A = (double*)malloc((size_t)local_rows * (size_t)N * sizeof(double));
    if (!local_A) { fprintf(stderr, "Rank %d: failed to alloc local_A\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    double* x = (double*)malloc((size_t)N * sizeof(double));
    double* local_y = (double*)malloc((size_t)local_rows * sizeof(double));
    if (!x || !local_y) { fprintf(stderr, "Rank %d: failed to alloc x/local_y\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    double* fullA = NULL;
    if (world_rank == 0) {
        fullA = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
        if (!fullA) { fprintf(stderr, "Root: failed to alloc fullA\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
       
        srand(42);
        for (int i = 0; i < N * N; ++i) fullA[i] = (double)(rand()) / RAND_MAX;
        for (int i = 0; i < N; ++i) x[i] = (double)(rand()) / RAND_MAX;
    }

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(fullA, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        double* row = local_A + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) sum += row[j] * x[j];
        local_y[i] = sum;
    }

    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int* recvcounts = NULL;
    int* rdispls = NULL;
    double* fully = NULL;
    if (world_rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        rdispls = (int*)malloc(world_size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            int rows = base_rows + (r < rem ? 1 : 0);
            recvcounts[r] = rows;
            rdispls[r] = offset;
            offset += rows;
        }
        fully = (double*)malloc((size_t)N * sizeof(double));
    }

    MPI_Gatherv(local_y, local_rows, MPI_DOUBLE,
                fully, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("%d,%.8f,%d\n", world_size, max_elapsed, N);
        fflush(stdout);
    }
    if (fullA) free(fullA);
    if (fully) free(fully);
    if (sendcounts) free(sendcounts);
    if (displs) free(displs);
    if (recvcounts) free(recvcounts);
    if (rdispls) free(rdispls);
    free(local_A);
    free(x);
    free(local_y);

    MPI_Finalize();
    return 0;
}
