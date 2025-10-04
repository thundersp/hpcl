#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int send_data, recv_data;
    int right_neighbor, left_neighbor;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("Please run with at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }
    right_neighbor = (rank + 1) % size;
    left_neighbor = (rank - 1 + size) % size;

    send_data = rank;

    printf("Rank %d is sending to %d and will receive from %d.\n", rank, right_neighbor, left_neighbor);

    MPI_Sendrecv(&send_data, 1, MPI_INT, right_neighbor, 0,
                 &recv_data, 1, MPI_INT, left_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank %d received data %d from its left neighbor.\n", rank, recv_data);

    MPI_Finalize();
    return 0;
}