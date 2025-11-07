#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int buffer_size = 1024 * 1024; 
    int* send_buffer = (int*)malloc(sizeof(int) * buffer_size);
    int* recv_buffer = (int*)malloc(sizeof(int) * buffer_size);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 8) {
        printf("This program is designed to run with exactly 8 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank % 2 == 0) { 
        printf("Rank %d: Sending data to Rank %d...\n", rank, rank + 1);
        MPI_Send(send_buffer, buffer_size, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

        printf("Rank %d: Waiting to receive from Rank %d...\n", rank, rank + 1);
        MPI_Recv(recv_buffer, buffer_size, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else { 
        printf("Rank %d: Sending data to Rank %d...\n", rank, rank - 1);
        MPI_Send(send_buffer, buffer_size, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);

        printf("Rank %d: Waiting to receive from Rank %d...\n", rank, rank - 1);
        MPI_Recv(recv_buffer, buffer_size, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(send_buffer);
    free(recv_buffer);
    MPI_Finalize();
    return 0;
}