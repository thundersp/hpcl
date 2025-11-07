#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("I am process with rank %d out of %d processes.\n", world_rank, world_size);

    MPI_Finalize();
    
    return 0;
}   