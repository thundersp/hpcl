#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int n = 10; 
    int subarray_size;
    int local_sum = 0;
    int total_sum = 0;
    int* array = NULL;
    int* subarray = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        printf("This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    subarray_size = n / 2;

    if (rank == 0) {
        array = (int*)malloc(n * sizeof(int));
        printf("Generating array of size %d...\n", n);
        for (int i = 0; i < n; i++) {
            array[i] = i + 9; 
            printf("%d ", array[i]);
        }
        printf("\n");
        MPI_Send(&array[subarray_size], subarray_size, MPI_INT, 1, 0, MPI_COMM_WORLD);

        for (int i = 0; i < subarray_size; i++) {
            local_sum += array[i];
        }
        printf("Process 0 calculated local sum: %d\n", local_sum);

        int received_sum;
        MPI_Recv(&received_sum, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        total_sum = local_sum + received_sum;
        printf("Final total sum is: %d\n", total_sum);

        free(array);

    } else if (rank == 1) {
        subarray = (int*)malloc(subarray_size * sizeof(int));

        MPI_Recv(subarray, subarray_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < subarray_size; i++) {
            local_sum += subarray[i];
        }
        printf("Process 1 calculated local sum: %d\n", local_sum);

        MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        free(subarray);
    }

    MPI_Finalize();
    return 0;
}