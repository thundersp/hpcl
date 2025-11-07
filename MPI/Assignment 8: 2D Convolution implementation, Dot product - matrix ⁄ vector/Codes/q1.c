#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#define DEFAULT_ITERATIONS 1

int conv_column(int *, int, int, int, int *, int);
int conv(int *, int, int, int, int *, int);
int * check(int *, int, int, int *, int);

int conv_column(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;
  
  for (int j = 1; j < (num_pads + 1); j++) {
    counter += sub_grid[i + j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) + j*kernel_dim];
    counter += sub_grid[i - j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) - j*kernel_dim];
  }
  counter += sub_grid[i] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2)];
  
  return counter;
}

int conv(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;
  counter += conv_column(sub_grid, i, nrows, DIM, kernel, kernel_dim);

  for (int j = 1; j < (num_pads + 1); j++) {
    int end = (((i / DIM) + 1) * DIM) - 1;
    if (i + j - end <= 0)
      counter += conv_column(sub_grid, i + j, nrows, DIM, kernel, kernel_dim);
    int first = (i / DIM) * DIM;
    if (i - j - first >= 0)
      counter += conv_column(sub_grid, i - j, nrows, DIM, kernel, kernel_dim);
  }
  return counter;
}

int * check(int * sub_grid, int nrows, int DIM, int * kernel, int kernel_dim) {
  int num_pads = (kernel_dim - 1) / 2;
  int * new_grid = calloc(DIM * nrows, sizeof(int));
  for(int i = (num_pads * DIM); i < (DIM * (num_pads + nrows)); i++) {
    int val = conv(sub_grid, i, nrows, DIM, kernel, kernel_dim);
    new_grid[i - (num_pads * DIM)] = val;
  }
  return new_grid;
}

int main(int argc, char** argv) {
  int num_procs, ID, DIM, KERNEL_DIM;
  int num_iterations = DEFAULT_ITERATIONS;
  double start_time, end_time;

  if (argc < 3) {
    if (ID == 0) printf("Usage: mpirun -np <p> ./q1 <DIM> <KERNEL_DIM>\n");
    MPI_Finalize();
    return -1;
  }

  DIM = atoi(argv[1]);
  KERNEL_DIM = atoi(argv[2]);
  int GRID_WIDTH = DIM * DIM;
  int main_grid[GRID_WIDTH];
  for (int i = 0; i < GRID_WIDTH; i++) main_grid[i] = 1;

  int KERNEL_SIZE = KERNEL_DIM * KERNEL_DIM;
  int kernel[KERNEL_SIZE];
  for (int i = 0; i < KERNEL_SIZE; i++) kernel[i] = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &ID);

  start_time = MPI_Wtime();

  int base_rows = DIM / num_procs;
  int remainder = DIM % num_procs;
  int nrows = base_rows + (ID < remainder ? 1 : 0);
  
  int start_row = ID * base_rows + (ID < remainder ? ID : remainder);
  int *sub_grid = &main_grid[start_row * DIM];

  int *result = check(sub_grid, nrows, DIM, kernel, KERNEL_DIM);

  end_time = MPI_Wtime();

  if (ID == 0) {
    double exec_time = end_time - start_time;
    FILE *f = fopen("results1.csv", "a");
    if (f) {
      fprintf(f, "%d,%.6f\n", num_procs, exec_time);
      fclose(f);
    }
    printf("Processes: %d | Time: %.6f sec\n", num_procs, exec_time);
  }

  free(result);
  MPI_Finalize();
  return 0;
}
