Installation of MPI & Implementation of basic functions of MPI

Steps of Installation:

1. For linux: use openmpi
    1. sudo apt install openmpi-bin openmpi-common libopenmpi-dev
2. Include the path in code editor:
    1. /usr/lib/x86_64-linux-gnu/openmpi/include
3. Compile and running of mpi-program:
    1. mpicc input_file_name.c -o output_file_name
    2. mpirun -np “no.of_processors” ./output_file_name

**PS1:** Implementing a simple hello world program by setting number of processors = 10

**MPI (Message Passing Interface)** is a standardized and portable **parallel programming model** that allows multiple processes to communicate with each other—usually across different CPUs or even different computers in a cluster.

Used where problems are divided into smaller subproblems and solved simultaneously by multiple processors

SPMD: Each process runs its own copy of the same program

       Each process has its own memory space

 Can communicate with other processes via message passing

 Uniquely identified by a rank within a communicator

1. MPI_Init: Initializing the MPI Environment

               Must be the first call in program

         Sets up the infrastructure that allows processes to communicate

         After this all MPI functions can be used

1. Getting the total number of processes:
    
           MPI_COMM_WORLD: Default communicator representing all processes launched in your MPI program.
    
           MPI_Comm_Size: Returns total number of processes in that communicator
    

int world_size;

MPI_Comm_Size(MPI_COMM_WORLD, &world_size)

This world size is taken from the line: as 4 from
mpirun -np 4 ./a.out

1. Getting the rank for each process: each process has a unique rank betn 0 and world_size - 1

      int world_rank;

MPI_Comm_rank(MPI_COMM_WORLD, &world_rank)

By it’s help MPI identifies processes when sending or receiving messages

1. MPI_Finalize(): 
    
    To clean up the MPI environment, we cannot call any other MPI functions after this
     
    

**PS2:** Implement a program to display the rank and the communicator group of five processes

Theory: MPI groups and communicator

Communicator: A communication context that defines which processes can talk to each other

                    MPI_COMM_WORLD: includes all process

Group: A set of process ranks that defines membership inside a communicator, each communicator is associated with exactly one group

1. Getting basic information
    
    world_size, world_rank
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size)
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank)
    

1. Extracting the group from a communicator
    
    MPI_Group world_group
    
    MPI_Comm_group(MPI_COMM_WORLD, &world_group)
    
    Associates group of processes that belong to the communicator MPI_COMM_WORLD
    
    world_group → Represents the group of all processes in program
    
    MPI_COMM_WORLD, as a network and world_group as list of members
    

1. Querying the group information
    
        MPI_Group_size(world_group, &group_size);
    
        MPI_Group_rank(world_group, &group_rank);
    

1. MPI_Group_free(&world_group) 

             Frees up the group resources

**PS3: Deadlock**

Deadlocks in message passing are crucial to understand how MPI’s blocking communication works

What is a deadlock: 

Two or more processes are waiting on each other indefinitely, and no one can proceed

In MPI: 

Every process is waiting for a message that will never arrive because the other process is also waiting

Both processes are doing MPI_Recv first

Each process is blocking, waiting for message to arrive

But no one has sent anything yet

Therefore both are stuck waiting on each other

To avoid this another process must first do MPI_Send

Non blocking calls

MPI_ISend, MPI_IRecv:  Functions start operations but returns immediately, the actual data transfer may still be in progress, the program can do other work while communication completes

**PS4: Implement blocking MPI send & receive to demonstrate Nearest neighbor exchange of data in a ring topology**

Ring Topology:

Each process is connected to two neighbours

Left neighbour → Sends data to you

Right neighbour → Receives data from you

Ring neighbour calculation →

To prevent deadlock:
Use alternate odd even message sending 

**PS:5 : Array manipulation**

Parallel Reduction Pattern: Dividing work across processes, then combining their partial results

1. Setting up the processes

Process P0: Sums first half of the array

Process P1: Sums up the second half

1. Giving ranks to each process to ensure calculation pathways are different

1. Process 0: First half:
    
    if (rank == 0) {
    // P0: Sum first half (0,1,2,3)
    for (int i = 0; i < 4; i++) {
    sum += array[i];
    }
    printf("P0 sum = %d\n", sum);
    
2. Process 1: Second half:
    
     
    
    } else {
    // P1: Sum second half (4,5,6,7)
    for (int i = 4; i < 8; i++) {
    sum += array[i];
    }
    printf("P1 sum = %d\n", sum);
    

Assignment-7:

**PS1:**

Implement Matrix-Vector Multiplication using MPI. Use a different number of processes and analyze the performance

## **1. Objective**

To implement **parallel Matrix–Vector Multiplication** using **MPI (Message Passing Interface)** and to analyze its performance with a varying number of processes.

---

## **2. Problem Statement**

Write an MPI program to perform multiplication of a square matrix `A` (size `N × N`) with a vector `x` (size `N × 1`).

Each process should compute a subset of the matrix rows and then send the partial results to the master process (`P0`) to combine and produce the final result.

Also, record the execution time using `MPI_Wtime()` and analyze performance when the program is executed with different numbers of processes.

---

## **3. Algorithm**

### **Step 1:** Initialize MPI environment.

- Call `MPI_Init(&argc, &argv)`.
- Obtain the process rank using `MPI_Comm_rank()`.
- Obtain the total number of processes using `MPI_Comm_size()`.

### **Step 2:** Initialize data.

- Define matrix `A[N][N]` and vector `x[N]`.
- Process `P0` initializes both with test data.
- Other processes allocate memory for their portions.

### **Step 3:** Broadcast the vector.

- Use `MPI_Bcast()` to send the complete vector from `P0` to all other processes since every process needs the entire vector for multiplication.

### **Step 4:** Divide the matrix among processes.

- Each process computes how many rows it should handle using:rows_per_process=sizeN
    
    \[
\text{rows\_per\_process} = \frac{N}{\text{size}}
\]

    
- Calculate the start and end row indices for each process.

### **Step 5:** Distribute the matrix.

- `P0` sends each process its subset of matrix rows using `MPI_Send()`.
- Each process receives its part using `MPI_Recv()`.

### **Step 6:** Local computation.

- Each process multiplies its assigned rows of the matrix with the full vector:yi=j=0∑N−1Aij×xj
    

\[
y_i = \sum_{j=0}^{N-1} A_{ij} \times x_j
\]



    
- Store results in a local result array.

### **Step 7:** Gather results.

- Each process sends its local result vector to `P0` using `MPI_Send()`.
- `P0` receives all partial results using `MPI_Recv()` and merges them into the final result array.

### **Step 8:** Record execution time.

- Use `MPI_Wtime()` to measure the computation time.

### **Step 9:** Display and save results.

- Process `P0` prints:
    - Matrix size
    - Number of processes
    - Computation time
    - First few result values
- Append the time result to a performance file `simple_performance.txt`.

### **Step 10:** Finalize MPI.

- Call `MPI_Finalize()` to end the MPI environment.

---

## PS2:

**Implement Matrix-Matrix Multiplication using MPI. Use a different number of processes and analyze the performance.**
 

## **1. Objective**

To implement **parallel Matrix–Matrix Multiplication** using **MPI (Message Passing Interface)** and analyze its performance with varying numbers of processes.

---

## **2. Problem Statement**

Perform multiplication of two square matrices `A` and `B` of size `N × N` using multiple MPI processes. Each process computes a subset of rows of the resulting matrix `C` and sends its partial results to the master process (`P0`) to assemble the final matrix. Execution time is measured and logged to analyze performance scaling.

---

## **3. Algorithm**

### **Step 1:** Initialize MPI environment

- Call `MPI_Init(&argc, &argv)`.
- Get the process rank using `MPI_Comm_rank()`.
- Get the total number of processes using `MPI_Comm_size()`.

### **Step 2:** Initialize matrices

- Only process `P0` initializes matrices `A` and `B` with sample values (1–10).
- Other processes allocate memory for local computations.

### **Step 3:** Broadcast matrix B

- Use `MPI_Bcast()` to send matrix `B` from `P0` to all other processes since all processes need `B` for multiplication.

### **Step 4:** Divide matrix A among processes

- Each process calculates:



\[
\text{rows\_per\_process} = \frac{\text{size}}{N}
\]


- Determine the starting and ending row indices for each process.

### **Step 5:** Distribute rows of matrix A

- `P0` sends chunks of matrix `A` to each worker process using `MPI_Send()`.
- Each worker receives its portion using `MPI_Recv()`.
- `P0` keeps its own portion of rows locally.

### **Step 6:** Local matrix multiplication

- Each process computes its assigned rows of the result matrix `C`:

\[
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
\]


- Store results in `local_C`.

### **Step 7:** Gather results

- Worker processes send their computed rows back to `P0` using `MPI_Send()`.
- `P0` receives partial results from each process using `MPI_Recv()` and assembles the full matrix `C`.

### **Step 8:** Measure execution time

- Use `MPI_Wtime()` to record computation time for local matrix multiplication.

### **Step 9:** Output results and performance

- `P0` prints:
    - Execution time
    - Number of processes used
    - First few elements of the result matrix for verification
- Save performance data to `matrix_matrix_performance.txt`.

### **Step 10:** Finalize MPI

- Call `MPI_Finalize()` to end MPI execution.
