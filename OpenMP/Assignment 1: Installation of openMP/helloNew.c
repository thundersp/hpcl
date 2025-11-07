#include <stdio.h>
#include <omp.h>

int main()
{
    int num_threads;
    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);

    omp_set_num_threads(num_threads);

    printf("\nSequential Execution:\n");
    for (int i = 0; i < num_threads; i++)
    {
        printf("Sequential: Hello, World! {%d}\n", i + 1);
    }

    printf("\nParallel Execution:\n");
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        printf("Parallel: Hello, World! {%d of %d threads}\n",
               thread_id + 1, total_threads);
    }

    printf("\nProgram completed.\n");

    return 0;
}