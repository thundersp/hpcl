#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

int n = 16;
int total_solutions = 0;

int is_safe(int board[], int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        if (board[i] == col ||
            board[i] - i == col - row ||
            board[i] + i == col + row)
        {
            return 0;
        }
    }
    return 1;
}

void solve_nqueens_sequential(int board[], int row, int *count)
{
    if (row == n)
    {
        (*count)++;
        return;
    }

    for (int col = 0; col < n; col++)
    {
        if (is_safe(board, row, col))
        {
            board[row] = col;
            solve_nqueens_sequential(board, row + 1, count);
        }
    }
}

void solve_nqueens_parallel(int board[], int row, int *count)
{
    if (row == n)
    {
#pragma omp atomic
        (*count)++;
        return;
    }

    if (row == 0)
    {
#pragma omp parallel for
        for (int col = 0; col < n; col++)
        {
            int local_board[n];
            memcpy(local_board, board, sizeof(int) * n);
            int local_count = 0;

            if (is_safe(local_board, row, col))
            {
                local_board[row] = col;
                solve_nqueens_sequential(local_board, row + 1, &local_count);

#pragma omp atomic
                *count += local_count;
            }
        }
    }
    else
    {
        for (int col = 0; col < n; col++)
        {
            if (is_safe(board, row, col))
            {
                board[row] = col;
                solve_nqueens_parallel(board, row + 1, count);
            }
        }
    }
}

void print_solution_info(int board[])
{
    printf("Sample solution for %d-Queens:\n", n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (board[i] == j)
            {
                printf("Q ");
            }
            else
            {
                printf(". ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    printf("=== N-Queens Problem - Exploratory Decomposition ===\n");
    printf("Board size: %dx%d\n", n, n);
    printf("Number of threads: %d\n\n", omp_get_max_threads());

    int board[n];
    double start_time, end_time;

    printf("Running Sequential N-Queens...\n");
    int sequential_count = 0;
    memset(board, -1, sizeof(board));

    start_time = omp_get_wtime();
    solve_nqueens_sequential(board, 0, &sequential_count);
    end_time = omp_get_wtime();

    double sequential_time = (end_time - start_time) * 1000; // Convert to ms
    printf("Sequential solutions found: %d\n", sequential_count);
    printf("Sequential time: %.2f ms\n\n", sequential_time);

   
    printf("Running Parallel N-Queens (Exploratory Decomposition)...\n");
    int parallel_count = 0;
    memset(board, -1, sizeof(board));

    start_time = omp_get_wtime();
    solve_nqueens_parallel(board, 0, &parallel_count);
    end_time = omp_get_wtime();

    double parallel_time = (end_time - start_time) * 1000; // Convert to ms
    printf("Parallel solutions found: %d\n", parallel_count);
    printf("Parallel time: %.2f ms\n\n", parallel_time);

    
    double speedup = sequential_time / parallel_time;
    printf("=== Results ===\n");
    printf("Sequential Time: %.2f ms\n", sequential_time);
    printf("Parallel Time: %.2f ms\n", parallel_time);
    printf("Speedup: %.2fx\n", speedup);
    printf("Wasted Computation: ~0%% (exploratory - no discarded work)\n");

    memset(board, -1, sizeof(board));
    int temp_count = 0;
    solve_nqueens_sequential(board, 0, &temp_count);

    return 0;
}
