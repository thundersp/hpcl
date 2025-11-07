#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define NUM_ITERATIONS 100000

int branch1_computed = 0;
int branch2_computed = 0;
int branch1_used = 0;
int branch2_used = 0;

double compute_branch1(double x)
{
#pragma omp atomic
    branch1_computed++;

    double result = 0;
    for (int i = 0; i < 500; i++)
    {
        result += sqrt(fabs(x + i * 0.001));
    }
    return result;
}

double compute_branch2(double x)
{
#pragma omp atomic
    branch2_computed++;

    double result = 0;
    for (int i = 0; i < 500; i++)
    {
        result += log(fabs(x + i * 0.001) + 1);
    }
    return result;
}

double sequential_computation(double x)
{
    if (x > 0)
    {
        branch1_used++;
        return compute_branch1(x);
    }
    else
    {
        branch2_used++;
        return compute_branch2(x);
    }
}

double speculative_computation(double x)
{
    double result1, result2;
    double final_result;

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            result1 = compute_branch1(x);
        }
#pragma omp section
        {
            result2 = compute_branch2(x);
        }
    }

    if (x > 0)
    {
        branch1_used++;
        final_result = result1;
    }
    else
    {
        branch2_used++;
        final_result = result2;
    }

    return final_result;
}

void reset_counters()
{
    branch1_computed = 0;
    branch2_computed = 0;
    branch1_used = 0;
    branch2_used = 0;
}

int main()
{
    printf("=== If-Else Branch Evaluation - Speculative Decomposition ===\n");
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Number of iterations: %d\n\n", NUM_ITERATIONS);

    double *test_data = (double *)malloc(NUM_ITERATIONS * sizeof(double));
    srand(time(NULL));

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        test_data[i] = ((rand() % 200) - 100) * 0.1; // Random values between -10 and 10
    }

    double start_time, end_time;
    double sequential_sum = 0, parallel_sum = 0;

    printf("Running Sequential Computation...\n");
    reset_counters();

    start_time = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        sequential_sum += sequential_computation(test_data[i]);
    }
    end_time = omp_get_wtime();

    double sequential_time = (end_time - start_time) * 1000; // Convert to ms
    int seq_branch1_computed = branch1_computed;
    int seq_branch2_computed = branch2_computed;
    int seq_branch1_used = branch1_used;
    int seq_branch2_used = branch2_used;

    printf("Sequential time: %.2f ms\n", sequential_time);
    printf("Branch1 (sqrt) computed: %d, used: %d\n", seq_branch1_computed, seq_branch1_used);
    printf("Branch2 (log) computed: %d, used: %d\n", seq_branch2_computed, seq_branch2_used);
    printf("Total computations: %d\n\n", seq_branch1_computed + seq_branch2_computed);

    printf("Running Speculative Parallel Computation...\n");
    reset_counters();

    start_time = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        parallel_sum += speculative_computation(test_data[i]);
    }
    end_time = omp_get_wtime();

    double parallel_time = (end_time - start_time) * 1000; // Convert to ms

    printf("Parallel time: %.2f ms\n", parallel_time);
    printf("Branch1 (sqrt) computed: %d, used: %d\n", branch1_computed, branch1_used);
    printf("Branch2 (log) computed: %d, used: %d\n", branch2_computed, branch2_used);
    printf("Total computations: %d\n\n", branch1_computed + branch2_computed);

    double speedup = sequential_time / parallel_time;
    double total_computations = branch1_computed + branch2_computed;
    double useful_computations = branch1_used + branch2_used;
    double wasted_computation = ((total_computations - useful_computations) / total_computations) * 100;

    double difference = fabs(sequential_sum - parallel_sum);
    printf("=== Results ===\n");
    printf("Sequential Time: %.2f ms\n", sequential_time);
    printf("Parallel Time: %.2f ms\n", parallel_time);
    printf("Speedup: %.2fx\n", speedup);
    printf("Wasted Computation: %.1f%%\n", wasted_computation);
    printf("Result difference (verification): %.10f\n", difference);

    if (difference < 1e-6)
    {
        printf("✓ Results match - Speculative execution correct!\n");
    }
    else
    {
        printf("✗ Results don't match - Check implementation!\n");
    }

    printf("\n=== Analysis ===\n");
    printf("In speculative execution:\n");
    printf("- Both branches computed: %d times each\n", NUM_ITERATIONS);
    printf("- Branch1 actually needed: %d times\n", branch1_used);
    printf("- Branch2 actually needed: %d times\n", branch2_used);
    printf("- Wasted Branch1 computations: %d\n", branch1_computed - branch1_used);
    printf("- Wasted Branch2 computations: %d\n", branch2_computed - branch2_used);

    free(test_data);
    return 0;
}