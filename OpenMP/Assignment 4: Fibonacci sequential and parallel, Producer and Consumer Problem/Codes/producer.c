#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS   5

int buffer[BUFFER_SIZE];
int in = 0, out = 0, count = 0;

void print_buffer() {
    /*
    printf("Buffer: [");
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (i < count)
            printf("%d ", buffer[(out + i) % BUFFER_SIZE]);
        else
            printf("- ");
    }
    printf("]\n");
    */
}

void producer(int id) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        int item = i + id * 100;

        while (1) {
            int produced = 0;

            #pragma omp critical
            {
                if (count < BUFFER_SIZE) {
                    buffer[in] = item;
                    in = (in + 1) % BUFFER_SIZE;
                    count++;
                    // printf("Producer %d produced %d\n", id, item);
                    produced = 1;
                }
            }

            if (produced) break;
            usleep(100000);
        }

        usleep(200000);
    }
}

void consumer(int id) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        int item;
        while (1) {
            int consumed = 0;

            #pragma omp critical
            {
                if (count > 0) {
                    item = buffer[out];
                    out = (out + 1) % BUFFER_SIZE;
                    count--;
                    // printf("Consumer %d consumed %d\n", id, item);
                    consumed = 1;
                }
            }

            if (consumed) break;
            usleep(100000);
        }

        usleep(300000);
    }
}

int main() {
    FILE *fp = fopen("results.csv", "w");
    fprintf(fp, "Threads,Time\n");   // CSV header

    for (int threads = 2; threads <= 14; threads++) {
        // Reset shared variables for each run
        in = 0; out = 0; count = 0;

        double start_time = omp_get_wtime();

        #pragma omp parallel num_threads(threads) shared(buffer, in, out, count)
        {
            int tid = omp_get_thread_num();
            if (tid % 2 == 0) {
                producer(tid);
            } else {
                consumer(tid);
            }
        }

        double end_time = omp_get_wtime();
        double elapsed = end_time - start_time;

        printf("Threads=%d -> %f sec\n", threads, elapsed);
        fprintf(fp, "%d,%f\n", threads, elapsed);
    }

    fclose(fp);
    return 0;
}
