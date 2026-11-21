
#include <stdio.h>
#include <omp.h>

int main() {

    printf("HPC OpenMP Test\n");
    printf("-------------------------\n");

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        printf("Hello from Thread %d out of %d threads\n",
               thread_id, total_threads);
    }

    return 0;
}
