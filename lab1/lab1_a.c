#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define VECTOR_SIZE           95000
#define NUMBER_OF_MEASURMENTS 5

void init_vectors(int *vector1, int *vector2) {
    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        vector1[i] = 1;
        vector2[i] = 2;
    }
}

double calc_time(int *vector1, int *vector2) {
    double time_start = 0;
    double time_end   = 0;
    double min_time   = ULLONG_MAX;
    long long int s   = 0;

    for (size_t k = 0; k < NUMBER_OF_MEASURMENTS; ++k) {
        s = 0;
        time_start = MPI_Wtime();
        for (size_t i = 0; i < VECTOR_SIZE; ++i) {
            for (size_t j = 0; j < VECTOR_SIZE; ++j)
                s += vector1[i] * vector2[j];
        }
        time_end   = MPI_Wtime();
        if (time_end - time_start < min_time) min_time = time_end - time_start;
    }

    printf("s = %lld\n", s);
    return min_time;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int *vector1 = malloc(VECTOR_SIZE * sizeof(int));
    int *vector2 = malloc(VECTOR_SIZE * sizeof(int));

    init_vectors(vector1, vector2);

    printf("Time taken: %f sec.\n", calc_time(vector1, vector2));

    free(vector1);
    free(vector2);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
