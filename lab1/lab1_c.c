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

long long mult(int *vector1, int *vector2, size_t size) {
    long long s = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < VECTOR_SIZE; ++j)
            s += vector1[i] * vector2[j];
    }
    return s;
}

int main(int argc, char *argv[]) {
    int rank = 0;
    int size = 0;
    long long int s   = 0;
    long long int tmp = 0;
    double time_start = 0;
    double time_end   = 0;
    double min_time   = ULLONG_MAX;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    size_t part_size = (VECTOR_SIZE % size == 0) ? VECTOR_SIZE / size : VECTOR_SIZE / size + 1;

    int *vector1;
    int *vector2 = malloc(VECTOR_SIZE * sizeof(int));

    if (rank == 0) {
        vector1 = malloc(VECTOR_SIZE * sizeof(int));

        init_vectors(vector1, vector2);
    }

    int *part_of_vector1 = malloc(part_size * sizeof(int));

    for (size_t k = 0; k < NUMBER_OF_MEASURMENTS; ++k) {
        tmp = 0;
        time_start = MPI_Wtime();
        MPI_Scatter(vector1, part_size, MPI_INT, part_of_vector1, part_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector2, VECTOR_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
        tmp = mult(part_of_vector1, vector2, part_size);
        MPI_Reduce(&tmp, &s, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        time_end   = MPI_Wtime();
        if (time_end - time_start < min_time) min_time = time_end - time_start;
    }

    if (rank == 0) {
        printf("s = %lld\n", s);
        printf("Time taken: %f sec.\n", min_time);
        free(vector1);
    }

    free(part_of_vector1);
    free(vector2);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
