#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define VECTOR_SIZE           95000
#define NUMBER_OF_MEASURMENTS 5
#define TAG                   123

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
    long long int s = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    size_t part_size = (VECTOR_SIZE % size == 0) ? VECTOR_SIZE / size : VECTOR_SIZE / size + 1;

    int *vector1, *vector2;

    if (rank == 0) {
        double time_start = 0;
        double time_end   = 0;
        double min_time   = ULLONG_MAX;
        long long int tmp = 0;

        vector1 = malloc(VECTOR_SIZE * sizeof(int));
        vector2 = malloc(VECTOR_SIZE * sizeof(int));

        init_vectors(vector1, vector2);

        for (size_t k = 0; k < NUMBER_OF_MEASURMENTS; ++k) {
            s = 0;
            time_start = MPI_Wtime();
            for (int i = 1; i < size; ++i) {
                MPI_Send(vector1 + i * part_size, part_size, MPI_INT, i, TAG, MPI_COMM_WORLD);
                MPI_Send(vector2, VECTOR_SIZE, MPI_INT, i, TAG, MPI_COMM_WORLD);
            }
            s = mult(vector1, vector2, part_size);
            for (size_t l = 1; l < size; ++l) {
                MPI_Recv(&tmp, 1, MPI_LONG_LONG_INT, l, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                s += tmp;
            }
            time_end   = MPI_Wtime();
            if (time_end - time_start < min_time) min_time = time_end - time_start;
        }
        printf("s = %lld\n", s);
        printf("Time taken: %f sec.\n", min_time);
    } else {
        vector1 = malloc(part_size * sizeof(int));
        vector2 = malloc(VECTOR_SIZE * sizeof(int));
        for (size_t k = 0; k < NUMBER_OF_MEASURMENTS; ++k) {
            s = 0;
            MPI_Recv(vector1, part_size, MPI_INT, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(vector2, VECTOR_SIZE, MPI_INT, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            s += mult(vector1, vector2, part_size);
            MPI_Send(&s, 1, MPI_LONG_LONG_INT, 0, TAG, MPI_COMM_WORLD);
        }
    }

    free(vector1);
    free(vector2);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
