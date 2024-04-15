#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define X        600 //Number of rows
#define Y        600 //Number of columns
#define MAX_ITER 2400
#define TAG1     123
#define TAG2     321

bool *gen_field() {
    bool *field = (bool *) calloc(X * Y, sizeof(bool));

    field[0 * Y + 1] = true;
    field[1 * Y + 2] = true;
    for (int i = 0; i < 3; ++i)
        field[2 * Y + i] = true;

    return field;
}

void set_matrix_part(int *sendcounts, int *displs, int size) {
    int offset = 0;
    int linecounts[size];
    int offsets[size];
    for (int i = 0; i < size; ++i) {
        linecounts[i] = X / size;
        if (i < X % size)
            ++linecounts[i];
        
        offsets[i] = offset;
        offset += linecounts[i];
        sendcounts[i] = linecounts[i] * Y;
        displs[i] = offsets[i] * Y;  
    }
}

void compare_with_prev(bool *field_part, bool **prev_field_parts, int len, bool *stop_flags, size_t iter) {
    for (size_t i = 0; i < iter; ++i) {
        if (!memcmp(field_part, prev_field_parts[i], len)) {
            stop_flags[i] = true;
        } else {
            stop_flags[i] = false;
        }
    }
}

void update_field(bool *field, bool *next_gen_field, int len) {
    for (int i = 0; i < len / Y; ++i) {
        for (int j = 0; j < Y; ++j) {
            int num_of_neighbours = field[(i - 1) * Y + (Y + j - 1) % Y] +
                                    field[(i - 1) * Y + j] +
                                    field[(i - 1) * Y + (j + 1) % Y] + 
                                    field[i * Y + (Y + j - 1) % Y] +
                                    field[i * Y + (j + 1) % Y] +
                                    field[(i + 1) * Y + (Y + j - 1) % Y] +
                                    field[(i + 1) * Y + j] +
                                    field[(i + 1) * Y + (j + 1) % Y];

            if (field[i * Y + j] == false) {
                next_gen_field[i * Y + j] = (num_of_neighbours == 3) ? true : false;
            } else if (field[i * Y + j] == true) {
                next_gen_field[i * Y + j] = 
                    (num_of_neighbours < 2 || num_of_neighbours > 3) ? false : true;  
            }                             
        }
    }
}

bool check_flags(bool *stop_flags_reduced, size_t iter) {
    for (size_t i = 0; i < iter; ++i)
        if (stop_flags_reduced[i]) return true;
    return false;
}

int main(int argc, char *argv[]) {
    int rank    = 0;
    int size    = 0;
    size_t iter = 0;
    double time_start, time_end;
    bool *field, *field_part, *next_gen_part;
    bool **prev_field_parts;
    int *sendcounts, *displs;
    bool *stop_flags, *stop_flags_reduced;
    int prev_proc_rank, next_proc_rank;
    MPI_Request requests[5];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        field = gen_field();
        time_start = MPI_Wtime();
    }

    sendcounts = (int *) malloc(size * sizeof(int));
    displs     = (int *) malloc(size * sizeof(int));
    set_matrix_part(sendcounts, displs, size);

    field_part         = (bool *)  malloc(sendcounts[rank] + 2 * Y * sizeof(bool));
    next_gen_part      = (bool *)  malloc(sendcounts[rank] + 2 * Y * sizeof(bool));
    prev_field_parts   = (bool **) malloc(MAX_ITER * sizeof(bool *));
    stop_flags         = (bool *)  malloc(MAX_ITER * sizeof(bool));
    stop_flags_reduced = (bool *)  malloc(MAX_ITER * sizeof(bool));

    MPI_Scatterv(field, sendcounts, displs, 
                 MPI_C_BOOL, field_part + Y, sendcounts[rank], 
                 MPI_C_BOOL, 
                 0, MPI_COMM_WORLD);
    
    prev_proc_rank = (size + (rank - 1)) % size;
    next_proc_rank = (rank + 1) % size;

    while (iter < MAX_ITER) {
        MPI_Isend(field_part + Y, Y, MPI_C_BOOL, prev_proc_rank, TAG1, 
                  MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(field_part + sendcounts[rank], Y, MPI_C_BOOL, next_proc_rank, TAG2,
                  MPI_COMM_WORLD, &requests[1]);
        MPI_Irecv(field_part, Y, MPI_C_BOOL, prev_proc_rank, TAG2,
                  MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(field_part + sendcounts[rank] + Y, Y, MPI_C_BOOL, next_proc_rank, TAG1,
                  MPI_COMM_WORLD, &requests[3]);

        prev_field_parts[iter] = (bool *) malloc(sendcounts[rank] * sizeof(bool));
        memcpy(prev_field_parts[iter], field_part + Y, sendcounts[rank] * sizeof(bool));
        compare_with_prev(field_part + Y, prev_field_parts, sendcounts[rank], stop_flags, iter);

        MPI_Iallreduce(stop_flags, stop_flags_reduced, iter + 1,
                       MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD, &requests[4]);

        update_field(field_part + 2 * Y, next_gen_part + 2 * Y, sendcounts[rank] - 2 * Y);

        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        update_field(field_part + Y, next_gen_part + Y, Y);

        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        update_field(field_part + sendcounts[rank], next_gen_part + sendcounts[rank], Y);

        MPI_Wait(&requests[4], MPI_STATUS_IGNORE);
        if (check_flags(stop_flags_reduced, iter)) break;

        bool *tmp = field_part;
        field_part = next_gen_part;
        next_gen_part = tmp;
        ++iter;
    }

    if (rank == 0) {
        time_end = MPI_Wtime();
        printf("Time taken: %f sec.\n", time_end - time_start);
        printf("%zu\n", iter);
        free(field);
    }
    for (size_t i = 0; i < iter; ++i)
        free(prev_field_parts[i]);
    free(prev_field_parts);
    free(sendcounts);
    free(displs);
    free(field_part);
    free(next_gen_part);
    free(stop_flags);
    free(stop_flags_reduced);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
