#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define P1    2
#define P2    12
#define N1    3000
#define N2    3600
#define N3    4200
#define NDIMS 2

double *gen_matrix(size_t matrix_rows, size_t matrix_cols) {
    double *A = (double *) malloc(matrix_rows * matrix_cols * sizeof(double));
    srand(0);

    for (size_t i = 0; i < matrix_rows; ++i) {
        for (size_t j = 0; j < matrix_cols; ++j) {
            A[i * matrix_cols + j] = (double) rand() / RAND_MAX * (double) rand();
        }
    }
    return A;
}

int main(int argc, char *argv[]) {
    int size      = 0;
    int grid_rank = 0;
    int row_width = N1 / P1;
    int col_width = N3 / P2;
    double time_start;
    double time_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != P1 * P2) {
        fprintf(stderr, "Expected %d, instead of %d processes\n", P1 * P2, size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm COMM_GRID;
    int dims[NDIMS] = {P1, P2};
    int periods[NDIMS] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, 
                    periods, false, &COMM_GRID);

    MPI_Comm COMM_ROW;
    int row_dims[NDIMS] = {false, true};
    MPI_Cart_sub(COMM_GRID, row_dims, &COMM_ROW);

    MPI_Comm COMM_COL;
    int col_dims[NDIMS] = {true, false};
    MPI_Cart_sub(COMM_GRID, col_dims, &COMM_COL);

    int coords[NDIMS] = {0, 0};
    MPI_Comm_rank(COMM_GRID, &grid_rank);
    MPI_Cart_coords(COMM_GRID, grid_rank, NDIMS, coords);

    MPI_Datatype TYPE_ROW;
    MPI_Type_contiguous(N2 * row_width, MPI_DOUBLE, &TYPE_ROW);
    MPI_Type_commit(&TYPE_ROW);

    MPI_Datatype TYPE_COL, TYPE_COL_RESIZED;
    MPI_Type_vector(N2, 
                    col_width, N3, MPI_DOUBLE, &TYPE_COL);
    MPI_Type_commit(&TYPE_COL);
    MPI_Type_create_resized(TYPE_COL, 
                            0, 
                            col_width * sizeof(double), 
                            &TYPE_COL_RESIZED);
    MPI_Type_commit(&TYPE_COL_RESIZED);

    MPI_Datatype TYPE_MINOR, TYPE_MINOR_RESIZED;
    MPI_Type_vector(row_width, 
                    col_width, N3, MPI_DOUBLE, &TYPE_MINOR);
    MPI_Type_commit(&TYPE_MINOR);
    MPI_Type_create_resized(TYPE_MINOR, 
                            0, 
                            col_width * sizeof(double), 
                            &TYPE_MINOR_RESIZED);
    MPI_Type_commit(&TYPE_MINOR_RESIZED);

    double *A, *B, *C;
    if (coords[0] == 0 && coords[1] == 0) {
        A = gen_matrix(N1, N2);
        B = gen_matrix(N2, N3);
        C = (double *) calloc(N1 * N3, sizeof(double));
        time_start = MPI_Wtime();
    }

    double *part_A = (double *) malloc(row_width * N2 * sizeof(double));
    double *part_B = (double *) malloc(col_width * N2 * sizeof(double));
    double *part_C = (double *) calloc(row_width * col_width, sizeof(double));

    if (coords[1] == 0)
        MPI_Scatter(A, 1, TYPE_ROW, 
                    part_A, 1, TYPE_ROW, 0, 
                    COMM_COL);
    MPI_Bcast(part_A, row_width * N2, MPI_DOUBLE, 0, 
              COMM_ROW);

    if (coords[0] == 0)
        MPI_Scatter(B, 1, TYPE_COL_RESIZED, 
                    part_B, col_width * N2, MPI_DOUBLE, 0, 
                    COMM_ROW);
    MPI_Bcast(part_B, col_width * N2, MPI_DOUBLE, 0, 
              COMM_COL);

    for (int i = 0; i < row_width; ++i) {
        for (int k = 0; k < N2; ++k) {
            for (int j = 0; j < col_width; ++j)
                part_C[i * col_width + j] += part_A[i * N2 + k] * part_B[k * col_width + j];
        }
    }

    int *recvcounts = (int *) malloc(size * sizeof(int));
    int *displs     = (int *) malloc(size * sizeof(int));
    int displ = coords[0] * row_width * P2 + coords[1];
    MPI_Gather(&displ, 1, MPI_INT, 
               displs, 1, MPI_INT, 0, COMM_GRID);

    for (int i = 0; i < size; ++i)  recvcounts[i] = 1;

    MPI_Gatherv(part_C, row_width * col_width, MPI_DOUBLE, 
                C, recvcounts, displs, TYPE_MINOR_RESIZED, 
                0, COMM_GRID);

    if (coords[0] == 0 && coords[1] == 0) {
        time_end = MPI_Wtime();
        printf("Time taken: %f sec.\n", time_end - time_start);
        free(A);
        free(B);
        free(C);
    }
    MPI_Type_free(&TYPE_ROW);
    MPI_Type_free(&TYPE_COL);
    MPI_Type_free(&TYPE_COL_RESIZED);
    MPI_Type_free(&TYPE_MINOR);
    MPI_Type_free(&TYPE_MINOR_RESIZED);
    free(recvcounts);
    free(displs);
    free(part_A);
    free(part_B);
    free(part_C);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
