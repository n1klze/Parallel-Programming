#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N                   10000
#define EPSILON             0.000001
#define MAX_ITERATION_COUNT 10000
#define DIAGONAL_DOMINANCE  120
#define MEASURES            1

double *fill_matrix(double *A) {
    srand(N);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) {
            A[i * N + j] = A[j * N + i];
        }
        for (size_t j = i; j < N; ++j) {
            A[i * N + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;  //in [-1; 1]
            if (i == j) A[i * N + j] += DIAGONAL_DOMINANCE;
        }
    }
    return A;
}

double *fill_vect(double *vect) {
    for (size_t i = 0; i < N; ++i)
        vect[i] = (double) rand() / RAND_MAX * (double) rand();
    return vect;
}

void matrix_to_vect_mul(const double *matrix, int num_of_lines, const double *vect, double *result) {
    for (int i = 0; i < num_of_lines; ++i) {
        double tmp = 0;
        for (int j = 0; j < N; j++) {
            tmp += matrix[i * N + j] * vect[j];
        }
        result[i] = tmp;
    }
}

void vect_to_scalar_mul(const double *vect, int len, double scalar, double *result) {
    for (size_t i = 0; i < len; ++i)
        result[i] = vect[i] * scalar;
}

void vect_sub(const double *vect1, const double *vect2, int len, double *result) {
    for (int i = 0; i < len; ++i)
        result[i] = vect1[i] - vect2[i];
}

double scalar_mul(const double *vect1, const double *vect2, int len) {
    double result = 0;

    for (size_t i = 0; i < len; ++i)
        result += vect1[i] * vect2[i];
    return result;
}

void set_matrix_part(int *sendcounts, int *displs, int *linecounts, int *offsets, int size) {
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        linecounts[i] = N / size;
        if (i < N % size)
            ++linecounts[i];
        
        offsets[i] = offset;
        offset += linecounts[i];
        sendcounts[i] = linecounts[i] * N;
        displs[i] = offsets[i] * N;  
    }
}

inline void solve(double *A, double *x, double *b, int rank, int size) {
    double epsilonb_norm_squared;
    double *y_n = (double *) malloc(N * sizeof(double));
    double tau;
    double temp;
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *) malloc(size * sizeof(int));
    int *displs     = (int *) malloc(size * sizeof(int));
    int *linecounts = (int *) malloc(size * sizeof(int));
    int *offsets    = (int *) malloc(size * sizeof(int));
    set_matrix_part(sendcounts, displs, linecounts, offsets, size);
    double *A_part  = (double *) malloc(sendcounts[rank] * sizeof(double));
    double *b_part  = (double *) malloc(linecounts[rank] * sizeof(double));
    double *x_part  = (double *) malloc(linecounts[rank] * sizeof(double));
    double *y_part  = (double *) malloc(linecounts[rank] * sizeof(double));
    double *Ay_part = (double *) malloc(linecounts[rank] * sizeof(double));
    double *tmp     = (double *) malloc(linecounts[rank] * sizeof(double));
    size_t iter     = 0;    

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_part, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, linecounts, offsets, MPI_DOUBLE, b_part, linecounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    temp = EPSILON * EPSILON * scalar_mul(b_part, b_part, linecounts[rank]);
    MPI_Allreduce(&temp, &epsilonb_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //calculate y_part
    matrix_to_vect_mul(A_part, linecounts[rank], x, tmp);
    vect_sub(tmp, b_part, linecounts[rank], y_part);
    while (iter < MAX_ITERATION_COUNT) {
        MPI_Allgatherv(y_part, linecounts[rank], MPI_DOUBLE, y_n, linecounts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
        
        matrix_to_vect_mul(A_part, linecounts[rank], y_n, Ay_part);

        double tau_numerator, tau_denominator;
        temp = scalar_mul(y_part, Ay_part, linecounts[rank]);
        MPI_Allreduce(&temp, &tau_numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        temp = scalar_mul(Ay_part, Ay_part, linecounts[rank]);
        MPI_Allreduce(&temp, &tau_denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //calculate tau
        tau = tau_numerator / tau_denominator;

        //calculate x_{n+1}
        vect_to_scalar_mul(y_part, linecounts[rank], tau, tmp);
        MPI_Scatterv(x, linecounts, offsets, MPI_DOUBLE, x_part, linecounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vect_sub(x_part, tmp, linecounts[rank], x_part);
        MPI_Allgatherv(x_part, linecounts[rank], MPI_DOUBLE, x, linecounts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);

        //calc y_{n+1}
        matrix_to_vect_mul(A_part, linecounts[rank], x, tmp);
        vect_sub(tmp, b_part, linecounts[rank], y_part);
        temp = scalar_mul(y_part, y_part, linecounts[rank]);
        double res;
        MPI_Allreduce(&temp, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        ++iter;
        if (res < epsilonb_norm_squared) break;
    }

    if (rank == 0) fprintf(stderr, "Number of iterations: %zu\n", iter);
    free(y_n);
    free(sendcounts);
    free(displs);
    free(linecounts);
    free(offsets);
    free(A_part);
    free(b_part);
    free(x_part);
    free(y_part);
    free(tmp);
}

int main(int argc, char *argv[]) {
    int rank = 0;
    int size = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    double *A;
    double *x = (double *) malloc(N * sizeof(double));
    double *b;
    double min_time = ULLONG_MAX;
    if (rank == 0) {
        A = (double *) malloc(N * N * sizeof(double));
        b = (double *) malloc(N * sizeof(double));
    }

    for (int i = 0; i < MEASURES; ++i) {
        if (rank == 0) {
            fill_matrix(A);
            fill_vect(x);
            fill_vect(b);
        }
        double time_start = MPI_Wtime();
        solve(A, x, b, rank, size);
        double time_end   = MPI_Wtime();
        if (time_end - time_start < min_time) min_time = time_end - time_start;
    }

    if (rank == 0) {
        printf("Time taken: %f sec.\n", min_time);
        free(A);
        free(b);
    }
    free(x);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
