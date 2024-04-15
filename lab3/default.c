#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N                   10000
#define EPSILON             0.000001
#define MAX_ITERATION_COUNT 10000
#define DIAGONAL_DOMINANCE  125
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

void matrix_to_vect_mul(const double *matrix, const double *vect, double *result) {
    for (size_t i = 0; i < N; ++i) {
        double tmp = 0;
        for (size_t j = 0; j < N; j++) {
            tmp += matrix[i * N + j] * vect[j];
        }
        result[i] = tmp;
    }
}

void vect_to_scalar_mul(const double *vect, double scalar, double *result) {
    for (size_t i = 0; i < N; ++i)
        result[i] = vect[i] * scalar;
}

void vect_sub(const double *vect1, const double *vect2, double *result) {
    for (size_t i = 0; i < N; ++i)
        result[i] = vect1[i] - vect2[i];
}

double scalar_mul(const double *vect1, const double *vect2) {
    double result = 0;

    for (size_t i = 0; i < N; ++i)
        result += vect1[i] * vect2[i];
    return result;
}

inline void solve(double *A, double *x, double *b) {
    double *y_n  = (double *) malloc(N * sizeof(double));
    double *Ay_n = (double *) malloc(N * sizeof(double));
    double *tmp  = (double *) malloc(N * sizeof(double));
    double tau;
    size_t iterations = 0;
    double epsilonb_norm_squared = EPSILON * EPSILON * scalar_mul(b, b);

    //calculate y_n
    matrix_to_vect_mul(A, x, tmp);
    vect_sub(tmp, b, y_n);
    while (iterations < MAX_ITERATION_COUNT) {
        matrix_to_vect_mul(A, y_n, Ay_n);   //Ay_n = A * y_n
        //calculate tau
        tau = scalar_mul(y_n, Ay_n) / scalar_mul(Ay_n, Ay_n);
        //calculate x_{n+1}
        vect_to_scalar_mul(y_n, tau, tmp);
        vect_sub(x, tmp, x);
        //calc y_{n+1}
        matrix_to_vect_mul(A, x, tmp);  //tmp = A * x
        vect_sub(tmp, b, y_n);          //y_n = A * x - b

        ++iterations;
        if (scalar_mul(y_n, y_n) < epsilonb_norm_squared) break;
    }
    fprintf(stderr, "Number of iterations: %zu\n", iterations);
    free(y_n);
    free(Ay_n);
    free(tmp);
}

int main(int argc, char *argv[]) {
    double time_start = 0;
    double time_end   = 0;
    double min_time   = ULLONG_MAX;
    double *A         = (double *) malloc(N * N * sizeof(double));
    double *x         = (double *) malloc(N * sizeof(double));
    double *b         = (double *) malloc(N * sizeof(double));

    for (int i = 0; i < MEASURES; ++i) {
        fill_matrix(A);
        fill_vect(x);
        fill_vect(b);
        time_start = omp_get_wtime();
        solve(A, x, b);
        time_end   = omp_get_wtime();
        if (time_end - time_start < min_time) min_time = time_end - time_start;
    }
    printf("Time taken: %f sec.\n", min_time);

    free(A);
    free(x);
    free(b);
    return EXIT_SUCCESS;
}
