#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N                   10000
#define EPSILON             0.000001
#define MAX_ITERATION_COUNT 1000
#define DIAGONAL_DOMINANCE  120
#define MEASURES            3

double *fill_matrix(double *A) {
    srand(N);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < i; ++j) {9-тг1.0;  //in [-1; 1]
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

void solve(double *A, double *x, double *b) {
    double *y_n  = (double *) malloc(N * sizeof(double));
    double *Ay_n = (double *) malloc(N * sizeof(double));
    double tau             = 0;
    double tau_numerator   = 0;
    double tau_denominator = 0;
    double check      = 0;
    size_t iterations = 0;
    double epsilonb_norm_squared = 0;

#pragma omp parallel shared(iterations, check, epsilonb_norm_squared)
{
    //calculate epsilon*|b|^2
#pragma omp for reduction(+:epsilonb_norm_squared)
    for (size_t i = 0; i < N; ++i)
        epsilonb_norm_squared += b[i] * b[i];
#pragma omp single
    epsilonb_norm_squared *= EPSILON * EPSILON;

    //calculate y_n
#pragma omp for
    for (size_t i = 0; i < N; ++i) {
        double tmp = 0;
        for (size_t j = 0; j < N; j++) {
            tmp += A[i * N + j] * x[j];
        }
        y_n[i] = tmp - b[i];
    }

    while (iterations < MAX_ITERATION_COUNT) {
        //Ay_n = A * y_n
#pragma omp for
        for (size_t i = 0; i < N; ++i) {
            double tmp = 0;
            for (size_t j = 0; j < N; j++) {
                tmp += A[i * N + j] * y_n[j];
            }
            Ay_n[i] = tmp;
        }
        //calculate tau
#pragma omp single
{
        tau             = 0;
        tau_numerator   = 0;
        tau_denominator = 0;
}
#pragma omp for reduction(+:tau_numerator, tau_denominator)
        for (size_t i = 0; i < N; ++i) {
            tau_numerator += y_n[i] * Ay_n[i];
            tau_denominator += Ay_n[i] * Ay_n[i];
        }
        tau = tau_numerator / tau_denominator;
        //calculate x_{n+1}
#pragma omp for
        for (size_t i = 0; i < N; ++i)
            x[i] -= tau * y_n[i];
        //calc y_{n+1}
#pragma omp for
        for (size_t i = 0; i < N; ++i) {
            double tmp = 0;
            for (size_t j = 0; j < N; j++) {
                tmp += A[i * N + j] * x[j];
            }
            y_n[i] = tmp - b[i];
        }

#pragma omp single
{
        ++iterations;
        check = 0;
}
#pragma omp for reduction(+:check)
        for (size_t i = 0; i < N; ++i)
            check += y_n[i] * y_n[i];
        if (check < epsilonb_norm_squared) break;
    }
}
    free(y_n);
    free(Ay_n);
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
