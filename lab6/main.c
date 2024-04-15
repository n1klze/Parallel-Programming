#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TASK_LIST_SIZE     1000
#define NUM_OF_TASK_LISTS  5
#define L                  100000
#define MIN_TASKS_TO_SHARE 30

#define ASK_TAG               100
#define TASK_NUMBER_REPLY_TAG 200
#define TASKS_SENDING_TAG     300

int rank, size;
pthread_mutex_t mutex;
pthread_t threads[2];
int *tasks;
int remainig_tasks, completed_tasks;
double global_res, res;
double total_disbalance_factor;

void init_task_list(int iter_counter) {
    for (int i = 0; i < TASK_LIST_SIZE; ++i)
        tasks[i] = abs(50 - i % 100) * abs(rank - (iter_counter % size)) * L;
}

void execute_task_list() {
    int i = 0;
    while (remainig_tasks > 0) {
        pthread_mutex_lock(&mutex);
        int repeat_num = tasks[i];
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < repeat_num; ++j)
            global_res += sin(j) / 2;

        pthread_mutex_lock(&mutex);
        ++completed_tasks;
        --remainig_tasks;
        pthread_mutex_unlock(&mutex); 
    }
}

void ask_for_tasks() {
    for (int i = 0; i < size; ++i) {
        int additional_tasks;
        if (i != rank) {
            MPI_Send(&rank, 1, MPI_INT, i, ASK_TAG, MPI_COMM_WORLD);
            MPI_Recv(&additional_tasks, 1, MPI_INT, i, TASK_NUMBER_REPLY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (additional_tasks != 0) {
                pthread_mutex_lock(&mutex);
                MPI_Recv(tasks, additional_tasks, MPI_INT, i, TASKS_SENDING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                remainig_tasks += additional_tasks;
                pthread_mutex_unlock(&mutex);
            }                 
        }
    }
}

void *executor_start_routine(void *args) {
    tasks = (int *) malloc(TASK_LIST_SIZE * sizeof(int));
    double time_start, time_end, total_time, min_time, max_time;

    for (int iter_counter = 0; iter_counter < NUM_OF_TASK_LISTS; ++iter_counter) {
        init_task_list(iter_counter);
        remainig_tasks  = TASK_LIST_SIZE;
        completed_tasks = 0;
        time_start = MPI_Wtime();
        execute_task_list();
        for (int i = 0; i < size; ++i) {
            ask_for_tasks();
            execute_task_list();
        }
        time_end = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        total_time = time_end - time_start;
        MPI_Reduce(&total_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&global_res, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
printf("Rank: %d, tasks done: %d, time: %lf\n", rank, completed_tasks, total_time);
        if (rank == 0) {
            double disbalance        = max_time - min_time;
            double disbalance_factor = (disbalance / max_time) * 100;
            total_disbalance_factor += disbalance_factor;
fprintf(stderr, "Max time: %lf, min time: %lf, result: %lf, disbalance: %lf, disbalance factor: %lf\n", max_time, min_time, global_res, disbalance, disbalance_factor);
        }
    }
    if (rank == 0) {
fprintf(stderr, "Average disbalance factor: %lf\n", total_disbalance_factor / NUM_OF_TASK_LISTS);        
    }
    int exit_flag = -1;
    MPI_Send(&exit_flag, 1, MPI_INT, rank, ASK_TAG, MPI_COMM_WORLD);
    free(tasks);
    pthread_exit(EXIT_SUCCESS);
}

void *receiver_start_routine(void *args) {
    while (true) {
        int transmitter_rank;
        MPI_Recv(&transmitter_rank, 1, MPI_INT, MPI_ANY_SOURCE, ASK_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (transmitter_rank == -1) {
            pthread_exit(EXIT_SUCCESS);
        }
        int num_of_sharing_tasks;
        pthread_mutex_lock(&mutex);
        if (remainig_tasks >= MIN_TASKS_TO_SHARE) {
            num_of_sharing_tasks = remainig_tasks / size;
            MPI_Send(&num_of_sharing_tasks, 1, MPI_INT, transmitter_rank, TASK_NUMBER_REPLY_TAG, MPI_COMM_WORLD);
            MPI_Send(&(tasks[TASK_LIST_SIZE - num_of_sharing_tasks]), num_of_sharing_tasks, MPI_INT, transmitter_rank, TASKS_SENDING_TAG, MPI_COMM_WORLD);
            remainig_tasks -= num_of_sharing_tasks;
            pthread_mutex_unlock(&mutex);
        } else {
            pthread_mutex_unlock(&mutex);
            num_of_sharing_tasks = 0;
            MPI_Send(&num_of_sharing_tasks, 1, MPI_INT, transmitter_rank, TASK_NUMBER_REPLY_TAG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[]) {
    int provided;
    pthread_attr_t thread_attr;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        fprintf(stderr, "Required level cannot be provided\n");
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_init(&mutex, NULL);
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &thread_attr, receiver_start_routine, NULL);
    pthread_create(&threads[1], &thread_attr, executor_start_routine, NULL);

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    pthread_attr_destroy(&thread_attr);
    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
