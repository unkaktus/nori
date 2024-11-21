#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void mpi_init() {
    MPI_Init(NULL, NULL);
}

void mpi_finalize() {
    MPI_Finalize();
}

int mpi_nranks() {
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    return nranks;
}

int mpi_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void mpi_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_waitall(MPI_Request *reqs, int n) {
    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
}

int mpi_exchange_blocking(int rank, double *out_buffer, int out_buffer_len, double *in_buffer, int in_buffer_len) {
    return MPI_Sendrecv(out_buffer, out_buffer_len, MPI_DOUBLE, rank, 0, in_buffer, in_buffer_len, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

MPI_Request *mpi_exchange_request(int rank, double *out_buffer, int out_buffer_len, double *in_buffer, int in_buffer_len) {
    MPI_Request* req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Isendrecv(out_buffer, out_buffer_len, MPI_DOUBLE, rank, 0, in_buffer, in_buffer_len, MPI_DOUBL/E, rank, 0, MPI_COMM_WORLD, req);
    return req;
}

double *dmalloc(size_t len) {
    double * x = (double *) malloc(len * sizeof(double));
    return x;
}


char *array_to_string(double *x, int n) {
    char *s=NULL;
    for (int i=0; i<n; i++) {
        if (s==NULL) {
            asprintf(&s, "%lf", x[i]);
        } else {
            char *s_prev = s;
            asprintf(&s, "%s %lf", s, x[i]);
            free(s_prev);
        }
    }
    return s;
}

int left_rank(int rank) {
    return abs(rank-1) % mpi_nranks();
}

int right_rank(int rank) {
    return abs(rank+1) % mpi_nranks();
}

void syncronize_buffers_blocking(int n, double * in_left, double *out_left, double *out_right, double *in_right) {
    for (int rank=0; rank<mpi_nranks(); rank++) {
            if (mpi_rank()==rank) { // left-to-right
                // printf("%d -> %d\n", mpi_rank(), right_rank(rank));
                mpi_exchange_blocking(right_rank(rank), out_right, n, in_right, n);
            }
            else if (mpi_rank()==right_rank(rank)) { // right-to-left
                // printf("%d <- %d\n", rank, mpi_rank());
                mpi_exchange_blocking(rank, out_left, n, in_left, n);
            }
        }
}

void syncronize_buffers(int n, double * in_left, double *out_left, double *out_right, double *in_right) {
    MPI_Request reqs[2];
    for (int rank=0; rank<mpi_nranks(); rank++) {
        if (mpi_rank()==rank) { // left-to-right
            // printf("%d -> %d\n", mpi_rank(), right_rank(rank));
            reqs[0] = *mpi_exchange_request(right_rank(rank), out_right, n, in_right, n);
        }
        else if (mpi_rank()==right_rank(rank)) { // right-to-left
            // printf("%d <- %d\n", rank, mpi_rank());
            reqs[1] = *mpi_exchange_request(rank, out_left, n, in_left, n);
        }
    }
    mpi_waitall(reqs, 2);
}



int main() {
    mpi_init();

    int n = 1;

    double *mine = dmalloc(n);
    for (int i=0; i<n; i++) {
        mine[i] = (double) mpi_rank();
    }

    double *inarr_left = dmalloc(n);
    double *inarr_right = dmalloc(n);

    syncronize_buffers(n, inarr_left, mine, mine, inarr_right);

    char *inarr_left_str = array_to_string(inarr_left, n);
    char *inarr_right_str = array_to_string(inarr_right, n);

    char *arr_str = array_to_string(mine, n);

    printf("rank %d: inarr: [%s] -> [%s] <- [%s]\n", mpi_rank(), inarr_left_str, arr_str, inarr_right_str);

    // Check the correctness
    int errors = 0;
    double expected_left = (double) ((mpi_nranks() + mpi_rank() - 1)%mpi_nranks());
    double expected_right = (double) ((mpi_nranks() + mpi_rank() + 1)%mpi_nranks());
    if ( inarr_left[0] != expected_left ) {
        printf("[!] rank %d: Incorrect left data: expected %lf, got %lf\n", mpi_rank(), expected_left, inarr_left[0]);
        errors++;
    }
    if ( inarr_right[0] != expected_right ) {
        printf("[!] rank %d: Incorrect right data: expected %lf, got %lf\n", mpi_rank(), expected_right, inarr_right[0]);
        errors++;
    }

    printf("[i] rank %d: finished with %d errors\n", mpi_rank(), errors);

    free(inarr_left_str);
    free(arr_str);
    free(inarr_right_str);
    free(mine);

    mpi_finalize();
}