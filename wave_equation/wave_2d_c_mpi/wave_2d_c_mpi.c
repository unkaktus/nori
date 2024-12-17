#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>

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

int left_rank(int rank) {
    return abs(rank-1) % mpi_nranks();
}

int right_rank(int rank) {
    return abs(rank+1) % mpi_nranks();
}

MPI_Request *mpi_exchange(int rank, double *out_buffer, int out_buffer_len, double *in_buffer, int in_buffer_len) {
    MPI_Request* req = (MPI_Request*) malloc(sizeof(MPI_Request));
    MPI_Isendrecv(out_buffer, out_buffer_len, MPI_DOUBLE, rank, 0, in_buffer, in_buffer_len, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, req);
    return req;
}


void synchronize_arrays(int size, double * in_left, double *out_left, double *out_right, double *in_right) {
    if (left_rank(mpi_rank()) == mpi_rank()) {
        for (int i=0; i<size; i++) {
            in_left[i] = out_right[i];
            in_right[i] = out_left[i];
        }
        return;
    }
    MPI_Request reqs[2];
    for (int rank=0; rank<mpi_nranks(); rank++) {
        if (mpi_rank()==rank) { // left-to-right
            reqs[0] = *mpi_exchange(right_rank(rank), out_right, size, in_right, size);
        }
        else if (mpi_rank()==right_rank(rank)) { // right-to-left
            reqs[1] = *mpi_exchange(rank, out_left, size, in_left, size);
        }
    }
    mpi_waitall(reqs, 2);
}


typedef struct NDarray
{
    int ndim;
    int *shape;
    int ghosts;
    void *pointer;
} NDarray;

NDarray new_ndarray(int ndim, int *shape)
{
    NDarray a = {
        .ndim = ndim,
        .shape = (int *)malloc(ndim * sizeof(int)),
    };
    for (int i = 0; i < ndim; i++)
    {
        a.shape[i] = shape[i];
    }
    if (ndim == 1)
    {
        a.pointer = malloc(shape[0] * sizeof(double));
    }
    if (ndim == 2)
    {
        double **p = (double **)malloc(shape[0] * sizeof(double *));
        for (int i = 0; i < shape[0]; i++)
        {
            p[i] = malloc(shape[1] * sizeof(double));
        }
        a.pointer = (void *)p;
    }

    return a;
}

NDarray new_ndarray_like(NDarray a) {
    NDarray b = new_ndarray(a.ndim, a.shape);
    b.ghosts = a.ghosts;
    return b;
}

#define PTR2(a) ((double **)a.pointer)

int write_binary(const char *fname, NDarray a)
{
    FILE *f = fopen(fname, "w");
    if (f == NULL)
    {
        return -1;
    }
    if (a.ndim == 2)
    {
        double **ap = PTR2(a);
        for (int i = 0+a.ghosts; i < a.shape[0]-a.ghosts; i++)
        {
            for (int j = 0+a.ghosts; j < a.shape[1]-a.ghosts; j++)
            {
                double xu = ap[i][j];
                fwrite(&xu, sizeof(double), 1, f);
            }
        }
    }
    fclose(f);
    return 0;
}

void free_ndarray(NDarray a)
{
    if (a.ndim == 1)
    {
        free(a.pointer);
    }
    if (a.ndim == 2)
    {
        double **p = (double **)a.pointer;
        for (int i = 0; i < a.shape[0]; i++)
        {
            free(p[i]);
        }
        free(p);
    }
    free(a.shape);
}

NDarray add_ghosts(NDarray y, int ghosts)
{
    NDarray yg;
    if (y.ndim == 2)
    {
        int shape[2] = {y.shape[0] + 2 * ghosts, y.shape[1] + 2 * ghosts};
        yg = new_ndarray(2, shape);
        yg.ghosts = ghosts;

        double **yp = PTR2(y);
        double **ygp = PTR2(yg);
        for (int i = 0; i < y.shape[0]; i++)
        {
            for (int j = 0; j < y.shape[1]; j++)
            {
                ygp[ghosts + i][ghosts + j] = yp[i][j];
            }
        }
    }
    return yg;
}

NDarray deghost(NDarray y, int ghosts)
{
    NDarray yg;
    if (y.ndim == 2)
    {
        int shape[2] = {y.shape[0] - 2 * ghosts, y.shape[1] - 2 * ghosts};
        yg = new_ndarray(2, shape);

        double **yp = PTR2(y);
        double **ygp = PTR2(yg);
        for (int i = ghosts; i < y.shape[0] - ghosts; i++)
        {
            for (int j = ghosts; j < y.shape[1] - ghosts; j++)
            {
                ygp[i - ghosts][j - ghosts] = yp[i][j];
            }
        }
    }
    return yg;
}

void apply_periodic_boundary(NDarray y)
{
    double **yp = PTR2(y);
    // Skip x-direction sync 

    for (int j = 0; j < y.shape[1]; j++)
    {
        for (int i = 0; i < y.ghosts; i++)
        {
            yp[i][j] = yp[y.shape[0] - 2 * y.ghosts + i][j];
        }
    }
    for (int j = 0; j < y.shape[1]; j++)
    {
        for (int i = 0; i < y.ghosts; i++)
        {
            yp[y.shape[0] - y.ghosts + i][j] = yp[y.ghosts + i][j];
        }
    }

    for (int i = 0; i < y.shape[0]; i++)
    {
        for (int j = 0; j < y.ghosts; j++)
        {
            yp[i][j] = yp[i][y.shape[1] - 2 * y.ghosts + j];
        }
    }
    for (int i = 0; i < y.shape[0]; i++)
    {
        for (int j = 0; j < y.ghosts; j++)
        {
            yp[i][y.shape[1] - y.ghosts + j] = yp[i][y.ghosts + j];
        }
    }
}

void synchronize_ghosts(NDarray y) {
    int size = y.ghosts*y.shape[1];
    // Fill the buffers
    double *out_left = (double *) malloc(size * sizeof(double));
    for (int i=0; i<y.ghosts; i++) {
        for (int j=0; j<y.shape[1]; j++) {
            int k = i*y.shape[1] + j;
            out_left[k] = PTR2(y)[y.ghosts + i][j];
        }
    }
    double *out_right = (double *) malloc(size * sizeof(double));
    for (int i=0; i<y.ghosts; i++) {
        for (int j=0; j<y.shape[1]; j++) {
            int k = i*y.shape[1] + j;
            out_right[k] = PTR2(y)[y.shape[0] - 2*y.ghosts + i][j];
        }
    }
    
    double *in_left = (double *) malloc(size * sizeof(double));
    double *in_right = (double *) malloc(size * sizeof(double));


    synchronize_arrays(size, in_left, out_left, out_right, in_right);

    // Unpack incoming buffers
    for (int i=0; i<y.ghosts; i++) {
        for (int j=0; j<y.shape[1]; j++) {
            int k = i*y.shape[1] + j;
            PTR2(y)[i][j] = in_left[k];
        }
    }
    for (int i=0; i<y.ghosts; i++) {
        for (int j=0; j<y.shape[1]; j++) {
            int k = i*y.shape[1] + j;
            PTR2(y)[y.shape[0]-y.ghosts + i][j] = in_right[k];
        }
    }
    free(out_left);
    free(out_right);
    free(in_left);
    free(in_right);
}


#define loop_2d_begin(a) \
    _Pragma("omp parallel for collapse(2)") \
    for (int i=0; i<a.shape[0]; i++) { \
        for (int j=0; j<a.shape[1]; j++) { \

#define loop_2d_end }}

#define loop_2d_physical_begin(a) \
    _Pragma("omp parallel for collapse(2)") \
    for (int i=a.ghosts; i<a.shape[0]-a.ghosts; i++) { \
        for (int j=a.ghosts; j<a.shape[1]-a.ghosts; j++) { \

#define loop_2d_physical_end }}


void ndarray_copy(NDarray dst, NDarray src) {
    loop_2d_begin(dst)
    PTR2(dst)[i][j] = PTR2(src)[i][j];
    loop_2d_end
}

void rhs_u(NDarray dst, NDarray u, NDarray A, double h) {
    loop_2d_physical_begin(dst)
    PTR2(dst)[i][j] = PTR2(A)[i][j];
    loop_2d_physical_end
}


void rhs_A(NDarray dst, NDarray u, NDarray A, double h) {
    loop_2d_physical_begin(dst)
    double pprime_x = (PTR2(u)[i-1][j] - 2 * PTR2(u)[i][j] + PTR2(u)[i+1][j])/(h*h);
    double pprime_y = (PTR2(u)[i][j-1] - 2 * PTR2(u)[i][j] + PTR2(u)[i][j+1])/(h*h);
    PTR2(dst)[i][j] = pprime_x + pprime_y;
    loop_2d_physical_end
}


int main()
{
    printf("[i] Started.\n");
    mpi_init();

    double L = 1.;
    int N = 400;
    int ghosts = 1;
    double CFL = 0.4;
    double t_end = 2;

    int Nx_local = N/mpi_nranks();
    // The last one gets the extra
    if (mpi_rank() == mpi_nranks()-1) {
         Nx_local += N%mpi_nranks();
    }
    printf("rank %d: Nx_local = %d\n", mpi_rank(), Nx_local);

    int shape[2] = {Nx_local, N};
    double h = L / ((double)N);
    double dt = CFL * h;

    NDarray x = new_ndarray(2, shape);
    NDarray y = new_ndarray(2, shape);

    loop_2d_begin(x)
        PTR2(x)[i][j] = (mpi_rank() * Nx_local + i) * h;
        PTR2(y) [i][j] = j * h;
    loop_2d_end

    NDarray xx = add_ghosts(x, ghosts);

    apply_periodic_boundary(xx);
    free_ndarray(x);
    x = xx;


    NDarray yy = add_ghosts(y, ghosts);
    apply_periodic_boundary(y);
    free_ndarray(y);
    y = yy;

    char *filename;
    asprintf(&filename, "data/x.%d.dat", mpi_rank());
    write_binary(filename, x);
    free(filename);
    asprintf(&filename, "data/y.%d.dat", mpi_rank());
    write_binary(filename, x);
    free(filename);

    // Initial data
    double omega = 2 * M_PI / L;

    NDarray u = new_ndarray_like(x);
    loop_2d_begin(u)
    PTR2(u)[i][j] = (sin(omega * PTR2(x)[i][j]) + sin(omega * PTR2(y)[i][j]))/2.;
    loop_2d_end

    NDarray A = new_ndarray_like(x);
    loop_2d_begin(A)
    PTR2(A)[i][j] = 0;
    loop_2d_end

    write_binary("data/u0.dat", u);

    NDarray u_cur = new_ndarray_like(u);
    NDarray A_cur = new_ndarray_like(u);

    NDarray u_next = new_ndarray_like(u);
    NDarray A_next = new_ndarray_like(u);

    NDarray k_u = new_ndarray_like(u);
    NDarray k_A = new_ndarray_like(u);

    double t = 0;
    int i = 0;

    while (t < t_end)
    {
        if (mpi_rank() == 0) {
            printf("i=%d, t=%.02lf\n", i, t);
        }
        if (i%10 == 0) {
            char *filename;
            asprintf(&filename, "data/u_rank%d_i%04d.dat", mpi_rank(), i/10);
            write_binary(filename, u);
            free(filename);
        }

        ndarray_copy(u_next, u);
        ndarray_copy(A_next, A);
    
        double RK_timestep[4] = {0., 1./2., 1./2., 1.};
        double RK_weight[4] = {1./6., 1./3., 1./3., 1./6.};
        // if (ghosts < 4) {
        //     printf("not enough ghost points!\n");
        //     exit(-1);
        // }
        for (int rk_step=0; rk_step<4; rk_step++) {
            // Prepare the sate vector for the RHS
            loop_2d_physical_begin(u)
            PTR2(u_cur)[i][j] = PTR2(u)[i][j] + dt * RK_timestep[rk_step] * PTR2(k_u)[i][j];
            PTR2(A_cur)[i][j] = PTR2(A)[i][j] + dt * RK_timestep[rk_step] * PTR2(k_A)[i][j];
            loop_2d_physical_end

            apply_periodic_boundary(u_cur);
            apply_periodic_boundary(A_cur);
            synchronize_ghosts(u_cur);
            synchronize_ghosts(A_cur);

            // Calculate the RHS
            rhs_u(k_u, u_cur, A_cur, h);
            rhs_A(k_A, u_cur, A_cur, h);

            // Append the weighted RK substeps
            loop_2d_physical_begin(u)
            PTR2(u_next)[i][j] += PTR2(k_u)[i][j]*dt * RK_weight[rk_step];
            PTR2(A_next)[i][j] += PTR2(k_A)[i][j]*dt * RK_weight[rk_step];
            loop_2d_physical_end
        }

        ndarray_copy(u, u_next);
        ndarray_copy(A, A_next);
        apply_periodic_boundary(u);
        apply_periodic_boundary(A);
        synchronize_ghosts(u);
        synchronize_ghosts(A);

        t += dt;
        i++;
    }


    free_ndarray(u_next);
    free_ndarray(A_next);
    free_ndarray(u_cur);
    free_ndarray(A_cur);
    free_ndarray(u);
    free_ndarray(A);
    free_ndarray(x);
    free_ndarray(y);

    mpi_finalize();
}