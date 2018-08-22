#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <fftw3.h>

#define PI 3.14159265358979323846

#define I3(i,j,k) i*Nx*Ny + j*Nx + k 

/**
* Corresponds to a 3D normal/Gaussian probability distribution with constant hard-coded means and
* variances. The covariance matrix is diagonal.
*
* @param x
* @param y
* @param z
*
* @return
*/
double trivariate_normal_distribution(double x, double y, double z) {
    static const double mu_x = 5.0;
    static const double mu_y = 5.0;
    static const double mu_z = 5.0;

    static const double sigma_x = 1.0;
    static const double sigma_y = 1.0;
    static const double sigma_z = 1.0;

    // double prefactor = 1 / (2 * PI * sigma_x * sigma_y * sigma_z);
    double prefactor = 100.0;

    double arg_x = - pow(x - mu_x, 2) / (2 * pow(sigma_x, 2));
    double arg_y = - pow(y - mu_y, 2) / (2 * pow(sigma_y, 2));
    double arg_z = - pow(z - mu_z, 2) / (2 * pow(sigma_z, 2));
    double arg = arg_x + arg_y + arg_z;

    // printf("prefactor=%f, arg_x=%f, arg_y=%f, arg_z=%f, ret=%f\n", prefactor, arg_x, arg_y, arg_z, prefactor*exp(arg));

    return prefactor * exp(arg);
}

double triple_cosine_waves(double x, double y, double z) {
    static const double k_x = 2*PI / 2.0;
    static const double k_y = 2*PI / 2.0;
    static const double k_z = 2*PI / 5.0;

    return cos(k_x * x) * cos(k_y * y) * sin(k_z * z);
}

void main(int argc, char* argv[]) {
    struct timeval t1, t2;

    // Number of gridpoints in each dimension.
    static const int Nx = 32;
    static const int Ny = 32;
    static const int Nz = 32;

    // Length of each dimension.
    static const double Lx = 10.0;
    static const double Ly = 10.0;
    static const double Lz = 10.0;

    printf("Number of gridpoints: (Nx,Ny,Nz) = (%d,%d,%d), Nx*Ny*Nz=%d\n", Nx, Ny, Nz, Nx*Ny*Nz);
    printf("Domain size: (Lx,Ly,Lz) = (%f,%f,%f)\n", Lx, Ly, Lz);

    double* in = (double*) malloc(sizeof(double) * Nx*Ny*Nz);
    double* out = (double*) malloc(sizeof(double) * Nx*Ny*Nz);
    double* rec = (double*) malloc(sizeof(double) * Nx*Ny*Nz);

    // Initialize in to 3D Gaussian.
    printf("Initializing input array... ");
    gettimeofday (&t1, NULL);

    double x, y, z;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                x = ( (double) i/ (double) Nx) * Lx;
                y = ( (double) j/ (double) Ny) * Ly;
                z = ( (double) k/ (double) Nz) * Lz;

                // in[I3(i,j,k)] = (double) trivariate_normal_distribution(x, y, z);
                in[I3(i,j,k)] = (double) triple_cosine_waves(x, y, z);
                
                // printf("in[I3(%d,%d,%d)] = in[%d] = %f\n", i, j, k, I3(i,j,k), (double) trivariate_normal_distribution(x, y, z));
            }
        }
    }

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Saving input array to in.dat... ");
    gettimeofday (&t1, NULL);

    FILE *f_in = fopen("in.dat", "wb");
    fwrite(in, sizeof(double), Nx*Ny*Nz, f_in);
    fclose(f_in);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    fftw_plan forward_plan, backward_plan;

    printf("Creating forward plan... ");
    gettimeofday (&t1, NULL);

    forward_plan = fftw_plan_r2r_3d(Nx, Ny, Nz, in, out,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_RODFT10, FFTW_MEASURE);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Executing forward plan... ");
    gettimeofday (&t1, NULL);

    fftw_execute(forward_plan);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Saving output array to out.dat... ");
    gettimeofday (&t1, NULL);

    FILE *f_out = fopen("out.dat", "wb");
    fwrite(out, sizeof(double), Nx*Ny*Nz, f_out);
    fclose(f_out);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Creating backward plan... ");
    gettimeofday (&t1, NULL);

    backward_plan = fftw_plan_r2r_3d(Nx, Ny, Nz, out, rec,
        FFTW_REDFT01, FFTW_REDFT01, FFTW_RODFT01, FFTW_MEASURE);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Executing backward plan... ");
    gettimeofday (&t1, NULL);

    fftw_execute(backward_plan);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("Saving reconstructed array to rec.dat... ");
    gettimeofday (&t1, NULL);

    FILE *f_rec = fopen("rec.dat", "wb");
    fwrite(rec, sizeof(double), Nx*Ny*Nz, f_rec);
    fclose(f_rec);

    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);

    fftw_cleanup();

    free(in);
    free(out);
}
