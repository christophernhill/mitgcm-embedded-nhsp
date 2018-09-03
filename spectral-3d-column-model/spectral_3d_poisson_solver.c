#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <fftw3.h>

#define PI 3.14159265358979323846

/**
* 
* @param myTime :: Current time in simulation
* @param myIter :: Current iteration number in simulation
* @param myThid :: Thread number for this instance of SOLVE_FOR_PRESSURE
* 
* Voodoo numbers controlling data layout:
* @param sNx :: Number of X points in tile.
* @param sNy :: Number of Y points in tile.
* @param OLx :: Tile overlap extent in X.
* @param OLy :: Tile overlap extent in Y.
* @param nSx :: Number of tiles per process in X.
* @param nSy :: Number of tiles per process in Y.
* @param nPx :: Number of processes to use in X.
* @param nPy :: Number of processes to use in Y.
* @param Nx  :: Number of points in X for the full domain.
* @param Ny  :: Number of points in Y for the full domain.
* @param Nr  :: Number of points in vertical direction.
*
* Non-hydrostatic field:
* @param phi_nh :: Non-hydrostatic potential (=NH-Pressure/rhoConst)
* @param source_term :: The source term or "right hand side". Called cg3d_b in solve_for_pressure.f.
*
* Notes:
*   - phi_nh_tiled and source_term_tiled are created with Fortran (column-major)
*     dimensions (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy).
*/
void spectral_3d_poisson_solver_(
    double* myTime_ptr, int* myIter_ptr, int* myThid_ptr,
    int* sNx_ptr, int* sNy_ptr, int* OLx_ptr, int* OLy_ptr, int* nSx_ptr, int* nSy_ptr,
    int* nPx_ptr, int* nPy_ptr, int* Nx_ptr, int* Ny_ptr, int* Nr_ptr,
    double* phi_nh_tiled, double* source_term_tiled)
{
    struct timeval t1, t2; // For timing crucial sections of code.

    double myTime = *myTime_ptr;
    int myIter = *myIter_ptr;
    int myThid = *myThid_ptr;

    int sNx = *sNx_ptr;
    int sNy = *sNy_ptr;
    int OLx = *OLx_ptr;
    int OLy = *OLy_ptr;
    int nSx = *nSx_ptr;
    int nSy = *nSy_ptr;
    int nPx = *nPx_ptr;
    int nPy = *nPy_ptr;
    int Nx  = *Nx_ptr;
    int Ny  = *Ny_ptr;
    int Nr  = *Nr_ptr;

    double cg3dNorm = 5e-2;
    double rhsMax   = 0.0;
    double rhsNorm  = 0.0;

    printf("[F2C] C function spectral_3d_poisson_solver_ called from Fortran77 SOLVE_FOR_PRESSURE.\n");
    printf("[F2C] myTime=%f, myIter=%d, myThid=%d\n", myTime, myIter, myThid);
    printf("[F2C] Total number of points: Nx=%d, Ny=%d, Nr=%d\n", Nx, Ny, Nr);
    printf("[F2C] Points per tile:        sNx=%d, sNy=%d\n", sNx, sNy);
    printf("[F2C] Tile overlap:           OLx=%d, OLy=%d\n", OLx, OLy);
    printf("[F2C] Tiles per process:      nSx=%d, nSy=%d\n", nSx, nSy);
    printf("[F2C] Processes in each dim:  nPx=%d, nPy=%d\n", nPx, nPy);

    int Sy_idx_max = nSy;
    int Sx_idx_max = nSx;
    int r_idx_max = Nr;
    int y_idx_max = sNy + 2*OLy;
    int x_idx_max = sNx + 2*OLx;
    int flat_idx_max = x_idx_max * y_idx_max * r_idx_max * Sx_idx_max * Sy_idx_max;

    printf("[F2C] Sy_idx_max=%d, Sx_idx_max=%d, r_idx_max=%d, y_idx_max=%d, x_idx_max=%d, flat_idx_max=%d\n",
        Sy_idx_max, Sx_idx_max, r_idx_max, y_idx_max, x_idx_max, flat_idx_max);

    double* phi_nh_global      = (double*) malloc(sizeof(double) * Nx*Ny*Nr);
    double* source_term_global = (double*) malloc(sizeof(double) * Nx*Ny*Nr);

    /* The following loop converts from 5D tiled coordinates (x,y,r,Sx,Sy) to 3D global coordinates (x,y,r)
     * and populates global versions of the fields allocated above.
     */
    printf("[F2C] Converting 5D (tiled) arrays to 3D (global) arrays...\n");

    int flat_idx = 0;
    for (; flat_idx < flat_idx_max;) {
        /* Convert from a flat index to the 5 indices used for tiled MITGCM fields. */
        
        int idx = flat_idx; // Mutable copy of flat_idx we can use to calculate the 5 indices.

        int Sy_idx = idx / (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);
        idx -= Sy_idx * (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);

        int Sx_idx = idx / (r_idx_max * y_idx_max * x_idx_max);
        idx -= Sx_idx * (r_idx_max * y_idx_max * x_idx_max);

        int r_idx = idx / (y_idx_max * x_idx_max);
        idx -= r_idx * (y_idx_max * x_idx_max);

        // TODO: Does this work when x_idx_max != y_idx_max?
        int y_idx = (idx / y_idx_max) - OLy;
        int x_idx = (idx % x_idx_max) - OLx;

        /* Convert from the 5 indices for tiled MITGCM fields to the 3 indices for global fields. */
        int i = sNx*Sx_idx + x_idx;
        int j = sNy*Sy_idx + y_idx;
        int k = r_idx;

        int idx_global = k*Nx*Ny + j*Nx + i; // This flat index is correspond to [k][j][i] for global fields.

        /* For tiled fields where the overlapping regions are filled in (e.g. phi_nh) we just make sure that (i,j)
         * is within the global domain (0 <= i < Nx, 0 <= j < Ny). Grid points inside the overlapping regions will
         * written to more than once but it's such a small inefficiency it's not worth worrying about.
         */
        if (i >= 0 && i < Nx && j >= 0 && j < Ny) {
            phi_nh_global[idx_global] = *phi_nh_tiled;
        }

        /* For tiled fields where the overlapping regions are "not filled in" (e.g. source_term = cg3d_b) where they're
         * filled with zeros we just make sure that (i,j) is both within the global domain (0 <= i < Nx, 0 <= j < Ny)
         * and within a tile (0 <= x_idx < sNx, 0 <= y_idx < sNy).
         */
        if (i >= 0 && i < Nx && j >= 0 && j < Ny &&
            x_idx >= 0 && x_idx < sNx && y_idx >= 0 && y_idx < sNy) {
            source_term_global[idx_global] = *source_term_tiled;

            if (fabs(source_term_global[idx_global]) > rhsMax) {
                rhsMax = fabs(source_term_global[idx_global]);
                printf("[F2C] rhsMax=%+e\n", rhsMax);
            }

            if (idx_global == 0)
                printf("[F2C] loop: source_term_global[0]=%+e\n", source_term_global[0]);
            if (idx_global == 100)
                printf("[F2C] loop: source_term_global[100]=%+e\n", source_term_global[100]);
            if (idx_global == 250000)
                printf("[F2C] loop: source_term_global[250000]=%+e\n", source_term_global[250000]);
        }

        flat_idx++;
        phi_nh_tiled++;
        source_term_tiled++;
    }

    printf("[F2C] after: source_term_global[0]=%+e\n", source_term_global[0]);
    printf("[F2C] after: source_term_global[100]=%+e\n", source_term_global[100]);
    printf("[F2C] after: source_term_global[250000]=%+e\n", source_term_global[250000]);

    rhsNorm = 1 / rhsMax;

    printf("[F2C] cg3dNorm=%+e, rhsNorm=%+e\n", cg3dNorm, rhsNorm);

    int stc = 0; // source term counter
    for (stc = 0; stc < Nx*Ny*Nr; stc++)
        source_term_global[stc] *= (cg3dNorm * rhsNorm);

    // Create filenames for all the fields we're saving to disk.
    char phi_nh_filename[50], phi_nh_hat_filename[50], phi_nh_rec_filename[50];
    char source_term_filename[50], source_term_hat_filename[50], source_term_rec_filename[50];

    sprintf(phi_nh_filename, "phi_nh.%d.dat", myIter);
    sprintf(phi_nh_hat_filename, "phi_nh_hat.%d.dat", myIter);
    sprintf(phi_nh_rec_filename, "phi_nh_rec.%d.dat", myIter);
    sprintf(source_term_filename, "source_term.%d.dat", myIter);
    sprintf(source_term_hat_filename, "source_term_hat.%d.dat", myIter);
    sprintf(source_term_rec_filename, "source_term_rec.%d.dat", myIter);

    printf("[F2C] Saving %s...\n", phi_nh_filename);
    FILE *f_phi_nh = fopen(phi_nh_filename, "wb");
    fwrite(phi_nh_global, sizeof(double), Nx*Ny*Nr, f_phi_nh);
    fclose(f_phi_nh);

    printf("[F2C] Saving %s...\n", source_term_filename);
    FILE *f_source_term = fopen(source_term_filename, "wb");
    fwrite(source_term_global, sizeof(double), Nx*Ny*Nr, f_source_term);
    fclose(f_source_term);

    fftw_plan forward_source_term_plan, backward_source_term_plan;
    fftw_plan backward_phi_nh_plan;

    /* source_term_hat_global is the (DCT/DST mixed) Fourier transform of source_term while
     * source_term_rec_global is the reconstruction of source_term_global from the Fourier
     * coefficients which we save to make sure that the forward and inverse transforms are
     * doing their job.
     */
    double* source_term_hat_global = (double*) malloc(sizeof(double) * Nx*Ny*Nr);
    double* source_term_rec_global = (double*) malloc(sizeof(double) * Nx*Ny*Nr);
    double* phi_nh_hat_global      = (double*) malloc(sizeof(double) * Nx*Ny*Nr);
    double* phi_nh_rec_global      = (double*) malloc(sizeof(double) * Nx*Ny*Nr);

    printf("[F2C] Creating forward source term FFTW plan...\n");
    forward_source_term_plan = fftw_plan_r2r_3d(Nr, Ny, Nx, source_term_global, source_term_hat_global,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);

    gettimeofday(&t1, NULL); // Start timing: source term FFT

    printf("[F2C] Executing forward source term FFTW plan... ");
    fftw_execute(forward_source_term_plan);

    // Stop timing: source term FFT
    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("[F2C] Saving %s...\n", source_term_hat_filename);
    FILE *f_source_term_hat = fopen(source_term_hat_filename, "wb");
    fwrite(source_term_hat_global, sizeof(double), Nx*Ny*Nr, f_source_term_hat);
    fclose(f_source_term_hat);

    printf("[F2C] Creating backward source term FFTW plan...\n");
    backward_source_term_plan = fftw_plan_r2r_3d(Nr, Ny, Nx, source_term_hat_global, source_term_rec_global,
        FFTW_RODFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);

    printf("[F2C] Executing backward source term FFTW plan...\n");
    fftw_execute(backward_source_term_plan);

    /* FFTW employs an unnormalized IFFT, so if we want to get back the original field we need
     * to divide the coefficients by 2N for each dimension, or 8*Nx*Ny*Nz in our case.
     */
    int fc; // Fourier coefficient counter.
    for (fc = 0; fc < Nx*Ny*Nr; fc++)
        source_term_rec_global[fc] /= 8.0*Nx*Ny*Nr;

    printf("[F2C] Saving %s...\n", source_term_rec_filename);
    FILE *f_source_term_rec = fopen(source_term_rec_filename, "wb");
    fwrite(source_term_rec_global, sizeof(double), Nx*Ny*Nr, f_source_term_rec);
    fclose(f_source_term_rec);

    // TODO: These are hard-coded for now but I should pass the domain size too.
    int delta_x = 2000 / Nx;
    int delta_y = 2000 / Ny;
    int delta_r = 1000 / Nr;

    gettimeofday(&t1, NULL); // Start timing: Fourier coefficient computation

    printf("[F2C] Computing phi_nh Fourier coefficients... ");

    for (fc = 0; fc < Nx*Ny*Nr; fc++) {
        int idx = fc;

        // Convert flat Fourier coefficient index to (l,m,n) wavenumber indices.
        int n = idx / (Nx*Ny);
        idx -= n * (Nx*Ny);

        // TODO: Does this work when Nx != Ny?
        int m = (idx / Ny);
        int l = (idx % Nx);

        double kx = (2 / pow(delta_x, 2)) * (cos( (PI*l) / Nx) - 1);
        double ky = (2 / pow(delta_y, 2)) * (cos( (PI*m) / Ny) - 1);
        double kr = (2 / pow(delta_r, 2)) * (cos( (PI*n) / Nr) - 1);
        
        double factor = 1 / (kx + ky + kr);

        // TODO: What to do if kx + ky + kz == 0?
        if (isinf(factor)) {
            phi_nh_hat_global[fc] = 0.0; // Assuming solvability condition that DC component is zero.
        } else {
            phi_nh_hat_global[fc] = factor * source_term_hat_global[fc];
        }
    }

    // Stop timing: Fourier coefficient computation
    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("[F2C] Saving %s...\n", phi_nh_hat_filename);
    FILE *f_phi_nh_hat = fopen(phi_nh_hat_filename, "wb");
    fwrite(phi_nh_hat_global, sizeof(double), Nx*Ny*Nr, f_phi_nh_hat);
    fclose(f_phi_nh_hat);

    printf("[F2C] Creating backward phi_nh FFTW plan...\n");
    backward_phi_nh_plan = fftw_plan_r2r_3d(Nr, Ny, Nx, phi_nh_hat_global, phi_nh_rec_global,
        FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);

    gettimeofday(&t1, NULL); // Start timing: phi_nh IFFT

    printf("[F2C] Executing backward phi_nh FFTW plan... ");
    fftw_execute(backward_phi_nh_plan);

    for (fc = 0; fc < Nx*Ny*Nr; fc++)
        phi_nh_rec_global[fc] /= (8.0*Nx*Ny*Nr * rhsNorm);

    // Stop timing: phi_nh IFFT
    gettimeofday (&t2, NULL);
    printf("(t=%ld us)\n", ((t2.tv_sec - t1.tv_sec) * 1000000L + t2.tv_usec) - t1.tv_usec);

    printf("[F2C] Saving %s...\n", phi_nh_rec_filename);
    FILE *f_phi_nh_rec = fopen(phi_nh_rec_filename, "wb");
    fwrite(phi_nh_rec_global, sizeof(double), Nx*Ny*Nr, f_phi_nh_rec);
    fclose(f_phi_nh_rec);

    fflush(stdout); // Make sure to flush stdout buffer so C output doesn't appear in the wrong order.

    fftw_destroy_plan(forward_source_term_plan);
    fftw_destroy_plan(backward_source_term_plan);
    fftw_destroy_plan(backward_phi_nh_plan);

    fftw_cleanup();

    free(phi_nh_global);
    free(source_term_global);
    free(source_term_hat_global);
    free(source_term_rec_global);
    free(phi_nh_hat_global);
    free(phi_nh_rec_global);
}
