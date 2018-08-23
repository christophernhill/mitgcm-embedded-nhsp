#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <fftw3.h>

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
* @param source_term :: The source term or "right hand side".
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

    printf("[F2C] Sy_idx_max=%d, Sx_idx_max=%d, r_idx_max=%d, y_idx_max=%d, x_idx_max=%d\n",
        Sy_idx_max, Sx_idx_max, r_idx_max, y_idx_max, x_idx_max);

    double* phi_nh_global      = (double*) malloc(sizeof(double) * Nx*Ny*Nr);
    double* source_term_global = (double*) malloc(sizeof(double) * Nx*Ny*Nr);

    // Convert from 5D tiled coordinates (x,y,r,Sx,Sy) to 3D global coordinates (x,y,r).
    printf("[F2C] Converting 5D (tiled) arrays to 3D (global) arrays...\n");

    int flat_idx = 0;
    for (; *phi_nh_tiled;) {
        // Convert from a flat index to the 5 indices used for tiled MITGCM fields.
        int idx = flat_idx;

        int Sy_idx = idx / (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);
        idx -= Sy_idx * (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);

        int Sx_idx = idx / (r_idx_max * y_idx_max * x_idx_max);
        idx -= Sx_idx * (r_idx_max * y_idx_max * x_idx_max);

        int r_idx = idx / (y_idx_max * x_idx_max);
        idx -= r_idx * (y_idx_max * x_idx_max);

        int y_idx = (idx / x_idx_max) - OLx;
        int x_idx = (idx % x_idx_max) - OLy;

        // Convert from the 5 indices for tiled MITGCM fields to the 3 indices for global fields.
        int i = sNx*Sx_idx + x_idx;
        int j = sNy*Sy_idx + y_idx;
        int k = r_idx;

        if (i >= 0 && i < Nx && j >= 0 && j < Ny) {
            // printf("[F2C] flat_idx=%d, i=%d, j=%d, k=%d, (k*Nx*Ny + j*Nx + i)=%d\n", flat_idx, i, j, k, k*Nx*Ny + j*Nx + i);

            phi_nh_global[k*Nx*Ny + j*Nx + i]      = *phi_nh_tiled;
            source_term_global[k*Nx*Ny + j*Nx + i] = *source_term_tiled;

            // printf("[F2C] phi_nh_tiled[%d] = phi_nh_tiled(x=%d/%d, y=%d/%d, r=%d/%d, Sx=%d/%d, Sy=%d/%d) = %g -> phi_nh_global[%d,%d,%d]\n", flat_idx,
            //     x_idx+1, x_idx_max - 2*OLx, y_idx+1, y_idx_max - 2*OLy, r_idx+1, r_idx_max,
            //     Sx_idx+1, Sx_idx_max, Sy_idx+1, Sy_idx_max, *phi_nh_tiled, i, j, k);
        }
        
        flat_idx++;
        phi_nh_tiled++;
        source_term_tiled++;
    }

    char phi_nh_filename[30], source_term_filename[30];
    sprintf(phi_nh_filename, "phi_nh.%d.dat", myIter);
    sprintf(source_term_filename, "source_term.%d.dat", myIter);

    printf("[F2C] Saving %s...\n", phi_nh_filename);
    FILE *f_phi_nh = fopen(phi_nh_filename, "wb");
    fwrite(phi_nh_global, sizeof(double), Nx*Ny*Nr, f_phi_nh);
    fclose(f_phi_nh);

    printf("[F2C] Saving %s...\n", source_term_filename);
    FILE *f_source_term = fopen(source_term_filename, "wb");
    fwrite(source_term_global, sizeof(double), Nx*Ny*Nr, f_source_term);
    fclose(f_source_term);

    fftw_plan forward_plan, backward_plan;

    double* source_term_hat_global = (double*) malloc(sizeof(double) * Nx*Ny*Nr);

    printf("[F2C] Creating forward FFTW plan...\n");
    forward_plan = fftw_plan_r2r_3d(Nx, Ny, Nr, source_term_global, source_term_hat_global,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_RODFT10, FFTW_MEASURE);

    printf("[F2C] Executing forward FFTW plan...\n");
    fftw_execute(forward_plan);

    char source_term_hat_filename[30];
    sprintf(source_term_hat_filename, "source_term_hat.%d.dat", myIter);

    printf("[F2C] Saving %s...\n", source_term_hat_filename);
    FILE *f_source_term_hat = fopen(source_term_hat_filename, "wb");
    fwrite(source_term_hat_global, sizeof(double), Nx*Ny*Nr, f_source_term_hat);
    fclose(f_source_term_hat);
}
