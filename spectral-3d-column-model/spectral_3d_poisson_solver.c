#include <stdio.h>

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
*
* Notes:
*   - phi_nh is created with dimensions phi_nh(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy).
*/
void spectral_3d_poisson_solver_(
    double* myTime, int* myIter, int* myThid,
    int* sNx, int* sNy, int* OLx, int* OLy, int* nSx, int* nSy, int* nPx, int* nPy, int* Nx, int* Ny, int* Nr,
    double* phi_nh)
{
    printf("[F2C] C function spectral_3d_poisson_solver_ was called from Fortran77 SOLVE_FOR_PRESSURE\n");
    printf("[F2C] myTime=%f, myIter=%d, myThid=%d\n", *myTime, *myIter, *myThid);
    printf("[F2C] Nx=%d, Ny=%d, Nr=%d\n", *Nx, *Ny, *Nr);
    printf("[F2C] sNx=%d, sNy=%d\n", *sNx, *sNy);
    printf("[F2C] OLx=%d, OLy=%d\n", *OLx, *OLy);
    printf("[F2C] nSx=%d, nSy=%d\n", *nSx, *nSy);
    printf("[F2C] nPx=%d, nPy=%d\n", *nPx, *nPy);

    int Sy_idx_max = *nSy;
    int Sx_idx_max = *nSx;
    int r_idx_max = *Nr;
    int y_idx_max = *sNy + 2*(*OLy);
    int x_idx_max = *sNx + 2*(*OLx);

    printf("[F2C] Sy_idx_max=%d, Sx_idx_max=%d, r_idx_max=%d, y_idx_max=%d, x_idx_max=%d\n",
        Sy_idx_max, Sx_idx_max, r_idx_max, y_idx_max, x_idx_max);

    double* phi_nh_arr = (double*) malloc(sizeof(double) * (*Nx) * (*Ny) * (*Nr));

    int flat_idx = 0;
    for (; *phi_nh; ++phi_nh) {
        int idx = flat_idx;

        int Sy_idx = idx / (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);
        idx -= Sy_idx * (Sx_idx_max * r_idx_max * y_idx_max * x_idx_max);

        int Sx_idx = idx / (r_idx_max * y_idx_max * x_idx_max);
        idx -= Sx_idx * (r_idx_max * y_idx_max * x_idx_max);

        int r_idx = idx / (y_idx_max * x_idx_max);
        idx -= r_idx * (y_idx_max * x_idx_max);

        int y_idx = (idx / x_idx_max) - *OLx;
        int x_idx = (idx % x_idx_max) - *OLy;

        printf("[F2C] phi_nh[%d] = phi_nh[x=%d/%d, y=%d/%d, r=%d/%d, Sx=%d/%d, Sy=%d/%d] = %g\n", flat_idx,
            x_idx+1, x_idx_max - 2*(*OLx), y_idx+1, y_idx_max - 2*(*OLy), r_idx+1, r_idx_max, Sx_idx+1, Sx_idx_max, Sy_idx+1, Sy_idx_max, *phi_nh);
        flat_idx++;

        // If within the non
        // if (x_idx >= OLx && x_idx <= Nx + OLx && y_idx >= OLy && y_idx <= Ny + OLy)
    }
}
