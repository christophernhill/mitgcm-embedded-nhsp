/*
  Discrete Cosine Transform II (DCT-II) definition:
  
  1D:
    X_k = \sqrt{2/N} * s_k * \sum_{n=0}^{n=N-1} [x_n * \cos(\pi*k*(n+1/2)/N), k = 0, 1, 2, ..., N-1
    where s_0 = 1/sqrt{2} and s_k = 1 for k = 1, 2, ..., N-1.
  
  2D:
    X_k1,k2 = \sqrt{2/N} * \sqrt{2/M} * \sum_{n=0}^{n=N-1} \sum_{m=0}^{m=M-1} s_k1 * s_k2 * x_n,m
              * \cos(pi*k1*(n+1/2)/N) * \cos(\pi*k2*(m+1/2)/M), k1 = 0, 1, ..., N-1, k2 = 0, 1, 2, ..., M-1
    where s_0i = 1/sqrt{2} and s_ki = 1 for i=1,2 and k1,k2 = 1, 2, ..., [N,M] - 1
  
  The Discrete Fourier Transform (DFT) equivalent of DCT-II involves creating a mirrored sequence padded with zeros
  between points. Scaled appropriately, the real part of the first N,M DFT entries will be equivalent to the DCT-II
  coefficients.
  */

/*
  To run using MPI:
    mpirun -n 2 ./a.out  30 20 5 30 10 1
 
  The DCT using FFTW MPI interface. This example code for 3-d (i,k,j) layout where domain is block distributed in the
  last dimension (j) per FFTW approach. 

  Running using the command
    mpirun -prepend-pattern 'RANK_%r: ' -n 2 ./a.out  7 6 1 7 3 1 
  would apply DCT and inverse for ni=7 x nj=6 x nk=1 domain (args 1,2,3) where domain is split in 2 equal parts in j,
  each of size 7 x 3 (args 4, 5). The transform and inverse is computed  x 1 (last arg, arg 6).
 
  Compilation instructions:
    cd eofe7.mit.edu:/nfs/cnhlab002/cnh/src/fftw/test_code
    module load intel/2017-01
    module load impi/2017-01
    mpiicc -qopenmp -O -xHost fftw_dct_mpi_nd.c -I../../../opt/fftw/usr/local/include -L../../../opt/fftw/usr/local/lib
    -lfftw3_mpi -lfftw3
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <mpi.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#undef  DIMS_1
#undef  DIMS_2
#define DIMS_3

// TODO: Describe the meaning of these variables.

#ifdef DIMS_1
#define N     5
#define M     1
#define L     1
#define NMID  2
#define MMID  0
#define LMID  0
#define NRANK 3
#define I3(a,b,c) a+b*N+c*M*N
#endif

#ifdef DIMS_2
#define N     5
#define M     5
#define L     1
#define NMID  2
#define MMID  2
#define LMID  0
#define NRANK 3
#define I3(a,b,c) a+b*N+c*M*N
#endif

#ifdef DIMS_3
#define NSX   1
#define SNX   192
#define NSY   1
#define SNY   192
#define NR    150
#define N     SNX
#define M     SNY
#define L     NR
#define NMID  95
#define MMID  95
#define LMID  74
#define NRANK 3
#define I3(a,b,c) a+b*N+c*M*N
#endif

void loc2glob(int mpiRank,
              int idl,int jdl,int kdl,
              int nx,int ny,int nr,int snx,int sny,
              int *ig,int *jg,int *kg)
 // Turn local index into equivalent global index
 {
  int npx, npy;
  int mypx, mypy;
  // Get calling rank index in process grid
  npx=nx/snx;
  npy=ny/sny;
  mypx=mpiRank%npx;
  mypy=mpiRank/npx;
  *ig=mypx*snx+idl;
  *jg=mypy*sny+jdl;
  *kg=kdl;
  return;
 }
void glob2loc(int mpiRank,
              int  ig,  int  jg,  int  kg,
              int nx, int ny, int nr, int snx, int sny, 
              int *il,  int *jl,  int *kl) {
 int  ilog,  jlog, klog;
 int   npx,   npy;
 int ioffl, joffl, koffl;
 int  mypx,  mypy;
 // Turn global index into process local index - return -1 if global index not in the local index space.
 npx=nx/snx;
 npy=ny/sny;
 mypx=mpiRank%npx;
 mypy=mpiRank/npx;
 ilog=mypx*snx;
 jlog=mypy*sny;
 ioffl=ig-ilog;
 joffl=jg-jlog;
 if ( ioffl >= 0 & ioffl < snx ) {
  *il=ioffl;
 } else {
  *il=-1; *jl=-1;
 }
 if ( joffl >= 0 & joffl < sny ) {
  *jl=joffl;
 } else {
  *il=-1; *jl=-1;
 }
 klog=0;
 koffl=kg-klog;
 *kl=koffl;

 // printf ("GLOB2LOC: r, ig, jg, kg = %d, %d, %d, %d\n",mpiRank,ig,jg,kg);
 // printf ("GLOB2LOC: r, ilog, jlog = %d, %d, %d    \n",mpiRank,ilog,jlog);
 // printf ("GLOB2LOC: r, ioffl,joffl= %d, %d, %d    \n",mpiRank,ioffl,joffl);
 // printf ("GLOB2LOC: r, il, jl, kl = %d, %d, %d, %d\n",mpiRank,*il,*jl,*kl);
 return;
}
              

main(int argc, char *argv[]){

      int i,j,k,ioff;
      double fac;
      fftw_plan plan;
      fftw_complex in[L*M*N], out[L*M*N];
      int  rank_list[NRANK];
      rank_list[0]=L;
      rank_list[1]=M;
      rank_list[2]=N;

      int mpiRank;
      int mpiSize;
      int mpiComm;

      // Read command line args
      // ./a.out  Nx  Ny  Nr sNx sNy NT
      //  Nx  - total size in X
      //  Ny  - total size in Y
      //  Nr  - total size in Z
      //  sNx - tile size in X (must divide Nx exactly)
      //  sNy - tile size in Y (must divide Ny exactly)
      //  NT  - number of transforms
      // ./a.out 192 192 150 36 36 100

      if ( argc != 7 ) {
       printf("ERROR: Not enough args set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      int Nx, Ny, Nr, sNx, sNy, NTrans, nmatch, nSx, nSy;
      nmatch=sscanf(argv[1],"%d",&Nx);
      if ( nmatch != 1 || Nx <= 0) {
       printf("ERROR: Nx not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      nmatch=sscanf(argv[2],"%d",&Ny);
      if ( nmatch != 1 ) {
       printf("ERROR: Ny not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      nmatch=sscanf(argv[3],"%d",&Nr);
      if ( nmatch != 1 ) {
       printf("ERROR: Nr not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      nmatch=sscanf(argv[4],"%d",&sNx);
      if ( nmatch != 1 ) {
       printf("ERROR: sNx not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      nmatch=sscanf(argv[5],"%d",&sNy);
      if ( nmatch != 1 ) {
       printf("ERROR: sNy not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }
      nmatch=sscanf(argv[6],"%d",&NTrans);
      if ( nmatch != 1 ) {
       printf("ERROR: NT not set?\n");
       printf("Usage: %s Nx Ny Nr sNx sNy NT\n",argv[0]);
       exit(-1);
      }

/*
    Try DCT-II using FFTW REDFT10 _N interface and its
    inverse REDFT01.
*/

    MPI_Init(&argc, &argv);
    mpiComm=MPI_COMM_WORLD;

    fftw_mpi_init();

    MPI_Comm_rank(mpiComm, &mpiRank);
    MPI_Comm_size(mpiComm, &mpiSize);

    // Check sizes
    if ( sNx*sNy*mpiSize != Nx*Ny ) {
     printf("ERROR: sNx*sNy*mpiSize != Nx*Ny\n");
     exit(-1);
    }
    nSx=floor(Nx/sNx);
    if ( sNx*nSx != Nx )
    {
     printf("ERROR: sNx*nSx != Nx\n");
     exit(-1);
    }
    nSy=floor(Ny/sNy);
    if ( sNy*nSy != Ny )
    {
     printf("ERROR: sNy*nSy != Ny\n");
     exit(-1);
    }
    if ( nSx*nSy != mpiSize )
    {
     printf("ERROR: nSy*nSx != mpiSize\n");
     exit(-1);
    }

    // Allocate local tile data array and scale factor array
#define _I3LOC(a,b,c) a+(c*sNx)+(b*sNx*Nr)
    double *mytData,*mytScale;
    double tr;
    int    sT,idl,jdl,kdl,idg,jdg,kdg;
    sT=sNx*sNy*Nr;
    mytData=(double *)malloc(sT*sizeof(double));
    mytScale=(double *)malloc(sT*sizeof(double));
    for(jdl=0;jdl<sNy;++jdl){
     for(kdl=0;kdl<Nr;++kdl){
      for(idl=0;idl<sNx;++idl){
       tr=(double)( rand() )/(double)(RAND_MAX)-0.5;
       mytData[_I3LOC(idl,jdl,kdl)]=0.+tr/1.e3;
    } } }
    idg=(int)(double)Nx/2.;
    jdg=(int)(double)Ny/2.;
    kdg=(int)(double)Nr/2.;
    double fracW;
    fracW=-1./4.;
    if ( Nr > 1 ) { fracW=-1./6.; }
    glob2loc(mpiRank,idg,jdg,kdg,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
    if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=1.; }

    glob2loc(mpiRank,idg,jdg+1,kdg,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
    if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }
    glob2loc(mpiRank,idg,jdg-1,kdg,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
    if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }

    glob2loc(mpiRank,idg+1,jdg,kdg,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
    if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }
    glob2loc(mpiRank,idg-1,jdg,kdg,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
    if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }

    if ( Nr > 1 ) {
     glob2loc(mpiRank,idg,jdg,kdg+1,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
     if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }
     glob2loc(mpiRank,idg,jdg,kdg-1,Nx,Ny,Nr,sNx,sNy,&idl,&jdl,&kdl);
     if ( idl > 0 ) { mytData[_I3LOC(idl,jdl,kdl)]=fracW; }
    }

    // Set scale factor array elements to eigenvalues x wavenumber bit
    // 3 factors
    // fac_i :  -[\frac{2\sin(\frac{\pi \kappa}{2nx})}{\Delta x}]^{2}
    // fac_j :  -[\frac{2\sin(\frac{\pi \kappa}{2ny})}{\Delta y}]^{2}
    // fac_k :  -[\frac{2\sin(\frac{\pi \kappa}{2nz})}{\Delta z}]^{2}
    // scaling is 1/(fac_i + fac_j + fac_k)
    // Temp arrays along each axis (full domain size on each process, but only 1d).
    double *lambda_fac_i, *lambda_fac_j, *lambda_fac_k;
    double dx, dy, dz;
    double tmp1, tmp2, tmp3;
    dx=1.;
    dy=1.;
    dz=1.;
    lambda_fac_i = (double *)malloc(Nx*sizeof(double));
    lambda_fac_j = (double *)malloc(Ny*sizeof(double));
    lambda_fac_k = (double *)malloc(Nr*sizeof(double));
    lambda_fac_i[0]=0.;
    for (i=1;i<Nx;++i) {
     tmp1=(M_PI*(double)i)/(2.*(double)Nx);
     tmp2=2.*sin(tmp1);
     tmp3=dx;
     lambda_fac_i[i]=-pow((tmp2/tmp3),2.);
    }
    lambda_fac_j[0]=0.;
    for (j=1;j<Ny;++j) {
     tmp1=(M_PI*(double)j)/(2.*(double)Ny);
     tmp2=2.*sin(tmp1);
     tmp3=dy;
     lambda_fac_j[j]=-pow((tmp2/tmp3),2.);
    }
    lambda_fac_k[0]=0.;
    for (k=1;k<Nr;++k) {
     tmp1=(M_PI*(double)k)/(2.*(double)Nr);
     tmp2=2.*sin(tmp1);
     tmp3=dz;
     lambda_fac_k[k]=-pow((tmp2/tmp3),2.);
    }
    // Now fill out tile local values
    int ig, jg, kg;
    for(jdl=0;jdl<sNy;++jdl){
     for(kdl=0;kdl<Nr;++kdl){
      for(idl=0;idl<sNx;++idl){
       loc2glob(mpiRank,
                idl,jdl,kdl,
                Nx,Ny,Nr,sNx,sNy,
                &ig,&jg,&kg);
       // ig=0;
       // jg=0;
       // kg=0;
       tmp1=lambda_fac_i[ig];
       tmp2=lambda_fac_j[jg];
       tmp3=lambda_fac_k[kg];
       mytScale[_I3LOC(idl,jdl,kdl)]=tmp1+tmp2+tmp3;
    } } }
    if ( mpiRank == 0 ) {
     mytScale[_I3LOC(0,0,0)]=1.;
    };
    for(jdl=0;jdl<sNy;++jdl){
     for(kdl=0;kdl<Nr;++kdl){
      for(idl=0;idl<sNx;++idl){
       mytScale[_I3LOC(idl,jdl,kdl)]=1./mytScale[_I3LOC(idl,jdl,kdl)];
    } } }

// void glob2loc(int mpiRank,
//               int  ig,  int  jg,  int  kg,
//               int nx, int ny, int nr, int snx, int sny,
//               int *il,  int *jl,  int *kl) {


    for (j=0;j<sNy;++j) {
     for (k=0;k<Nr;++k) {
      for (i=0;i<sNx;++i) {
       ioff=_I3LOC(i,j,k);
       loc2glob(mpiRank,
                i,j,k,
                Nx,Ny,Nr,sNx,sNy,
                &ig,&jg,&kg);
       if ( abs(idg - ig) < 2 &
            abs(jdg - jg) < 2 &
            abs(kdg - kg) < 2 ) {
        printf("INITIAL mytData: ioff, mytData(%d,%d,%d) = %d, %f\n",i,j,k,ioff,mytData[ioff]);
       }
      }
     }
    }

    ptrdiff_t alloc_local_n, local_n0_n, local_0_start_n;
    int       nrsizes=3;
    ptrdiff_t howmany=1;
    ptrdiff_t block0=FFTW_MPI_DEFAULT_BLOCK;
    ptrdiff_t *rsizes;
    rsizes=(ptrdiff_t *)malloc(nrsizes*sizeof(ptrdiff_t));
    // Slowest first, fastest last. Start decomp is on first dimension.
    rsizes[0]=Ny;
    rsizes[1]=Nr;
    rsizes[2]=Nx;
    alloc_local_n = fftw_mpi_local_size_many(nrsizes,rsizes,howmany,block0,mpiComm,
                     &local_n0_n, &local_0_start_n);
    printf("local_n0_n = %d\n",local_n0_n);
    printf("local_0_start_n = %d\n",local_0_start_n);

    if ( local_n0_n != sNy ) {
     // Check have input data that is distributed using 1-d block decomp, along slowest (left-most/first in C)
     // dimension.
     // If we don't then we would need to resdistribute the data to be that way.
     printf("ERROR: local_n0_n != sNy\n");
     exit(-1);
    }
    // if ( Nr != 1 ) {
    //  printf("Nr != 1\n");
    //  exit(-1);
    // }

    // Nd
    double *data_nd;
    data_nd = fftw_alloc_real(alloc_local_n);
    fftw_r2r_kind kindArr[3];
    kindArr[0]=FFTW_REDFT10;
    kindArr[1]=FFTW_REDFT10;
    kindArr[2]=FFTW_REDFT10;
    fftw_plan plan_n1;
    plan_n1 = fftw_mpi_plan_r2r(nrsizes,rsizes,data_nd,data_nd,mpiComm,
                               kindArr,FFTW_EXHAUSTIVE);
    kindArr[0]=FFTW_REDFT01;
    kindArr[1]=FFTW_REDFT01;
    kindArr[2]=FFTW_REDFT01;
    fftw_plan plan_n2;
    plan_n2 = fftw_mpi_plan_r2r(nrsizes,rsizes,data_nd,data_nd,mpiComm,
                               kindArr,FFTW_EXHAUSTIVE);
  
    for (j=0; j<local_n0_n; ++j) {
     for (k=0; k<Nr; ++k) {
      for (i=0; i<Nx; ++i ) {
       data_nd[j*Nx*Nr+k*Nx+i]=mytData[j*Nx*Nr+k*Nx+i];
      }
     }
    }

    for (k=0;k<Nr;++k) {
     for (j=0;j<sNy;++j) {
      for (i=0;i<sNx;++i) {
       ioff=_I3LOC(i,j,k);
       loc2glob(mpiRank,
                i,j,k,
                Nx,Ny,Nr,sNx,sNy,
                &ig,&jg,&kg);
       if ( abs(idg - ig) < 2 &
            abs(jdg - jg) < 2 &
            abs(kdg - kg) < 2 ) {
        printf("BEFORE REDFT10: ioff, data_nd(%d,%d,%d) = %d, %f\n",i,j,k,ioff,data_nd[ioff]);
       }
      }
     }
    }
  
    double wt0mpi, wt1mpi;
    wt0mpi=omp_get_wtime();
    fftw_execute(plan_n1);
    wt1mpi=omp_get_wtime();
    printf ("Forward MPI wall time (plan_n1) = %f\n",wt1mpi-wt0mpi);
    fftw_destroy_plan(plan_n1);

    for (k=0;k<Nr;++k) {
     for (j=0;j<sNy;++j) {
      for (i=0;i<sNx;++i) {
       ioff=_I3LOC(i,j,k);
       loc2glob(mpiRank,
                i,j,k,
                Nx,Ny,Nr,sNx,sNy,
                &ig,&jg,&kg);
       if ( abs(idg - ig) < 2 &
            abs(jdg - jg) < 2 &
            abs(kdg - kg) < 2 ) {
        printf("AFTER REDFT10: ioff, data_nd(%d,%d,%d) = %d, %f\n",i,j,k,ioff,data_nd[ioff]);
       }
      }
     }
    }

    fac=1/(2.*Nx*2.*Ny*2.*Nr);  // NOW CHANGED
    for (j=0; j<local_n0_n; ++j) {
     for (k=0; k<Nr; ++k) {
      for (i=0; i<Nx; ++i ) {
       data_nd[j*Nx*Nr+k*Nx+i]=data_nd[j*Nx*Nr+k*Nx+i]*fac*mytScale[j*Nx*Nr+k*Nx+i];
       // data_nd[j*Nx*Nr+k*Nx+i]=data_nd[j*Nx*Nr+k*Nx+i]*fac;
      }
     }
    }

    wt0mpi=omp_get_wtime();
    fftw_execute(plan_n2);
    wt1mpi=omp_get_wtime();
    printf ("Inverse MPI wall time (plan_n) = %f\n",wt1mpi-wt0mpi);

    for (k=0;k<Nr;++k) {
     for (j=0;j<sNy;++j) {
      for (i=0;i<sNx;++i) {
       ioff=_I3LOC(i,j,k);
       loc2glob(mpiRank,
                i,j,k,
                Nx,Ny,Nr,sNx,sNy,
                &ig,&jg,&kg);
       if ( abs(idg - ig) < 2 &
            abs(jdg - jg) < 2 &
            abs(kdg - kg) < 2 ) {
        printf("AFTER REDFT01: ioff, data_nd(%d,%d,%d) = %d, %f\n",i,j,k,ioff,data_nd[ioff]);
       }
      }
     }
    }

    free(mytData);
    free(mytScale);
    free(lambda_fac_i);
    free(lambda_fac_j);
    free(lambda_fac_k);
    free(rsizes);

    fftw_destroy_plan(plan_n2);
    fftw_free(data_nd);

    MPI_Finalize();
 
}
