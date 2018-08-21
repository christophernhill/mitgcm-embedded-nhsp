/*
  DCT-II definition
  1-d
  X(k) = (2/N)^0.5*s(k)*sum_(n=0)^(n=N-1)[x(n)cos((pi*k*(n+0.5)/N) : k=0,N-1
  { s(0) = 1/(2^0.5); s(k=1,N-1)=1
  2-d
  X(k1,k2)=((2/N)^0.5)*((2/M)^0.5)*sum_(n=0)^(n=N-1),sum_(m=0)^(m=M-1)
           s(k1)*s(k2)*
           x(n,m)*
           cos(pi*k1*(n+0.5)/N)*
           cos(pi*k2*(m+0.5)/M
           : k1=0,N-1;k2=0,M-1
  { s(k[1,2]=0)=1/(2^0.5); s(k[1,2]=1,[N,M]-1)=1}
  DFT equivalent of DCT-II involves creating mirrored sequence
  padded with zero between points. Scaled appropriately real
  part of the first N,M DFT entries will be equivalent to DCT-II
  coefficients.
*/
#include <math.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N     192
#define M     192
#define L     150
#define NMID  95
#define MMID  95
#define LMID  74
#define NRANK 3
#define I3(a,b,c) a+b*N+c*M*N

main(int argc, char* argv[]) {
    
    int i,j,k,ioff;
    double fac;
    
    fftw_plan plan;
    fftw_complex in[L*M*N], out[L*M*N];
    
    int  rank_list[NRANK];
    rank_list[0] = L;
    rank_list[1] = M;
    rank_list[2] = N;

    /*
    Try DCT-II using FFTW REDFT10 _N interface and its
    inverse REDFT01.
    */

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // N-d form
    double inr[L*M*N], outr[L*M*N];
    fftw_r2r_kind kArr[NRANK];

    for (k = 0; k < L; ++k) {
        for (j = 0; j < M; ++j) {
            for (i = 0; i < N; ++i) {
                ioff = I3(i,j,k);
                inr[ioff]=0.;
            }
        }
    }

    ioff = I3(NMID, MMID, LMID);
    inr[ioff] = 1.;

    printf("inr (N-d)\n");
    
    for (k = 0; k < L; ++k) {
        for (j = 0; j < M; ++j) {
            for (i = 0; i < N; ++i) {
                ioff = I3(i,j,k);
                // printf("(%d,%d,%d): %f\n",i,j,k,inr[ioff]);
            }
        }
    }

    for (i = 0; i < NRANK; ++i)
        kArr[i] = FFTW_REDFT10;

    // FFTW_MEASURE, FFTW_PATIENT, or FFTW_EXHAUSTIVE
    
    // plan = fftw_plan_r2r(NRANK,rank_list,inr,outr,kArr,FFTW_ESTIMATE);
    plan = fftw_plan_r2r(NRANK, rank_list, inr, outr, kArr, FFTW_MEASURE);
    // plan = fftw_plan_r2r(NRANK,rank_list,inr,outr,kArr,FFTW_EXHAUSTIVE);
    
    // Start forward timing
    clock_t t0, t1;
    printf ("Forward transform(s) start \n");
    
    t0 = clock();
    struct timespec rt0, rt1;
    clock_gettime(CLOCK_MONOTONIC, &rt0);
    
    printf ("Calling plan execute \n");
    
    double wt0, wt1;
    wt0 = omp_get_wtime();

    fftw_execute(plan);

    wt1 = omp_get_wtime();
    clock_gettime(CLOCK_MONOTONIC, &rt1);
    t1 = clock();
    
    printf ("Forward transform(s) end \n");
    printf ("Forward CPU time = %f\n",((double)(t1-t0))/(double)CLOCKS_PER_SEC);
    
    struct timespec {
        time_t   tv_sec;        /* seconds */
        long     tv_nsec;       /* nanoseconds */
    };

    printf ("Forward RT wall time = %f\n",((rt1.tv_sec+rt1.tv_nsec/1.e9)-(rt0.tv_sec+rt0.tv_nsec/1.e9)));
    printf ("Forward OMP wall time = %f\n",wt1-wt0);

    // End forward timing
    fftw_destroy_plan(plan);

    fac = 1 / (2.*N * 2.*L * 2.*M);
    printf("outr (N-d)\n");
    
    for (k = 0; k < L; ++k) {
        for (j = 0; j < M; ++j) {
            for (i = 0; i < N; ++i) {
                ioff = I3(i,j,k);
                outr[ioff] = outr[ioff] * fac;
                // printf("(%d,%d,%d): %f\n",i,j,k,outr[ioff]);
            }
        }
    }

    for (i = 0; i < NRANK; ++i)
        kArr[i] = FFTW_REDFT01;

    plan = fftw_plan_r2r(NRANK, rank_list, outr, inr, kArr, FFTW_ESTIMATE);
    // plan = fftw_plan_r2r(NRANK,rank_list,outr,inr,kArr,FFTW_EXHAUSTIVE);

    t0 = clock();
    fftw_execute(plan);
    t1 = clock();
    printf ("Backward CPU time = %f\n",((double)(t1-t0))/(double)CLOCKS_PER_SEC);
    
    fftw_destroy_plan(plan);

    printf("inr (N-d)\n");
    for (k = 0; k < L; ++k) {
        for (j = 0; j < M; ++j) {
            for (i = 0; i < N; ++i) {
                ioff = I3(i,j,k);
                // printf("(%d,%d,%d): %f\n",i,j,k,inr[ioff]);
            }
        }
    }

    fftw_cleanup_threads();
}
