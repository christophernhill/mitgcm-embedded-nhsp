#include <stdio.h>

void spectral_3d_poisson_solver_(double* myTime, int* myIter, int* myThid) {
    printf("[F2C] This is a C function being called from inside the MITGCM!\n");
    printf("[F2C] myTime=%f, myIter=%d, myThid=%d\n", *myTime, *myIter, *myThid);
    printf("[F2C] myTime=%f, myIter=%d, myThid=%d (short)\n", *myTime, (short) *myIter, (short) *myThid);
    printf("[F2C] myTime=%f, myIter=%d, myThid=%d (long)\n", *myTime, (long) *myIter, (long) *myThid);
}
