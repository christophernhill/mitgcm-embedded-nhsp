# Spectral 3D column model

Here we are building a 3D column model which solves the equations of fluid motion in the ocean using a Fourier-Galerkin spectral method with the aim of resolving small-scale/submesoscale processes. It is meant to be embedded into each grid point of a coarser resolution model and thus is a super-parameterization.

For now the aim is to get an isolated version of this model to run on a CPU cluster using the [FFTW](http://www.fftw.org/) library to perform the fast Fourier transforms. Then we will try to port the solver to a GPU cluster and may switch to CUDA's [cuFFT](https://developer.nvidia.com/cufft) library.

### CPU compilation instructions
Make sure to load the Intel compiler and MPI compiler(?): <tt>module load intel/2017-01 impi/2017-01</tt>

To compile:
<pre>
mpiicc -qopenmp -O -xHost fftw_dct_mpi_nd.c -I /nfs/cnhlab002/cnh/opt/fftw/usr/local/include/ -L /nfs/cnhlab002/cnh/opt/fftw/usr/local/lib/ -lfftw3_mpi -lfftw3 -lm -o fftw_dct_mpi_nd
</pre>
Notes on compiling:
* I include the source files and libraries from Chris' local installation of FFTW3, although you could download the latest FFTW3 release and use that instead. Make sure to compile FFTW3 with MPI support though.
* Argument/flag ordering matters very much. Make sure the includes and library linking is done after specifying the file name. And make sure <tt>-lfftw3</tt> comes after <tt>-lfftw3_mpi</tt>!

Example run:
<pre>
mpirun -n 2 ./fftw_dct_mpi_nd 30 20 5 30 10 1
</pre>
