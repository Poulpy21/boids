
#ifdef __CUDACC__

#include "kernel_utilities.cuh"

namespace kernel {
   
    //Constants
    __device__ __constant__ Real dt;

    __device__ __constant__ Real wCohesion;
    __device__ __constant__ Real wAlignment;
    __device__ __constant__ Real wSeparation;

    __device__ __constant__ Real rCohesion;
    __device__ __constant__ Real rAlignment;
    __device__ __constant__ Real rSeparation;

    __device__ __constant__ Real maxVelocity;
    __device__ __constant__ Real domainSize;

    __device__ __constant__ Real minInitValues[9];
    __device__ __constant__ Real maxInitValues[9];

}

#endif


