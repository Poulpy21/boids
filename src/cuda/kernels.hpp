
#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "headers.hpp"

#ifdef CUDA_ENABLED

namespace kernel {

    extern void computeForcesKernel(Real *boidData, 
            Real *meanBoidData, 
            int *meanBoidWeights, 
            const int nBoids, 
            const int nMeanBoids, 
            const struct Options *opt);

    extern void applyForcesKernel(Real *boidData, 
            const int nBoids, 
            const struct Options *opt);

    extern void computeMeanBoidKernel(Real *boidData, 
            const int nBoids, 
            Vector *meanBoid);

    extern void initializeBoidsKernel(unsigned int nBoids, 
            float *rand_d, Real *agents_d, 
            const InitBounds<Real> initBounds, cudaStream_t &stream);

}

#endif
#endif



