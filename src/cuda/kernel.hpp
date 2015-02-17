
#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "headers.hpp"
#include "initBounds.hpp"

#ifdef CUDA_ENABLED

namespace kernel {
    
    extern __CONSTANT__ Real dt;

    extern __CONSTANT__ Real wCohesion;
    extern __CONSTANT__ Real wAlignment;
    extern __CONSTANT__ Real wSeparation;

    extern __CONSTANT__ Real rCohesion;
    extern __CONSTANT__ Real rAlignment;
    extern __CONSTANT__ Real rSeparation;

    extern __CONSTANT__ Real maxVelocity;
    extern __CONSTANT__ Real domainSize;

    extern __CONSTANT__ Real minInitValues[9];
    extern __CONSTANT__ Real maxInitValues[9];

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
            float *rand_d, Real *agents_d);

#ifdef THRUST_ENABLED
    extern void thrustSort(Real *agents_h, unsigned int nAgents);
#endif

}

#endif
#endif



