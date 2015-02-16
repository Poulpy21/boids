
#include "initBounds.hpp"

namespace kernel {

    __constant__ Real dt;

    __constant__ Real wCohesion;
    __constant__ Real wAlignment;
    __constant__ Real wSeparation;

    __constant__ Real rCohesion;
    __constant__ Real rAlignment;
    __constant__ Real rSeparation;

    __constant__ Real maxVelocity;
    __constant__ Real domainSize;


    __launch_bounds__(MAX_THREAD_PER_BLOCK)
        __global__ void initializeBoids(const unsigned int nBoids, float *rand, Real* agents, const InitBounds<Real> initBounds) {

            unsigned long int id = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

            if(id > 9ul * nBoids)
                return;

            switch(id/nBoids) {
                case(0): 
                    agents[id] = 3.14;
                case(1): 
                    agents[id] = 4.13;
                default:
                    agents[id] = 888;
            }
        }

    void initializeBoidsKernel(unsigned int nBoids, float *rand_d, Real *agents_d, const InitBounds<Real> initBounds, cudaStream_t &stream) {
        dim3 dimBlock(MAX_THREAD_PER_BLOCK);
        dim3 dimGrid((unsigned int)ceil(nBoids/MAX_THREAD_PER_BLOCK) % 65535, ceil(nBoids/(MAX_THREAD_PER_BLOCK*65535.0f)));
        log4cpp::log_console->infoStream() << "[KERNEL::InitializeBoids] <<<" 
            << utils::toStringDim(dimBlock) << ", " 
            << utils::toStringDim(dimGrid) <<  ", "
            << stream
            << ">>>";

        initializeBoids<<<dimGrid,dimBlock,0,stream>>>(nBoids, rand_d, agents_d, initBounds);
        CHECK_KERNEL_EXECUTION();
    }

}
