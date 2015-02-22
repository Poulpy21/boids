
#include "initBounds.hpp"
#include "options.hpp"
#include <string>
#include <iostream>

#include "kernel_utilities.cuh"

namespace kernel {

    __launch_bounds__(MAX_THREAD_PER_BLOCK)
        __global__ void initializeBoids(const unsigned int nBoids, float *rand, Real* agents) {

            unsigned long int id = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

            if(id >= 6u*nBoids)
                return;

            unsigned int idd = id/nBoids;

            agents[id] = mix(rand[id], minInitValues[idd], maxInitValues[idd]);
        }

    void initializeBoidsKernel(unsigned int nBoids, float *rand_d, Real *agents_d) {
        float nReals = nBoids * 6u;
        dim3 dimBlock(MAX_THREAD_PER_BLOCK);
        dim3 dimGrid((unsigned int)ceil(nReals/MAX_THREAD_PER_BLOCK) % 65535, ceil(nReals/(MAX_THREAD_PER_BLOCK*65535.0f)));
        log4cpp::log_console->infoStream() << "[KERNEL::InitializeBoids] <<<" 
            << utils::toStringDim(dimBlock) << ", " 
            << utils::toStringDim(dimGrid)
            << ">>>";

        initializeBoids<<<dimGrid,dimBlock>>>(nBoids, rand_d, agents_d);
        CHECK_KERNEL_EXECUTION();
    }

}
