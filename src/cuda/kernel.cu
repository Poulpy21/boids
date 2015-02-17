
#include "initBounds.hpp"
#include "options.hpp"
#include <string>

namespace kernel {

    __constant__ __device__ Real dt = 2;

    __constant__ Real wCohesion;
    __constant__ Real wAlignment;
    __constant__ Real wSeparation;

    __constant__ Real rCohesion;
    __constant__ Real rAlignment;
    __constant__ Real rSeparation;

    __constant__ Real maxVelocity;
    __constant__ Real domainSize;

    __constant__ Real minInitValues[9];
    __constant__ Real maxInitValues[9];

    template <typename T>
    __device__ inline T mix(T alpha, T a, T b) {
        return (a + alpha*(b-a));
    }

    std::string toStringDim(const dim3 &dim) {
        std::stringstream ss;
        ss << "(" << dim.x << "," << dim.y << "," << dim.z << ")";
        return ss.str();
    }


    __launch_bounds__(MAX_THREAD_PER_BLOCK)
        __global__ void initializeBoids(const unsigned int nBoids, float *rand, Real* agents) {

            unsigned long int id = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

            if(id >= 9u*nBoids)
                return;

            unsigned int idd = id/nBoids;

            agents[id] = mix(rand[id], minInitValues[idd], maxInitValues[idd]);
        }

    void initializeBoidsKernel(unsigned int nBoids, float *rand_d, Real *agents_d) {
        float nReals = nBoids * 9u;
        dim3 dimBlock(MAX_THREAD_PER_BLOCK);
        dim3 dimGrid((unsigned int)ceil(nReals/MAX_THREAD_PER_BLOCK) % 65535, ceil(nReals/(MAX_THREAD_PER_BLOCK*65535.0f)));
        log4cpp::log_console->infoStream() << "[KERNEL::InitializeBoids] <<<" 
            << toStringDim(dimBlock) << ", " 
            << toStringDim(dimGrid)
            << ">>>";

        initializeBoids<<<dimGrid,dimBlock>>>(nBoids, rand_d, agents_d);
        CHECK_KERNEL_EXECUTION();
    }
    
}
