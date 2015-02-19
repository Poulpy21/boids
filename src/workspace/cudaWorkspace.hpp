
#ifndef CUDA_WORKSPACE_H
#define CUDA_WORKSPACE_H

#include "headers.hpp"

#ifdef CUDA_ENABLED

#include "options.hpp"
#include "vector.hpp"
#include "messenger.hpp"
#include "agentData.hpp"
#include "PinnedCPUResource.hpp"
#include "devicePool.hpp"
#include "initBounds.hpp"
#include "boidMemoryView.hpp"
#include "localBoidDataStructure.hpp"

class CudaWorkspace {

    public:
        CudaWorkspace(const Options &options, const InitBounds<Real> &initBounds, 
                LocalBoidDataStructure<Real> *boidDataStructure, unsigned int nStreamsPerDevice = 2u);

        void update();

    private:
        void initStreams();
        void initSymbols();
        void initBoids();
        
        void computeAndApplyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights);
        
#ifdef CURAND_ENABLED
        unsigned int computeMaxAgentsAtInit(unsigned int deviceId);
#endif
    
    private:
        
        const Options options;
        const InitBounds<Real> initBounds;
        LocalBoidDataStructure<Real>* const boidDataStructure;
        const unsigned int nStreamsPerDevice;

        unsigned int nAgents;
        unsigned int stepId;
        
        std::vector<std::vector<cudaStream_t>> streams;
   
        PinnedCPUResource<Real> agents_h;
        BoidMemoryView<Real> agents_view_h;
};

#endif

#endif
