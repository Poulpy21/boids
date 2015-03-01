
#ifndef CUDA_DISTRIBUTED_WORKSPACE_H
#define CUDA_DISTRIBUTED_WORKSPACE_H

#include "headers.hpp"

#ifdef CUDA_ENABLED

#include "options.hpp"
#include "vector.hpp"
#include "messenger.hpp"
#include "agentData.hpp"
#include "PinnedCPUResource.hpp"
#include "UnpagedCPUResource.hpp"
#include "devicePool.hpp"
#include "boidGrid.hpp"
#include "computeGrid.hpp"
#include "initBounds.hpp"
#include "boidMemoryView.hpp"
#include "localBoidDataStructure.hpp"

class CudaDistributedWorkspace {

    public:
        
        CudaDistributedWorkspace(const BoundingBox<3u,Real> &globalDomain, bool keepBoidsInGlobalDomain, 
                const Options &opt,
                unsigned int rank, unsigned int size, unsigned int masterRank, 
                const MPI_Comm &comm, const std::string &name,
                unsigned int deviceId);

        void update();

    protected:
        void initSymbols();
        void initBoids();
        
        void computeAndApplyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights);

#ifdef CURAND_ENABLED
        unsigned int computeMaxAgentsAtInit(unsigned int deviceId);
#endif
    
    protected:
        const Options &options;
        ComputeGrid<Real> computeGrid;
        BoidGrid<Real, UnpagedCPUResource<Real> > localBoidGrid; //Pinned resource => MPI crash...
       
        const unsigned int rank, size, masterRank;
        const MPI_Comm &comm;
        const std::string name;

        const unsigned int deviceId;

        unsigned int nGlobalAgents;
        unsigned int nLocalAgents;
        unsigned int stepId;

        //PinnedCPUResource<Real> agents_h;  MPI crash ...
        UnpagedCPUResource<Real> agents_h;
        BoidMemoryView<Real> agents_view_h;
        InitBounds<Real> initBounds;
};

#endif

#endif
