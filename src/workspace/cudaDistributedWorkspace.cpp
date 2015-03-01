
#include "cudaDistributedWorkspace.hpp"

#ifdef CUDA_ENABLED

#include "agent.hpp"
#include "GPUMemory.hpp"
#include "GPUResource.hpp"
#include "rand.hpp"
#include "initBounds.hpp"
#include "kernel.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>

#define PRINT(s)    { MPI_Barrier(comm); if (rank == masterRank) { std::cout << s; } }
#define PAUSE()     { MPI_Barrier(comm); std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
#define JUMP_LINE() { PRINT("\n") }

CudaDistributedWorkspace::CudaDistributedWorkspace(const BoundingBox<3u,Real> &globalDomain, bool keepBoidsInGlobalDomain, 
        const Options &opt,
        unsigned int rank, unsigned int size, unsigned int masterRank,
        const MPI_Comm &comm, const std::string &name, unsigned int deviceId_) :
    options(opt),
    computeGrid(globalDomain, opt, size, rank, masterRank),
    localBoidGrid(rank, computeGrid.getSubdomain(rank), globalDomain, false, 
            std::max<Real>(options.rCohesion, std::max<Real>(options.rAlignment, options.rSeparation)),
            deviceId_),
    rank(rank), size(size), masterRank(masterRank), comm(comm), name(name),
    deviceId(deviceId_),
    nGlobalAgents(opt.nAgents), 
    nLocalAgents((((rank == size) - 1 && (nGlobalAgents % size != 0)) ? nGlobalAgents % size : nGlobalAgents/size)), 
    stepId(1u)
{

    log_console->infoStream() << "Rank " << rank << "/" << size << " choosed domain " << localBoidGrid.getLocalDomain();
    PAUSE();
    JUMP_LINE();

    //Initialize init bounds
    const Vec3<Real> &xmin = localBoidGrid.getLocalDomain().min;
    const Vec3<Real> &xmax = localBoidGrid.getLocalDomain().max;
    Real minValues[9] = {xmin.x, xmin.y, xmin.z,  0,0,0,  0,0,0};
    Real maxValues[9] = {xmax.x, xmax.y, xmax.z,  0,0,0,  0,0,0};
    initBounds = InitBounds<Real>(minValues, maxValues);
    
    initSymbols();
    initBoids();
    
    localBoidGrid.init(agents_view_h, nLocalAgents);
}

void CudaDistributedWorkspace::initSymbols() {

    // Upload options to devices
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::dt, &options.dt, sizeof(Real)));

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::wCohesion, &options.wCohesion, sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::wAlignment, &options.wAlignment, sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::wSeparation, &options.wSeparation, sizeof(Real)));

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::rCohesion, &options.rCohesion, sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::rAlignment, &options.rAlignment, sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::rSeparation, &options.rSeparation, sizeof(Real)));

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::maxVelocity, &options.maxVel, sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(&kernel::domainSize, &options.domainSize, sizeof(Real)));

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(kernel::minInitValues, initBounds.minValues, 9u*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(kernel::maxInitValues, initBounds.maxValues, 9u*sizeof(Real)));

    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    MPI_Barrier(comm);
}



void CudaDistributedWorkspace::initBoids() {

    //init agents
    agents_h = UnpagedCPUResource<Real>(BoidMemoryView<Real>::N*nLocalAgents); 
    agents_h.allocate();
    agents_view_h = BoidMemoryView<Real>(agents_h.data(), nLocalAgents);

#ifdef CURAND_ENABLED
    std::cout << std::endl;
    std::cout << ":: Initializing " << nLocalAgents << " boids with Curand !";
    std::cout << std::endl;
    std::cout << std::endl;

    unsigned int agentsToInitialize = nLocalAgents;
    unsigned int devAgents = 0u, i = 0u, offset = 0u; 
    unsigned int devMaxAgents;
    GPUResource<float> *random_d;
    GPUResource<Real>  *agents_d;

    while(agentsToInitialize > 0) {
        if(i == 0) {
            devMaxAgents = computeMaxAgentsAtInit(deviceId);
            devAgents = std::min(devMaxAgents, agentsToInitialize);
        }

        unsigned int size = devAgents*(BoidMemoryView<Real>::N-1u);
        random_d = new GPUResource<float>(deviceId, size, NEXT_POW_2(size));
        agents_d = new GPUResource<Real> (deviceId, size, NEXT_POW_2(size));
        random_d->allocate();
        agents_d->allocate();

        devAgents = std::min(devMaxAgents, agentsToInitialize);

        log4cpp::log_console->infoStream() << "\tDevice " << deviceId << " allocating " << devAgents << " boids !";

        curandGenerator_t generator;
        unsigned long long int seed = Random::randl();
        CHECK_CURAND_ERRORS(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MRG32K3A));
        CHECK_CURAND_ERRORS(curandSetPseudoRandomGeneratorSeed(generator, seed));
        CHECK_CURAND_ERRORS(curandGenerateUniform(generator, random_d->data(), (BoidMemoryView<Real>::N-1u)*devAgents));
        CHECK_CURAND_ERRORS(curandDestroyGenerator(generator));

        kernel::initializeBoidsKernel(devAgents, random_d->data(), agents_d->data());
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

        //Copy data back
        cudaMemcpy(agents_h.data() + (BoidMemoryView<Real>::N-1u)*offset, agents_d->data(), (BoidMemoryView<Real>::N-1u)*devAgents*sizeof(Real), cudaMemcpyDeviceToHost);

        offset += devAgents;
        agentsToInitialize -= devAgents;
        i++;
    }
    
    delete agents_d;
    delete random_d;

#else
    //TODO CURAND NOT PRESENT - CPU GENERATION
    std::cout << std::endl;
    std::cout << "Initializing " << nLocalAgents << " boids with CPU !";
    std::cout << std::endl;
    std::cout << std::endl;
    NOT_IMPLEMENTED_YET;
#endif
    
    MPI_Barrier(comm);
}

#ifdef CURAND_ENABLED
unsigned int CudaDistributedWorkspace::computeMaxAgentsAtInit(unsigned int deviceId) {
    float safeCoef = 0.25f;
    return static_cast<unsigned int>(safeCoef*GPUMemory::memoryLeft(deviceId)/((BoidMemoryView<Real>::N-1)*(sizeof(Real) + sizeof(float))));
}
#endif

void CudaDistributedWorkspace::update() {
    if(rank == masterRank) {
        log4cpp::log_console->debugStream() << "Computing step " << stepId;
    }
    
    localBoidGrid.computeLocalStep();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    stepId++;
    MPI_Barrier(comm);
}

void CudaDistributedWorkspace::computeAndApplyForces(Container &receivedMeanLocalAgents, std::vector<int> &receivedMeanLocalAgentsWeights) {
#if FALSE
    for (size_t k = 0; k < agents.size(); k++) {
        int countSeparation = 0, countCohesion = 0, countAlignment = 0;
        Vec3<Real> forceSeparation, forceCohesion, forceAlignment;
        Compute "internal forces"
            for (size_t i = 0; i < agents.size(); i++) {
                if (i != k) {
                    Real dist = (agents[k].position - agents[i].position).norm();
                    if (dist < opt.rSeparation) {
                        forceSeparation -= (agents[k].position - agents[i].position).normalized();
                        ++countSeparation;
                    }
                    if (dist < opt.rCohesion) {
                        forceCohesion += agents[i].position;
                        ++countCohesion;
                    }
                    if (dist < opt.rAlignment) {
                        forceAlignment += agents[i].velocity;
                        ++countAlignment;
                    }
                }
            }
        Compute "external forces"
            for (size_t i = 0; i < receivedMeanLocalAgents.size(); i++) {
                Real dist = (agents[k].position - receivedMeanLocalAgents[i].position).norm();
                Real weight = receivedMeanLocalAgentsWeights[i]; 
                if (dist < opt.rSeparation) {
                    forceSeparation -= weight * (agents[k].position - receivedMeanLocalAgents[i].position).normalized();
                    countSeparation += weight;
                }
                if (dist < opt.rCohesion) {
                    forceCohesion += weight * receivedMeanLocalAgents[i].position;
                    countCohesion += weight;
                }
                if (dist < opt.rAlignment) {
                    forceAlignment += weight * receivedMeanLocalAgents[i].velocity;
                    countAlignment += weight;
                }
            }   
        agents[k].direction = opt.wSeparation * ( countSeparation>0 ? forceSeparation/static_cast<Real>(countSeparation) : forceSeparation) +
            opt.wCohesion   * ( countCohesion  >0 ? forceCohesion  /static_cast<Real>(countCohesion)   : forceCohesion  ) +
            opt.wAlignment  * ( countAlignment >0 ? forceAlignment /static_cast<Real>(countAlignment)  : forceAlignment );
    }

    Integration in time using euler method
        for(size_t k = 0; k < agents.size(); k++){
            agents[k].velocity += agents[k].direction;

            Real speed = agents[k].velocity.norm();
            if (speed > opt.maxVel) {
                agents[k].velocity *= opt.maxVel/speed;
            }
            agents[k].position += opt.dt*agents[k].velocity;

            Real modX = fmod(agents[k].position.x, opt.domainSize);
            Real modY = fmod(agents[k].position.y, opt.domainSize);
            Real modZ = fmod(agents[k].position.z, opt.domainSize);
            agents[k].position.x = modX > 0 ? modX : modX + opt.domainSize;
            agents[k].position.y = modY > 0 ? modY : modY + opt.domainSize;
            agents[k].position.z = modZ > 0 ? modZ : modZ + opt.domainSize;
        }
#endif
}

#undef PRINT
#undef PAUSE
#undef JUMP_LINE

#endif
