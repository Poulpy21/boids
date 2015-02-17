
#include "cudaWorkspace.hpp"

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

CudaWorkspace::CudaWorkspace(const Options &options, const InitBounds<Real> &initBounds) :
    options(options), initBounds(initBounds), 
    nStreamsPerDevice(2u)
{

    nAgents = options.nAgents;
    agentsPerKernel = 1024u;
    nKernels = (nAgents + agentsPerKernel - 1u)/agentsPerKernel;
    
    initStreams();
    initSymbols();
    initBoids();
}
        
void CudaWorkspace::initStreams() {
    for (unsigned int k = 0; k < nStreamsPerDevice; k++) {

        std::vector<cudaStream_t> deviceStreams;

        for (unsigned int i = 0; i < DevicePool::nDevice; i++) {
            cudaStream_t buffer;
            cudaStreamCreate(&buffer);
            deviceStreams.push_back(buffer);
        }

        streams.push_back(deviceStreams);
    }
}

void CudaWorkspace::initSymbols() {
    for(unsigned int i = 0u; i < DevicePool::nDevice; i++) {

        // Upload options to devices
        CHECK_CUDA_ERRORS(cudaSetDevice(i));

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
    }
}



void CudaWorkspace::initBoids() {
    
    //init agents
    
    agents_h = PinnedCPUResource<Real>(9u*nAgents); 
    agents_h.allocate();
   
#ifdef CURAND_ENABLED
    unsigned int agentsToInitialize = nAgents;
    unsigned int deviceId = 0u, devAgents = 0u, i = 0u, offset = 0u; 
    std::vector<unsigned int> devMaxAgents;
    std::vector<GPUResource<float>*> random_d;
    std::vector<GPUResource<Real>*> agents_d;

    while(agentsToInitialize > 0) {
        deviceId = i % DevicePool::nDevice;
        CHECK_CUDA_ERRORS(cudaSetDevice(deviceId));

        if(i < DevicePool::nDevice) {
            devMaxAgents.push_back(computeMaxAgentsAtInit(deviceId));

            devAgents = std::min(devMaxAgents[i], agentsToInitialize);

            random_d.push_back(new GPUResource<float>(deviceId, devAgents*9ul));
            agents_d.push_back(new GPUResource<Real>(deviceId, devAgents*9ul));
            random_d[i]->allocate();
            agents_d[i]->allocate();

            log4cpp::log_console->debugStream() << "Device " << i << " can allocate " << devMaxAgents[i] << " boids !";
        }

        devAgents = std::min(devMaxAgents[deviceId], agentsToInitialize);

        log4cpp::log_console->debugStream() << "Device " << deviceId << " allocating " << devAgents << " boids !";

        curandGenerator_t generator;
        unsigned long long int seed = Random::randl();
        //CHECK_CURAND_ERRORS(curandSetStream(generator, streams[deviceId][0])); //CURAND CRASH ...
        CHECK_CURAND_ERRORS(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MRG32K3A));
        CHECK_CURAND_ERRORS(curandSetPseudoRandomGeneratorSeed(generator, seed));
        CHECK_CURAND_ERRORS(curandGenerateUniform(generator, random_d[deviceId]->data(), 9u*devAgents));
        CHECK_CURAND_ERRORS(curandDestroyGenerator(generator));

        kernel::initializeBoidsKernel(devAgents, random_d[deviceId]->data(), agents_d[deviceId]->data());

        //Copy data back
        cudaMemcpy(agents_h.data() + 9ul*offset, agents_d[deviceId]->data(), 9u*devAgents*sizeof(Real), cudaMemcpyDeviceToHost);

        offset += devAgents;
        agentsToInitialize -= devAgents;
        i++;
    }

    for(auto resource : random_d)
        delete resource;
    
    for(auto resource : agents_d)
        delete resource;
#else
    //TODO CURAND NOT PRESENT - CPU GENERATION
#endif
    
    sortBoids();
}

#ifdef CURAND_ENABLED
unsigned int CudaWorkspace::computeMaxAgentsAtInit(unsigned int deviceId) {
    float safeCoef = 0.5f;
    return static_cast<unsigned int>(safeCoef*GPUMemory::memoryLeft(deviceId)/(9u*(sizeof(Real) + sizeof(float))));
}
#endif


void CudaWorkspace::sortBoids() {
#ifdef THRUST_ENABLED
    kernel::thrustSort(agents_h.data(), nAgents);
#endif
}

void CudaWorkspace::update() {

}



void CudaWorkspace::computeAndApplyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights) {
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
        for (size_t i = 0; i < receivedMeanAgents.size(); i++) {
            Real dist = (agents[k].position - receivedMeanAgents[i].position).norm();
            Real weight = receivedMeanAgentsWeights[i]; 
            if (dist < opt.rSeparation) {
                forceSeparation -= weight * (agents[k].position - receivedMeanAgents[i].position).normalized();
                countSeparation += weight;
            }
            if (dist < opt.rCohesion) {
                forceCohesion += weight * receivedMeanAgents[i].position;
                countCohesion += weight;
            }
            if (dist < opt.rAlignment) {
                forceAlignment += weight * receivedMeanAgents[i].velocity;
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

#endif
