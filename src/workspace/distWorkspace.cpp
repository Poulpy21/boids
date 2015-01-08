#include "headers.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "distWorkspace.hpp"
#include "messenger.hpp"
#include "agentKernel.cu"

DistWorkspace::DistWorkspace(size_t localAgentsCount, MPI_Comm comm, int root) : 
    agents(), comm(comm), mess(comm), rootID(root)
{
    MPI_Comm_rank(comm, &myID);
    init(localAgentsCount);
}

void DistWorkspace::init(size_t localAgentsCount) {
    mess.broadcastOptions(&opt, rootID);

    // Upload options to device
    // TODO

    // Init boids
    srand48(std::time(0));

    // This loop may be quite expensive due to random number generation
    for(size_t i = 0; i < localAgentsCount; i++){
        // Create random position
        Vec3<Real> position(drand48(), drand48(), drand48());

        // Create random velocity
        agents.push_back(Agent(position, Vec3<Real>(), Vec3<Real>()));
    }
}

void DistWorkspace::update() {
    // Update neighborhood info
    // TODO
    // neighbors = ...
   
#ifdef CUDA_ENABLED 
    // Upload boids to device
    // TODO
    // memcpy : agents -> d_agents
#endif

    // Compute mean boid
    // TODO
    // mean(d_agents, d_meanAgent)

    // Get mean boids from the neighborhood and send ours (approximation)
    // TODO
    // mess.exchangeMeanAgents(receivedMeanAgents, meanAgent, neighbors);

    // Compute and apply forces to local boids
    // TODO
#ifdef CUDA_ENABLED
    // memcpy -> d_meanAgents
    // applyInternalForcesKernel(d_agents, d_opt)
    // applyExternalForcesKernel(d_agents, d_meanAgents, d_opt)
    // memcpy -> agents
#else
    // TODO code workspace.cpp + meanAgents
#endif
    
    // Sort out boids that cross domain boundaries
    // TODO
    
    // Exchange boids that cross domain boundaries
    // TODO
    // mess.exchangeAgents(agents, agentsForNeighborsMap, neighbors);

    // Note: if practically there is too many agents to send, we should compute and send agents by batches
}

void DistWorkspace::save(int stepId) {
    std::ofstream myfile;

    std::stringstream ss;
    ss << "data/boids" << std::setw(3) << std::setfill('0') << myID << ".xyz";
    myfile.open(ss.str(), stepId==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << agents.size() << std::endl;
    for (Agent a : agents)
        myfile << "B " << a.position;
        //TODO: boidType
    myfile.close();

    //TODO zlib?
}

