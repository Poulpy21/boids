#include "headers.hpp"
#include "distWorkspace.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "agent.hpp"

#undef CUDA_ENABLED //FIXME when the kernels are done

DistWorkspace::DistWorkspace(Options options, MPI_Comm comm, int root) : 
    agents(), opt(options), comm(comm), mess(comm), rootID(root)
#ifdef CUDA_ENABLED
    , h_agents(), h_meanAgents()
#endif
{
    MPI_Comm_rank(comm, &myID);
    init();
}

void DistWorkspace::init() {
#ifdef CUDA_ENABLED 
    // Upload options to device
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_opt, sizeof(struct Options)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_opt, &opt, sizeof(struct Options), cudaMemcpyHostToDevice));
    
    // Malloc d_meanAgent
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgent, sizeof(Vec3<Real>)));
#endif

    // Init boids
    srand48(std::time(0));
    //srand48(myID * std::time(0));

    
    // Initalize boids on root and scatter them to other processes after sorting
    if (myID == rootID) {
    // This loop may be quite expensive due to random number generation
        for(size_t i = 0; i < opt.nAgents; i++){
            // Create random position
            Vec3<Real> position(drand48(), drand48(), drand48());

            // Create random velocity
            agents.push_back(Agent(position, Vec3<Real>(), Vec3<Real>()));
        }
    }

    // TODO call sort boids method / kernel
    //sortAgents(agents);
    //mess.exchangeAgents(...);
    

}

void DistWorkspace::update() {
    // Update neighborhood info
    // TODO
    std::vector<int> neighbors = {};
   
    Agent meanAgent;
#ifdef CUDA_ENABLED 
    makeArrayFromContainer(agents, h_agents);
    
    // Malloc and upload boids to device
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_agents1, h_agents.size()*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_agents2, h_agents.size()*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_agents1, &h_agents, h_agents.size()*sizeof(Real), cudaMemcpyHostToDevice));

    // Compute mean boid
    computeMeanBoidKernel(d_agents1, agents.size(), d_meanAgent);
    
    CHECK_CUDA_ERRORS(cudaMemcpy(&meanAgent, d_meanAgent, sizeof(Vector), cudaMemcpyDeviceToHost));
#else
    computeMeanAgent(meanAgent);
#endif

    // Get mean boids from the neighborhood and send ours (approximation)
    Container receivedMeanAgents;
    std::vector<int> receivedMeanAgentsWeight;
    mess.exchangeMeanAgents(meanAgent, agents.size(), receivedMeanAgents, receivedMeanAgentsWeight, neighbors);

    // Compute and apply forces to local boids
#ifdef CUDA_ENABLED
    makeArrayFromContainer(receivedMeanAgents, h_meanAgents);

    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgents, receivedMeanAgents.size()*9*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_meanAgents, &receivedMeanAgents, receivedMeanAgents.size()*9*sizeof(Real), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgentsWeights, receivedMeanAgentsWeight.size()*sizeof(int)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_meanAgentsWeights, &receivedMeanAgentsWeight, receivedMeanAgentsWeight.size()*sizeof(int), cudaMemcpyHostToDevice));
    
    applyForcesKernel(d_currentAgents, d_newAgents, d_meanAgents, d_meanAgentsWeights, agents.size(), receivedMeanAgents.size(), d_opt);
    
    CHECK_CUDA_ERRORS(cudaMemcpy(&h_agents, d_agents2, h_agents.size()*sizeof(Real), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaFree(d_agents1));
    CHECK_CUDA_ERRORS(cudaFree(d_agents2));

    makeContainerFromArray(h_agents, agents);
#else
    applyForces(receivedMeanAgents, receivedMeanAgentsWeight);
#endif
    
    // Sort out boids that cross domain boundaries
    std::map<int, Container> agentsForNeighborsMap;
    sortAgents(agentsForNeighborsMap);
    
    // Exchange boids that cross domain boundaries
    mess.exchangeAgents(agents, agentsForNeighborsMap, neighbors);
}


void DistWorkspace::computeMeanAgent(Agent &meanAgent) {
    Vec3<Real> meanPos;
    int count = 0;
    for (Agent a : agents) {
        meanPos += a.position;
        count++;
    }
    meanAgent.position = meanPos / static_cast<Real>(count);
}

void DistWorkspace::applyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights) {
    for (size_t k = 0; k < agents.size(); k++) {
        int countSeparation = 0, countCohesion = 0, countAlignment = 0;
        Vec3<Real> forceSeparation, forceCohesion, forceAlignment;
        // Compute "internal forces"
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
        // Compute "external forces"
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

    // Integration in time using euler method
    for(size_t k = 0; k < agents.size(); k++){
        agents[k].velocity += agents[k].direction;

        Real speed = agents[k].velocity.norm();
        if (speed > opt.maxVel) {
            agents[k].velocity *= opt.maxVel/speed;
        }
        agents[k].position += opt.dt*agents[k].velocity;

        agents[k].position.x= fmod(agents[k].position.x,opt.domainSize);
        agents[k].position.y= fmod(agents[k].position.y,opt.domainSize);
        agents[k].position.z= fmod(agents[k].position.z,opt.domainSize);
    }
}

void DistWorkspace::sortAgents(std::map<int, Container> &agentsForNeighborsMap) {
    // TODO
    /*for (size_t i = 0; i < agents.size(); i++) {
        if (agents[i].position.x > ..... {
          agentsForNeighborsMap[neighbor].push_back(agents[i]);
          agents.erase(agents.begin()+i););
          }
    }*/
}

void DistWorkspace::makeArrayFromContainer(Container &c, std::vector<Real> &array) {
    //TODO
}

void DistWorkspace::makeContainerFromArray(std::vector<Real> &array, Container &c) {
    //TODO
}


void DistWorkspace::save(int stepId) {
    std::ofstream myfile;

    std::stringstream ss;
    ss << "data/boids_" << std::setw(3) << std::setfill('0') << myID << ".xyz";
    myfile.open(ss.str(), stepId==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << agents.size() << std::endl;
    for (Agent a : agents)
        myfile << "B " << a.position << std::endl;
        //TODO: boidType
    myfile.close();

    //TODO zlib?
}

