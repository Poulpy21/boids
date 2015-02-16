#include "headers.hpp"
#include "distWorkspace.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include "agent.hpp"

extern void computeForcesKernel(Real *boidData, Real *meanBoidData, int *meanBoidWeights, 
                                const int nBoids, const int nMeanBoids, const struct Options *opt);
extern void applyForcesKernel(Real *boidData, const int nBoids, const struct Options *opt);
extern void computeMeanBoidKernel(Real *boidData, const int nBoids, Vector *meanBoid);

DistWorkspace::DistWorkspace(Options options, MPI_Comm comm, int root) : 
    agents(), opt(options), comm(comm), mess(comm), rootID(root)
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
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgent, sizeof(Vector)));
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
    AgentData h_agents(agents.size());
    makeAgentDataFromContainer(agents, h_agents);
    
    // Malloc and upload boids to device
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_agents, agents.size()*9*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_agents, h_agents.data, agents.size()*9*sizeof(Real), cudaMemcpyHostToDevice));

    // Compute mean boid FIXME
    /*computeMeanBoidKernel(d_agents, agents.size(), d_meanAgent);
    
    CHECK_CUDA_ERRORS(cudaMemcpy(&meanAgent, d_meanAgent, sizeof(Vector), cudaMemcpyDeviceToHost));
#else*/
#endif //FIXME
    computeMeanAgent(meanAgent);
//#endif

    // Get mean boids from the neighborhood and send ours (approximation)
    Container receivedMeanAgents;
    std::vector<int> receivedMeanAgentsWeight;
    mess.exchangeMeanAgents(meanAgent, agents.size(), receivedMeanAgents, receivedMeanAgentsWeight, neighbors);

    // Compute and apply forces to local boids
#ifdef CUDA_ENABLED 
    AgentData h_meanAgents(receivedMeanAgents.size());
    makeAgentDataFromContainer(receivedMeanAgents, h_meanAgents);

    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgents, agents.size()*9*sizeof(Real)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_meanAgents, h_meanAgents.data, receivedMeanAgents.size()*9*sizeof(Real), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMalloc((void **) &d_meanAgentsWeights, receivedMeanAgentsWeight.size()*sizeof(int)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_meanAgentsWeights, &receivedMeanAgentsWeight, receivedMeanAgentsWeight.size()*sizeof(int), cudaMemcpyHostToDevice));
    
    computeForcesKernel(d_agents, d_meanAgents, d_meanAgentsWeights, agents.size(), receivedMeanAgents.size(), d_opt);
    applyForcesKernel(d_agents, agents.size(), d_opt);
    
    CHECK_CUDA_ERRORS(cudaMemcpy(h_agents.data, d_agents, agents.size()*9*sizeof(Real), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERRORS(cudaFree(d_agents));
    CHECK_CUDA_ERRORS(cudaFree(d_meanAgents));

    makeContainerFromAgentData(h_agents, agents);
#else
    computeAndApplyForces(receivedMeanAgents, receivedMeanAgentsWeight);
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

void DistWorkspace::computeAndApplyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights) {
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

        Real modX = fmod(agents[k].position.x, opt.domainSize);
        Real modY = fmod(agents[k].position.y, opt.domainSize);
        Real modZ = fmod(agents[k].position.z, opt.domainSize);
        agents[k].position.x = modX > 0 ? modX : modX + opt.domainSize;
        agents[k].position.y = modY > 0 ? modY : modY + opt.domainSize;
        agents[k].position.z = modZ > 0 ? modZ : modZ + opt.domainSize;
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

void DistWorkspace::makeAgentDataFromContainer(Container &c, AgentData &d) {
    for (size_t i = 0; i < c.size(); i++) {
        Vec3<Real> pos = c[i].position;
        Vec3<Real> vel = c[i].velocity;
        Vec3<Real> dir = c[i].direction;
        d.setPosition (i, pos.x, pos.y, pos.z);
        d.setVelocity (i, vel.x, vel.y, vel.z);
        d.setDirection(i, dir.x, dir.y, dir.z);
    }
}

void DistWorkspace::makeContainerFromAgentData(AgentData &d, Container &c) {
    for (int i = 0; i < d.nAgents; i++) {
        Vector pos = d.getPosition(i);
        Vector vel = d.getPosition(i);
        Vector dir = d.getPosition(i);
        c[i].position.setValue(pos.x,pos.y,pos.z);
        c[i].velocity.setValue(vel.x,vel.y,vel.z);
        c[i].direction.setValue(dir.x,dir.y,dir.z);
    }
}


void DistWorkspace::save(int stepId) {
    std::ofstream myfile;

    std::stringstream ss;
    ss << "data/boids_" << std::setw(3) << std::setfill('0') << myID << ".xyz";
    myfile.open(ss.str(), stepId==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << agents.size() << std::endl;
    for (Agent a : agents)
        //myfile << "B " << a.position << std::endl;
    myfile.close();
}

