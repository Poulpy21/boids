#include "headers.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "distWorkspace.hpp"
#include "messenger.hpp"
#include "agent.hpp"
#include "agentKernel.cu"

DistWorkspace::DistWorkspace(Options options, MPI_Comm comm, int root) : 
    agents(), opt(options), comm(comm), mess(comm), rootID(root)
{
    MPI_Comm_rank(comm, &myID);
    init();
}

void DistWorkspace::init() {
    // Upload options to device
    // TODO

    // Init boids
    srand48(std::time(0));
    //srand48(myID * std::time(0));

    /*
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
    */

}

void DistWorkspace::update() {
    // Update neighborhood info
    // TODO
    // neighbors = ...
   
    Agent meanAgent;
#ifdef CUDA_ENABLED 
    // Upload boids to device
    // TODO convert Container to Vec3<Real>[]
    // memcpy : agents -> d_agents

    // Compute mean boid
    // TODO
    // computeMeanBoidKernel(d_agents, d_meanAgent)
    // memcpy : d_meanAgent -> meanAgent
#else
    //computeMeanAgent(meanAgent);
#endif

    // Get mean boids from the neighborhood and send ours (approximation)
    // mess.exchangeMeanAgents(meanAgent, agents.size(), receivedMeanAgents, receivedMeanAgentsWeight, neighbors);

    // Compute and apply forces to local boids
#ifdef CUDA_ENABLED
    // TODO
    // memcpy : receivedMeanAgents -> d_meanAgents
    // memcpy : receivedMeanAgentsWeight -> d_meanAgentsWeightss
    //applyInternalForcesKernel(d_currentAgents, d_newAgents, d_opt)
    //applyExternalForcesKernel(d_currentAgents, d_newAgents, d_meanAgents, d_meanAgentsWeights, d_opt)
    // memcpy : newAgents -> agents
#else
    // TODO code workspace.cpp + meanAgents
    //applyInternalForces();
    //applyExternalForces(receivedMeanAgents, receivedMeanAgentsWeight);
#endif
    
    // Sort out boids that cross domain boundaries
    static std::map<int, Container> agentsForNeighborsMap;
    agentsForNeighborsMap.clear();
    //sortAgents(agentsForNeighborsMap);
    
    // Exchange boids that cross domain boundaries
    // mess.exchangeAgents(agents, agentsForNeighborsMap, neighbors);

    // Note: if practically there is too many agents to send, we should compute and send agents by batches
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

void DistWorkspace::applyInternalForces() {
    Vec3<Real> s,c,a;

    for(size_t k = 0; k< agents.size(); k++){
        s = agents[k].separation(agents, k, opt.rSeparation);
        c = agents[k].cohesion(agents, k, opt.rCohesion);
        a = agents[k].alignment(agents, k, opt.rAlignment);

        agents[k].direction = opt.wCohesion*c + opt.wAlignment*a + opt.wSeparation*s;
    }

    // Integration in time using euler method
    for(size_t k = 0; k< agents.size(); k++){
        agents[k].velocity += agents[k].direction;

        double speed = agents[k].velocity.norm();
        if (speed > opt.maxVel) {
            agents[k].velocity *= opt.maxVel/speed;
        }
        agents[k].position += opt.dt*agents[k].velocity;

        agents[k].position.x= fmod(agents[k].position.x,opt.domainSize);
        agents[k].position.y= fmod(agents[k].position.y,opt.domainSize);
        agents[k].position.z= fmod(agents[k].position.z,opt.domainSize);
    }
}

void DistWorkspace::applyExternalForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeight) {
    //TODO : create new / modify methods for agents to use weights
    //TODO : factor code with applyInternalForces()
    /*Vec3<Real> s,c,a;

    // Use SIZE_MAX so the computation is done for each meanAgent
    for(size_t k = 0; k< agents.size(); k++){
        s = agents[k].separation(receivedMeanAgents, SIZE_MAX, opt.rSeparation);
        c = agents[k].cohesion(receivedMeanAgents, SIZE_MAX, opt.rCohesion);
        a = agents[k].alignment(receivedMeanAgents, SIZE_MAX, opt.rAlignment);

        agents[k].direction = opt.wCohesion*c + opt.wAlignment*a + opt.wSeparation*s;
    }

    // Integration in time using euler method
    for(size_t k = 0; k< agents.size(); k++){
        agents[k].velocity += agents[k].direction;

        double speed = agents[k].velocity.norm();
        if (speed > opt.maxVel) {
            agents[k].velocity *= opt.maxVel/speed;
        }
        agents[k].position += opt.dt*agents[k].velocity;

        agents[k].position.x= fmod(agents[k].position.x,opt.domainSize);
        agents[k].position.y= fmod(agents[k].position.y,opt.domainSize);
        agents[k].position.z= fmod(agents[k].position.z,opt.domainSize);
    }*/
}
        
void DistWorkspace::sortAgents(std::map<int, Container> &agentsForNeighborsMap) {
    // TODO
    for (size_t i = 0; i < agents.size(); i++) {
        /*if (agents[i].position.x > ..... {
            agentsForNeighborsMap[neighbor].push_back(agents[i]);
            agents.erase(agents.begin()+i););
            }*/
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
        myfile << "B " << a.position << std::endl;
        //TODO: boidType
    myfile.close();

    //TODO zlib?
}

