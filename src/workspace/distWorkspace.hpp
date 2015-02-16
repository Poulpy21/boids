#ifndef DIST_WORKSPACE_HPP
#define DIST_WORKSPACE_HPP

#include "headers.hpp"
#include "options.hpp"
#include "vector.hpp"
#include "messenger.hpp"
#include "agentData.hpp"

class DistWorkspace {

    public:
        DistWorkspace(Options options, MPI_Comm comm, int root = 0);

        void update();
        void save(int stepId);

    private:
        void init();
        void computeMeanAgent(Agent &meanAgent);
        void computeAndApplyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights);
        void sortAgents(std::map<int, Container> &agentsForNeighborsMap);
        void makeAgentDataFromContainer(Container &c, AgentData &d);
        void makeContainerFromAgentData(AgentData &d, Container &c);

        Container agents;
        Options opt;
        MPI_Comm comm;
        Messenger mess;
        int rootID, myID;
        // TODO: affectation domaine hypercube
        // TODO: structure pour stockage voisinage

#ifdef CUDA_ENABLED
        struct Options* d_opt;
        Real *d_agents, *d_meanAgents;
        Vector *d_meanAgent;
        int *d_meanAgentsWeights;
#endif

};

#endif
