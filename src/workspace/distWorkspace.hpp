#ifndef DIST_WORKSPACE_HPP
#define DIST_WORKSPACE_HPP

#include "utils/headers.hpp"
#include "options.hpp"
#include "utils/types.hpp"
#include "vector.hpp"
#include "messenger.hpp"

class DistWorkspace {

    public:
        DistWorkspace(Options options, MPI_Comm comm, int root = 0);

        void update();
        void save(int stepId);

    private:
        void init();
        void computeMeanAgent(Agent &meanAgent);
        void applyForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeights);
        void sortAgents(std::map<int, Container> &agentsForNeighborsMap);
        void makeArrayFromContainer(Container &c, std::vector<Real> &array);
        void makeContainerFromArray(std::vector<Real> &array, Container &c);

        Container agents;
        Options opt;
        MPI_Comm comm;
        Messenger mess;
        int rootID, myID;
        // TODO: affectation domaine hypercube
        // TODO: structure pour stockage voisinage
#ifdef CUDA_ENABLED
        struct Options* d_opt;
        std::vector<Real> h_agents, h_meanAgents;
        Real *d_agents1, *d_agents2, *d_meanAgents;
        Vector *d_meanAgent;
        int *d_meanAgentsWeights;
#endif

};

#endif
