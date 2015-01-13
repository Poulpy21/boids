#ifndef DIST_WORKSPACE_HPP
#define DIST_WORKSPACE_HPP

#include "utils/headers.hpp"
#include "options.hpp"
#include "utils/types.hpp"
#include "messenger.hpp"

class DistWorkspace {

    public:
        DistWorkspace(Options options, MPI_Comm comm, int root = 0);

        void update();
        void save(int stepId);

    private:
        void init();
        void computeMeanAgent(Agent &meanAgent);
        void applyInternalForces();
        void applyExternalForces(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeight);
        void sortAgents(std::map<int, Container> &agentsForNeighborsMap);

        Container agents;
        Options opt;
        MPI_Comm comm;
        Messenger mess;
        int rootID, myID;
        // TODO: affectation domaine hypercube
        // TODO: structure pour stockage voisinage

};

#endif
