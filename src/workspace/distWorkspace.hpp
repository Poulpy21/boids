#ifndef DIST_WORKSPACE_HPP
#define DIST_WORKSPACE_HPP

#include "utils/headers.hpp"
#include "options.hpp"
#include "utils/types.hpp"
#include "messenger.hpp"

class DistWorkspace {

    public:
        DistWorkspace(size_t localAgentsCount, MPI_Comm comm, int root = 0);

        void update();
        void save(int stepId);

    private:
        void init(size_t localAgentsCount);

        Container agents;
        Options opt;
        MPI_Comm comm;
        Messenger mess;
        int rootID, myID;
        // TODO: affectation domaine hypercube
        // TODO: structure pour stockage voisinage

};

#endif
