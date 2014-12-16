#ifndef DIST_WORKSPACE_HPP
#define DIST_WORKSPACE_HPP

#include "utils/headers.hpp"
#include "options.hpp"
#include "utils/types.hpp"
#include "messenger.hpp"

class DistWorkspace {

    public:
        DistWorkspace(MPI_Comm comm, int root = 0);

        void update();
        void save(int stepId);

    private:
        void init();

        Container agents;
        Options opt;
        MPI_Comm comm;
        Messenger mess;
        int rootID, myID;
        // TODO: affectation domaine hypercube

};

#endif
