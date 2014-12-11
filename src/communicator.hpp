#ifndef COMMUNICATOR_HPP
#define COMMUNICATOR_HPP

#include <mpi.h>
#include "boid/agent.hpp"
#include "utils/types.hpp"


class Communicator {

    public:

        static inline void exchangeAgents(MPI_Comm comm, Container &agentsToSend, Container &agentsToReceive, int targetRank) {
            MPI_Request req;
            MPI_Status statSend, statRecv;
            int tagSend = 0, tagReceive = 0;
            int recvBufferSize;
            int realsPerAgent = sizeof(Agent)/sizeof(Real);

            MPI_Isend(&agentsToSend[0], agentsToSend.size()*realsPerAgent, MPI_DOUBLE, targetRank, tagSend, comm, &req);
            MPI_Probe(targetRank, tagReceive, comm, &statRecv);
            MPI_Get_count(&statRecv, MPI_DOUBLE, &recvBufferSize);
            agentsToReceive.resize(recvBufferSize/realsPerAgent);
            MPI_Recv(&agentsToReceive[0], recvBufferSize/realsPerAgent, MPI_DOUBLE, targetRank, tagReceive, comm, &statRecv);
            MPI_Wait(&req, &statSend);
        }

    private: 

        static inline int getRankFromDirection(MPI_Comm comm, int localRank, int direction[3]) {
            int currentRank = localRank;

            // Since MPI_Cart_shift can only shift in 1 direction, we're doing one shift per direction
            if (direction[0] != 0)
                MPI_Cart_shift(comm, 0, direction[0], &currentRank, &currentRank);
            if (direction[1] != 0)
                MPI_Cart_shift(comm, 1, direction[1], &currentRank, &currentRank);
            if (direction[2] != 0)
                MPI_Cart_shift(comm, 2, direction[2], &currentRank, &currentRank);
            return currentRank;
        }
};

#endif
