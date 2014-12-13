#ifndef MESSENGER_HPP
#define MESSENGER_HPP

#include <mpi.h>
#include <vector>
#include "options.hpp"
#include "boid/agent.hpp"
#include "utils/types.hpp"


class Messenger {

    public:

        /*
         * comm : Current communicator
         * opt : Options to broadcast
         * root : process that is the source of the broacast
         */
        inline void broadcastOptions(MPI_Comm comm, Options *opt, int root) {
            MPI_Bcast(opt, sizeof(Options)/sizeof(double), MPI_DOUBLE, 0, comm);
        }

        /*
         * comm : Current communicator
         * agents : Container to which to append received boids
         * agentsForRanks : Maps Containers of boids to send to ranks
         * sourceRanks : ranks from which to receive boids (typically the keys of agentsForRanks)
         */
        inline void exchangeAgents(MPI_Comm comm, Container &agents, 
                std::map<int, Container> &agentsForRanks, 
                std::vector<int> &sourceRanks) 
        {
            sendAgents(comm, agentsForRanks);
            receiveAgents(comm, agents, sourceRanks);
            waitForSendCompletion();
            MPI_Barrier(comm);
        }

        /*
         * comm : Current communicator
         * receivedMeanBoids : Container that will contain received boids
         * meanBoidToSend : Mean boid to send
         * sourceRanks : ranks with which to exchange mean boids
         */
        inline void exchangeMeanBoids(MPI_Comm comm, Container &receivedMeanBoids, 
                Agent &meanBoidToSend, 
                std::vector<int> &sourceRanks) 
        {
            sendMeanBoid(comm, meanBoidToSend, sourceRanks);
            receiveMeanBoids(comm, receivedMeanBoids, sourceRanks);
            waitForSendCompletion();
            MPI_Barrier(comm);
        }

    private:

        void sendAgents(MPI_Comm comm, std::map<int, Container> &agentsForRanks);
        void sendMeanBoid(MPI_Comm comm, Agent &meanBoidToSend, std::vector<int> &sourceRanks); 
        // Note: this method appends the received agents to the container
        void receiveAgents(MPI_Comm comm, Container &agents, std::vector<int> &sourceRanks); 
        void receiveMeanBoids(MPI_Comm comm, Container &receivedMeanBoids, std::vector<int> &sourceRanks); 
        void waitForSendCompletion();

        std::vector<MPI_Request> pendingRequests;
        static const int realsPerAgent = sizeof(Agent)/sizeof(Real);
};

#endif
