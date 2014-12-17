#ifndef MESSENGER_HPP
#define MESSENGER_HPP

#include "utils/headers.hpp"
#include <vector>
#include "options.hpp"
#include "boid/agent.hpp"
#include "utils/types.hpp"


class Messenger {

    public:

        Messenger(MPI_Comm c) : comm(c) {};

        /*
         * opt : Options to broadcast
         * root : process that is the source of the broacast
         */
        inline void broadcastOptions(Options *opt, int root) {
            MPI_Bcast(opt, sizeof(Options)/sizeof(double), MPI_DOUBLE, 0, comm);
        }

        /*
         * agents : Container to which to append received boids
         * agentsForRanks : Maps Containers of boids to send to ranks
         * sourceRanks : ranks from which to receive boids (typically the keys of agentsForRanks)
         */
        inline void exchangeAgents(Container &agents, 
                std::map<int, Container> &agentsForRanks, 
                std::vector<int> &sourceRanks) 
        {
            sendAgents(agentsForRanks);
            receiveAgents(agents, sourceRanks);
            waitForSendCompletion();
            MPI_Barrier(comm);
        }

        /*
         * receivedMeanAgents : Container that will contain received boids
         * meanAgentToSend : Mean boid to send
         * sourceRanks : ranks with which to exchange mean boids
         */
        inline void exchangeMeanAgents(Container &receivedMeanAgents, 
                Agent &meanAgentToSend, 
                std::vector<int> &sourceRanks) 
        {
            sendMeanAgent(meanAgentToSend, sourceRanks);
            receiveMeanAgents(receivedMeanAgents, sourceRanks);
            waitForSendCompletion();
            MPI_Barrier(comm);
        }

        /*
         * root : Rank that gathers everything
         * localLoad : Local load value
         * loads : Resulting array of all loads, gathered on root and ordered by rank number (root included)
         */
        inline void gatherAgentLoads(int root, int localLoad, std::vector<int> &loads) {
            MPI_Gather(&localLoad, 1, MPI_INT, &loads[0], 1, MPI_INT, root, comm);
        }


    private:

        void sendAgents(std::map<int, Container> &agentsForRanks);
        void sendMeanAgent(Agent &meanAgentToSend, std::vector<int> &sourceRanks); 
        // Note: this method appends the received agents to the container
        void receiveAgents(Container &agents, std::vector<int> &sourceRanks); 
        void receiveMeanAgents(Container &receivedMeanAgents, std::vector<int> &sourceRanks); 
        void waitForSendCompletion();

        std::vector<MPI_Request> pendingRequests;
        static const int realsPerAgent = sizeof(Agent)/sizeof(Real);
        MPI_Comm comm;
};

#endif
