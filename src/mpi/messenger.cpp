#include "messenger.hpp"

void Messenger::sendAgents(std::map<int, Container> &agentsForRanks) {
    MPI_Request req;

    // Send everything immediately to prevent deadlocks 
    for (auto it : agentsForRanks) {
        int targetRank = it.first;
        Container agents = it.second;
        int tag = 0;

        MPI_Isend(&agents[0], agents.size()*realsPerAgent, MPI_DOUBLE, targetRank, tag, comm, &req);
        pendingRequests.push_back(req);
    }
}

void Messenger::sendMeanAgent(Agent &meanAgentToSend, int meanAgentToSendWeight, std::vector<int> &sourceRanks) {
    MPI_Request req1, req2;
    int tag1 = 0, tag2 = 1;

    for (int rank : sourceRanks) {
        MPI_Isend(&meanAgentToSend, realsPerAgent, MPI_DOUBLE, rank, tag1, comm, &req1);
        pendingRequests.push_back(req1);
        MPI_Isend(&meanAgentToSendWeight, 1, MPI_INT, rank, tag2, comm, &req2);
        pendingRequests.push_back(req2);
    }
}

void Messenger::receiveAgents(Container &agents, std::vector<int> &sourceRanks) {
    MPI_Status stat;
    int nAgents = 0;
    int sizes[sourceRanks.size()];

    // Wait until everything arrives
    int i = 0;
    for (int sourceRank : sourceRanks) {
        MPI_Probe(sourceRank, MPI_ANY_TAG, comm, &stat);
        MPI_Get_count(&stat, MPI_DOUBLE, &sizes[i]);
        nAgents += sizes[i] / realsPerAgent;
        i++;
    }

    // Save previous number of Agents
    int pos = agents.size();

    // Make room for copying
    agents.resize(agents.size() + nAgents);

    // Concatenate the Containers
    i = 0;
    for (int sourceRank : sourceRanks) {
        MPI_Recv(&agents[pos], sizes[i], MPI_DOUBLE, sourceRank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        pos += sizes[i] / realsPerAgent;
        i++;
    }
}

void Messenger::receiveMeanAgents(Container &receivedMeanAgents, std::vector<int> &receivedMeanAgentsWeight, std::vector<int> &sourceRanks) { 
    // Clear and reserve space in the containers
    receivedMeanAgents.clear();
    receivedMeanAgents.resize(sourceRanks.size());
    receivedMeanAgentsWeight.clear();
    receivedMeanAgentsWeight.resize(sourceRanks.size());

    int pos = 0;
    for (int sourceRank : sourceRanks) {
        MPI_Recv(&receivedMeanAgents[pos], realsPerAgent, MPI_DOUBLE, sourceRank, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&receivedMeanAgentsWeight[pos], 1, MPI_INT, sourceRank, 1, comm, MPI_STATUS_IGNORE);
        pos++;
    }
}

void Messenger::waitForSendCompletion() {
    for (MPI_Request req : pendingRequests) {    
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    pendingRequests.clear();
}

/*static void exchangeAgents(Container &agentsToSend, Container &agentsToReceive, int targetRank) {
  MPI_Request req;
  MPI_Status statSend, statRecv;
  int tagSend = 0, tagReceive = 0;
  int recvBufferSize;

  MPI_Isend(&agentsToSend[0], agentsToSend.size()*realsPerAgent, MPI_DOUBLE, targetRank, tagSend, comm, &req);
  MPI_Probe(targetRank, tagReceive, comm, &statRecv);
  MPI_Get_count(&statRecv, MPI_DOUBLE, &recvBufferSize);
  agentsToReceive.reserve(recvBufferSize/realsPerAgent);
  MPI_Recv(&agentsToReceive[0], recvBufferSize, MPI_DOUBLE, targetRank, tagReceive, comm, &statRecv);
  MPI_Wait(&req, &statSend);
  }

  static int getRankFromDirection(int localRank, int direction[3]) {
  int currentRank = localRank;

// Since MPI_Cart_shift can only shift in 1 direction, we're doing one shift per direction
if (direction[0] != 0)
MPI_Cart_shift(comm, 0, direction[0], &currentRank, &currentRank);
if (direction[1] != 0)
MPI_Cart_shift(comm, 1, direction[1], &currentRank, &currentRank);
if (direction[2] != 0)
MPI_Cart_shift(comm, 2, direction[2], &currentRank, &currentRank);
return currentRank;
}*/


