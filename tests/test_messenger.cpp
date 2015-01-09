#include "headers.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>

#include "agent.hpp"
#include "types.hpp"
#include "vec3.hpp"
#include "parser.hpp"
#include "options.hpp"
#include "messenger.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    MPI_Init(&argc,&argv); 
    
    int rank, size;
    MPI_Comm comm;

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get the name of this processor (usually the hostname).  We call                                                      
    // memset to ensure the string is null-terminated.  Not all MPI                                                        
    // implementations null-terminate the processor name since the MPI                                                     
    // standard specifies that the name is *not* supposed to be returned                                                   
    // null-terminated.                                                                                                    
    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    memset(name,0,MPI_MAX_PROCESSOR_NAME);
    MPI_Get_processor_name(name, &len);
    memset(name+len,0,MPI_MAX_PROCESSOR_NAME-len);

    std::cout << " My rank=" << rank << " My name="<< name << std::endl;
   
    MPI_Barrier(comm);

    Options opt;
    if (rank == 0) {

        // Create parser
        ArgumentParser parser;

        // Add options to parser
        parser.addOption("agents", 640);
        parser.addOption("steps", 500);
        parser.addOption("wc", 12);
        parser.addOption("wa", 15);
        parser.addOption("ws", 35);

        parser.addOption("rc", 0.11);
        parser.addOption("ra", 0.15);
        parser.addOption("rs", 0.01);
        parser.addOption("dt", 0.05);
        parser.addOption("mv", 2.0);

        // Parse command line arguments
        parser.setOptions(argc, argv);

        opt = Options(parser);
    }
   
    Messenger mess(comm);
    mess.broadcastOptions(&opt, 0);
    std::cout << "Options for rank " << rank << " : " << opt << std::endl; 

    // exchangeAgents() and exchangeMeanAgents()
    Container localBoids, receivedBoids;
    Agent meanBoid(Vec3<Real>(rank+1,0,0), Vec3<Real>(), Vec3<Real>());
    Agent boid(Vec3<Real>(0,rank+1,0), Vec3<Real>(), Vec3<Real>());
    Container tmpContainer = {boid};
    std::vector<int> otherRanks;
    std::map<int, Container> boidsToSend;
    std::vector<int> receivedBoidsWeight;
    if (rank == 0) {
        otherRanks.push_back(1);
        boidsToSend.emplace(1, tmpContainer);
    }
    if (rank == 1) { 
        otherRanks.push_back(0);
        boidsToSend.emplace(0, tmpContainer);
    }
    if (rank < 2) {
        mess.exchangeMeanAgents(meanBoid, rank, receivedBoids, receivedBoidsWeight, otherRanks);
        std::cout << "RANK: " << rank  << " MEANBOIDS: " << receivedBoids.size() << " pos="; 
        for (Agent boid : receivedBoids)
            std::cout << boid.position << "/";
        std::cout << "\tWEIGHTS : " << receivedBoidsWeight.size() << " values=";
        for (int w : receivedBoidsWeight)
            std::cout << w << "/";
        mess.exchangeAgents(localBoids, boidsToSend, otherRanks);
        std::cout << "\tRECEIVEDBOIDS :" << localBoids.size() << " pos=";
        for (Agent boid : localBoids) {
            std::cout << boid.position << "/";
        }
        std::cout << std::endl;
    }
    // gatherAgentLoads()
    std::vector<int> loads(size);
    if (rank == 1)
        localBoids.push_back(Agent(Vec3<Real>(1337,1337,1337), Vec3<Real>(), Vec3<Real>()));
    mess.gatherAgentLoads(0, localBoids.size(), loads);
    if (rank == 0) {
        std::cout << std::endl << "LOADS: ";
        for (int load : loads)
            std::cout  << load << "//";  
        std::cout << std::endl;
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
