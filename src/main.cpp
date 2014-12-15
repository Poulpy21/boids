#include <mpi.h>

// Agent (particle model)
#include <cstdio>
#include <iostream>
#include <cmath>

#include "utils/headers.hpp"
#include "agent.hpp"
#include "types.hpp"
#include "parser.hpp"
#include "options.hpp"
#include "workspace.hpp"
#include "messenger.hpp"

#include "rootNode.hpp"
#include "treeNode.hpp"
#include "leafNode.hpp"
#include "hypercubeTree.hpp"
#include "abstractContainer.hpp"
#include "abstractContainerFactory.hpp"
#include "simpleContainerFactory.hpp"
#include "vecBool.hpp"
#include "boundingBox.hpp"
#include "hypercube.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();
    
    Vec3<float> v0(0.0f,0.0f,0.0f);
    Vec3<float> v1(1.0f,1.0f,1.0f);
    HyperCube<3u,float> cube(v0,v1);

    std::cout << cube << std::endl;

    return EXIT_SUCCESS;

    MPI_Init(&argc,&argv); 
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm;
    int dimensions[3] = {0,0,0};
    MPI_Dims_create(size, 3, dimensions);
    if (rank == 0) {
        log_console->infoStream() << "[MPI] Cartesian grid dimensions : " << dimensions[0] << "/" << dimensions[1] << "/" << dimensions[2]; 
    }
    int wrap[3] = {1,1,1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, wrap, reorder, &comm);
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

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);

    std::cout << "Number of tasks=" << size << " My rank=" << rank << " My name="<< name 
              << " My coords=" << coords[0] << "/" << coords[1] << "/" << coords[2] << "." << std::endl;
   
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

        // Parse command line arguments
        parser.setOptions(argc, argv);

        opt = Options(parser);
    }
   
    Messenger mess;
    mess.broadcastOptions(comm, &opt, 0);
    std::cout << "Options for rank " << rank << " : " << opt << std::endl; 

    //test
    Container localBoids, receivedBoids;
    Agent meanBoid(Vector(rank+1,0,0), Vector(), Vector());
    Agent boid(Vector(0,rank+1,0), Vector(), Vector());
    Container tmpContainer = {boid};
    std::vector<int> otherRanks;
    std::map<int, Container> boidsToSend;
    if (rank == 0) {
        otherRanks.push_back(1);
        boidsToSend.emplace(1, tmpContainer);
    }
    if (rank == 1) { 
        otherRanks.push_back(0);
        boidsToSend.emplace(0, tmpContainer);
    }
    if (rank < 2) {
        mess.exchangeMeanAgents(comm, receivedBoids, meanBoid, otherRanks);
        std::cout << "RANK: " << rank  << " MEANBOIDS: " << receivedBoids.size() << " pos="; 
        for (Agent boid : receivedBoids)
            std::cout << boid.position << "/";
        mess.exchangeAgents(comm, localBoids, boidsToSend, otherRanks);
        std::cout << "\tRECEIVEDBOIDS :" << localBoids.size() << " pos=";
        for (Agent boid : localBoids) {
            std::cout << boid.position << "/";
        }
        std::cout << std::endl;
    }

    //Workspace workspace(opt, ....)

    /*
    // Create workspace
    Workspace workspace(parser);

    // Launch simulation
    int nSteps = parser("steps").asInt();
    workspace.simulate(nSteps);
    */

    MPI_Finalize();

    return EXIT_SUCCESS;
}
