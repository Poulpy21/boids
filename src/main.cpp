
// Agent (particle model)
#include <cstdio>

#include "headers.hpp"
#include "agent.hpp"
#include "types.hpp"
#include "parser.hpp"
#include "workspace.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    //using log4cpp::log_console;
    //log4cpp::initLogs();

    //static const char* MPI_levels[4] = {"MPI_THREAD_SINGLE", "MPI_THREAD_FUNNELED", "MPI_THREAD_SERIALIZED", "MPI_THREAD_MULTIPLE"}; 
    
    //int claimed = MPI_THREAD_MULTIPLE;
    //int provided = -1;
    //MPI_Init_thread(&argc, &argv, claimed, &provided);
    
    //int myid, numprocs;

    //MPI_Init(&argc,&argv);
    //MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    //MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    /* print out my rank and this run's PE size*/
    //printf("Hello from %d\n",myid);
    //printf("Numprocs is %d\n",numprocs);

    //MPI_Finalize();

    //return EXIT_SUCCESS;

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

    // Create workspace
    Workspace workspace(parser);

    // Launch simulation
    int nSteps = parser("steps").asInt();
    workspace.simulate(nSteps);

    return 0;
}
