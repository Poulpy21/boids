#include "headers.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>

#include "agent.hpp"
#include "types.hpp"
#include "vec3.hpp"
#include "parser.hpp"
#include "options.hpp"
#include "distWorkspace.hpp"
#include "messenger.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();

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
    Options opt(parser);
    if (rank == 0)
        log_console->infoStream() << opt;

    DistWorkspace workspace(opt, comm);

    // Launch simulation
    for (size_t step = 1; step <= opt.nSteps; step++) {
        workspace.update();
        workspace.save(step);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
