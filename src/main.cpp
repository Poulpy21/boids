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

    parser.addOption("save", true);

    // Parse command line arguments
    parser.setOptions(argc, argv);
    Options opt(parser);
    if (rank == 0) {
        log_console->infoStream() << opt;
        //TODO create tree
    }
    //TODO broadcast map<domain, procID>

    DistWorkspace workspace(opt, comm);
    
    // Store initial positions
    if (parser("save").asBool())
        workspace.save(0);

    // Launch simulation
    for (size_t step = 1; step <= opt.nSteps; step++) {
        workspace.update();
        if (parser("save").asBool())
            workspace.save(step);
        //TODO load balancing
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
