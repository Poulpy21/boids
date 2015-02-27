
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
#include "boundingBox.hpp"
#include "device.hpp"
#include "devicePool.hpp"
#include "CPUMemory.hpp"
#include "GPUMemory.hpp"
#include <pthread.h>

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {
    
    constexpr unsigned int MASTER_RANK = 0u;

    using log4cpp::log_console;
    log4cpp::initLogs();

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided == MPI_THREAD_MULTIPLE) {
        log_console->infoStream() << "The MPI library has full thread support !";
    }
    else if(provided == MPI_THREAD_SERIALIZED) {
        log_console->warnStream() << "The MPI library does not have full thread support !";
    }
    else {
        log_console->errorStream() << "The MPI library does not have thread support !";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int rank, size;
    MPI_Comm comm;

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //null terminate the string
    char name[MPI_MAX_PROCESSOR_NAME+1];
    int len;
    memset(name,0,MPI_MAX_PROCESSOR_NAME+1);
    MPI_Get_processor_name(name, &len);
    memset(name+len,0,MPI_MAX_PROCESSOR_NAME-len);

    if(rank == MASTER_RANK)
        log_console->infoStream() << "Processes :";
    MPI_Barrier(comm);
    log_console->infoStream() << "\t" << rank << "/" << size - 1<< "\t"<< name;
    MPI_Barrier(comm);
    if(rank == MASTER_RANK)
        log_console->infoStream() << "\n";
    MPI_Barrier(comm);
    
    
    if(rank == MASTER_RANK)
        std::cout << ":: Looking for cuda device..." << std::endl << std::endl;

    // Thread safe init
    DevicePool::init();
    CPUMemory::init();
    GPUMemory::init();

    // Choose GPU and reset
    CHECK_CUDA_ERRORS(cudaSetDevice(rank % DevicePool::nDevice));
    CHECK_CUDA_ERRORS(cudaFree(0));

    log_console->infoStream() << "Rank " << rank << " choosed device " << rank % DevicePool::nDevice;

    double *options;
    if (rank == MASTER_RANK) {
        // Create parser
        ArgumentParser parser;

        // Add options to parser
        parser.addOption("agents", 10000);
        parser.addOption("steps", 100);
        parser.addOption("wc", 12);
        parser.addOption("wa", 15);
        parser.addOption("ws", 35);

        parser.addOption("rc", 0.11);
        parser.addOption("ra", 0.15);
        parser.addOption("rs", 0.01);
        parser.addOption("dt", 0.05);
        parser.addOption("mv", 2.0);

        parser.addOption("size", 10);
        parser.addOption("save", false);

        // Parse command line arguments
        parser.setOptions(argc, argv);
       
        // Broadcast options to other threads
        Options opt(parser);

        options = opt.serialize();
        log_console->infoStream() << opt;
    }
    else {
        options = new double[Options::nData];
    }

    //Broadcast options
    MPI_Bcast(options, Options::nData, MPI_DOUBLE, MASTER_RANK, comm);
    MPI_Barrier(comm);

    //Build global domain from options
    Options opt(options);
    Real ds = opt.domainSize;
    BoundingBox<3u, Real> globalDomain(Vec3<Real>(0,0,0), Vec3<Real>(ds,ds,ds));

    
    MPI_Barrier(comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
