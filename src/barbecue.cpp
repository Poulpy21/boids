
#include "headers.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <pthread.h>

#include "agent.hpp"
#include "types.hpp"
#include "vec3.hpp"
#include "parser.hpp"
#include "options.hpp"
#include "messenger.hpp"
#include "boundingBox.hpp"
#include "device.hpp"
#include "devicePool.hpp"
#include "CPUMemory.hpp"
#include "GPUMemory.hpp"
#include "computeGrid.hpp"
#include "rand.hpp"
#include "cudaDistributedWorkspace.hpp"

void mpiInit(int argc, char **argv, int &rank, int &size, MPI_Comm &comm, std::string &name, int masterrank);
void broadcastOptions(int argc, char **argv, int rank, int masterrank, const MPI_Comm &comm, Options &opt);
void printStep(unsigned int step, unsigned int nSteps);
void resetDevice(int rank, int masterrank);
void header(int rank, int masterrank);
void footer(int rank, int masterrank);

#define PRINT(s) {MPI_Barrier(comm); if (rank == MASTER_RANK) { std::cout << s; } }
#define JUMP_LINE() { PRINT("\n") }

// Main class for running the parallel cuda flocking sim
int main(int argc, char **argv) {

    using log4cpp::log_console;
    constexpr unsigned int MASTER_RANK = 0u;

    
    // Initialize logs and random
    log4cpp::initLogs();
    Random::init();


    // Initialize MPI
    int rank, size;
    MPI_Comm comm;
    std::string name;
    mpiInit(argc, argv, rank, size, comm, name, MASTER_RANK);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    JUMP_LINE();

    // Print header
    header(rank, MASTER_RANK);

    PRINT(":: Looking for cuda devices...\n\n");

    
    // Some more init 
    DevicePool::init();
    CPUMemory::init();
    GPUMemory::init();

    if(rank == MASTER_RANK) {
        DevicePool::display(std::cout);
        std::cout << std::endl;
        CPUMemory::display(std::cout);
        std::cout << std::endl;
        GPUMemory::display(std::cout);
        std::cout << std::endl;
    }
    MPI_Barrier(comm);


    // Choose GPU and reset it
    unsigned int myGPU = rank % DevicePool::nDevice;
    CHECK_CUDA_ERRORS(cudaSetDevice(myGPU));
    CHECK_CUDA_ERRORS(cudaFree(0));

    log_console->infoStream() << "Rank " << rank << " choosed device " << myGPU << " on computer " << name << " !";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    JUMP_LINE();
    MPI_Barrier(comm);


    // Parse master options and broadcast to slaves
    Options opt;
    broadcastOptions(argc, argv, rank, MASTER_RANK, comm, opt);


    //Build global domain bounding box according to options
    Real ds = opt.domainSize;
    BoundingBox<3u, Real> globalDomain(Vec3<Real>(0,0,0), Vec3<Real>(ds,ds,ds));


    // Create workspace
    PRINT("\n:: Initializing workspace...\n\n");

    CudaDistributedWorkspace workspace(globalDomain, true, opt,
            rank, size, MASTER_RANK, comm, name);


    // Launch simulation
    PRINT("\n:: Computing...\n\n");

    unsigned int nSteps = opt.nSteps;
    for (unsigned int step = 1; step <= nSteps; step++) {
        if(rank == MASTER_RANK)
            printStep(step, nSteps);

        workspace.update();
    }
    PRINT("\n:: Computing done !\n");
    MPI_Barrier(comm);


    // Reset GPUs
    resetDevice(rank, MASTER_RANK);


    // Finalize MPI
    MPI_Barrier(comm);
    MPI_Finalize();


    //Print footer
    footer(rank, MASTER_RANK);

    return EXIT_SUCCESS;
}

void printStep(unsigned int step, unsigned int nSteps) {
    if(nSteps < 10u || step == nSteps) {
        std::cout << "Step " << step << "/" << nSteps << " (" 
            << 100*static_cast<float>(step)/nSteps << "%)" << std::endl;
    }
    else if(step == 1) {
        std::cout << "Step " << step << "/" << nSteps << " (0%)" << std::endl;
    }
    else if((step % (nSteps/10u)) == 0) {
        std::cout << "Step " << step << "/" << nSteps << " (" 
            << (10u*step)/nSteps*10u << "%)" << std::endl;
    }
}

void resetDevice(int rank, int masterrank) {
    if(rank == masterrank) {
        std::cout << std::endl;
        std::cout << ":: Resetting devices..." << std::endl;
    }
    CHECK_CUDA_ERRORS(cudaDeviceReset());
}

void header(int rank, int masterrank) {
    if(rank == masterrank) {
        std::cout << std::endl;
        std::cout
            << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
            << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
            << "::::::::::::::   Flocking boids implementation v." STRINGIFY_MACRO(BOIDS_VERSION) "   :::::::::::::::\n"
            << ":::::   High Performance Computing -- GPU Computing -- 2014-2015   :::::\n"
            << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
            << "::                                                                    ::\n"
            << "::  Authors:   Keck Jean-Baptiste -- Ensimag - M2 MSIAM               ::\n"
            << "::             Zirnhelt Gauthier  -- Ensimag                          ::\n" 
            << "::                                                                    ::\n"
            << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
            << "::  Program is running in " STRINGIFY_MACRO(COMPILE_MODE) " mode !                                ::\n"
            << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
            << "\n";
    }
}

void footer(int rank, int masterrank) {
    if(rank == masterrank) {
        std::cout << "\n:: All done, exiting !" << std::endl;
        std::cout << std::endl;
    }
}

void mpiInit(int argc, char **argv, int &rank, int &size, MPI_Comm &comm, std::string &name, int masterrank) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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

    //null terminate the string
    char _name[MPI_MAX_PROCESSOR_NAME+1];
    int len;
    memset(_name,0,MPI_MAX_PROCESSOR_NAME+1);
    MPI_Get_processor_name(_name, &len);
    memset(_name+len,0,MPI_MAX_PROCESSOR_NAME-len);

    name = std::string(_name);

    MPI_Barrier(comm);

    if(rank == masterrank)
        log_console->infoStream() << "Processes :";

    MPI_Barrier(comm);

    log_console->infoStream() << "\t" << rank << "/" << size - 1 << "\t" << name;

    MPI_Barrier(comm);
}

void broadcastOptions(int argc, char **argv, int rank, int masterrank, const MPI_Comm &comm, Options &opt) {

    double *options;
    if (rank == masterrank) {
        // Create parser
        ArgumentParser parser;

        // Add options to parser
        parser.addOption("agents", 10000);
        parser.addOption("steps", 1);
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
    MPI_Bcast(options, Options::nData, MPI_DOUBLE, masterrank, comm);
    MPI_Barrier(comm);

    //Build global domain from options
    opt = Options(options);
    delete [] options;
    options = nullptr;
}

