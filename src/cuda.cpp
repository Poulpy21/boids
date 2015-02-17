#include "headers.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>

#include "device.hpp"
#include "devicePool.hpp"
#include "CPUMemory.hpp"
#include "GPUMemory.hpp"
#include "cudaWorkspace.hpp"
#include "rand.hpp"
#include "initBounds.hpp"

void compute(int argc, char **argv);
void initDevices();
void resetDevices();


// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();

    Random::init();

    DevicePool::init();
    DevicePool::display(std::cout);

    CPUMemory::init();
    CPUMemory::display(std::cout);
    
    GPUMemory::init();
    GPUMemory::display(std::cout);

    initDevices();
    compute(argc, argv);
    resetDevices();
  
    return EXIT_SUCCESS;
}

void initDevices() {
    for (unsigned int i = 0; i < DevicePool::nDevice; i++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(i));
        CHECK_CUDA_ERRORS(cudaFree(0));
    }
}

void resetDevices() {
    for (unsigned int i = 0; i < DevicePool::nDevice; i++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(i));
        CHECK_CUDA_ERRORS(cudaDeviceReset());
    }
}

void compute(int argc, char **argv) {

    //InitBounds
    Real minValues[9] = {0,0,0, 0,0,0, 0,0,0};
    Real maxValues[9] = {10,10,10,0,0,0,0,0,0};
    InitBounds<Real> initBounds(minValues, maxValues);
    
    // Add options to parser
    ArgumentParser parser;
    parser.addOption("agents", 100000000);
    parser.addOption("steps", 1000);
    parser.addOption("wc", 12);
    parser.addOption("wa", 15);
    parser.addOption("ws", 35);

    parser.addOption("rc", 0.11);
    parser.addOption("ra", 0.15);
    parser.addOption("rs", 0.01);
    parser.addOption("dt", 0.05);
    parser.addOption("mv", 2.0);

    parser.addOption("save", false);

    parser.setOptions(argc, argv);
   
    Options options(parser);
    CudaWorkspace workspace(options, initBounds);

    for (unsigned long int step = 1; step <= options.nSteps; step++) {
        workspace.update();
    }
}
