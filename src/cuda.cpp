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

    compute(argc, argv);
    
    cudaDeviceReset();

    return EXIT_SUCCESS;
}



void compute(int argc, char **argv) {

    //InitBounds
    BoundingBox<3u,Real> position(Vec3<Real>(0,0,0), Vec3<Real>(10,10,10));
    InitBounds<Real> initBounds(position, position, position);
    
    // Add options to parser
    ArgumentParser parser;
    parser.addOption("agents", 10000);
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
