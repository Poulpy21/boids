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
#include "boidGrid.hpp"

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

    // Add options to parser
    ArgumentParser parser;
    //parser.addOption("agents", 100000000);
    parser.addOption("agents", 1000);
    parser.addOption("steps", 1000);
    parser.addOption("wc", 12);
    parser.addOption("wa", 15);
    parser.addOption("ws", 35);

    parser.addOption("rc", 0.11);
    parser.addOption("ra", 0.15);
    parser.addOption("rs", 0.1);
    parser.addOption("dt", 0.05);
    parser.addOption("mv", 2.0);

    parser.addOption("save", false);

    parser.setOptions(argc, argv);
   
    // Parse options
    Options options(parser);
    
    //create simulation domain
    Real ds = options.domainSize;
    BoundingBox<3u, Real> domain(Vec3<Real>(0,0,0), Vec3<Real>(ds,ds,ds));
    
    // Set up bounds for boid initialization
    // format is XYZ  VX VY VZ  AX AY AZ
    Real minValues[9] = {0, 0, 0,  0,0,0,  0,0,0};
    Real maxValues[9] = {ds,ds,ds, 0,0,0,  0,0,0};
    InitBounds<Real> initBounds(minValues, maxValues);
    

    //Create data structure
    Real minRadius = std::min<Real>(options.rCohesion, std::min<Real>(options.rAlignment, options.rSeparation));

    BoidGrid<Real> *grid = new BoidGrid<Real>(domain, minRadius);
    std::cout << *grid << std::endl;
   
    
    // Create workspace and simulate
    CudaWorkspace workspace(options, initBounds, grid);

    for (unsigned long int step = 1; step <= options.nSteps; step++) {
        workspace.update();
    }

    delete grid;
}
