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

void init();
void header();
void initDevices();
void compute(int argc, char **argv);
void resetDevices();
void footer();

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    init();
    header();

    std::cout << ":: Looking for cuda device..." << std::endl << std::endl;
    DevicePool::init();
    DevicePool::display(std::cout);
    std::cout << std::endl;
    
    initDevices();

    CPUMemory::init();
    CPUMemory::display(std::cout);
    std::cout << std::endl;
    
    GPUMemory::init();
    GPUMemory::display(std::cout);
    std::cout << std::endl;

    compute(argc, argv);
    resetDevices();

    footer();
  
    return EXIT_SUCCESS;
}

void compute(int argc, char **argv) {

    // Add options to parser
    ArgumentParser parser;
    parser.addOption("agents", 1000000);
    parser.addOption("steps", 100);
    parser.addOption("wc", 12);
    parser.addOption("wa", 15);
    parser.addOption("ws", 35);

    parser.addOption("rc", 0.11);
    parser.addOption("ra", 0.15);
    parser.addOption("rs", 0.01);
    parser.addOption("dt", 0.05);
    parser.addOption("mv", 2.0);
    
    parser.addOption("size", 100);

    parser.addOption("save", false);

    parser.setOptions(argc, argv);
   
    // Parse options
    Options options(parser);
    
    //create simulation domain
    Real ds = options.domainSize;
    BoundingBox<3u, Real> domain(Vec3<Real>(0,0,0), Vec3<Real>(ds,ds,ds));
    Globals::setGlobalDomainSize(Vec3<unsigned int>(1,1,1));
    
    // Set up bounds for boid initialization
    // format is XYZ  VX VY VZ  AX AY AZ
    Real minValues[9] = {0, 0, 0,  0,0,0,  0,0,0};
    Real maxValues[9] = {ds,ds,ds, 0,0,0,  0,0,0};
    InitBounds<Real> initBounds(minValues, maxValues);
    

    //Create data structure
    Real maxRadius = std::max<Real>(options.rCohesion, std::max<Real>(options.rAlignment, options.rSeparation));

    BoidGrid<Real> *grid = new BoidGrid<Real>(0u, domain, domain, true, maxRadius);
    std::cout << ":: Data structure used ::" <<std::endl;
    std::cout << *grid << std::endl;
    std::cout << std::endl;
   
    
    // Create workspace and simulate
    std::cout << "\n:: Initializing workspace...\n\n";
    CudaWorkspace workspace(options, initBounds, grid);

    std::cout << "\n:: Computing...\n\n";
    unsigned int nSteps = options.nSteps;
    for (unsigned long int step = 1; step <= nSteps; step++) {
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
        workspace.update();
    }
    std::cout << "\n:: Computing done !\n\n";

    delete grid;
}

void initDevices() {
    std::cout << ":: Initializing devices..." << std::endl;
    for (unsigned int i = 0; i < DevicePool::nDevice; i++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(i));
        CHECK_CUDA_ERRORS(cudaFree(0));
    }
    std::cout << std::endl;
}

void resetDevices() {
    std::cout << std::endl;
    std::cout << ":: Resetting devices..." << std::endl;
    for (unsigned int i = 0; i < DevicePool::nDevice; i++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(i));
        CHECK_CUDA_ERRORS(cudaDeviceReset());
    }
    std::cout << std::endl;
}


void header() {
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

void footer() {
    std::cout << ":: All done, exiting !" << std::endl;
    std::cout << std::endl;
}

void init() {
    using log4cpp::log_console;
    log4cpp::initLogs();
    Random::init();
}

