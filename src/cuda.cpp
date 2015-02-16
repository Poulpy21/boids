#include "headers.hpp"

#include <cstdio>
#include <iostream>
#include <cmath>

#include "device.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();

    int nDevices;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&nDevices));

    std::vector<Device> devices;
    for (int i = 0; i < nDevices; i++) {
        devices.push_back(Device(i));
        std::cout << devices[i] << std::endl;
    }

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
