
#ifndef DEVICEPOOL_H
#define DEVICEPOOL_H

#include "headers.hpp"
#include "device.hpp"
#include <iostream>

#ifdef CUDA_ENABLED

class DevicePool {

    public:
        ~DevicePool();

        static void init();
        static void display(std::ostream &os);
        static std::string toString();

        static unsigned int nDevice;
        static unsigned int nCores;
        static std::vector<Device> devices;

    private: 
        DevicePool();
        

};

#endif

#endif /* end of include guard: DEVICEPOOL_H */
