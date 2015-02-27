
#ifndef DEVICEPOOL_H
#define DEVICEPOOL_H

#include "headers.hpp"
#include "device.hpp"
#include <iostream>
#include <pthread.h>

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
        
        static pthread_mutex_t mtx;
        static bool _init;

};

#endif

#endif /* end of include guard: DEVICEPOOL_H */
