
#ifndef DEVICE_H
#define DEVICE_H

#include "headers.hpp"
#include "vec2.hpp"
#include "vec3.hpp"

#ifdef CUDA_ENABLED

class Device {

    public:
        Device(unsigned int deviceId); 
        ~Device();

        std::string toString() const;

    public:
        std::string name;
        unsigned int deviceId;
        unsigned int pciBusID;
        unsigned int pciDeviceID;
        unsigned int pciDomainID;
        unsigned int clockRate;
        unsigned int major;
        unsigned int minor;

        unsigned int multiProcessorCount;
        unsigned int maxThreadsPerMultiProcessor;
        unsigned int coresPerSM;
        unsigned int coresCount;
        Vec3<unsigned int> maxThreadsDim;
        Vec3<unsigned int> maxGridSize;

        size_t totalGlobalMem;
        size_t totalConstMem;
        size_t memPitch;
        size_t sharedMemPerBlock;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        size_t surfaceAlignment;

        unsigned int l2CacheSize;
        unsigned int memoryClockRate;
        unsigned int memoryBusWidth;
        unsigned int maxThreadsPerBlock;
        unsigned int regsPerBlock;
        unsigned int warpSize;
        unsigned int computeMode;

        bool ECCEnabled;
        bool integrated;
        bool unifiedAddressing;
        bool deviceOverlap;
        bool kernelExecTimeoutEnabled;
        bool canMapHostMemory;
        bool concurrentKernels;
        bool asyncEngineCount;
        bool tccDriver;

        unsigned int maxTexture1D;
        unsigned int maxTexture1DLinear;
        unsigned int maxTextureCubemap;
        unsigned int maxSurface1D;
        unsigned int maxSurfaceCubemap;

        Vec2<unsigned int> maxTexture1DLayered;
        Vec2<unsigned int> maxTextureCubemapLayered;
        Vec2<unsigned int> maxTexture2D;
        Vec2<unsigned int> maxTexture2DGather;
        Vec2<unsigned int> maxSurface1DLayered;
        Vec2<unsigned int> maxSurfaceCubemapLayered;
        Vec2<unsigned int> maxSurface2D;

        Vec3<unsigned int> maxTexture2DLinear; 
        Vec3<unsigned int> maxSurface2DLayered; 
        Vec3<unsigned int> maxTexture2DLayered;
        Vec3<unsigned int> maxTexture3D;
        Vec3<unsigned int> maxTexture3DLinear;
        Vec3<unsigned int> maxTexture3DLayered;
        Vec3<unsigned int> maxSurface3D;
        Vec3<unsigned int> maxSurface3DLayered;

};

std::ostream& operator<<(std::ostream &os, const Device &device);

#endif /* end of cuda enabled */

#endif /* end of include guard: DEVICE_H */
