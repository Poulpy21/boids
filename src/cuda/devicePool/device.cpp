

#include "device.hpp"

#ifdef CUDA_ENABLED

Device::Device(unsigned int deviceId) :
    deviceId(deviceId) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        name = std::string(prop.name);

        totalGlobalMem = prop.totalGlobalMem;
        sharedMemPerBlock = prop.sharedMemPerBlock;
        regsPerBlock = prop.regsPerBlock;
        warpSize = prop.warpSize;
        memPitch = prop.memPitch;
        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        clockRate = prop.clockRate;
        totalConstMem = prop.totalConstMem;
        major = prop.major;
        minor = prop.minor;
        textureAlignment = prop.textureAlignment;
        texturePitchAlignment = prop.texturePitchAlignment;
        deviceOverlap = prop.deviceOverlap;
        multiProcessorCount = prop.multiProcessorCount;
        kernelExecTimeoutEnabled = prop.kernelExecTimeoutEnabled;
        integrated = prop.integrated;
        canMapHostMemory = prop.canMapHostMemory;
        computeMode = prop.computeMode;
        maxTexture1D = prop.maxTexture1D;
        maxTexture1DLinear = prop.maxTexture1DLinear;
        surfaceAlignment = prop.surfaceAlignment;
        concurrentKernels = prop.concurrentKernels;
        ECCEnabled = prop.ECCEnabled;
        pciBusID = prop.pciBusID;
        pciDeviceID = prop.pciDeviceID;
        pciDomainID = prop.pciDomainID;
        tccDriver = prop.tccDriver;
        asyncEngineCount = prop.asyncEngineCount;
        unifiedAddressing = prop.unifiedAddressing;
        memoryClockRate = prop.memoryClockRate;
        memoryBusWidth = prop.memoryBusWidth;
        l2CacheSize = prop.l2CacheSize;
        maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;

        maxSurfaceCubemap = prop.maxSurfaceCubemap;
        maxTextureCubemap = prop.maxTextureCubemap;
        maxSurface1D = prop.maxSurface1D;

        for (unsigned int i = 0; i < 2; i++) {
            maxTexture2D[i] = prop.maxTexture2D[i];
            maxTexture2DGather[i] = prop.maxTexture2DGather[i];
            maxTexture1DLayered[i] = prop.maxTexture1DLayered[i];
            maxTextureCubemapLayered[i] = prop.maxTextureCubemapLayered[i];
            maxSurfaceCubemapLayered[i] = prop.maxSurfaceCubemapLayered[i];
            maxSurface1DLayered[i] = prop.maxSurface1DLayered[i];
            maxSurface2D[i] = prop.maxSurface2D[i];
        }

        for (unsigned int i = 0; i < 2; i++) {
            maxTexture3D[i] = prop.maxTexture3D[i];
            maxSurface3D[i] = prop.maxSurface3D[i];
            maxThreadsDim[i] = prop.maxThreadsDim[i];
            maxGridSize[i] = prop.maxGridSize[i];
            maxTexture2DLinear[i] = prop.maxTexture2DLinear[i];
            maxSurface2DLayered[i] = prop.maxSurface2DLayered[i];
            maxTexture2DLayered[i] = prop.maxTexture2DLayered[i];
        }

        coresPerSM = utils::SMVersionToCores(major, minor);
        coresCount = multiProcessorCount * coresPerSM;
    }

Device::~Device() {
}

std::string Device::toString() const {

    std::stringstream os;

    char buffer[200];

    os << "Device Id : " << deviceId << std::endl;
    os << "\tDevice name:                   "  << name << std::endl;
    os << "\tPCI Device:                    " 
        << pciBusID << ":" << pciDeviceID << ":" << pciDomainID << std::endl;
    os << "\tMajor revision number:         " << major << std::endl;
    os << "\tMinor revision number:         " <<   minor << std::endl;
    os << "\tMemory Clock Rate :            " << memoryClockRate/1000 << " MHz" << std::endl;
    os << "\tMemory Bus Width:              " << memoryBusWidth << " bits" << std::endl;
    os << "\tPeak Memory Bandwidth:         " 
        << 2.0*memoryClockRate*(memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
    os << "\tTotal global memory:           " <<   totalGlobalMem/(1024*1024) << " MB" << std::endl;
    os << "\tTotal shared memory per block: " <<   sharedMemPerBlock/1024 << " kB" << std::endl;
    os << "\tTotal registers per block:     " <<   regsPerBlock/1024 << " kB" << std::endl;
    os << "\tTotal constant memory:         " <<   totalConstMem/1024 << " kB" << std::endl;
    os << "\tMaximum memory pitch:          " <<   memPitch/(1024*1024) << " MB" << std::endl;
    os << "\tNumber of multiprocessors:     " <<   multiProcessorCount << std::endl;
    os << "\tMaximum threads per SM:        " <<   maxThreadsPerMultiProcessor << std::endl;
    os << "\tMaximum threads per block:     " <<   maxThreadsPerBlock << std::endl;

    sprintf(buffer, "%ix%ix%i", maxThreadsDim[0], maxThreadsDim[1], maxThreadsDim[2]);
    os << "\tMaximum thread block dimension " <<  buffer << std::endl;
    sprintf(buffer, "%ix%ix%i", maxGridSize[0], maxGridSize[1], maxGridSize[2]);
    os << "\tMaximum grid dimension         " <<  buffer << std::endl;
    os << "\tWarp size:                     " <<   warpSize << std::endl;
    os << "\tTexture alignment:             " <<   textureAlignment << std::endl;
    os << "\tTexture picth alignment:       " <<   texturePitchAlignment << std::endl;
    os << "\tSurface alignment:             " <<   surfaceAlignment << std::endl;
    os << "\tConcurrent copy and execution: " <<   (deviceOverlap ? "Yes" : "No") << std::endl;
    os << "\tKernel execution timeout:      " <<   (kernelExecTimeoutEnabled ?"Yes" : "No") << std::endl;
    os << "\tDevice has ECC support:        " <<   (ECCEnabled ?"Yes" : "No") << std::endl;
    os << "\tCompute mode:                  " 
        <<   (computeMode == 0 ? "Default" : computeMode == 1 ? "Exclusive" :
                computeMode == 2 ? "Prohibited" : "Exlusive Process") << std::endl;

    os << "======================" << std::endl;

    return os.str();
}

std::ostream& operator<<(std::ostream &os, const Device &device) {
    os << device.toString();
    return os;
}

#endif
