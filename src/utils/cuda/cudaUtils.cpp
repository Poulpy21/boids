
#include "cudaUtils.hpp"

#ifdef CUDA_ENABLED

namespace utils {
    void printCudaDevices(std::ostream &outputStream) {

        int nDevices;
        char buffer[100];

        cudaGetDeviceCount(&nDevices);
        outputStream << "==== CUDA DEVICES ====" << std::endl;
        outputStream << "Found " << nDevices << " devices !" << std::endl;
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            outputStream << "Device Number: " << i << std::endl;
            outputStream << "\tDevice name:                   "  << prop.name << std::endl;
            outputStream << "\tPCI Device:                    " 
                << prop.pciBusID << ":" << prop.pciDeviceID << ":" << prop.pciDomainID << std::endl;
            outputStream << "\tMajor revision number:         " << prop.major << std::endl;
            outputStream << "\tMinor revision number:         " <<   prop.minor << std::endl;
            outputStream << "\tMemory Clock Rate :            " << prop.memoryClockRate/1000 << " MHz" << std::endl;
            outputStream << "\tMemory Bus Width:              " << prop.memoryBusWidth << " bits" << std::endl;
            outputStream << "\tPeak Memory Bandwidth:         " 
                << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
            outputStream << "\tTotal global memory:           " <<   prop.totalGlobalMem/(1024*1024) << " MB" << std::endl;
            outputStream << "\tTotal shared memory per block: " <<   prop.sharedMemPerBlock/1024 << " kB" << std::endl;
            outputStream << "\tTotal registers per block:     " <<   prop.regsPerBlock/1024 << " kB" << std::endl;
            outputStream << "\tTotal constant memory:         " <<   prop.totalConstMem/1024 << " kB" << std::endl;
            outputStream << "\tMaximum memory pitch:          " <<   prop.memPitch/(1024*1024) << " MB" << std::endl;
            outputStream << "\tNumber of multiprocessors:     " <<   prop.multiProcessorCount << std::endl;
            outputStream << "\tMaximum threads per SM:        " <<   prop.maxThreadsPerMultiProcessor << std::endl;
            outputStream << "\tMaximum threads per block:     " <<   prop.maxThreadsPerBlock << std::endl;

            sprintf(buffer, "%ix%ix%i", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            outputStream << "\tMaximum thread block dimension " <<  buffer << std::endl;
            sprintf(buffer, "%ix%ix%i", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            outputStream << "\tMaximum grid dimension         " <<  buffer << std::endl;
            outputStream << "\tWarp size:                     " <<   prop.warpSize << std::endl;
            outputStream << "\tTexture alignment:             " <<   prop.textureAlignment << std::endl;
            outputStream << "\tTexture picth alignment:       " <<   prop.texturePitchAlignment << std::endl;
            outputStream << "\tSurface alignment:             " <<   prop.surfaceAlignment << std::endl;
            outputStream << "\tConcurrent copy and execution: " <<   (prop.deviceOverlap ? "Yes" : "No") << std::endl;
            outputStream << "\tKernel execution timeout:      " <<   (prop.kernelExecTimeoutEnabled ?"Yes" : "No") << std::endl;
            outputStream << "\tDevice has ECC support:        " <<   (prop.ECCEnabled ?"Yes" : "No") << std::endl;
            outputStream << "\tCompute mode:                  " 
                <<   (prop.computeMode == 0 ? "Default" : prop.computeMode == 1 ? "Exclusive" :
                        prop.computeMode == 2 ? "Prohibited" : "Exlusive Process") << std::endl;
        }

        outputStream << "======================" << std::endl;
    }

    void logCudaDevices(log4cpp::Category &log_output) {
        int nDevices;

        cudaGetDeviceCount(&nDevices);
        char buffer[100];
        log_output.infoStream() << "==== CUDA DEVICES ====";
        log_output.infoStream() << "Found " << nDevices << " devices !";
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            log_output.infoStream() << "Device Number: " << i;
            log_output.infoStream() << "\tDevice name:                   "  << prop.name;
            log_output.infoStream() << "\tPCI Device:                    " 
                << prop.pciBusID << ":" << prop.pciDeviceID << ":" << prop.pciDomainID;
            log_output.infoStream() << "\tMajor revision number:         " << prop.major;
            log_output.infoStream() << "\tMinor revision number:         " <<   prop.minor;
            log_output.infoStream() << "\tMemory Clock Rate :            " << prop.memoryClockRate/1000 << " MHz";
            log_output.infoStream() << "\tMemory Bus Width:              " << prop.memoryBusWidth << " bits";
            log_output.infoStream() << "\tPeak Memory Bandwidth:         " 
                << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " GB/s";
            log_output.infoStream() << "\tTotal global memory:           " <<   prop.totalGlobalMem/(1024*1024) << " MB";
            log_output.infoStream() << "\tTotal shared memory per block: " <<   prop.sharedMemPerBlock/1024 << " kB";
            log_output.infoStream() << "\tTotal registers per block:     " <<   prop.regsPerBlock/1024 << " kB";
            log_output.infoStream() << "\tTotal constant memory:         " <<   prop.totalConstMem/1024 << " kB";
            log_output.infoStream() << "\tMaximum memory pitch:          " <<   prop.memPitch/(1024*1024) << " MB";
            log_output.infoStream() << "\tNumber of multiprocessors:     " <<   prop.multiProcessorCount;
            log_output.infoStream() << "\tMaximum threads per SM:        " <<   prop.maxThreadsPerMultiProcessor;
            log_output.infoStream() << "\tMaximum threads per block:     " <<   prop.maxThreadsPerBlock;

            sprintf(buffer, "%ix%ix%i", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            log_output.infoStream() << "\tMaximum thread block dimension " <<  buffer;
            sprintf(buffer, "%ix%ix%i", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            log_output.infoStream() << "\tMaximum grid dimension         " <<  buffer;
            log_output.infoStream() << "\tWarp size:                     " <<   prop.warpSize;
            log_output.infoStream() << "\tTexture alignment:             " <<   prop.textureAlignment;
            log_output.infoStream() << "\tTexture picth alignment:       " <<   prop.texturePitchAlignment;
            log_output.infoStream() << "\tSurface alignment:             " <<   prop.surfaceAlignment;
            log_output.infoStream() << "\tConcurrent copy and execution: " <<   (prop.deviceOverlap ? "Yes" : "No");
            log_output.infoStream() << "\tKernel execution timeout:      " <<   (prop.kernelExecTimeoutEnabled ?"Yes" : "No");
            log_output.infoStream() << "\tDevice has ECC support:        " <<   (prop.ECCEnabled ?"Yes" : "No");
            log_output.infoStream() << "\tCompute mode:                  " 
                <<   (prop.computeMode == 0 ? "Default" : prop.computeMode == 1 ? "Exclusive" :
                        prop.computeMode == 2 ? "Prohibited" : "Exlusive Process");
        }

        log_output.infoStream() << "======================";
    }

    int SMVersionToCores(unsigned int major, unsigned int minor) {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct
        {
            unsigned int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        {
            { 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
            { 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
            { 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
            { 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
            { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
            { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
            { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            { 0, -1 }
        };

        int index = 0;
        while (nGpuArchCoresPerSM[index].SM != 0)
        {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            {
                return nGpuArchCoresPerSM[index].Cores;
            }
            index++;
        }

        // If we don't find the values, we default use the previous one to run properly
        printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);

        return nGpuArchCoresPerSM[7].Cores;
    }
}

void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort) {
    if (code != cudaSuccess) 
    {
        log4cpp::log_console->errorStream() << "GPU Assert => " << cudaGetErrorString(code) << " in file " <<  file << ":" << line << ".";
        std::cout << std::endl;
        if (abort) 
            exit(1);
    }
}

void checkKernelExecution() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        log4cpp::log_console->errorStream() << "Kernel launch failed : " << cudaGetErrorString(error);
        std::cout << std::endl;
        exit(1);
    }
}
#endif

#ifdef CURAND_ENABLED
void curandAssert(curandStatus status, const std::string &file, int line, bool abort) {

    std::string str;

    switch(status) {
        case CURAND_STATUS_SUCCESS:
            return;
        case CURAND_STATUS_VERSION_MISMATCH:
            str = "Header file and linkedlibrary version do not match.";
            break;
        case CURAND_STATUS_NOT_INITIALIZED:
            str = "Generator not initialized.";
            break;
        case CURAND_STATUS_ALLOCATION_FAILED:
            str = "Memory allocationfailed.";
            break;
        case CURAND_STATUS_TYPE_ERROR:
            str = "Generator is wrong type.";
            break;
        case CURAND_STATUS_OUT_OF_RANGE:
            str = "Argument out of range.";
            break;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            str = "Length requested isnot a multple of dimension.";
            break;
        case CURAND_STATUS_LAUNCH_FAILURE:
            str = "Kernel launch failure.";
            break;
        case CURAND_STATUS_PREEXISTING_FAILURE:
            str = "Preexisting failure onlibrary entry.";
            break;
        case CURAND_STATUS_INITIALIZATION_FAILED:
            str = "Initialization ofCUDA failed.";
            break;
        case CURAND_STATUS_ARCH_MISMATCH:
            str = "Architecture mismatch, GPU does not support requested feature.";
            break;
        case CURAND_STATUS_INTERNAL_ERROR:
            str = "Internal library error !";
            break;
        default:
            str = "The impossible happened !";
    }

    log4cpp::log_console->errorStream() << "CURAND Assert => " << str << " in file " <<  file << ":" << line << ".";
    std::cout << std::endl;
    if (abort) 
        exit(1);
}
#endif
