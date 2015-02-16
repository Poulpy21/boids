
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include "headers.hpp"
#include <log4cpp/Category.hh>

#ifdef CUDA_ENABLED
#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_EXECUTION(ans) { checkKernelExecution(); }

namespace utils {
    void printCudaDevices(std::ostream &outputStream);
    void logCudaDevices(log4cpp::Category &log_output);
    int SMVersionToCores(unsigned int major, unsigned int minor);
}

void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
void checkKernelExecution();

#endif

#ifdef CURAND_ENABLED
#define CHECK_CURAND_ERRORS(ans) { curandAssert((ans), __FILE__, __LINE__); }
    void curandAssert(curandStatus status, const std::string &file, int line, bool abort = true);
#endif

#endif
