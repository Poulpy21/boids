
#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include "headers.hpp"
#include <log4cpp/Category.hh>
#include <string>
#include <sstream>

#ifdef CUDA_ENABLED

namespace utils {
    extern void printCudaDevices(std::ostream &outputStream);
    extern void logCudaDevices(log4cpp::Category &log_output);
    extern int SMVersionToCores(unsigned int major, unsigned int minor);
    extern std::string toStringDim(const dim3 &dim);
}

extern void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
extern void checkKernelExecution();
#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_EXECUTION(ans) { checkKernelExecution(); }

#endif


#ifdef CURAND_ENABLED
    extern void curandAssert(curandStatus status, const std::string &file, int line, bool abort = true);
#define CHECK_CURAND_ERRORS(ans) { curandAssert((ans), __FILE__, __LINE__); }
#endif


#endif /* endifdef cudautils */
