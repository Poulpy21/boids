
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include "headers.hpp"
#ifdef CUDA_ENABLED

#include <log4cpp/Category.hh>

#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }

namespace utils {
    void printCudaDevices(std::ostream &outputStream);
    void logCudaDevices(log4cpp::Category &log_output);
}

void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
void checkKernelExecution();

#endif
#endif
