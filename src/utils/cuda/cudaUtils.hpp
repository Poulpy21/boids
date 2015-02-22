
#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_

#include "headers.hpp"
#include <log4cpp/Category.hh>

#ifdef CUDA_ENABLED
namespace utils {
    std::string toStringDim(const dim3 &dim);
    void printCudaDevices(std::ostream &outputStream);
    void logCudaDevices(log4cpp::Category &log_output);
    int SMVersionToCores(unsigned int major, unsigned int minor);
}
#endif


#ifdef _DEBUG //_DEBUG FLAG SET

#ifdef CUDA_ENABLED
#define CHECK_CUDA_ERRORS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_EXECUTION(ans) { checkKernelExecution(); }
void gpuAssert(cudaError_t code, const std::string &file, int line, bool abort = true);
void checkKernelExecution();
#endif

#ifdef CURAND_ENABLED
#define CHECK_CURAND_ERRORS(ans) { curandAssert((ans), __FILE__, __LINE__); }
    void curandAssert(curandStatus status, const std::string &file, int line, bool abort = true);
#endif

#ifdef THRUST_ENABLED
#define CHECK_THRUST_ERRORS(ans) { try { (ans); } catch(const std::exception &e) { checkThrustError(e, __FILE__, __LINE__); }}
    void checkThrustError(const std::exception &e, const std::string &file, int line, bool abort = true);
#endif

#else //_DEBUG FLAG NOT SET

#ifdef CUDA_ENABLED
#define CHECK_CUDA_ERRORS(ans) { (ans); }
#define CHECK_KERNEL_EXECUTION(ans) { }
#endif

#ifdef CURAND_ENABLED
#define CHECK_CURAND_ERRORS(ans) { (ans); }
#endif

#ifdef THRUST_ENABLED
#define CHECK_THRUST_ERRORS(ans) { (ans); }
#endif

#endif //_DEBUG

#endif //CUDA UTILS
