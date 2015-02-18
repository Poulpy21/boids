
#ifndef DEFINES_H
#define DEFINES_H

#include "config.hpp.out"

#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for ALIGN macro for your host compiler (in utils/defines.hpp) !"
#endif

#ifdef __CUDACC__
#define __CONSTANT__ __constant__
#define __SHARED__ __shared__
#define __DEVICE__ __device__
#define __HOST__ __host__
#else
#define __CONSTANT__
#define __SHARED__
#define __DEVICE__
#define __HOST__
#endif

#if __cplusplus < 201103L
#define override
#define final
#define constexpr
#define nullptr 0
#endif
    
#ifdef GUI_ENABLED
#define RENDETREE ,public RenderTree
#define RENDETREE_ONLY : public RenderTree
#endif

#define STRINGIFY(X) #X
#define STRINGIFY_MACRO(X) STRINGIFY(X)

//System reserved memory in MB
#define _CPU_MIN_RESERVED_MEMORY 256ul*1024ul*1024ul
#define _GPU_MIN_RESERVED_MEMORY 256ul*1024ul*1024ul

#define MAX_THREAD_PER_BLOCK 512u //opti arch sm20

#endif /* end of include guard: DEFINES_H */
