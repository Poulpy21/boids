
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
#define __DEVICE__ __device__
#define __HOST__ __host__
#else
#define __DEVICE__
#define __HOST__
#endif
    
#ifdef GUI_ENABLED
#define RENDETREE ,public RenderTree
#define RENDETREE_ONLY : public RenderTree
#endif

#define STRINGIFY(X) #X
#define STRINGIFY_MACRO(X) STRINGIFY(X)

#endif /* end of include guard: DEFINES_H */
