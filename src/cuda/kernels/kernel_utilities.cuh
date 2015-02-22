
#ifdef __CUDACC__

#ifndef KERNEL_UTILITIES_H
#define KERNEL_UTILITIES_H

#include "headers.hpp"
#include <vector_types.h>

//Common cuda templates and variables

namespace kernel {
   
    //Common constants
    extern __device__ __constant__ Real dt;

    extern __device__ __constant__ Real wCohesion;
    extern __device__ __constant__ Real wAlignment;
    extern __device__ __constant__ Real wSeparation;

    extern __device__ __constant__ Real rCohesion;
    extern __device__ __constant__ Real rAlignment;
    extern __device__ __constant__ Real rSeparation;

    extern __device__ __constant__ Real maxVelocity;
    extern __device__ __constant__ Real domainSize;

    extern __device__ __constant__ Real minInitValues[9];
    extern __device__ __constant__ Real maxInitValues[9];


    //float3 and double3 wrapper type (because nvidia you know...)
    template <typename T, unsigned int N> struct MakeCudaVec;
    template <> struct MakeCudaVec<float,  3>;     
    template <> struct MakeCudaVec<float,  3> { typedef float3 type;  };
    template <> struct MakeCudaVec<double, 3> { typedef double3 type; };
    extern template struct MakeCudaVec<float,  3u>; //explicit instantiation of full specializations -- only in this compilation unit !
    extern template struct MakeCudaVec<double, 3u>;

    template <typename T>  __inline__ __device__ 
        T distance(const typename MakeCudaVec<T,3>::type &v1, const typename MakeCudaVec<T,3>::type &v2) {
            return sqrt(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
        }
    
    template <> __inline__ __device__ 
        float distance<float>(const float3 &v1, const float3 &v2) {
            return sqrtf(fmaf(v1.z,v2.z, fmaf(v1.y,v2.y, v1.x*v2.x)));
        }

    template <> __inline__ __device__ 
        double distance<double>(const double3 &v1, const double3 &v2) {
            return sqrt(fma(v1.z,v2.z, fma(v1.y,v2.y, v1.x*v2.x)));
        }
    
    template <typename T>  __inline__ __device__ 
        T norm(const typename MakeCudaVec<T,3>::type &v) {
            return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        }
    
    template <> __inline__ __device__ 
        float norm<float>(const float3 &v) {
            return sqrtf(fmaf(v.z,v.z, fmaf(v.y,v.y, v.x*v.x)));
        }

    template <> __inline__ __device__ 
        double norm<double>(const double3 &v) {
            return sqrt(fma(v.z,v.z, fma(v.y,v.y, v.x*v.x)));
        }
   

    //other utils
    template <typename T>
    __device__ __inline__ T mix(T alpha, T a, T b) {
        return (a + alpha*(b-a));
    }
    
    template <>
    __device__ __inline__ float mix<float>(float alpha, float a, float b) {
        return fmaf(alpha,b-a,a);
    }
    
    template <>
    __device__ __inline__ double mix<double>(double alpha, double a, double b) {
        return fmaf(alpha,b-a,a);
    }

    
    template <typename T>
    __device__ __inline__ T distance2(T x1, T y1, T z1, 
                              T x2, T y2, T z2) {
        return x1*x2 + y1*y2 + z1*z2;
    }
    
    template <>
    __device__ __inline__ float distance2<float>(float x1, float y1, float z1, 
                              float x2, float y2, float z2) {
        return fmaf(x1,x2, fmaf(y1,y2, z1*z2));
    }
    
    template <>
    __device__ __inline__ double distance2<double>(double x1, double y1, double z1, 
                              double x2, double y2, double z2) {
        return fma(x1,x2, fma(y1,y2, z1*z2));
    }
    

    template <typename T>
    __device__ __inline__ T distance(T x1, T y1, T z1, 
                              T x2, T y2, T z2) {
        return sqrt(x1*x2 + y1*y2 + z1*z2);
    }
    
    template <>
    __device__ __inline__ float distance<float>(float x1, float y1, float z1, 
                              float x2, float y2, float z2) {
        return sqrtf(fmaf(x1,x2, fmaf(y1,y2, z1*z2)));
    }
    
    template <>
    __device__ __inline__ double distance<double>(double x1, double y1, double z1, 
                              double x2, double y2, double z2) {
        return sqrt(fma(x1,x2, fma(y1,y2, z1*z2)));
    }
    
    template <typename T>
    __device__ __inline__ T norm2(T x1, T y1, T z1) {
        return x1*x1 + y1*y1 + z1*z1;
    }
    
    template <>
    __device__ __inline__ float norm2<float>(float x1, float y1, float z1) {
        return fmaf(x1,x1, fmaf(y1,y1, z1*z1));
    }
    
    template <>
    __device__ __inline__ double norm2<double>(double x1, double y1, double z1) { 
        return fma(x1,x1, fma(y1,y1, z1*z1));
    }
    
    template <typename T>
    __device__ __inline__ T norm(T x1, T y1, T z1) { 
        return sqrt(x1*x1 + y1*y1 + z1*z1);
    }
    
    template <>
    __device__ __inline__ float norm<float>(float x1, float y1, float z1) { 
        return sqrtf(fmaf(x1,x1, fmaf(y1,y1, z1*z1)));
    }
    
    template <>
    __device__ __inline__ double norm<double>(double x1, double y1, double z1) { 
        return sqrt(fma(x1,x1, fma(y1,y1, z1*z1)));
    }

    
    //Deprecated utils (still here for compatibility issues)
    template <typename T>
        struct CudaVec {
    
            T x, y, z;
        
            __device__ CudaVec() : x(0), y(0), z(0) {}
            __device__ CudaVec(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
            __device__ CudaVec(const CudaVec<T> &other) : x(other.x), y(other.y), z(other.z) {}
            __device__ CudaVec<T>& operator=(const CudaVec<T> &other) { 
                x = other.x;
                y = other.y;
                z = other.z;
                return *this;
            }
            __device__ ~CudaVec() {}
    };
    
    template <typename T>
    __device__ __inline__ T distance(const CudaVec<T> &a, const CudaVec<T> &b) {
        return distance<T>(a.x,a.y,a.z, b.x,b.y,b.z);
    }
    
    template <typename T>
    __device__ __inline__ T distance2(const CudaVec<T> &a, const CudaVec<T> &b) {
        return distance2<T>(a.x,a.y,a.z, b.x,b.y,b.z);
    }
}

#endif /* end of include guard: KERNEL_UTILITIES_H */

#endif
    
