
#include "initBounds.hpp"
#include "options.hpp"
#include <string>
#include <iostream>

namespace kernel {

    __constant__ Real dt;

    __constant__ Real wCohesion;
    __constant__ Real wAlignment;
    __constant__ Real wSeparation;

    __constant__ Real rCohesion;
    __constant__ Real rAlignment;
    __constant__ Real rSeparation;

    __constant__ Real maxVelocity;
    __constant__ Real domainSize;

    __constant__ Real minInitValues[9];
    __constant__ Real maxInitValues[9];

    template <typename T>
    __device__ inline T mix(T alpha, T a, T b) {
        return (a + alpha*(b-a));
    }
    
    template <>
    __device__ inline float mix<float>(float alpha, float a, float b) {
        return fmaf(alpha,b-a,a);
    }
    
    template <>
    __device__ inline double mix<double>(double alpha, double a, double b) {
        return fmaf(alpha,b-a,a);
    }

    template <typename T>
    __device__ inline T distance2(T x1, T y1, T z1, 
                              T x2, T y2, T z2) {
        return x1*x2 + y1*y2 + z1*z2;
    }

    template <>
    __device__ inline float distance2<float>(float x1, float y1, float z1, 
                              float x2, float y2, float z2) {
        return fmaf(x1,x2, fmaf(y1,y2, z1*z2));
    }
    
    template <>
    __device__ inline double distance2<double>(double x1, double y1, double z1, 
                              double x2, double y2, double z2) {
        return fma(x1,x2, fma(y1,y2, z1*z2));
    }
    
    template <typename T>
    __device__ inline T distance(T x1, T y1, T z1, 
                              T x2, T y2, T z2) {
        return sqrt(x1*x2 + y1*y2 + z1*z2);
    }
    
    template <>
    __device__ inline float distance<float>(float x1, float y1, float z1, 
                              float x2, float y2, float z2) {
        return sqrtf(fmaf(x1,x2, fmaf(y1,y2, z1*z2)));
    }
    
    template <>
    __device__ inline double distance<double>(double x1, double y1, double z1, 
                              double x2, double y2, double z2) {
        return sqrt(fma(x1,x2, fma(y1,y2, z1*z2)));
    }
    
    template <typename T>
    __device__ inline T norm2(T x1, T y1, T z1) {
        return x1*x1 + y1*y1 + z1*z1;
    }

    template <>
    __device__ inline float norm2<float>(float x1, float y1, float z1) {
        return fmaf(x1,x1, fmaf(y1,y1, z1*z1));
    }
    
    template <>
    __device__ inline double norm2<double>(double x1, double y1, double z1) { 
        return fma(x1,x1, fma(y1,y1, z1*z1));
    }
    
    template <typename T>
    __device__ inline T norm(T x1, T y1, T z1) { 
        return sqrt(x1*x1 + y1*y1 + z1*z1);
    }
    
    template <>
    __device__ inline float norm<float>(float x1, float y1, float z1) { 
        return sqrtf(fmaf(x1,x1, fmaf(y1,y1, z1*z1)));
    }
    
    template <>
    __device__ inline double norm<double>(double x1, double y1, double z1) { 
        return sqrt(fma(x1,x1, fma(y1,y1, z1*z1)));
    }

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
    __device__ inline T distance(const CudaVec<T> &a, const CudaVec<T> &b) {
        return distance<T>(a.x,a.y,a.z, b.x,b.y,b.z);
    }
    
    template <typename T>
    __device__ inline T distance2(const CudaVec<T> &a, const CudaVec<T> &b) {
        return distance2<T>(a.x,a.y,a.z, b.x,b.y,b.z);
    }


    __launch_bounds__(MAX_THREAD_PER_BLOCK)
        __global__ void initializeBoids(const unsigned int nBoids, float *rand, Real* agents) {

            unsigned long int id = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

            if(id >= 9u*nBoids)
                return;

            unsigned int idd = id/nBoids;

            agents[id] = mix(rand[id], minInitValues[idd], maxInitValues[idd]);
        }

    void initializeBoidsKernel(unsigned int nBoids, float *rand_d, Real *agents_d) {
        float nReals = nBoids * 9u;
        dim3 dimBlock(MAX_THREAD_PER_BLOCK);
        dim3 dimGrid((unsigned int)ceil(nReals/MAX_THREAD_PER_BLOCK) % 65535, ceil(nReals/(MAX_THREAD_PER_BLOCK*65535.0f)));
        log4cpp::log_console->infoStream() << "[KERNEL::InitializeBoids] <<<" 
            << utils::toStringDim(dimBlock) << ", " 
            << utils::toStringDim(dimGrid)
            << ">>>";

        initializeBoids<<<dimGrid,dimBlock>>>(nBoids, rand_d, agents_d);
        CHECK_KERNEL_EXECUTION();
    }



    /*__launch_bounds__(MAX_THREAD_PER_BLOCK)*/
        /*__global__ void computeInternalForces(const unsigned int nBoids, Real* agents) {*/

            /*unsigned long int id = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;*/
            /*if(id >= nBoids)*/
                /*return;*/
        
            /*const Real *agentsPositionX = agents     + 0ul*nBoids;*/
            /*const Real *agentsPositionY = agents     + 1ul*nBoids;*/
            /*const Real *agentsPositionZ = agents     + 2ul*nBoids;*/
            /*const Real *agentsVelocityX = agents     + 3ul*nBoids;*/
            /*const Real *agentsVelocityY = agents     + 4ul*nBoids;*/
            /*const Real *agentsVelocityZ = agents     + 5ul*nBoids;*/

            /*CudaVec<Real> myPosition(agentsPositionX[id], agentsPositionY[id], agentsPositionZ[id]);*/
            
            /*CudaVec<Real> forceSeparation, forceCohesion, forceAlignment;*/
            /*unsigned int nSeparation = 0u, nCohesion = 0u, nAlignment = 0u;*/

            /*for(unsigned int i = 0u; i < nBoids; i++) {*/
                /*if(i == id)*/
                    /*continue;*/

                /*CudaVec<Real> otherPosition(agentsPositionX[i], agentsPositionY[i], agentsPositionZ[i]);*/
                /*CudaVec<Real> otherVelocity(agentsVelocityX[i], agentsVelocityY[i], agentsVelocityZ[i]);*/

                /*Real dist = distance<Real>(myPosition, otherPosition); */

                /*if(dist < rSeparation) {*/
                    
                    /*nSeparation++;*/
                /*}*/
                

            /*}*/

        /*}*/

//#ifdef THRUST_ENABLED
    //void thrustSort(Real *agents_d, unsigned int nAgents) {
    
        //log4cpp::log_console->infoStream() << "Sorting agents with thrust...";

        //thrust::device_ptr<Real> agents_d_ptr[9u];
        //for(unsigned int i = 0u; i < 9u; i++)
            //agents_d_ptr[i] = thrust::device_ptr<Real>(agents_d+i*nAgents);


        //thrust::device_vector<unsigned int> keys(nAgents);
        //thrust::sequence(keys.begin(), keys.end());

        //for(int i = 2; i >= 0; i--) {
            //thrust::stable_sort_by_key(agents_d_ptr[i], agents_d_ptr[i] + nAgents, keys.begin(), thrust::less<Real>());
        //}
        
        //for(unsigned int i = 0u; i < 9u; i++) {
            //thrust::stable_sort_by_key(keys.begin(), keys.end(), agents_d_ptr[i]);
        //}


        //thrust::copy(keys.begin(), keys.end(), std::ostream_iterator<Real>(std::cout, " "));
    //}
//#endif
    
}
