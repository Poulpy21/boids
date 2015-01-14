#include "headers.hpp"

#ifdef CUDA_ENABLED

#include "defines.hpp"
#include "vec3.hpp"
#include "options.hpp"


__global__ void applyForces(Vec3<Real> *currentBoidList,
                            Vec3<Real> *newBoidList,
                            Vec3<Real> *meanBoidList, 
                            int *meanBoidWeights, 
                            const int nBoids, 
                            const int nMeanBoids,
                            const struct Options *opt) 
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nBoids)
        return;

    // Compute "internal forces"
    int countSeparation = 0, countCohesion = 0, countAlignment = 0;
    Vec3<Real> forceSeparation, forceCohesion, forceAlignment;
    Vec3<Real> thisBoidPosition = currentBoidList[3*id];
    Vec3<Real> otherBoidPosition;
    for (size_t i = 0; i < nBoids; i++) {
        if (i != id) {
            otherBoidPosition = currentBoidList[3*i];
            Real dist = (thisBoidPosition - otherBoidPosition).norm();
            if (dist < opt.rSeparation) {
                forceSeparation -= (thisBoidPosition - otherBoidPosition).normalized();
                ++countSeparation; 
            }
            if (dist < opt.rCohesion) {
                forceCohesion += thisBoidPosition;
                ++countCohesion;
            }
            if (dist < opt.rAlignment) {
                forceAlignment += currentBoidList[3*i+1];
                ++countAlignment;
            }
        }
    }

    // Compute "external forces"
    for (size_t i = 0; i < nMeanBoids; i++) {
        otherBoidPosition = meanBoidList[3*i];
        Real dist = (thisBoidPosition - otherBoidPosition).norm();
        Real weight = meanBoidWeights[i];
        if (dist < opt.rSeparation) {
            forceSeparation -= weight * (thisBoidPosition - otherBoidPosition).normalized();
            countSeparation += weight; 
        }
        if (dist < opt.rCohesion) {
            forceCohesion += weight * thisBoidPosition;
            countCohesion += weight; 
        }
        if (dist < opt.rAlignment) {
            forceAlignment += weight * meanBoidList[3*i+1];
            countAlignment += weight;
        }
    }

    // Update direction
    newBoidList[3*id+2] = opt.wSeparation * ( countSeparation>0 ? forceSeparation/static_cast<Real>(countSeparation) : forceSeparation) +
                          opt.wCohesion   * ( countCohesion  >0 ? forceCohesion  /static_cast<Real>(countCohesion)   : forceCohesion  ) +
                          opt.wAlignment  * ( countAlignment >0 ? forceAlignment /static_cast<Real>(countAlignment)  : forceAlignment );
    // Update velocity
    newBoidList[3*id+1] = currentBoidList[3*id+1] + newBoidList[3*id+2];
    Real speed = newBoidList[3*id+1].norm();
    newBoidList[3*id+1] = (speed > opt.maxVel ? newBoidList[3*id+1]*opt.maxVel/speed : newBoidList[3*id+1]);

    // Update position
    Vec3<Real> pos = currentBoidList[3*id] + dt * newBoidList[3*id+1];

    // Make sure the boid stays inside the domain
    pos.x = fmod(pos.x, opt.domainSize);
    pos.y = fmod(pos.y, opt.domainSize);
    pos.z = fmod(pos.z, opt.domainSize);
    newBoidList[3*id] = pos;

}

void applyForcesKernel(Vec3<Real> *currentBoidList,
                       Vec3<Real> *newBoidList,
                       Vec3<Real> *meanBoidList, 
                       int *meanBoidWeights, 
                       const int nBoids, 
                       const int nMeanBoids,
                       const struct Options *opt) 
{

    dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    applyForces<<<<gridDim,blockDim,0,0>>>(currentBoidList, 
                                           newBoidList, 
                                           meanBoidList, 
                                           meanBoidWeights, 
                                           nBoids, 
                                           nMeanBoids, 
                                           opt);
    
    cudaDeviceSynchronize();
    checkKernelExecution();
}



#endif
