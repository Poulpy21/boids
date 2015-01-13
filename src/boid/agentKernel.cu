#include "headers.hpp"

#ifdef CUDA_ENABLED

#include "defines.hpp"
#include "vec3.hpp"
#include "options.hpp"

__global__ void applyInternalForces(Vec3<Real> *currentBoidList, Vec3<Real>* newBoidList, const int nBoids, const struct Options *opt) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nBoids)
        return;

    // Compute "forces"
    int countSeparation = 0, countCohesion = 0, countAlignment = 0;
    Vec3<Real> forceSeparation, forceCohesion, forceAlignment;
    Vec3<Real> thisBoidPosition = currentBoidList[3*id];
    Vec3<Real> otherBoidPosition;
    for (size_t i = 0; i < nBoids; i++) {
        if (i != id) {
            otherBoidPosition = currentBoidList[3*i];
            double dist = (thisBoidPosition - otherBoidPosition).norm();
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
    
    // Update direction
    newBoidList[3*id+2] = opt.wSeparation * ( count>0 ? forceSeparation/static_cast<Real>(countSeparation) : force) +
                          opt.wCohesion   * ( count>0 ? forceCohesion/static_cast<Real>(countCohesion)     : force) +
                          opt.wAlignment  * ( count>0 ? forceAlignment/static_cast<Real>(countAlignment)   : force);
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

void applyInternalForcesKernel(Vec3<Real> *currentBoidList, Vec3<Real>* newBoidList, const int nBoids, const struct Options *opt) {
    dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    applyInternalForces<<<<gridDim,blockDim,0,0>>>(currentBoidList, newBoidList, nBoids, opt);
    
    cudaDeviceSynchronize();
    checkKernelExecution();
}

//TODO applyExternalForces
#endif
