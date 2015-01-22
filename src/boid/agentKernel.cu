#include "headers.hpp"

#ifdef CUDA_ENABLED

#include "options.hpp"
#include "agentData.hpp"

#include "vector.hpp"

__global__ void computeForces(Real *boidData,
                              Real *meanBoidData, 
                              int *meanBoidWeights, 
                              const int nBoids, 
                              const int nMeanBoids,
                              const struct Options *opt) 
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nBoids)
        return;

    // Rebuild AgentData
    AgentData boidList(boidData, nBoids), meanBoidList(meanBoidData, nMeanBoids);

    // Compute "internal forces"
    int countSeparation = 0, countCohesion = 0, countAlignment = 0;
    Vector forceSeparation, forceCohesion, forceAlignment;
    Vector thisBoidPosition = boidList.getPosition(id);
    Vector otherBoidPosition;
    for (int i = 0; i < nBoids; i++) {
        if (i != id) {
            otherBoidPosition = boidList.getPosition(i);
            Real dist = (thisBoidPosition - otherBoidPosition).norm();
            if (dist < opt->rSeparation) {
                forceSeparation -= (thisBoidPosition - otherBoidPosition).normalized();
                ++countSeparation; 
            }
            if (dist < opt->rCohesion) {
                forceCohesion += thisBoidPosition;
                ++countCohesion;
            }
            if (dist < opt->rAlignment) {
                forceAlignment += boidList.getVelocity(i);
                ++countAlignment;
            }
        }
    }

    // Compute "external forces"
    for (int i = 0; i < nMeanBoids; i++) {
        otherBoidPosition = boidList.getPosition(i);
        Real dist = (thisBoidPosition - otherBoidPosition).norm();
        Real weight = meanBoidWeights[i];
        if (dist < opt->rSeparation) {
            forceSeparation -= weight * (thisBoidPosition - otherBoidPosition).normalized();
            countSeparation += weight;
        }
        if (dist < opt->rCohesion) {
            forceCohesion += weight * thisBoidPosition;
            countCohesion += weight;
        }
        if (dist < opt->rAlignment) {
            forceAlignment += weight * boidList.getVelocity(i);
            countAlignment += weight;
        }
    }
    
    // Update direction
    Vector direction( opt->wSeparation * ( countSeparation>0 ? forceSeparation/static_cast<Real>(countSeparation) : forceSeparation) +
                      opt->wCohesion   * ( countCohesion  >0 ? forceCohesion  /static_cast<Real>(countCohesion)   : forceCohesion  ) +
                      opt->wAlignment  * ( countAlignment >0 ? forceAlignment /static_cast<Real>(countAlignment)  : forceAlignment ));
    boidList.setDirection(id, direction);
}

void computeForcesKernel(Real *boidData,
                         Real *meanBoidData, 
                         int *meanBoidWeights, 
                         const int nBoids, 
                         const int nMeanBoids,
                         const struct Options *opt) 
{
    dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    computeForces<<<gridDim,blockDim,0,0>>>(boidData, 
                                            meanBoidData, 
                                            meanBoidWeights, 
                                            nBoids, 
                                            nMeanBoids, 
                                            opt);
    
    cudaDeviceSynchronize();
    checkKernelExecution();
}



__global__ void applyForces(Real *boidData, const int nBoids, const struct Options *opt) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nBoids)
        return;

    // Rebuild AgentData
    AgentData boidList(boidData, nBoids);

    // Update velocity
    Vector velocity = boidList.getVelocity(id) + boidList.getDirection(id);
    Real speed = velocity.norm();
    velocity = (speed > opt->maxVel ? velocity*opt->maxVel/speed : velocity);
    boidList.setVelocity(id, velocity);

    // Update position
    Vector pos = boidList.getPosition(id) + opt->dt * boidList.getVelocity(id);

    // Make sure the boid stays inside the domain
    pos.x = fmod(pos.x, opt->domainSize);
    pos.y = fmod(pos.y, opt->domainSize);
    pos.z = fmod(pos.z, opt->domainSize);
    boidList.setPosition(id, pos);
}

void applyForcesKernel(Real*boidData, const int nBoids, const struct Options *opt) {
    dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    applyForces<<<gridDim,blockDim,0,0>>>(boidData, nBoids, opt);
    
    cudaDeviceSynchronize();
    checkKernelExecution();
}


//FIXME reduce sum
__global__ void computeMeanBoid(Real *boidData, const int nBoids, Vector *meanBoid) {

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= nBoids)
        return;

    //Rebuild AgentData
    AgentData boidList(boidData, nBoids);

    /*__shared__ Vector sumVector //= Vector(0,0,0);
    sumVector += boidList.getPosition(i);
    *meanBoid =  sumVector / (nBoids>0 ? static_cast<Real>(nBoids) : 1.0);*/
}

void computeMeanBoidKernel(Real *boidData, const int nBoids, Vector *meanBoid) {
    dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    computeMeanBoid<<<gridDim,blockDim,0,0>>>(boidData, nBoids, meanBoid); 
    
    cudaDeviceSynchronize();
    checkKernelExecution();
}

#endif
