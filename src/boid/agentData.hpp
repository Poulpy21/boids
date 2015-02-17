#ifndef __AGENT_DATA_HPP__
#define __AGENT_DATA_HPP__

#include "headers.hpp"
#include "vector.hpp"

#define AGENTDATA_POSX(index) data[index]
#define AGENTDATA_POSY(index) data[nAgents+index]
#define AGENTDATA_POSZ(index) data[2*nAgents+index]
#define AGENTDATA_VELX(index) data[3*nAgents+index]
#define AGENTDATA_VELY(index) data[4*nAgents+index]
#define AGENTDATA_VELZ(index) data[5*nAgents+index]
#define AGENTDATA_DIRX(index) data[6*nAgents+index]
#define AGENTDATA_DIRY(index) data[7*nAgents+index]
#define AGENTDATA_DIRZ(index) data[8*nAgents+index]

struct AgentData {
        Real *data;
        int nAgents;
        
        // Alloc contiguous memory 
        AgentData(int n) : data(new Real[9*n]), nAgents(n) {}

        __DEVICE__ AgentData(Real *d, int n) : data(d), nAgents(n) {}

        __DEVICE__ ~AgentData() {
            delete[] data;
        }

        __DEVICE__ Vector getPosition (int index) {
            return Vector(AGENTDATA_POSX(index),AGENTDATA_POSY(index),AGENTDATA_POSZ(index));
        }
        __DEVICE__ Vector getVelocity (int index) {
            return Vector(AGENTDATA_VELX(index),AGENTDATA_VELY(index),AGENTDATA_VELZ(index));
        }
        __DEVICE__ Vector getDirection(int index) {
            return Vector(AGENTDATA_DIRX(index),AGENTDATA_DIRY(index),AGENTDATA_DIRZ(index));
        }

        __DEVICE__ void setPosition (int index, Vector v) {
            AGENTDATA_POSX(index)=v.x;
            AGENTDATA_POSY(index)=v.y;
            AGENTDATA_POSZ(index)=v.z;
        }
        __DEVICE__ void setPosition (int index, Real x, Real y, Real z) {
            AGENTDATA_POSX(index)=x;
            AGENTDATA_POSY(index)=y;
            AGENTDATA_POSZ(index)=z;
        }
        __DEVICE__ void setVelocity (int index, Vector v) {
            AGENTDATA_VELX(index)=v.x;
            AGENTDATA_VELY(index)=v.y;
            AGENTDATA_VELZ(index)=v.z;
        }
        __DEVICE__ void setVelocity (int index, Real x, Real y, Real z) {
            AGENTDATA_VELX(index)=x;
            AGENTDATA_VELY(index)=y;
            AGENTDATA_VELZ(index)=z;
        }
        __DEVICE__ void setDirection(int index, Vector v) {
            AGENTDATA_DIRX(index)=v.x;
            AGENTDATA_DIRY(index)=v.y;
            AGENTDATA_DIRZ(index)=v.z;
        }
        __DEVICE__ void setDirection(int index, Real x, Real y, Real z) {
            AGENTDATA_DIRX(index)=x;
            AGENTDATA_DIRY(index)=y;
            AGENTDATA_DIRZ(index)=z;
        }
};

#endif
