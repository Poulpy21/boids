#ifndef __AGENT_DATA_HPP__
#define __AGENT_DATA_HPP__

#include "headers.hpp"
#include "vector.hpp"


struct AgentData {
        Real *posX;
        Real *posY;
        Real *posZ;
        Real *velX;
        Real *velY;
        Real *velZ;
        Real *dirX;
        Real *dirY;
        Real *dirZ;
        int nAgents;
   
        // Alloc contiguous memory 
        DEVICE AgentData(int n) : posX(new Real[9*n]), nAgents(n) {
            posY = posX + n*sizeof(Real);
            posZ = posY + n*sizeof(Real);
            velX = posZ + n*sizeof(Real);
            velY = velX + n*sizeof(Real);
            velZ = velY + n*sizeof(Real);
            dirX = velZ + n*sizeof(Real);
            dirY = dirX + n*sizeof(Real);
            dirZ = dirY + n*sizeof(Real);
        }
        DEVICE ~AgentData() {
            delete[] posX;
        }

        DEVICE int size() {return nAgents*9*sizeof(Real);}

        DEVICE Vector getPosition (int index) {return Vector(posX[index],posY[index],posZ[index]);}
        DEVICE Vector getVelocity (int index) {return Vector(velX[index],velY[index],velZ[index]);}
        DEVICE Vector getDirection(int index) {return Vector(dirX[index],dirY[index],dirZ[index]);}

        DEVICE void setPosition (int index, Vector &v) {posX[index]=v.x;posY[index]=v.y;posZ[index]=v.z;}
        DEVICE void setVelocity (int index, Vector &v) {velX[index]=v.x;velY[index]=v.y;velZ[index]=v.z;}
        DEVICE void setDirection(int index, Vector &v) {dirX[index]=v.x;dirY[index]=v.y;dirZ[index]=v.z;}
};

#endif
