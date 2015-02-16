
#ifndef INITBOUNDS_H
#define INITBOUNDS_H

#include "headers.hpp"
#include "boundingBox.hpp"

template <typename T>
struct InitBounds {
    T min_X, max_X, min_Y, max_Y, min_Z, max_Z;
    T min_VX, max_VX, min_VY, max_VY, min_VZ, max_VZ;
    T min_AX, max_AX, min_AY, max_AY, min_AZ, max_AZ;

    __HOST__ __DEVICE__ InitBounds(const BoundingBox<3u,T> &positionBox,
            const BoundingBox<3u,T> &velocityBox,
            const BoundingBox<3u,T> &accelerationBox) :
        min_X(positionBox.min[0]),
        max_X(positionBox.max[0]),
        min_Y(positionBox.min[1]),
        max_Y(positionBox.max[1]),
        min_Z(positionBox.min[2]),
        max_Z(positionBox.max[2]),
        min_VX(velocityBox.min[0]),
        max_VX(velocityBox.max[0]),
        min_VY(velocityBox.min[1]),
        max_VY(velocityBox.max[1]),
        min_VZ(velocityBox.min[2]),
        max_VZ(velocityBox.max[2]),
        min_AX(accelerationBox.min[0]),
        max_AX(accelerationBox.max[0]),
        min_AY(accelerationBox.min[1]),
        max_AY(accelerationBox.max[1]),
        min_AZ(accelerationBox.min[2]),
        max_AZ(accelerationBox.max[2])
    {
    }

};

#endif /* end of include guard: INITBOUNDS_H */
