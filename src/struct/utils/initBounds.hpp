
#ifndef INITBOUNDS_H
#define INITBOUNDS_H

#include "headers.hpp"

template <typename T>
struct InitBounds {
    T minValues[9]; //position velocity acceleration in format (X Y Z)
    T maxValues[9];
    
    __HOST__ __DEVICE__ InitBounds()
    {
        for (unsigned int i = 0; i < 9u; i++) {
            minValues[i] = T(0);
            maxValues[i] = T(0);
        }
    }

    __HOST__ __DEVICE__ InitBounds(const Real *_minValues, const Real *_maxValues)
    {
        for (unsigned int i = 0; i < 9u; i++) {
            minValues[i] = _minValues[i];
            maxValues[i] = _maxValues[i];
        }
    }

};

#endif /* end of include guard: INITBOUNDS_H */
