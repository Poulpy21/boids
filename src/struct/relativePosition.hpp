
#ifndef RELATIVEPOSITION_H
#define RELATIVEPOSITION_H

#include "triState.hpp"

struct RelativePosition {
    TriState x, y, z;
    
    RelativePosition() : x(0), y(0), z(0) {};
    
    RelativePosition(TriState x, TriState y, TriState z) :
        x(x), y(y), z(z) {
    }
};

#endif /* end of include guard: RELATIVEPOSITION_H */
