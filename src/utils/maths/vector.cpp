#include "vector.hpp"

#ifndef __CUDACC__
Vector& Zeros() {
    static Vector u(0.,0.,0.);
    return u;
}

std::ostream &operator<< (std::ostream &stream, const Vector & u){
    stream<<u.x<<" "<<u.y<<" "<<u.z;
    return stream;
}
#endif
