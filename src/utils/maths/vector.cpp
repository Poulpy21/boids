#include "vector.hpp"

Vector& Zeros()
{
    static Vector u(0.,0.,0.);
    return u;
}

Vector operator*( Real s, Vector &u) {
    return u*s;
}

#ifndef __CUDACC__
std::ostream &operator<< (std::ostream &stream, const Vector & u){
    stream<<u.x<<" "<<u.y<<" "<<u.z;
    return stream;
}
#endif
