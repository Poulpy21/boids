#ifndef VECTOR
#define VECTOR

#include "headers.hpp"
#include <cmath>
#include "types.hpp"
#ifndef __CUDACC__
#include <iostream>
#include <limits>
#endif

#ifndef __CUDACC__
#define EPSILON std::numeric_limits<Real>::epsilon()
#else
#define EPSILON 1.19209e-07
#endif

class Vector
{
    public:
        Real x, y ,z;

        // Default constructor
        DEVICE Vector(){}

        // Constructor from three real numbers
        DEVICE Vector(Real x0, Real y0, Real z0){
            this->x = x0; this->y = y0; this->z = z0;
        }

        // Operators
        DEVICE Vector operator+( const Vector& rhs ) const {
            return Vector( x + rhs.x, y + rhs.y, z + rhs.z );
        }

        DEVICE Vector& operator+=( const Vector& rhs ) {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            return *this;
        }

        DEVICE Vector operator-( const Vector& rhs ) const {
            return Vector( x - rhs.x, y - rhs.y, z - rhs.z );
        }

        DEVICE Vector& operator-=( const Vector& rhs ) {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            return *this;
        }

        DEVICE Vector operator*( Real s ) const {
            return Vector( x * s, y * s, z * s );
        }

        DEVICE Vector& operator*=( Real s ) {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        DEVICE Vector operator/( Real s ) const {
            Real inv = 1.0 / s;
            return Vector( x * inv, y * inv, z * inv );
        }

        DEVICE Vector& operator/=( Real s ) {
            Real inv = 1.0 / s;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }

        DEVICE Vector operator-() const {
            return Vector( -x, -y, -z );
        }

        DEVICE bool operator==( const Vector& rhs ) const {
            return std::abs(x - rhs.x)<EPSILON && std::abs(y - rhs.y)<EPSILON && std::abs(z -rhs.z)<EPSILON;
        }

        DEVICE bool operator!=( const Vector& rhs ) const {
            return !operator==( rhs );
        }

        DEVICE Real norm() {
            return sqrt(x * x + y * y + z * z);
        }

        DEVICE Vector normalized(){
            double inorm = 1./this->norm();
            return Vector(x*inorm,y*inorm,z*inorm);
        }

        DEVICE void normalize(){
            double inorm = 1./this->norm();
            x*=inorm;y*=inorm;z*=inorm;
        }

        DEVICE friend Vector operator*( Real s, const Vector &u) {
            return u * s;
        }
};

#ifndef __CUDACC__
Vector& Zeros();

std::ostream &operator<< (std::ostream &stream, const Vector & u);
#endif

#endif
