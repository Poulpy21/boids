#ifndef VECTOR
#define VECTOR

#include "headers.hpp"
#include <cmath>
#include "types.hpp"


#ifndef __CUDACC__
#include <iostream>
#include <limits>

#define EPSILON std::numeric_limits<Real>::epsilon()
#else
#define EPSILON 1.19209e-07

#endif // ifdef __CUDACC__

class Vector
{
    public:
        Real x, y ,z;

        // Default constructor
        __DEVICE__ Vector(){
            this->x = 0; this->y = 0; this->z = 0;
        }

        // Constructor from three real numbers
        __DEVICE__ Vector(Real x0, Real y0, Real z0){
            this->x = x0; this->y = y0; this->z = z0;
        }

        // Operators
        __DEVICE__ Vector operator+( const Vector& rhs ) const {
            return Vector( x + rhs.x, y + rhs.y, z + rhs.z );
        }

        __DEVICE__ Vector& operator+=( const Vector& rhs ) {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            return *this;
        }

        __DEVICE__ Vector operator-( const Vector& rhs ) const {
            return Vector( x - rhs.x, y - rhs.y, z - rhs.z );
        }

        __DEVICE__ Vector& operator-=( const Vector& rhs ) {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            return *this;
        }

        __DEVICE__ Vector operator*( Real s ) const {
            return Vector( x * s, y * s, z * s );
        }

        __DEVICE__ Vector& operator*=( Real s ) {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        __DEVICE__ Vector operator/( Real s ) const {
            Real inv = Real(1.0) / s;
            return Vector( x * inv, y * inv, z * inv );
        }

        __DEVICE__ Vector& operator/=( Real s ) {
            Real inv = Real(1.0) / s;
            x *= inv;
            y *= inv;
            z *= inv;
            return *this;
        }

        __DEVICE__ Vector operator-() const {
            return Vector( -x, -y, -z );
        }

        __DEVICE__ bool operator==( const Vector& rhs ) const {
            return std::abs(x - rhs.x)<EPSILON && std::abs(y - rhs.y)<EPSILON && std::abs(z -rhs.z)<EPSILON;
        }

        __DEVICE__ bool operator!=( const Vector& rhs ) const {
            return !operator==( rhs );
        }

        __DEVICE__ Real norm() {
            return sqrt(x * x + y * y + z * z);
        }

        __DEVICE__ Vector normalized(){
            Real inorm = Real(1.0)/this->norm();
            return Vector(x*inorm,y*inorm,z*inorm);
        }

        __DEVICE__ void normalize(){
            Real inorm = Real(1.0)/this->norm();
            x*=inorm;y*=inorm;z*=inorm;
        }

        __DEVICE__ friend Vector operator*( Real s, const Vector &u) {
            return u * s;
        }
};

#ifndef __CUDACC__
Vector& Zeros();

std::ostream &operator<< (std::ostream &stream, const Vector & u);
#endif

#endif
