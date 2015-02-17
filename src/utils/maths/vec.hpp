
#ifndef VEC_H
#define VEC_H

#include "headers.hpp"
#include "utils.hpp"

#include <iostream>
#include <cstring>

/*
* ND vector structure of arithmetic type T
* T should be an arithmetic type, T& = T and T(int) should be defined.
* Operators %, %=, == and != are safe for integer types and floating point types as well (see utils.hpp for the definition of ARE_EQUAL(T) and modulo<T>).
* VecBool, Vec2 and Vec3 extends this class for more convenience (see vecBool.hpp, vec2.hpp and vec3.hpp)
* The behaviour of the output stream operator << is specialized in the case N=1 to avoid unnecessary brackets (scalar case).
*/

template <unsigned int N>
struct VecBool;

template <unsigned int N, typename T>
struct Vec {
#ifndef __CUDACC__
    static_assert(std::is_arithmetic<T>(),         "Vectors can only contain arithmetic types !");
    static_assert(std::is_assignable<T&, T>(),     "T should be assignable !");
    static_assert(std::is_constructible<T, int>(), "T should be constructible from int !");
#endif

    __HOST__ __DEVICE__ Vec<N,T>();
    __HOST__ __DEVICE__ Vec(const Vec<N,T> &v);
    __HOST__ __DEVICE__ explicit Vec(const T data[]);
    __HOST__ __DEVICE__ virtual ~Vec();

    template <typename S>
    __HOST__ __DEVICE__ explicit Vec(const Vec<N,S> &v);

    __HOST__ __DEVICE__ Vec<N,T>& operator= (const Vec<N,T> &v);

    __HOST__ __DEVICE__ T& operator[](unsigned int k);
    __HOST__ __DEVICE__ T  operator[](unsigned int k) const;
    
    __HOST__ __DEVICE__ Vec<N,T> & operator+= (const Vec<N,T> &a);
    __HOST__ __DEVICE__ Vec<N,T> & operator-= (const Vec<N,T> &a);
    __HOST__ __DEVICE__ Vec<N,T> & operator*= (const Vec<N,T> &a);
    __HOST__ __DEVICE__ Vec<N,T> & operator%= (const Vec<N,T> &a);
    __HOST__ __DEVICE__ Vec<N,T> & operator/= (const Vec<N,T> &a);
    __HOST__ __DEVICE__ Vec<N,T> & operator^= (const Vec<N,T> &a);

    __HOST__ __DEVICE__ Vec<N,T> & operator+= (T k);
    __HOST__ __DEVICE__ Vec<N,T> & operator-= (T k);
    __HOST__ __DEVICE__ Vec<N,T> & operator*= (T k);
    __HOST__ __DEVICE__ Vec<N,T> & operator%= (T k);
    __HOST__ __DEVICE__ Vec<N,T> & operator/= (T k);

    __HOST__ __DEVICE__ T normalize();

    __HOST__ __DEVICE__ T norm() const;
    __HOST__ __DEVICE__ T squaredNorm() const;

    __HOST__ __DEVICE__ Vec<N,T> normalized() const;
    
protected: 
    T data[N];
};

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T>::Vec() {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = T(0);
    }
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T>::Vec(const Vec<N,T> &v) {
    memcpy(this->data, v.data, N*sizeof(T));
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T>::Vec(const T data[]) {
    memcpy(this->data, data, N*sizeof(T));
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T>::~Vec() {}
    
template <unsigned int N, typename T>
template <typename S>
__HOST__ __DEVICE__ Vec<N,T>::Vec(const Vec<N,S> &v) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = static_cast<T>(v[i]);
    }
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T>& Vec<N,T>::operator= (const Vec<N,T> &v) {
    memcpy(this->data, v.data, N*sizeof(T));
    return *this;
}
    
template <unsigned int N, typename T>
__HOST__ __DEVICE__ T& Vec<N,T>::operator[](unsigned int k) {
    return this->data[k];
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ T  Vec<N,T>::operator[](unsigned int k) const {
    return this->data[k];
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator+= (const Vec<N,T> &a) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] += a.data[i];
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator-= (const Vec<N,T> &a) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] -= a.data[i];
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator*= (const Vec<N,T> &a) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] *= a.data[i];
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator%= (const Vec<N,T> &a) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = MODULO(T)(this->data[i], a.data[i]);
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator/= (const Vec<N,T> &a) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] /= a.data[i];
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator+= (T k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] += k; 
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator-= (T k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] -= k; 
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator*= (T k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] *= k; 
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator%= (T k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = MODULO(T)(this->data[i] , k); 
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> & Vec<N,T>::operator/= (T k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] /= k; 
    }
    return *this;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator+ (const Vec<N,T> &a, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] + b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator- (const Vec<N,T> &a, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] - b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator* (const Vec<N,T> &a, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] * b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator% (const Vec<N,T> &a, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = MODULO(T)(a[i], b[i]);
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator/ (const Vec<N,T> &a, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] / b[i];
    }
    return Vec<N,T>(buffer);
}


template <unsigned int N, typename T>
__HOST__ __DEVICE__ T operator| (const Vec<N,T> &a, const Vec<N,T> &b) {
    T scalarProduct(0);
    for (unsigned int i = 0; i < N; i++) {
        scalarProduct += a[i] * b[i];
    }
    return scalarProduct;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator* (const Vec<N,T> &a, T k) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] * k;
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator% (const Vec<N,T> &a, T k) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = MODULO(T)(a[i], k);
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator+ (const Vec<N,T> &a, T k) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] + k;
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator+ (T k, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = k + b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator- (const Vec<N,T> &a, T k) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] - k;
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator- (T k, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = k - b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator/ (const Vec<N,T> &a, T k) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = a[i] / k;
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator* (T k, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = k * b[i];
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator% (T k, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = MODULO(T)(k, b[i]);
    }
    return Vec<N,T>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> operator/ (T k, const Vec<N,T> &b) {
    T buffer[N];
    for (unsigned int i = 0; i < N; i++) {
        buffer[i] = k / b[i];
    }
    return Vec<N,T>(buffer);
}


template <unsigned int N, typename T>
__HOST__ __DEVICE__ bool operator!= (const Vec<N,T> &a, const Vec<N,T> &b) {
    for (unsigned int i = 0; i < N; i++) {
        if (ARE_EQUAL(T)(a[i],b[i]))
            return false;
    }
    return true;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ bool operator== (const Vec<N,T> &a, const Vec<N,T> &b) {
    for (unsigned int i = 0; i < N; i++) {
        if (!ARE_EQUAL(T)(a[i],b[i]))
            return false;
    }
    return true;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ VecBool<N> operator<= (const Vec<N,T> &a, const Vec<N,T> &b) {
    bool buffer[N];
    for (unsigned int i = 0; i < N; i++) {
       buffer[i] = ARE_EQUAL(T)(a[i],b[i]) || (a[i] < b[i]);
    }
    return VecBool<N>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ VecBool<N> operator>= (const Vec<N,T> &a, const Vec<N,T> &b) {
    bool buffer[N];
    for (unsigned int i = 0; i < N; i++) {
       buffer[i] = ARE_EQUAL(T)(a[i],b[i]) || (a[i] > b[i]);
    }
    return VecBool<N>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ VecBool<N> operator< (const Vec<N,T> &a, const Vec<N,T> &b) {
    bool buffer[N];
    for (unsigned int i = 0; i < N; i++) {
       buffer[i] = (a < b) && (!ARE_EQUAL(T)(a[i],b[i]));
    }
    return VecBool<N>(buffer);
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ VecBool<N> operator> (const Vec<N,T> &a, const Vec<N,T> &b) {
    bool buffer[N];
    for (unsigned int i = 0; i < N; i++) {
       buffer[i] = (a[i] > b[i]) && (!ARE_EQUAL(T)(a[i],b[i]));
    }
    return VecBool<N>(buffer);
}


template <unsigned int N, typename T>
__HOST__ __DEVICE__ T Vec<N,T>::normalize () {
    T norm = this->norm();
    for (unsigned int i = 0; i < N; i++) {
        data[i] /= norm;
    }
    return norm;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ Vec<N,T> Vec<N,T>::normalized () const {
    Vec<N,T> v(*this);
    v.normalize();
    return v;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ T Vec<N,T>::squaredNorm () const {
    T norm2(0);
    for (unsigned int i = 0; i < N; i++) {
        norm2 += data[i] * data[i];
    }
    return norm2;
}

template <unsigned int N, typename T>
__HOST__ __DEVICE__ T Vec<N,T>::norm () const {
    return sqrt(this->squaredNorm());
}

template <unsigned int N, typename T>
__HOST__ std::ostream & operator << (std::ostream &os, const Vec<N,T> &v) {
    os << "(";
    for (unsigned int i = 0; i < N-1; i++) {
        os << v[i] << ",";
    }
    os << v[N-1] << ")";
    
    return os;

    //for (unsigned int i = 0; i < N-1; i++) {
        //os << v[i] << " ";
    //}
    //os << v[N-1];
}
    
#endif /* end of include guard: VEC_H */
