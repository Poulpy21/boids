
#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

/* 
 * N-D Bounding box 
 * N is the dimension of the box
 * T is the vectorial position type
 */ 

#include "headers.hpp"
#include <bitset>

template <unsigned int N, typename T>
struct BoundingBox {
    static_assert(std::is_arithmetic<T>(),         "Positions can only contain arithmetic types !");
    static_assert(std::is_assignable<T&, T>(),     "T should be assignable !");
    static_assert(std::is_constructible<T, int>(), "T should be constructible from int !");

    BoundingBox();
    BoundingBox(const BoundingBox<N,T> &other);
    explicit BoundingBox(const Vec<N,T> &min, const Vec<N,T> &max);
    template <typename S> 
    explicit BoundingBox(const BoundingBox<N,S> &v);
    virtual ~BoundingBox();
    
    BoundingBox<N,T>& operator= (const BoundingBox<N,T> &v);

    Vec<N,T> center() const;
    Vec<N,T> corner(unsigned long id) const;
    Vec<N,T> corner(const std::bitset<N> &bitset) const;

    Vec<N,T> min;
    Vec<N,T> max;
};

template <unsigned int N, typename T>
BoundingBox<N,T>::BoundingBox() : min(), max() {}

template <unsigned int N, typename T>
BoundingBox<N,T>::BoundingBox(const BoundingBox<N,T> &other) :
    min(other.min), max(other.max) {
    }

template <unsigned int N, typename T>
BoundingBox<N,T>::BoundingBox(const Vec<N,T> &min, const Vec<N,T> &max) :
    min(min), max(max) {
    }

template <unsigned int N, typename T>
BoundingBox<N,T>& BoundingBox<N,T>::operator= (const BoundingBox<N,T> &v) {
    this->min = v.min;
    this->max = v.max;
}

template <unsigned int N, typename T>
BoundingBox<N,T>::~BoundingBox() {}

template <unsigned int N, typename T>
template <typename S>
BoundingBox<N,T>::BoundingBox(const BoundingBox<N,S> &v) {
    this->min = Vec<N,T>(v.min);
    this->max = Vec<N,T>(v.max);
}

template <unsigned int N, typename T>
Vec<N,T> BoundingBox<N,T>::center() const {
    return (max+min)/T(2);
}

template <unsigned int N, typename T>
Vec<N,T> BoundingBox<N,T>::corner(unsigned long id) const {
    VecBool<N> dir(id);
    return (Vec<N,T>(dir)*max + Vec<N,T>(~dir)*min)/T(2);
}

template <unsigned int N, typename T>
Vec<N,T> BoundingBox<N,T>::corner(const std::bitset<N> &bitset) const {
    VecBool<N> dir(bitset);
    return (Vec<N,T>(dir)*max + Vec<N,T>(~dir)*min)/T(2);
}

#endif /* end of include guard: BOUNDINGBOX_H */
