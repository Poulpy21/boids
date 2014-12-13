
#ifndef VECBOOL_H
#define VECBOOL_H

#include "headers.hpp"
#include "vec.hpp"

#include <bitset>

/*
* Boolean vector structure of size N
* See vec.hpp for more informations 
* Boolean operators are defined.
* LSB = data[0] // MSB = data[N-1]
*/

template <unsigned int N> 
struct VecBool final : public Vec<N,bool> {
    static_assert(N <= 8*sizeof(unsigned long), "N should be less then 8*sizeof(unsigned long) !");
    
    VecBool();
    VecBool(const Vec<N,bool> &v);
    explicit VecBool(unsigned long value);
    explicit VecBool(const std::bitset<N> &bitset);
    explicit VecBool(const bool data[]);
    ~VecBool();

    void setValue(unsigned long value);
    void setValue(const std::bitset<N> &bitset);

    unsigned long to_ulong() const;

    VecBool<N> & complement();
    VecBool<N> & operator ^=  (const VecBool<N> &other);
    VecBool<N> & operator |=  (const VecBool<N> &other);
    VecBool<N> & operator &=  (const VecBool<N> &other);
    VecBool<N> & operator <<= (unsigned int k);
    VecBool<N> & operator >>= (unsigned int k);
};

template <unsigned int N>
VecBool<N>::VecBool() : Vec<N,bool>() {}

template <unsigned int N>
VecBool<N>::VecBool(unsigned long value) : Vec<N,bool>() {
    this->setValue(value);
}

template <unsigned int N>
VecBool<N>::VecBool(const std::bitset<N> &bitset) : Vec<N,bool>() {
    this->setValue(bitset);
}

template <unsigned int N>
VecBool<N>::VecBool(const Vec<N,bool> &v) : Vec<N,bool>(v) {}

template <unsigned int N>
VecBool<N>::VecBool(const bool data[]) : Vec<N,bool>(data) {}

template <unsigned int N>
VecBool<N>::~VecBool() {}

template <unsigned int N>
void VecBool<N>::setValue(unsigned long value) {
    unsigned long buffer = value;
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = buffer & 0x01;
        buffer >>= 1;
    }
}

template <unsigned int N>
void VecBool<N>::setValue(const std::bitset<N> &bitset) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = bitset.test(i);
    }
}
     
template <unsigned int N>
VecBool<N> & VecBool<N>::complement() {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = ~this->data[i];
    }
    return *this;
}

template <unsigned int N>
VecBool<N> & VecBool<N>::operator ^= (const VecBool<N> &other) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] ^= other.data[i];
    }
    return *this;
}

template <unsigned int N>
VecBool<N> & VecBool<N>::operator |= (const VecBool<N> &other) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] |= other.data[i];
    }
    return *this;
}

template <unsigned int N>
VecBool<N> & VecBool<N>::operator &= (const VecBool<N> &other) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] &= other.data[i];
    }
    return *this;
}

template <unsigned int N>
VecBool<N> & VecBool<N>::operator <<= (unsigned int k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = ( i-k > 0u ? this.data[i-k] : false);
    }
    return *this;
}

template <unsigned int N>
VecBool<N> & VecBool<N>::operator >>= (unsigned int k) {
    for (unsigned int i = 0; i < N; i++) {
        this->data[i] = ( i+k < N ? this.data[i+k] : false);
    }
    return *this;
}
    
template <unsigned int N>
unsigned long VecBool<N>::to_ulong() const {
    unsigned long val = 0;
    for (unsigned int i = 0; i < N; i++) {
        val += (this->data[i] << i);
    }
    return val;
}

template <unsigned int N>
VecBool<N> operator !(const VecBool<N> &a) {
    return VecBool<N>(~ a.to_ulong());
}

template <unsigned int N>
VecBool<N> operator ~(const VecBool<N> &a) {
    return VecBool<N>(~ a.to_ulong());
}

template <unsigned int N>
VecBool<N> operator &  (const VecBool<N> &a, const VecBool<N> &b) {
    return VecBool<N>(a.to_ulong() & b.to_ulong());
}

template <unsigned int N>
VecBool<N> operator |  (const VecBool<N> &a, const VecBool<N> &b) {
    return VecBool<N>(a.to_ulong() | b.to_ulong());
}

template <unsigned int N>
VecBool<N> operator ^  (const VecBool<N> &a, const VecBool<N> &b) {
    return VecBool<N>(a.to_ulong() ^ b.to_ulong());
}

template <unsigned int N>
VecBool<N> operator << (const VecBool<N> &a, unsigned int k) {
    return VecBool<N>(a.to_ulong() << k);
}

template <unsigned int N>
VecBool<N> operator >> (const VecBool<N> &a, unsigned int k) {
    return VecBool<N>(a.to_ulong() >> k);
}
    


#endif /* end of include guard: VECBOOL_H */
