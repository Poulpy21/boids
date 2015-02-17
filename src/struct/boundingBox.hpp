
#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

/* 
 * D-d Bounding box 
 * D is the Dimension of the box
 * A is the Arithmetic position type
 */ 

#include "headers.hpp"
#include "vec.hpp"

#include <bitset>

template <unsigned int D, typename A>
struct BoundingBox {

    static_assert(D != 0u,                         "Dimension can't be zero !");
    static_assert(std::is_arithmetic<A>(),         "Positions can only contain arithmetic types !");
    static_assert(std::is_assignable<A&, A>(),     "A should be assignable !");
    static_assert(std::is_constructible<A, int>(), "A should be constructible from int !");

     BoundingBox();
     BoundingBox(const BoundingBox<D,A> &other);
     explicit BoundingBox(const Vec<D,A> &min, const Vec<D,A> &max);
    
    template <typename S> 
     explicit BoundingBox(const BoundingBox<D,S> &v);
     virtual ~BoundingBox();
    
     BoundingBox<D,A>& operator= (const BoundingBox<D,A> &v);

     Vec<D,A> center() const;
     Vec<D,A> corner(unsigned long id) const;

    Vec<D,A> corner(const std::bitset<D> &bitset) const;
    virtual std::string toString() const;

    Vec<D,A> min;
    Vec<D,A> max;

};

template <unsigned int D, typename A>
 BoundingBox<D,A>::BoundingBox() : min(), max() {}

template <unsigned int D, typename A>
 BoundingBox<D,A>::BoundingBox(const BoundingBox<D,A> &other) :
    min(other.min), max(other.max) {
    }

template <unsigned int D, typename A>
 BoundingBox<D,A>::BoundingBox(const Vec<D,A> &min, const Vec<D,A> &max) :
    min(min), max(max) {
        for (unsigned int i = 0; i < D; i++) {
            if(min[i] > max[i])
                throw std::runtime_error("Trying to construct an hypercube with something that is not a cube !");
        }
    }

template <unsigned int D, typename A>
 BoundingBox<D,A>& BoundingBox<D,A>::operator= (const BoundingBox<D,A> &v) {
    this->min = v.min;
    this->max = v.max;
    return *this;
}

template <unsigned int D, typename A>
 BoundingBox<D,A>::~BoundingBox() {}

template <unsigned int D, typename A>
template <typename S>
 BoundingBox<D,A>::BoundingBox(const BoundingBox<D,S> &v) {
    this->min = Vec<D,A>(v.min);
    this->max = Vec<D,A>(v.max);
}

template <unsigned int D, typename A>
 Vec<D,A> BoundingBox<D,A>::center() const {
    return (max+min)/A(2);
}

template <unsigned int D, typename A>
 Vec<D,A> BoundingBox<D,A>::corner(unsigned long id) const {
    VecBool<D> dir(id);
    return (Vec<D,A>(dir)*max + Vec<D,A>(~dir)*min)/A(2);
}

#ifndef __CUDACC__
template <unsigned int D, typename A>
Vec<D,A> BoundingBox<D,A>::corner(const std::bitset<D> &bitset) const {
    VecBool<D> dir(bitset);
    return (Vec<D,A>(dir)*max + Vec<D,A>(~dir)*min)/A(2);
}
    
template <unsigned int D, typename A>
std::string BoundingBox<D,A>::toString() const {
    std::stringstream ss;
    ss << "BoundingBox<" << D << ",";
    utils::templatePrettyPrint<A>(ss);
    ss << ">" << std::endl;
    ss << "\tMin: " << min << std::endl;
    ss << "\tMax: " << max << std::endl;
    return ss.str();
}

template <unsigned int D, typename A>
    std::ostream & operator <<(std::ostream &os, const BoundingBox<D,A> &bbox) {
        os << bbox.toString();
        return os;
}
#endif /*end ifndef __CUDACC__*/


#endif /* end of include guard: BOUNDINGBOX_H */

