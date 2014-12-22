

#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include <stdexcept>

/* 
 * D-HyperCube
 * D is the Dimension of the hypercube
 * A is the Arithmetic position type
 */ 

template <unsigned int D, typename A>
struct HyperCube : public BoundingBox<D,A> {
    
    HyperCube();
    HyperCube(const HyperCube<D,A> &other);
    explicit HyperCube(const Vec<D,A> &min, const Vec<D,A> &max);
    
    template <typename S> 
    explicit HyperCube(const HyperCube<D,S> &v);
    
    virtual ~HyperCube();

    HyperCube<D,A>& operator= (const HyperCube<D,A> &v);

    HyperCube<D,A> generateSubcube(unsigned long cubeId);

    virtual std::string toString() const;
};

template <unsigned int D, typename A>
HyperCube<D,A>::HyperCube() : BoundingBox<D,A>() {}

template <unsigned int D, typename A>
HyperCube<D,A>::HyperCube(const HyperCube<D,A> &other) : 
    BoundingBox<D,A>(other) {}

template <unsigned int D, typename A>
HyperCube<D,A>::HyperCube(const Vec<D,A> &min, const Vec<D,A> &max) :
    BoundingBox<D,A>(min, max) {
        Vec<D,A> dv = max - min;
        A val(dv[0]);

        for (unsigned int i = 1u; i < D; i++) {
            if(!utils::areEqual<A>(val, dv[i])) 
                    throw std::runtime_error("Trying to construct an hypercube with something that is not a cube !");
        }
}

template <unsigned int D, typename A>
HyperCube<D,A>& HyperCube<D,A>::operator= (const HyperCube<D,A> &v) {
    this->min = v.min;
    this->max = v.max;
}

template <unsigned int D, typename A>
HyperCube<D,A>::~HyperCube() {}

template <unsigned int D, typename A>
template <typename S>
HyperCube<D,A>::HyperCube(const HyperCube<D,S> &v) {
    this->min = Vec<D,A>(v.min);
    this->max = Vec<D,A>(v.max);
}
    
template <unsigned int D, typename A>
HyperCube<D,A> HyperCube<D,A>::generateSubcube(unsigned long cubeId) {
        
    Vec<D,A> size = (this->max - this->min)/A(2);
    Vec<D,A> origin = Vec<D,A>(VecBool<D>(cubeId)) * size;

    return HyperCube<D,A>(origin, origin + size);
}

template <unsigned int D, typename A>
std::string HyperCube<D,A>::toString() const {
    std::stringstream ss;
    ss << "HyperCube<" << D << ",";
    utils::templatePrettyPrint<A>(os);
    ss << ">" << std::endl;
    ss << "\tMin: " << this->min << std::endl;
    ss << "\tMax: " << this->max << std::endl;
    return ss.str();
}


#endif /* end of include guard: HYPERCUBE_H */
