
#ifndef LOCALIZED_H
#define LOCALIZED_H

#include "headers.hpp"

template <unsigned int D, typename A>
struct Localized {
    Localized() {};
    virtual ~Localized() {};

    virtual Vec<D,A> position() const = 0;
};


#endif /* end of include guard: LOCALIZED_H */
