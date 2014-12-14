
#ifndef LOCALIZED_H
#define LOCALIZED_H

#include "headers.hpp"

template <unsigned int N, typename T>
class Localized {
    Localized() {};
    virtual ~Localized() {};

    virtual Vec<N,T> getPosition() const = 0;
};


#endif /* end of include guard: LOCALIZED_H */
