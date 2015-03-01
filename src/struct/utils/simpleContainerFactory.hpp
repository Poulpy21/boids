#ifndef SIMPLECONTAINERFACTORY_H
#define SIMPLECONTAINERFACTORY_H

#include "abstractContainerFactory.hpp"

/*
 * C should inherit from AbstractContainer<E>
 * E are the elements container in the container
 */

template <typename C, typename E, typename... ContainerArgs>
class SimpleContainerFactory final : public AbstractContainerFactory<C,E,ContainerArgs...> {

    static_assert(std::is_base_of<AbstractContainer<E>, C>(), "The container should inherit from AbstractContainer<E>  !");
    static_assert(std::is_constructible<C, unsigned int, ContainerArgs...>(), "The container should be constructible from C(unsigned int nMinElements, ContainerArgs...) !");

    SimpleContainerFactory() {}
    ~SimpleContainerFactory() {}

    //creates and allocate a container of type C of elements E
    C* operator()(unsigned int nMinElements, ContainerArgs... args) const {
        return new C(nMinElements, args...);
    }
};

#endif /* end of include guard: SIMPLECONTAINERFACTORY_H */
