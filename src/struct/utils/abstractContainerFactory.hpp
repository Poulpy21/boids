
#ifndef ABSTRACTCONTAINERFACTORY_H
#define ABSTRACTCONTAINERFACTORY_H

/*
 * C should inherit from AbstractContainer<E>
 * E are the elements container in the container
 * ContainerArgs... may be additional arguments for the Container construction (optional)
 */

template <typename C, typename E, typename... ContainerArgs>
class AbstractContainerFactory {
    static_assert(std::is_base_of<AbstractContainer<E>, C>(), "The container should inherit from AbstractContainer<E>  !");
    static_assert(std::is_constructible<C, unsigned int, ContainerArgs...>(), "The container should be constructible from C(unsigned int nMinElements, ContainerArgs...) !");

    AbstractContainerFactory() {}
    virtual ~AbstractContainerFactory() {}

    //creates and allocate a container of type C of elements E
    virtual C* operator()(unsigned int nMinElements, ContainerArgs... args) const = 0;
};

#endif /* end of include guard: CONTAINERFACTORY_H */
