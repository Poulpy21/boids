
#ifndef HYPERCUBETREE_H
#define HYPERCUBETREE_H

#include "headers.hpp"
#include "rootNode.hpp"

namespace BoxTree {

    template <unsigned int D, typename A, typename T, typename L, typename E>
        class HyperCubeTree : public BoxTree<D,utils::get_power_of_two<unsigned int>(D), A, T, L, E> {
        };

}

#endif /* end of include guard: HYPERCUBETREE_H */
