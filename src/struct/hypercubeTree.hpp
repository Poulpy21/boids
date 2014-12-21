
#ifndef HYPERCUBETREE_H
#define HYPERCUBETREE_H

#include "headers.hpp"
#include "rootNode.hpp"
#include "hypercube.hpp"

namespace BoxTree {

#define _N_ utils::get_power_of_two(D)

    template <unsigned int D, typename A, typename T, typename L, typename E>
        class HyperCubeTree : public BoxTree<D,utils::get_power_of_two<unsigned int>(D), A, T, L, E> {

            public:
                static constexpr unsigned int N = utils::get_power_of_two<unsigned int>(D);

                HyperCubeTree();
                HyperCubeTree(unsigned int maxElementsPerLeaf, float fillThreshHold, const HyperCube<D,A> &cube);
                HyperCubeTree(const HyperCubeTree<D,A,T,L,E> &other);
                virtual ~HyperCubeTree();
                
                virtual unsigned int targetChild(const TreeNode<D,N,A,T> &node, const E &e);
                virtual TreeNode<D,N,A,T>& splitLeaf(LeafNode<D,N,A,T,L,E> &leaf);
                virtual void mergeChilds();
        };

}

#endif /* end of include guard: HYPERCUBETREE_H */
