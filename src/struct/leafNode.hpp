
#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "headers.hpp"
#include "treeNode.hpp"

namespace Tree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class LeafNode final : public TreeNode<D,N,A,T> {

            public:
                LeafNode();
                ~LeafNode();
                explicit LeafNode(const LeafNode<D,N,A,T,L,E> &other);

                bool isLeaf() final const;

            private:
                L _leafData;
        };



    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode() : TreeNode<D,N,A,T>(), _leafData() {}

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::~LeafNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const LeafNode<D,N,A,T,L,E> &other) : TreeNode<D,N,A,T>(other), _leafData(other._leafData) {
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool TreeLeaf<D,N,A,T,L,E>::isLeaf() const {
            return true;
        }
}


#endif /* end of include guard: LEAFNODE_H */
