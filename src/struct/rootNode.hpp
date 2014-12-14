
#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "headers.hpp"
#include "treeNode.hpp"

namespace Tree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode final : public TreeNode<D,N,A,T> {

            public:
                RootNode();
                ~RootNode();
                explicit RootNode(const RootNode<D,N,A,T,L,E> &other);
    
                bool isRoot() final const;

            private:
                L _leafData;
        };



    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode() : TreeNode<D,N,A,T>(), _leafData() {}

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::~RootNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const RootNode<D,N,A,T,L,E> &other) : TreeNode<D,N,A,T>(other), _leafData(other._leafData) {
        }
    
    template <unsigned int D, unsigned int N, typename A, typename T>
    bool TreeNode<D,N,A,T>::isRoot() const {
        return true;
    }
}


#endif /* end of include guard: LEAFNODE_H */
