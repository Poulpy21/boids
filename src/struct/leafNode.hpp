
#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "headers.hpp"
#include "treeNode.hpp"
#include <stdexcept>

namespace BoxTree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class LeafNode final : public TreeNode<D,N,A,T> {

            public:
                LeafNode();
                ~LeafNode();
                explicit LeafNode(const LeafNode<D,N,A,T,L,E> &other);

                bool isLeaf() const final;

                unsigned int elements() const;
                void insert(const E &e);

                void setData()

            private:
                std::shared_ptr<L> _leafData;
        };



    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode() : TreeNode<D,N,A,T>(), _leafData(nullptr) {}

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::~LeafNode() {
            delete _leafData;
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const LeafNode<D,N,A,T,L,E> &other) : TreeNode<D,N,A,T>(other), _leafData(other._leafData) {
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool LeafNode<D,N,A,T,L,E>::isLeaf() const {
            return true;
        }
                
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int LeafNode<D,N,A,T,L,E>::elements() const {
            if(_leafData == nullptr)
                return 0u;
            else
                return _leafData->elements();
        }
                
    template <unsigned int D, unsigned int N, typename A, typename T>
        void insert(const E &e) {
            if(_leafData == nullptr)
                throw new std::runtime_error("Trying to insert in leaf but leafData is not initialized !");
            else
                _leafData->insert(e);
        }
    }

#endif /* end of include guard: LEAFNODE_H */
