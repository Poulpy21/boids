
#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "headers.hpp"
#include "treeNode.hpp"
#include <stdexcept>

namespace Tree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class LeafNode final : public TreeNode<D,N,A,T> {

            public:
                LeafNode();
                LeafNode(const LeafNode<D,N,A,T,L,E> &other);
                explicit LeafNode(const BoundingBox<D,A> &domain, std::shared_ptr<L> data = nullptr);
                ~LeafNode();

                bool isLeaf() const final;

                unsigned int elements() const;
                void insert(const E &e);

                std::shared_ptr<L> data();

            private:
                std::shared_ptr<L> _leafData;
        };



    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode() : TreeNode<D,N,A,T>(), _leafData(nullptr) {}

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::~LeafNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const LeafNode<D,N,A,T,L,E> &other) : 
            TreeNode<D,N,A,T>(other), _leafData(other._leafData) {
            }
                
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const BoundingBox<D,A> &domain, std::shared_ptr<L> data) :
            TreeNode<D,N,A,T>(domain), _leafData(data) {
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        bool LeafNode<D,N,A,T,L,E>::isLeaf() const {
            return true;
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        unsigned int LeafNode<D,N,A,T,L,E>::elements() const {
            if(_leafData == nullptr)
                return 0u;
            else
                return _leafData->elements();
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void LeafNode<D,N,A,T,L,E>::insert(const E &e) {
            if(_leafData == nullptr)
                throw new std::runtime_error("Trying to insert in leaf but leafData is not initialized !");
            else
                _leafData->insert(e);
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        std::shared_ptr<L> LeafNode<D,N,A,T,L,E>::data() {
            return _leafData;
        }
}

#endif /* end of include guard: LEAFNODE_H */
