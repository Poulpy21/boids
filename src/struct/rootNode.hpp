
#ifndef LEAFNODE_H
#define LEAFNODE_H

#include "headers.hpp"
#include "treeNode.hpp"
#include "leafNode.hpp"
#include "abstractContainer.hpp"

namespace BoxTree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class LeafNode;
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode;
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
    using BoxTree = RootNode<D,N,A,T,L,E>;

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode final : public TreeNode<D,N,A,T> {

            public:
                explicit RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf);
                explicit RootNode(const RootNode<D,N,A,T,L,E> &other);
                virtual ~RootNode();
    
                void insert(const E &e);
                void insert(const AbstractContainer<E> &container);

                //void balance();
                
                virtual void onPreLeafSplit(LeafNode<D,N,A,T,L,E> &leaf) {};
                virtual void onPostLeafSplit(TreeNode<D,N,A,T> &trunk) {};

                virtual void onPreInsert(LeafNode<D,N,A,T,L,E> &leaf, const E &e) {};
                virtual void onPostInsert(LeafNode<D,N,A,T,L,E> &leaf, const E &e) {};

                virtual unsigned int targetChild(const TreeNode<D,N,A,T> &node, const E &e) = 0;
                virtual TreeNode<D,N,A,T>& splitLeaf(LeafNode<D,N,A,T,L,E> &leaf)    = 0;
                virtual void mergeChilds()  = 0;
                
                bool isRoot() const final;

            private:
                unsigned int _maxElementsPerLeaf;
        };


    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf) : 
            TreeNode<D,N,A,T>(domain),  _maxElementsPerLeaf(maxElementsPerLeaf) {

                //create initialize root childs
                this->splitLeaf();
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::~RootNode() {}

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const RootNode<D,N,A,T,L,E> &other):
            TreeNode<D,N,A,T>(other), _maxElementsPerLeaf(other._maxElementsPerLeaf) {}
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
    bool RootNode<D,N,A,T,L,E>::isRoot() const {
        return true;
    }
                
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
    void RootNode<D,N,A,T,L,E>::insert(const E &e) {
        TreeNode<D,N,A,T> &buffer = *this;

        while(!buffer.isLeaf())
            buffer = buffer[targetChild(buffer, e)];

        LeafNode<D,N,A,T,L,E> &leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>&>(buffer);

        //leaf container is already full
        if(leaf.elements() == _maxElementsPerLeaf) {    
            onPreLeafSplit(leaf);
            buffer = splitLeaf(leaf);
            leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>>(buffer[targetChild(buffer, e)]);
            onPostLeafSplit(buffer);
        }

        onPreInsert(leaf, e);
        leaf.insert(e);
        onPostInsert(leaf, e);
    }
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
    void RootNode<D,N,A,T,L,E>::insert(const AbstractContainer<E> &container) {
        for (unsigned int i = 0; i < container.elements(); i++) {
            this->insert(container[i]);
        }
    }
    
    //template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
    //void RootNode<D,N,A,T,L,E>::balance() {
    //}
}


#endif /* end of include guard: LEAFNODE_H */
