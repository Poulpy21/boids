
#ifndef ROOTNODE_H
#define ROOTNODE_H

#include "headers.hpp"
#include "treeNode.hpp"
#include "leafNode.hpp"
#include "abstractContainer.hpp"
#include "localized.hpp"

namespace Tree {

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class LeafNode;

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode;

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        using BoxTree = RootNode<D,N,A,T,L,E>;

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode : public TreeNode<D,N,A,T> {
            
            static_assert(D>=1u, "D should be greater than 0 !");
            static_assert(N<=32u, "N should be less than 32 !");

            static_assert(std::is_arithmetic<A>(), "A should be arithmetic !");
            static_assert(std::is_default_constructible<A>(), "Arithmetic type A should be default constructible !");
            static_assert(std::is_copy_constructible<A>(), "Arithmetic type A should be copy constructible !");
            static_assert(std::is_constructible<A, int>(), "Arithmetic type A should be constructible from int !");
            static_assert(std::is_assignable<A&,A>(), "Arithmetic type A should be assignable !");
            
            static_assert(std::is_default_constructible<T>(), "Trunk type T should be default constructible !");
            static_assert(std::is_copy_constructible<T>(), "Trunk type T should be copy constructible !");
            
            static_assert(std::is_default_constructible<L>(), "Leaf type L should be default constructible !");
            static_assert(std::is_copy_constructible<L>(), "Leaf type L should be copy constructible !");
            static_assert(std::is_base_of<AbstractContainer<E>,L>(), "Leaf type L should inherit AbstractContainer<E>");

            static_assert(std::is_default_constructible<E>(), "Element type E should be default constructible !");
            static_assert(std::is_copy_constructible<E>(), "Element type E should be copy constructible !");
            static_assert(std::is_base_of<Localized<D,A>,E>(), "Element type E should inherit Localized<D,A>");

            public:
                RootNode(const RootNode<D,N,A,T,L,E> &other);
                virtual ~RootNode();

                void insert(E &e);
                void insert(AbstractContainer<E> &container);

                bool isRoot() const final;

                virtual unsigned int targetChild(const TreeNode<D,N,A,T> &node, const E &e) const = 0;
                virtual TreeNode<D,N,A,T>& splitLeaf(std::shared_ptr<LeafNode<D,N,A,T,L,E>> leaf) = 0;
                virtual LeafNode<D,N,A,T,L,E>&  mergeChilds(std::shared_ptr<TreeNode<D,N,A,T>> father) = 0;

                virtual void onPreLeafSplit(LeafNode<D,N,A,T,L,E> &leaf) {}
                virtual void onPostLeafSplit(TreeNode<D,N,A,T> &father) {}
                virtual void onPreInsert(const E &e, LeafNode<D,N,A,T,L,E> &leaf) {}
                virtual void onPostInsert(const E &e, LeafNode<D,N,A,T,L,E> &leaf) {}

            protected:
                explicit RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf, float fillThreshold);

                unsigned int _maxElementsPerLeaf;
                float _fillThreshold;
        };


    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf, float fillThreshold) : 
            TreeNode<D,N,A,T>(domain),  
            _maxElementsPerLeaf(maxElementsPerLeaf),
            _fillThreshold(fillThreshold) {
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::~RootNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const RootNode<D,N,A,T,L,E> &other):
            TreeNode<D,N,A,T>(other), 
            _maxElementsPerLeaf(other._maxElementsPerLeaf),
            _fillThreshold(other._fillThreshold) {
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        bool RootNode<D,N,A,T,L,E>::isRoot() const {
            return true;
        }

template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::insert(E &e) {

            TreeNode<D,N,A,T> &buffer = *this;

            while(!buffer.isLeaf()) {
                //std::cout << "targetChild is " << targetChild(buffer,e) << std::endl;
                buffer = buffer[targetChild(buffer, e)];
            }

            std::shared_ptr<LeafNode<D,N,A,T,L,E>> leaf;
            leaf.reset(dynamic_cast<LeafNode<D,N,A,T,L,E>*>(&buffer));

            //leaf container is already full
            if(leaf->elements() == _maxElementsPerLeaf) {    
                onPreLeafSplit(*leaf);
                buffer = splitLeaf(leaf);
                leaf.reset(dynamic_cast<LeafNode<D,N,A,T,L,E>*>(&buffer[targetChild(buffer, e)]));
                onPostLeafSplit(buffer);
            }

            onPreInsert(e, *leaf);
            leaf->insert(e);
            onPostInsert(e, *leaf);
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::insert(AbstractContainer<E> &container) {
            for (unsigned int i = 0; i < container.elements(); i++) {
                this->insert(container[i]);
            }
        }
}


#endif /* end of include guard: LEAFNODE_H */
