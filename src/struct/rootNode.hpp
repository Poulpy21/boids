
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

                void insert(const E &e);
                void insert(const AbstractContainer<E> &container);

                bool isRoot() const final;

                virtual unsigned int targetChild(const TreeNode<D,N,A,T> *node, const E &e) const = 0;
                virtual TreeNode<D,N,A,T>* splitLeaf(LeafNode<D,N,A,T,L,E>* &leaf) = 0;
                virtual LeafNode<D,N,A,T,L,E>* mergeChilds(TreeNode<D,N,A,T>* &father) = 0;

                virtual void onPreLeafSplit(LeafNode<D,N,A,T,L,E> *leaf) {}
                virtual void onPostLeafSplit(TreeNode<D,N,A,T> *father) {}
                virtual void onPreInsert(const E &e, LeafNode<D,N,A,T,L,E> *leaf) {}
                virtual void onPostInsert(const E &e, LeafNode<D,N,A,T,L,E> *leaf) {}
      
            protected:
                explicit RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf, float fillThreshold);

                unsigned int _maxElementsPerLeaf;
                float _fillThreshold;

#ifdef GUI_ENABLED
            public:
                virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
                virtual void drawUpwards(const float *currentTransformationMatrix);
#endif
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
        void RootNode<D,N,A,T,L,E>::insert(const E &e) {

            TreeNode<D,N,A,T> *buffer = this;

            unsigned int level = 0u;
            while(!buffer->isLeaf()) {
                buffer = buffer->child(targetChild(buffer, e));
            }
            
            LeafNode<D,N,A,T,L,E> *leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>*>(buffer);

            //leaf container is already full
            if(leaf->elements() == _maxElementsPerLeaf) {    
                onPreLeafSplit(leaf);
                buffer = splitLeaf(leaf);
                onPostLeafSplit(buffer);
                leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>*>(buffer->child(targetChild(buffer, e)));
            }

            onPreInsert(e, leaf);
            leaf->insert(e);
            onPostInsert(e, leaf);
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::insert(const AbstractContainer<E> &container) {
            for (unsigned int i = 0; i < container.elements(); i++) {
                this->insert(container[i]);
            }
        }

#ifdef GUI_ENABLED
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
                void RootNode<D,N,A,T,L,E>::drawDownwards(const float *currentTransformationMatrix) {
                    
                    this->_drawBoxProgram->use();

                    glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
                    glUniformMatrix4fv(this->_drawBoxUniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);


                    glBindBuffer(GL_ARRAY_BUFFER, this->_cubeVBO);           
                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

                    for (unsigned int i = 0; i < N; i++) {
                        this->_childs[i]->drawDownwards(currentTransformationMatrix);
                    }
                }
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
                void RootNode<D,N,A,T,L,E>::drawUpwards(const float *currentTransformationMatrix) {
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    glUseProgram(0);
                }
#endif

}


#endif /* end of include guard: LEAFNODE_H */
