
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
                explicit LeafNode(const BoundingBox<D,A> &domain, unsigned int maxData, TreeNode<D,N,A,T> *father, unsigned int childId);
                ~LeafNode();

                bool isLeaf() const final;

                unsigned int elements() const;
                void insert(E e);

                L& data() ;

            private:
                L _leafData;
                unsigned int _maxData;

#ifdef GUI_ENABLED
            public:
                void drawDownwards(const float *currentTransformationMatrix) override;
#endif

        };


    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode() : TreeNode<D,N,A,T>(), _leafData() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::~LeafNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const LeafNode<D,N,A,T,L,E> &other) : 
            TreeNode<D,N,A,T>(other), _leafData(other._leafData), _maxData(other._maxData) {
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        LeafNode<D,N,A,T,L,E>::LeafNode(const BoundingBox<D,A> &domain, unsigned int maxData, TreeNode<D,N,A,T> *father, unsigned int childId) :
            TreeNode<D,N,A,T>(domain), _leafData(), _maxData(maxData) {
                this->_father = father;
                _leafData.allocate(maxData);
                
#ifdef GUI_ENABLED
                Vec<D,A> dx = domain.max - domain.min;
                std::cout << "Creating leaf node !";
                std::cout << domain << std::endl;
                this->move(domain.center());
                this->scale(dx[0], dx[1], dx[2]);
#endif
            }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        bool LeafNode<D,N,A,T,L,E>::isLeaf() const {
            return true;
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        unsigned int LeafNode<D,N,A,T,L,E>::elements() const {
            return _leafData.elements();
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void LeafNode<D,N,A,T,L,E>::insert(E e) {
            _leafData.insert(e);
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        L& LeafNode<D,N,A,T,L,E>::data() {
            return _leafData;
        }

#ifdef GUI_ENABLED
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void LeafNode<D,N,A,T,L,E>::drawDownwards(const float *currentTransformationMatrix) {
            static_assert(D==3, "Tree display only possible in dimension 3 !");
	
            std::cout << "Leaf Node draw" << std::endl;
            glUniformMatrix4fv(this->_drawBoxUniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
            glDrawArrays(GL_TRIANGLES, 0, this->_nTriangles*3);
        }
#endif /* GUI_ENABLED */

}

#endif /* end of include guard: LEAFNODE_H */
