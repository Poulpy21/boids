
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
                explicit LeafNode(const BoundingBox<D,A> &domain, unsigned int level, unsigned int id,
                        unsigned int maxData, TreeNode<D,N,A,T> *father);
                ~LeafNode();

                bool isLeaf() const final;

                L& data() ;
                unsigned int elements() const;

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
        LeafNode<D,N,A,T,L,E>::LeafNode(const BoundingBox<D,A> &domain, unsigned int level, unsigned int id,
                unsigned int maxData, TreeNode<D,N,A,T> *father) :
            TreeNode<D,N,A,T>(domain, level, id),
            _leafData(), _maxData(maxData) {
                this->_father = father;
                _leafData.allocate(maxData);
                
#ifdef GUI_ENABLED
                Vec<D,A> dx = domain.max - domain.min;
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
        L& LeafNode<D,N,A,T,L,E>::data() {
            return _leafData;
        }

#ifdef GUI_ENABLED
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void LeafNode<D,N,A,T,L,E>::drawDownwards(const float *currentTransformationMatrix) {
            static_assert(D==3, "Tree display only possible in dimension 3 !");
                    
            this->_drawBoxProgram->use();

            glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);

            if(Globals::wireframe)
                glBindBuffer(GL_ARRAY_BUFFER, this->_wireCubeVBO);           
            else
                glBindBuffer(GL_ARRAY_BUFFER, this->_cubeVBO);           

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
                        
            glUniformMatrix4fv(this->_drawBoxUniformLocs["modelMatrix"], 1, GL_TRUE, this->relativeModelMatrix);
            glUniform1i(this->_drawBoxUniformLocs["level"], this->_level);
            glUniform1f(this->_drawBoxUniformLocs["fillrate"], this->_leafData.fillRate());
         
            if(this->level() >= Globals::minTreeLevelDisplay) {
                if(Globals::wireframe) {
                    for (unsigned int i = 0; i < 6; i++) {
                        glDrawArrays(GL_LINE_LOOP,4*i,4);
                    }
                }
                else {
                    glDrawArrays(GL_TRIANGLES, 0, this->_nTriangles*3);
                }
            }

            _leafData.template drawDownwards<float>(currentTransformationMatrix);
    }
#endif /* GUI_ENABLED */

}

#endif /* end of include guard: LEAFNODE_H */
