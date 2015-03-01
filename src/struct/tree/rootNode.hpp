
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
                virtual ~RootNode();

                // Leaf monitoring

                // Element management
                AbstractContainer<E> & getLeafContainer(unsigned long leafId);

                void insertElement(const E &e);
                void insertElements(const AbstractContainer<E> &container);

                void removeElement(unsigned long leafId, unsigned int elementId);
                void removeElements(unsigned long leafId, unsigned int minElementId, unsigned int maxElementId);
                void removeElements(unsigned long leafId, unsigned int nElements, unsigned int *elementIds);
               
                void moveElement(unsigned long fromNodeId, unsigned long toNodeId, unsigned int oldElementId);
                void moveElements(unsigned long fromNodeId, unsigned long toNodeId, unsigned int oldMinElementId, unsigned int oldMaxElementId);
                void moveElements(unsigned long fromNodeId, unsigned long toNodeId, unsigned int nElements, unsigned int *oldElementIds);

                //void overwriteElement(unsigned long fromLeafId, unsigned int elementId, const E &e);
                //void exchangeElement(unsigned long firstLeafId, unsigned long secondLeafId, unsigned int fistElementId, unsigned int secondElementId);
                ////////////////////////////////
                
            protected:
                explicit RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf, float fillThreshold);
                RootNode(const RootNode<D,N,A,T,L,E> &other);
                
                bool isRoot() const final;

                LeafNode<D,N,A,T,L,E>* getLeaf(unsigned long leafId);
                
                virtual unsigned int targetChild(const TreeNode<D,N,A,T> *node, const E &e) const = 0;
                virtual TreeNode<D,N,A,T>* splitLeaf(LeafNode<D,N,A,T,L,E>* &leaf) = 0;
                virtual LeafNode<D,N,A,T,L,E>* mergeChilds(TreeNode<D,N,A,T>* &father) = 0;
                
                virtual void onLeafSplit(TreeNode<D,N,A,T> *oldleaf) {}
                virtual void onChildMerge(TreeNode<D,N,A,T> *father) {};

                virtual void onInsert(const E &e, LeafNode<D,N,A,T,L,E> *leaf) {}
                virtual void onDelete(const E &e, LeafNode<D,N,A,T,L,E> *leaf) {}

                unsigned int _maxElementsPerLeaf;
                float _fillThreshold;

#ifdef GUI_ENABLED
            public:
                virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) override;
                virtual void drawUpwards(const float *currentTransformationMatrix) override;
                virtual void keyPressEvent(QKeyEvent*) override;
#endif
        };


    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        RootNode<D,N,A,T,L,E>::RootNode(const BoundingBox<D,A> &domain, unsigned int maxElementsPerLeaf, float fillThreshold) : 
            TreeNode<D,N,A,T>(domain, 0u, 0u),  
            _maxElementsPerLeaf(maxElementsPerLeaf),
            _fillThreshold(fillThreshold) {
                log4cpp::log_console->debugStream() << "Creating rootnode ! Max tree level is " << this->_maxLevel << " !";
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
    LeafNode<D,N,A,T,L,E>* RootNode<D,N,A,T,L,E>::getLeaf(unsigned long leafId) {
            
        TreeNode<D,N,A,T> *buffer = this;
    
        unsigned int ids[this->_maxLevel];
        unsigned long id = leafId;
        int i = 0; 

        while(id > 1ul) {
            ids[i++] = id%N;
            id /= N;
        }

        while(i-- > 0 && !buffer->isLeaf()) {
            buffer = buffer->child(ids[i]);
        }

        assert(buffer->isLeaf());
    }

template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::insertElement(const E &e) {

            TreeNode<D,N,A,T> *buffer = this;

            while(!buffer->isLeaf()) {
                buffer = buffer->child(targetChild(buffer, e));
            }
            
            LeafNode<D,N,A,T,L,E> *leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>*>(buffer);

            //leaf container is already full
            while(leaf->elements() == _maxElementsPerLeaf) {    

                if(leaf->level() > this->_maxLevel) {
                    log4cpp::log_console->warnStream() << "Max tree level achieved, you are inserting too many elements at the same location ! Insertion aborted !";
                    return;
                }

                buffer = splitLeaf(leaf);
                onLeafSplit(buffer);
                leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>*>(buffer->child(targetChild(buffer, e)));
            }
            
            onInsert(e, leaf);
            leaf->data().insert(e);
        }

    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::insertElements(const AbstractContainer<E> &container) {
            for (unsigned int i = 0; i < container.elements(); i++) {
                this->insertElement(container[i]);
            }
        }

template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        void RootNode<D,N,A,T,L,E>::removeElement(unsigned long leafId, unsigned int elementId) {
            this->getLeaf(leafId)->remove(elementId);
        }

#ifdef GUI_ENABLED
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
                void RootNode<D,N,A,T,L,E>::drawDownwards(const float *currentTransformationMatrix) {
                    
                    static_assert(D==3, "Tree display only possible in dimension 3 !");

                    glEnable(GL_POINT_SMOOTH);
                    glEnable(GL_LINE_SMOOTH);
                    glEnable(GL_POINT_SPRITE);
                    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

                    for (unsigned int i = 0; i < N; i++) {
                        this->_childs[i]->drawDownwards(currentTransformationMatrix);
                    }
                    
                    glDisable(GL_POINT_SMOOTH);
                    glDisable(GL_LINE_SMOOTH);
                    glDisable(GL_POINT_SPRITE);
                    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
                }
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
                void RootNode<D,N,A,T,L,E>::drawUpwards(const float *currentTransformationMatrix) {
                    static_assert(D==3, "Tree display only possible in dimension 3 !");

                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    glUseProgram(0);
                }
                
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
                void RootNode<D,N,A,T,L,E>::keyPressEvent(QKeyEvent* e) {
                
                    if (e->key() == Qt::Key_X && e->modifiers() == Qt::NoButton) {
                        if(Globals::wireframe) {
                            log_console->infoStream() << "Tree switched in plain mode !";
                            Globals::wireframe = false;
                        }
                        else {
                            log_console->infoStream() << "Tree switched in wireframe mode !";
                            Globals::wireframe = true;
                        }

                    }
                    else if (e->key() == 45) {
                        Globals::minTreeLevelDisplay++;
                        log_console->infoStream() << "Tree view level set to " << Globals::minTreeLevelDisplay << " !";
                    }
                    else if (e->key() == 43 && Globals::minTreeLevelDisplay >= 1) {
                        Globals::minTreeLevelDisplay--;
                        log_console->infoStream() << "Tree view level set to " << Globals::minTreeLevelDisplay << " !";
                    }
                    else {
                        log_console->infoStream() << "event : " << e->key() << " !";
                    }
            }
#endif

}


#endif /* end of include guard: LEAFNODE_H */
