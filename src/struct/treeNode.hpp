
#ifndef TREENODE_H
#define TREENODE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include <memory>

/*
 * Binary space partitioning tree with up to 2^32 nodes
 * Sparse implementation (linked lists)
 * Templates:
 * D is the dimension of the spatial domain (D-d domain)
 * N is the number of childs per node (max value is 32)
 * A is the arithmetic position type (either floating-point or integers)
 * T is the Trunk data type (present at leaf too)
 * L is the Leaf data type and should inherit AbstractConrainer<E>
 * E is the Element data type and should inherit Localized<N,A>
 * Conditions:
 * N : unsigned int, power of two > 1
 * D : unsigned int >= 1
 * A : integral type, default constructible, constructible from int, copy constructible, assignable
 * T : default constructible, copy constructible
 * L : default constructible, copy constructible, inherit AbstractContainer<E>
 * E : default constructible, copy constructible, inherit Localized<D> 
 */

namespace Tree {
    
    template <unsigned int D, unsigned int N, typename A, typename T, typename L, typename E>
        class RootNode;

    template <unsigned int D, unsigned int N, typename A, typename T>
        class TreeNode {
          
            public:
                TreeNode();
                TreeNode(const TreeNode<D,N,A,T> &other);
                explicit TreeNode(const BoundingBox<D,A> &domain);
                virtual ~TreeNode();

                //parent-child handling
                unsigned int getParentId() const;
                unsigned int getFirstChildId() const;
                unsigned int getChildId(unsigned int childNum) const;

                std::weak_ptr<TreeNode<D,N,A,T>> father() const;
                std::weak_ptr<TreeNode<D,N,A,T>> weakChild(unsigned int childNum) const;
                std::shared_ptr<TreeNode<D,N,A,T>>& child(unsigned int childNum);

                TreeNode<D,N,A,T>& operator[](unsigned int k);
                TreeNode<D,N,A,T>  operator[](unsigned int k) const;

                //acessors
                unsigned long id() const;
                unsigned int nChilds() const;
                unsigned int nSubchilds() const;
                BoundingBox<D,A> bbox() const;

                //helper funcs
                unsigned int level() const;

                virtual bool isRoot() const;
                virtual bool isLeaf() const;

            protected:
                unsigned long _id;
                unsigned int _nChilds;
                unsigned int _nSubchilds;

                BoundingBox<D,A> _bbox;
                T _nodeData;

                std::weak_ptr<TreeNode> _father;
                std::shared_ptr<TreeNode> _childs[N];
        };

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode() :
            _id(0ul), _nChilds(0u), _nSubchilds(0u),
            _bbox(),  _nodeData() {
                _father = nullptr;
                for (unsigned int i = 0; i < N; i++) {
                    _childs[i].reset(nullptr);
                }
            }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::~TreeNode() {
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode(const BoundingBox<D,A> &domain) :
            _id(0ul), _nChilds(0u), _nSubchilds(0u),
            _bbox(),  _nodeData() {
                _father.reset();
                for (unsigned int i = 0; i < N; i++) {
                    _childs[i].reset();
                }
            }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode(const TreeNode<D,N,A,T> &other) :
            _id(other._id), _nChilds(other._nChilds), _nSubchilds(other._nSubchilds),
            _bbox(other._bbox),  _nodeData(other._nodeData) 
    {
        _father = other._father;
        for (unsigned int i = 0; i < N; i++) {
            _childs[i] = other.weakChild(i).lock();
        }
    }

    //parent-child handling
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::getParentId() const {
            return _id/N;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::getFirstChildId() const {
            return _id*N;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::getChildId(unsigned int childNum) const {
            return _id*N + childNum;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>& TreeNode<D,N,A,T>::operator[](unsigned int k) {
            return *_childs[k].get();
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T> TreeNode<D,N,A,T>::operator[](unsigned int k) const {
            return *_childs[k].get();
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        std::weak_ptr<TreeNode<D,N,A,T>> TreeNode<D,N,A,T>::father() const {
            return _father;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        std::weak_ptr<TreeNode<D,N,A,T>> TreeNode<D,N,A,T>::weakChild(unsigned int childNum) const {
            return _childs[childNum];
        }
                
    template <unsigned int D, unsigned int N, typename A, typename T>
    std::shared_ptr<TreeNode<D,N,A,T>>& TreeNode<D,N,A,T>::child(unsigned int childNum) {
            return _childs[childNum];
        }

    //acessors
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned long TreeNode<D,N,A,T>::id() const {
            return _id;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::nChilds() const {
            return _nChilds;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::nSubchilds() const {
            return _nSubchilds;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        BoundingBox<D,A> TreeNode<D,N,A,T>::bbox() const {
            return _bbox;
        }

    //helper funcs
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::level() const {
            unsigned int level = 0ul;
            unsigned long buf = _id;
            while(buf) {
                buf >>= D;
                level++;
            }

            return level;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool TreeNode<D,N,A,T>::isRoot() const {
            return false;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool TreeNode<D,N,A,T>::isLeaf() const {
            return false;
        }

}

#endif /* end of include guard: TREENODE_H */
