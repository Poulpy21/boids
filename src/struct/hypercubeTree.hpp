
#ifndef HYPERCUBETREE_H
#define HYPERCUBETREE_H

#include "headers.hpp"
#include "rootNode.hpp"
#include "hypercube.hpp"

namespace Tree {

#define _N_ utils::compute_power_of_two<unsigned int>(D)

    template <unsigned int D, typename A, typename T, typename L, typename E>
        class HyperCubeTree : public RootNode<D,_N_, A, T, L, E> {

            public:
                static constexpr unsigned int N = utils::compute_power_of_two<unsigned int>(D);

                HyperCubeTree(const HyperCubeTree<D,A,T,L,E> &other);
                explicit HyperCubeTree(const HyperCube<D,A> &cube, unsigned int maxElementsPerLeaf, float fillThreshHold);
                virtual ~HyperCubeTree();
                
                unsigned int targetChild(const TreeNode<D,_N_,A,T> *node, const E &e) const override;
                TreeNode<D,_N_,A,T>* splitLeaf(LeafNode<D,_N_,A,T,L,E>* &leaf) override;
                LeafNode<D,_N_,A,T,L,E>* mergeChilds(TreeNode<D,_N_,A,T>* &father) override;
        };
                
    template <unsigned int D, typename A, typename T, typename L, typename E>
    HyperCubeTree<D,A,T,L,E>::HyperCubeTree(const HyperCubeTree<D,A,T,L,E> &other) : 
        BoxTree<D,_N_,A,T,L,E>(other) {
        }

    template <unsigned int D, typename A, typename T, typename L, typename E>
    HyperCubeTree<D,A,T,L,E>::HyperCubeTree(const HyperCube<D,A> &cube, 
            unsigned int maxElementsPerLeaf, float fillThreshHold) :
        BoxTree<D,_N_,A,T,L,E>(cube, maxElementsPerLeaf, fillThreshHold) {
            
            //create first leafs
            const Vec<D,A> &xmin = this->bbox().min;
            const Vec<D,A> &xmax = this->bbox().max;
            Vec<D,A> dx2 = (xmax - xmin)/A(2); 
            for (unsigned int i = 0; i < N; i++) {
                Vec<D,A> v = Vec<D,A>(VecBool<D>(i));
                BoundingBox<D,A> bbox(xmin+v*dx2, xmin+(A(1)+v)*dx2);
                this->child(i) = new LeafNode<D,N,A,T,L,E>(bbox, this->_maxElementsPerLeaf, this, i);
            }
    }

    template <unsigned int D, typename A, typename T, typename L, typename E>
    HyperCubeTree<D,A,T,L,E>::~HyperCubeTree() {
    }

    template <unsigned int D, typename A, typename T, typename L, typename E>
    unsigned int HyperCubeTree<D,A,T,L,E>::targetChild(const TreeNode<D,_N_,A,T> *node, const E &e) const {
        return (e.position() > node->bbox().center()).to_uint(); 
    }

    template <unsigned int D, typename A, typename T, typename L, typename E>
    TreeNode<D,_N_,A,T>* HyperCubeTree<D,A,T,L,E>::splitLeaf(LeafNode<D,_N_,A,T,L,E>* &leaf) {

        assert(leaf->isLeaf()); 
        
        //clone leaf base treenode and get data
        TreeNode<D,_N_,A,T>* father = new TreeNode<D,N,A,T>(*leaf);
        L data(leaf->data());
        
        delete leaf;
        leaf = dynamic_cast<LeafNode<D,N,A,T,L,E>*>(father);

        //create new leafs
        const Vec<D,A> &xmin = father->bbox().min;
        const Vec<D,A> &xmax = father->bbox().max;
        Vec<D,A> dx2 = (xmax - xmin)/A(2); 
        for (unsigned int i = 0; i < N; i++) {
            Vec<D,A> v = Vec<D,A>(VecBool<D>(i));
            BoundingBox<D,A> bbox(xmin+v*dx2, xmin+(A(1)+v)*dx2);
            father->child(i) = new LeafNode<D,N,A,T,L,E>(bbox, this->_maxElementsPerLeaf, father, i);
        }
    
        //put back data in the tree
        this->insert(data);

        return father;
    }

    template <unsigned int D, typename A, typename T, typename L, typename E>
    LeafNode<D,_N_,A,T,L,E>*  HyperCubeTree<D,A,T,L,E>::mergeChilds(TreeNode<D,_N_,A,T>* &father) {
        return dynamic_cast<LeafNode<D,N,A,T,L,E>*>(father->child(0));
    }

#undef _N_

}

#endif /* end of include guard: HYPERCUBETREE_H */
