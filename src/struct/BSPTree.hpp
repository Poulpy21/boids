
#ifndef BSPTREE_H
#define BSPTREE_H

#include "headers.hpp"

/*
 * Binary space partitioning tree with up to 2^32 - 1 childs
 * Linked list implementation
 * N is the dimension of the domain 
 * T is the trunk data type
 * L is the leaf data type and is a container for E
 * E is the element data type (element are inserted into the tree)
 * L should be default constructible
 * T and L sould be copy constructible
*/

template <typename T, typename L>
union TreeNodeData {
    T trunkData;
    L leafData;

    TreeNodeData();
    explicit TreeNodeData(const T &trunkData);
    explicit TreeNodeData(const L &leafData);
    explicit TreeNodeData(const TreeNodeData<T,L> &other);
};

template <unsigned int N, typename T, typename L, typename E>
struct TreeNode {
    unsigned int id;
    unsigned int nChilds;
    unsigned int nSubchilds;
    
    TreeNode *childs[N];
    TreeNodeData<T,L> data;

    TreeNode();
    virtual ~TreeNode();
    explicit TreeNode(const TreeNode<N,T,L,E> &other);
    explicit TreeNode(unsigned long int id, const T &data);

    inline bool isLeaf() const;
    Vec<N,double> center() const;

    inline unsigned int getParentId() const;
    inline unsigned int getFirstChildId() const;
    inline unsigned int getChildId(unsigned int nChild) const;
};

template <unsigned int N, typename T, typename L, typename E>
class BSPTree {
    static_assert(utils::is_power_of_two<unsigned int>(N), "N should be a power of two !");
    static_assert(N != 1u, "You are building an unary BSP tree ...");
    
    private:
        T *_data;
        unsigned int _levels;
        
        static const unsigned int nChilds = N;
        static const unsigned int dim = utils::get_power_of_two(N);

    public:
        
        BSPTree();
        virtual ~BSPTree();
        explicit BSPTree(const BSPTree<N,T,L,E> &other);
        explicit BSPTree(unsigned int initialLevels);

        T* data();
        unsigned int levels() const;

        //insert(

        const TreeNode<N,T,L,E> operator[](unsigned int id) const;
        TreeNode<N,T,L,E> & operator[](unsigned int id);

        inline TreeNode<N,T,L,E> & getFather(unsigned int nodeId);
        inline TreeNode<N,T,L,E> & getFather(const TreeNode<N,T,L,E> &node);
        inline TreeNode<N,T,L,E> & getChild(unsigned int nodeId, unsigned int childNum);
        inline TreeNode<N,T,L,E> & getChild(const TreeNode<N,T,L,E> &node, unsigned int childNum);
};

template <typename T, typename L, typename E>
using BinaryTree = BSPTree<2u,T,L,E>;

template <typename T, typename L, typename E>
using QuadTree   = BSPTree<4u,T,L,E>;

template <typename T, typename L, typename E>
using OcTree     = BSPTree<8u,T,L,E>;

template <typename T, typename L, typename E>
using SedecTree  = BSPTree<16u,T,L,E>;

#endif /* end of include guard: BSPTREE_H */

