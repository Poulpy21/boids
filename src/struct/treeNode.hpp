
#ifndef TREENODE_H
#define TREENODE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include "renderTree.hpp"
#include "program.hpp"
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
#ifdef GUI_ENABLED
        class TreeNode : public RenderTree {
#else
        class TreeNode {
#endif

            public:
                TreeNode();
                TreeNode(const TreeNode<D,N,A,T> &other);
                explicit TreeNode(const BoundingBox<D,A> &domain, unsigned int level, unsigned int id);
                virtual ~TreeNode();

                //parent-child handling
                unsigned int getParentId() const;
                unsigned int getFirstChildId() const;
                unsigned int getChildId(unsigned int childNum) const;

                TreeNode<D,N,A,T>* father() const;
                TreeNode<D,N,A,T>* child(unsigned int childNum) const;

                TreeNode<D,N,A,T>*& father();
                TreeNode<D,N,A,T>*& child(unsigned int childNum);

                //acessors
                unsigned long id() const;
                BoundingBox<D,A> bbox() const;

                //helper funcs
                unsigned int level() const;

                virtual bool isRoot() const;
                virtual bool isLeaf() const;

            protected:
                unsigned long _id;
                unsigned int _level;
                
                static unsigned int _maxLevel;

                BoundingBox<D,A> _bbox;
                T _nodeData;

                TreeNode *_father;
                TreeNode *_childs[N];

#ifdef GUI_ENABLED
            public:
                virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) override;

            protected:
                static unsigned int constexpr _nTriangles = 6*2;
                static const float _unitCube[_nTriangles*3*3];
                static const float _unitWireframeCube[6*4*3];

                static Program *_drawBoxProgram; 
                static unsigned int _cubeVBO, _wireCubeVBO;
                static std::map<std::string, int> _drawBoxUniformLocs;

                static void makeDrawBoxProgram();
                static void makeCube();
                static void initProgram();
#endif
        };
        
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::_maxLevel = floor(64u*log(2.0f)/log(N));

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode() :
            _id(0ul), _level(0u),
            _bbox(),  _nodeData() {
                _father = nullptr;
                for (unsigned int i = 0; i < N; i++) {
                    _childs[i] = nullptr;
                }
            }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::~TreeNode() {
            for (unsigned int i = 0; i < N; i++) {
                delete _childs[i];
            }
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode(const BoundingBox<D,A> &domain, unsigned int level, unsigned int id) :
            _id(id), _level(level),
            _bbox(domain),  _nodeData() {
                _father = nullptr;
                for (unsigned int i = 0; i < N; i++) {
                    _childs[i] = nullptr;
                }
#ifdef GUI_ENABLED
                initProgram();
#endif
            }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>::TreeNode(const TreeNode<D,N,A,T> &other) :
#ifdef GUI_ENABLED
            RenderTree(other),
#endif
            _id(other._id), _level(other._level),
            _bbox(other._bbox),  _nodeData(other._nodeData)
    {
        _father = other._father;
        for (unsigned int i = 0; i < N; i++) {
            _childs[i] = other.child(i);
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
        TreeNode<D,N,A,T>* TreeNode<D,N,A,T>::father() const {
            return _father;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>* TreeNode<D,N,A,T>::child(unsigned int childNum) const {
            return _childs[childNum];
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>*& TreeNode<D,N,A,T>::father() {
            return _father;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        TreeNode<D,N,A,T>*& TreeNode<D,N,A,T>::child(unsigned int childNum) {
            return _childs[childNum];
        }

    //acessors
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned long TreeNode<D,N,A,T>::id() const {
            return _id;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        BoundingBox<D,A> TreeNode<D,N,A,T>::bbox() const {
            return _bbox;
        }

    //helper funcs
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::level() const {
            return this->_level;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool TreeNode<D,N,A,T>::isRoot() const {
            return false;
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        bool TreeNode<D,N,A,T>::isLeaf() const {
            return false;
        }

#ifdef GUI_ENABLED

    template <unsigned int D, unsigned int N, typename A, typename T>
        void TreeNode<D,N,A,T>::drawDownwards(const float *currentTransformationMatrix) {
            for (unsigned int i = 0; i < N; i++) {
                this->_childs[i]->drawDownwards(currentTransformationMatrix);
            }
        }
            

    template <unsigned int D, unsigned int N, typename A, typename T>
        void TreeNode<D,N,A,T>::initProgram() {
            static_assert(D==3, "Tree display only possible in dimension 3 !");
            makeCube();
            makeDrawBoxProgram();
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        void TreeNode<D,N,A,T>::makeDrawBoxProgram() {
            if(_drawBoxProgram == nullptr) {
                _drawBoxProgram = new Program("Tree Node Draw");
                _drawBoxProgram->bindAttribLocation(0, "vertex_position");
                _drawBoxProgram->bindFragDataLocation(0, "out_colour");
                _drawBoxProgram->bindUniformBufferLocations("0", "projectionView");

                _drawBoxProgram->attachShader(Shader(Globals::shaderFolder + "/box/box_vs.glsl", GL_VERTEX_SHADER));
                _drawBoxProgram->attachShader(Shader(Globals::shaderFolder + "/box/box_fs.glsl", GL_FRAGMENT_SHADER));

                _drawBoxProgram->link();
                _drawBoxUniformLocs = _drawBoxProgram->getUniformLocationsMap("modelMatrix level", true);
            }
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        void TreeNode<D,N,A,T>::makeCube() {
            if(_cubeVBO == 0) {
                glGenBuffers(1, &_cubeVBO);
                glBindBuffer(GL_ARRAY_BUFFER, _cubeVBO);
                glBufferData(GL_ARRAY_BUFFER, 6*2*3*3*sizeof(float), _unitCube, GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
            if(_wireCubeVBO == 0) {
                glGenBuffers(1, &_wireCubeVBO);
                glBindBuffer(GL_ARRAY_BUFFER, _wireCubeVBO);
                glBufferData(GL_ARRAY_BUFFER, 6*4*3*sizeof(float), _unitWireframeCube, GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }
        }

    template <unsigned int D, unsigned int N, typename A, typename T>
        Program* TreeNode<D,N,A,T>::_drawBoxProgram = nullptr; 

    template <unsigned int D, unsigned int N, typename A, typename T>
        std::map<std::string, int> TreeNode<D,N,A,T>::_drawBoxUniformLocs;

    template <unsigned int D, unsigned int N, typename A, typename T>
        const float TreeNode<D,N,A,T>::_unitCube[108] = {
            -0.5f, -0.5f,  0.5f,  //0
            0.5f, -0.5f,  0.5f,  //1
            0.5f,  0.5f,  0.5f,  //2
            0.5f,  0.5f,  0.5f,  //2
            -0.5f,  0.5f,  0.5f,  //3
            -0.5f, -0.5f,  0.5f,  //0

            0.5f,  0.5f, -0.5f,  //6 
            0.5f,  0.5f,  0.5f,  //2
            0.5f, -0.5f,  0.5f,  //1
            0.5f, -0.5f,  0.5f,  //1
            0.5f, -0.5f, -0.5f,  //5
            0.5f,  0.5f, -0.5f,  //6 

            -0.5f, -0.5f, -0.5f,  //4
            -0.5f,  0.5f, -0.5f,  //7
            0.5f,  0.5f, -0.5f,  //6 
            0.5f,  0.5f, -0.5f,  //6 
            0.5f, -0.5f, -0.5f,  //5
            -0.5f, -0.5f, -0.5f,  //4

            -0.5f,  0.5f,  0.5f,  //3
            -0.5f,  0.5f, -0.5f,  //7
            -0.5f, -0.5f, -0.5f,  //4
            -0.5f, -0.5f, -0.5f,  //4
            -0.5f, -0.5f,  0.5f,  //0
            -0.5f,  0.5f,  0.5f,  //3

            0.5f,  0.5f,  0.5f,  //2
            0.5f,  0.5f, -0.5f,  //6 
            -0.5f,  0.5f, -0.5f,  //7
            -0.5f,  0.5f, -0.5f,  //7
            -0.5f,  0.5f,  0.5f,  //3
            0.5f,  0.5f,  0.5f,  //2

            -0.5f, -0.5f, -0.5f,  //4
            0.5f, -0.5f, -0.5f,  //5
            0.5f, -0.5f,  0.5f,  //1
            0.5f, -0.5f,  0.5f,  //1
            -0.5f, -0.5f,  0.5f,  //0
            -0.5f, -0.5f, -0.5f   //4
        };
    
    template <unsigned int D, unsigned int N, typename A, typename T>
        const float TreeNode<D,N,A,T>::_unitWireframeCube[6*4*3] = {
            -0.5f, -0.5f,  0.5f,  //0
            0.5f, -0.5f,  0.5f,  //1
            0.5f,  0.5f,  0.5f,  //2
            -0.5f,  0.5f,  0.5f,  //3

            0.5f, -0.5f,  0.5f,  //1
            0.5f,  0.5f,  0.5f,  //2
            0.5f,  0.5f, -0.5f,  //6 
            0.5f, -0.5f, -0.5f,  //5

            -0.5f, -0.5f, -0.5f,  //4
            -0.5f,  0.5f, -0.5f,  //7
            0.5f,  0.5f, -0.5f,  //6 
            0.5f, -0.5f, -0.5f,  //5

            -0.5f,  0.5f,  0.5f,  //3
            -0.5f,  0.5f, -0.5f,  //7
            -0.5f, -0.5f, -0.5f,  //4
            -0.5f, -0.5f,  0.5f,  //0

            0.5f,  0.5f,  0.5f,  //2
            0.5f,  0.5f, -0.5f,  //6 
            -0.5f,  0.5f, -0.5f,  //7
            -0.5f,  0.5f,  0.5f,  //3

            -0.5f, -0.5f, -0.5f,  //4
            0.5f, -0.5f, -0.5f,  //5
            0.5f, -0.5f,  0.5f,  //1
            -0.5f, -0.5f,  0.5f,  //0
        };

    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::_cubeVBO = 0u;
    
    template <unsigned int D, unsigned int N, typename A, typename T>
        unsigned int TreeNode<D,N,A,T>::_wireCubeVBO = 0u;

#endif /* GUI_ENABLED */

}

#endif /* end of include guard: TREENODE_H */
