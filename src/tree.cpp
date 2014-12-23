
#include <cstdio>
#include <iostream>

#include "headers.hpp"
#include "rootNode.hpp"
#include "treeNode.hpp"
#include "leafNode.hpp"
#include "hypercubeTree.hpp"
#include "abstractContainer.hpp"
#include "abstractContainerFactory.hpp"
#include "simpleContainerFactory.hpp"
#include "vecBool.hpp"
#include "boundingBox.hpp"
#include "hypercube.hpp"
#include "localized.hpp"
#include "hypercubeTree.hpp"
#include "arrayContainer.hpp"

struct Trunk {
    unsigned int foo;
    double bar;
};

struct Boid : public Localized<3u,float> {
    Boid() {}
    ~Boid() {}
    Vec<3u,float> position() const override {
        return static_cast<Vec<3u,float>>(Vec3<float>(0.0f,0.0f,0.0f));
    }
};
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();

    Vec3<float> v0(0.0f,0.0f,0.0f);
    Vec3<float> v1(1.0f,1.0f,1.0f);
    HyperCube<3u,float> hypercube(v0,v1);
    Tree::HyperCubeTree<3u,float,Trunk,ArrayContainer<Boid>,Boid> tree(hypercube, 10000u, 0.8f);

    Boid B1, B2, B3;
    tree.insert(B1);
    tree.insert(B2);
    tree.insert(B3);

    return EXIT_SUCCESS;
}
