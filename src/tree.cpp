
#include <cstdio>
#include <iostream>

#include "headers.hpp"
#include "rand.hpp"
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

#include "texture.hpp"
#include "renderRoot.hpp"
#include "boids.hpp"

struct Trunk {
    unsigned int foo;
    double bar;
};

struct Boid : public Localized<3u,float> {
    Boid() {}
    ~Boid() {}
    Vec<3u,float> position() const override {
        using Random::randf;
        return static_cast<Vec<3u,float>>(Vec3<float>(randf(),randf(),randf()));
    }
};
int main(int argc, char **argv) {

    using log4cpp::log_console;
    log4cpp::initLogs();

    log_console->infoStream() << "[Log Init] ";

    srand(time(NULL));
    log_console->infoStream() << "[Rand Init] ";

#ifdef GUI_ENABLED
    // glut initialisation (mandatory) 
    glutInit(&argc, argv);
    log_console->infoStream() << "[Glut Init] ";

    // Read command lines arguments.
    QApplication application(argc,argv);
    log_console->infoStream() << "[Qt Init] ";

    // Instantiate the viewer (mandatory)
    Viewer *viewer = new Viewer();
    viewer->setWindowTitle("Flocking boids");
    viewer->show();
    Globals::viewer = viewer;

    //glew initialisation (mandatory)
    log_console->infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

    //global vars
    Globals::init();
    Globals::print(std::cout);
    Globals::check();

    //texture manager
    Texture::init();

    log_console->infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
    //FIN INIT//
#endif

    log_console->infoStream() << "Let the test begin !";
    Vec3<float> v0(0.0f,0.0f,0.0f);
    Vec3<float> v1(1.0f,1.0f,1.0f);
    HyperCube<3u,float> hypercube(v0,v1);
    Tree::HyperCubeTree<3u,float,Trunk,ArrayContainer<Boid>,Boid> tree(hypercube, 10u, 0.8f);

    for (unsigned int i = 0; i < 100; i++) {
        tree.insert(Boid());
    }

#ifdef GUI_ENABLED
    RenderRoot *root = new RenderRoot(); 
    root->addChild("Tree", &tree);
    viewer->addRenderable(root);
    application.exec();
#endif

    return EXIT_SUCCESS;
}
