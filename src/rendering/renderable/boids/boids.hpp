#ifndef __BOIDS_RENDER_HPP__
#define __BOIDS_RENDER_HPP__

#include "utils/headers.hpp"
#ifdef GUI_ENABLED

#include <string>
#include "rendering/renderable/renderTree.hpp"
#include "utils/opengl/program/program.hpp"
#include "utils/opengl/texture/texture2D.hpp"

class Boids : public RenderTree {
    public:
        Boids(/*boids*/);
        ~Boids();
        
        void upateBoids(/*boids*/);
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        void makeProgram();

        Program *_program;
        std::map<std::string, int> _uniformLocations;
        unsigned int _vertexVBO;
        Texture **_boidTextures;
};

#endif
#endif

