#ifndef __BOIDS_RENDER_HPP__
#define __BOIDS_RENDER_HPP__

#include "utils/headers.hpp"
#ifdef GUI_ENABLED

#include <string>
#include <fstream>
#include "rendering/renderable/renderTree.hpp"
#include "utils/opengl/program/program.hpp"
#include "utils/opengl/texture/texture2D.hpp"

class Boids : public RenderTree {
    public:
        Boids();
        ~Boids();
        
        void updateBoids(/*boids*/);
        void readBoidsFromFile(std::string fileName);
        void resetFile();

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards();
        void keyPressEvent(QKeyEvent* e);

    private:
        void makeProgram();
        void makeVAO();
        void parseBoidsFile();

        Program *_program;
        std::map<std::string, int> _uniformLocations;
        GLuint _VAO, _VBO;
        Texture **_boidTextures;

        bool boidsUpdated;
        std::string boidsFile;
        int currentStep;
        std::ifstream boidsFs;
        std::vector<float> boids;
};

#endif
#endif

