#include "utils/headers.hpp"
#include "boids.hpp"

#ifdef GUI_ENABLED

#include <fstream>
#include <string>
#include <sstream>
#include "utils/globals.hpp"
#include "utils/types.hpp"

Boids::Boids(/*boids*/) {
    makeProgram();
}

Boids::~Boids () {
    for (int i = 0; i < 4; i++) {
		delete _boidTextures[i];
	}

	delete[] _boidTextures;

    delete _program;
}

void Boids::drawDownwards(const float *currentTransformationMatrix) {
    static float *proj = new float[16], *view = new float[16];

	_program->use();

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(_uniformLocations["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_uniformLocations["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(_uniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
	
	/*glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, nBoids);
	glBindVertexArray(0);*/

	glUseProgram(0);
}

void Boids::updateBoids(/*boids*/) {
}

void Boids::makeProgram() {
    _program = new Program("Boids");

    _program->bindAttribLocations("0", "vertexPosition");
    _program->bindFragDataLocation(0, "outColor");
    _program->bindUniformBufferLocations("0", "projectionView");

    _program->attachShader(Shader("src/shaders/boids_vs.glsl", GL_VERTEX_SHADER));
    _program->attachShader(Shader("src/shaders/boids_fs.glsl", GL_FRAGMENT_SHADER));

    _program->link();

    _uniformLocations = _program->getUniformLocationsMap("modelMatrix", true);
    
    /*
    _boidTextures = new Texture*[4];

    _boidTextures[0] = new Texture2D("textures/normalBoid.png","png");
    _boidTextures[0]->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
    _boidTextures[0]->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
    _boidTextures[0]->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    _boidTextures[0]->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    _boidTextures[0]->generateMipMap();

    _boidTextures[1] = new Texture2D("textures/predatorBoid.png", "png");
    _boidTextures[1]->addParameters(_boidTextures[0]->getParameters());
    _boidTextures[1]->generateMipMap();

    _boidTextures[2] = new Texture2D("textures/preyBoid.png", "png");
    _boidTextures[2]->addParameters(_boidTextures[0]->getParameters());
    _boidTextures[2]->generateMipMap();

    _boidTextures[3] = new Texture2D("textures/wallBoid.png", "png");
    _boidTextures[3]->addParameters(_boidTextures[0]->getParameters());
    _boidTextures[3]->generateMipMap();

    _program->bindTextures(&_boidTextures, "normalBoid predatorBoid preyBoid wallBoid", true);
    */
}

void Boids::readFile(std::string fileName) {
    /*std::ifstream file;
    int nAgents;
    Real value; 

    file.open(fileName);
   
    file.close();  */
}

#endif
