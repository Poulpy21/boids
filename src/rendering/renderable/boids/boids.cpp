#include "utils/headers.hpp"
#include "boids.hpp"

#ifdef GUI_ENABLED

#include <GL/freeglut.h>
#include "utils/globals.hpp"
#include "utils/types.hpp"
#include "utils/maths/vec3.hpp"

using namespace log4cpp;

Boids::Boids() : boidsUpdated(false), boidsFile(""), currentStep(-1), boids()
{
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    makeProgram();
    makeVAO();
}

Boids::~Boids () {
    boidsFs.close();

    /*for (int i = 0; i < 4; i++) {
		delete _boidTextures[i];
	}*/

	//delete[] _boidTextures;

    delete _program;
}

void Boids::drawDownwards(const float *currentTransformationMatrix) {
    if (boidsUpdated) {
        glBindBuffer(GL_ARRAY_BUFFER, _VBO);
        glBufferData(GL_ARRAY_BUFFER, boids.size()*sizeof(float), boids.data(), GL_DYNAMIC_DRAW);
        currentStep++;
    }
    boidsUpdated = false;

	_program->use();

	glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
	//glUniformMatrix4fv(_uniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
    /*GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glUniform2i(_uniformLocations["screenSize"], viewport[2], viewport[3]);*/

	glBindVertexArray(_VAO);
		glDrawArrays(GL_POINTS, 0, boids.size() / 3);
	glBindVertexArray(0);

	glUseProgram(0);

    // Draw frame
    glWindowPos2i(10,10);
    std::stringstream ss;
    ss << "Step : " << currentStep;
    glutBitmapString(GLUT_BITMAP_HELVETICA_12, reinterpret_cast<const unsigned char*>(ss.str().c_str()));

}

void Boids::animateDownwards() {
    static int i = 0;
    if (i++%2) return;

    if (boidsFile.compare("") != 0)
        parseBoidsFile();
}

void Boids::updateBoids(/*boids*/) {

    //TODO
    //boidsUpdated = true;
}

void Boids::makeProgram() {
    _program = new Program("Boids");

    _program->bindAttribLocations("0", "vertexPosition");
    _program->bindFragDataLocation(0, "outColor");
    _program->bindUniformBufferLocations("0", "projectionView");

    _program->attachShader(Shader(Globals::shaderFolder + "/boids/boids_vs.glsl", GL_VERTEX_SHADER));
    _program->attachShader(Shader(Globals::shaderFolder + "/boids/boids_fs.glsl", GL_FRAGMENT_SHADER));

    _program->link();

    //_uniformLocations = _program->getUniformLocationsMap("modelMatrix", true);
    //_uniformLocations = _program->getUniformLocationsMap("screenSize", false);
    
    
    /*_boidTextures = new Texture*[4];

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
    */

    //_program->bindTextures(&_boidTextures, "normalBoid predatorBoid preyBoid wallBoid", true);
    //_program->bindTextures(_boidTextures, "boidTexture0", false);
}

void Boids::makeVAO() {
    glGenVertexArrays(1, &_VAO);
    glBindVertexArray(_VAO);

    glGenBuffers(1, &_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    //TODO attrib #1 : boidType

    glBindVertexArray(0);
}

void Boids::readBoidsFromFile(std::string fileName) {
    boidsFile = fileName;

    boidsFs.open(boidsFile, std::ios::in);

    if (! boidsFs.is_open()) {
        log_console->errorStream() << "Can't open " << fileName << " !";
        return;
    }
    log_console->infoStream() << "Reading boids data file : " << boidsFile;

    parseBoidsFile();
}

void Boids::parseBoidsFile() {
    long long int nAgents;
    Real value;
    std::string tmp;

    if (! boidsFs.is_open())
        return;

    if (! (boidsFs >> nAgents)) {
        //EOF
        return;
    }
    if (nAgents < 0)
        log_console->errorStream() << "Negative number of boids in file " << boidsFile << " !";
    boids.clear();
    boids.reserve(nAgents*3);

    for (int i = 0; i < nAgents; i++) {
        if (! (boidsFs >> tmp))
            log_console->errorStream() << "Number of lines in " << boidsFile << " doesn't match number of boids !";
        if (tmp.compare("B") != 0)
            log_console->errorStream() << "Bad token in file " << boidsFile << " !";
        for (int n = 0; n < 3; n++) {
            if (! (boidsFs >> value))
                log_console->errorStream() << "Cannot read boid position data in file " << boidsFile << " !";
            boids.push_back(value);
        }
        //TODO boidType
    }

    boidsUpdated = true;
}

void Boids::resetFile() {
    if (! boidsFs.is_open())
        return;

    boidsFs.clear();   
    boidsFs.seekg(0, std::ios::beg);

    parseBoidsFile();

    currentStep = -1;
}
        
void Boids::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_R && e->modifiers() == Qt::NoButton) {
        log_console->infoStream() << "Boids file reset";
        resetFile();
    }
}

#endif
