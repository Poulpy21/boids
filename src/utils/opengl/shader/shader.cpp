
#include "shader.hpp"
#include "log.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>


Shader::Shader(const char* location_, GLenum shaderType_) :
shader(0), location(location_)
{

	// Load shader file
std::stringstream shaderS;
std::string line;

std::ifstream shaderFile(location);
	
	if (shaderFile.is_open() && getline(shaderFile, line))
	{
		shaderS << line;

		while (getline(shaderFile,line))
			shaderS << "\n" << line;

		shaderFile.close();

		log_console->debugStream() << "Loading shader from file " << location;
		log_console->debugStream() << "\n-- SHADER -- \n" << shaderS << "\n-- END --";
	}

	else {
		log_console->errorStream() << "\nUnable to load shader from file " << location; 
		exit(1);
	}
	
	// Create and compile shader
	this->shaderType = shaderType_;
	this->shader = glCreateShader(shaderType_);
	
	const std::string program = shaderS.str();
	const GLint prog_length = program.length();
	const char* prog = program.c_str();

	glShaderSource(shader, 1, &prog , &prog_length);
	glCompileShader(shader);

	if (GL_COMPILE_STATUS == GL_FALSE) {
		log_console->errorStream() << "\nCompilation failed !";

		char* buffer = new char[1000];
		int length;

		glGetShaderInfoLog(shader, 1000, &length, buffer);
		log_console->errorStream() << "\n" << buffer;

		delete [] buffer;
		exit(1);
	}
}

unsigned int Shader::getShader() const {
	return shader;
}

GLenum Shader::getShaderType() const {
	return shaderType;	
}

const std::string Shader::toStringShaderType() const {
	switch(shaderType) {
		case GL_VERTEX_SHADER:
			return "VERTEX_SHADER";
		case GL_TESS_CONTROL_SHADER:
			return "TESSELATION_CONTROL_SHADER";
		case GL_TESS_EVALUATION_SHADER:
			return "TESSELATION_EVALUATION_SHADER";
		case GL_GEOMETRY_SHADER:
			return "GEOMETRY_SHADER";
		case GL_FRAGMENT_SHADER:
			return "FRAGMENT_SHADER";
		default:
			return "UNKNOWN_SHADER_TYPE";
	}
}

const std::string Shader::getLocation() const {
	return location;
}
