
#ifndef SHADER_H
#define SHADER_H

#include "headers.hpp"
#ifdef GUI_ENABLED

#include <string>

class Shader {

	private:
		unsigned int shader;
		std::string location;
		GLenum shaderType;

	public:
		Shader(const char* location, GLenum shaderType);
		Shader(std::string const &location, GLenum shaderType);

		unsigned int getShader() const;
		GLenum getShaderType() const;
		const std::string toStringShaderType() const;
		const std::string getLocation() const;
};

#endif
#endif /* end of include guard: SHADER_H */
