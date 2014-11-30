
#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include "headers.hpp"
#include "texture.hpp"

class Texture2D : public Texture {

	public: 
		Texture2D(std::string const &src, std::string const &type);
		Texture2D(float *data);
		~Texture2D();

		void bindAndApplyParameters(unsigned int location);

	private:
		QImage image;
		const std::string src;
		const std::string type;

};

#endif /* end of include guard: TEXTURE2D_H */
