

#ifndef TEXTURE3D_H
#define TEXTURE3D_H

#include "headers.hpp"
#ifdef GUI_ENABLED

#include "texture.hpp"

//SIZES SHOULD BE POWER OF TWO !
//Initial data is undefined if data is set to a NULL pointer (default)
//If a buffer is bind to GL_PIXEL_UNPACK_BUFFER, the data will be taken from it
//and not from actual data pointer
class Texture3D : public Texture {

	public: 
		Texture3D(unsigned int width, unsigned int height, unsigned int length,
				GLint internalFormat, 
				void *sourceData=0, GLenum sourceFormat=0, GLenum sourceType=0);

		virtual ~Texture3D();

		//allocate data, transfers data if not NULL and bind to texture unit location 
		void bindAndApplyParameters(unsigned int location);
	
		//data only updated when bind is called !!
		void setData(void *data, GLenum sourceFormat = 0, GLenum sourceType = 0);

	protected:
		unsigned int _width, _height, _length;

		void * _texels;
		GLint _internalFormat;
		
		GLenum _sourceFormat, _sourceType;
};

#endif
#endif /* end of include guard: TEXTURE2D_H */
