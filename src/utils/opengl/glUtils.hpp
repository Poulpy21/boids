
#ifndef GLUTILS_H
#define GLUTILS_H

#ifdef GUI_ENABLED

#include "headers.hpp"
#include <string>

namespace utils {
		void glAssert(const std::string &file, int line, bool abort = true);
		void createOpenGLContext(Display **display, GLXContext *ctx, Window *win, Colormap *cmap, GLXContext shareList);
		bool isExtensionSupported(const char *extList, const char *extension);

		int contextErrorHandler(Display *dpy, XErrorEvent *ev);
        
        //texture initialization helpers
        GLenum internalFormatToValidExternalFormat(unsigned int internalFormat);
        GLenum internalFormatToValidExternalType(unsigned int internalFormat);

        size_t externalTypeToBytes(GLenum externalType);
        unsigned int externalFormatToChannelNumber(GLenum externalFormat);

        const std::string toStringInternalFormat(unsigned int internalFormat);
        const std::string toStringExternalFormat(GLenum externalFormat);
        const std::string toStringExternalType(GLenum externalType);
        
        const std::string TextureTarget(GLenum textureTarget);
	
        void checkFrameBufferStatus();
}

#endif
#endif /* end of include guard: GLUTILS_H */
