
#include "glUtils.hpp"

#ifdef GUI_ENABLED

namespace utils {

	bool contextError = false;

	void glAssert(const std::string &file, int line, bool abort) {

		GLenum error = glGetError();

		switch(error) {
			case GL_NO_ERROR:
				break; 
			case GL_INVALID_ENUM:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_ENUM\n\t\t"
					"An unacceptable value is specified for an enumerated argument. "
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;

			case GL_INVALID_VALUE:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_VALUE\n\t\t"
					"A numeric argument is out of range."
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_INVALID_OPERATION:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_OPERATION\n\t\t"
					"The specified operation is not allowed in the current state."
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_INVALID_FRAMEBUFFER_OPERATION\n\t\t"
					"The framebuffer object is not complete. "
					"The offending command is ignored and has no other side effect than to set the error flag.";
				break;
			case GL_OUT_OF_MEMORY:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_OUT_OF_MEMORY\n\t\t"
					"There is not enough memory left to execute the command. The state of the case GL"
					"The state of the GL is undefined, except for the state of the error flags, after this error is recorded.";
					break;
			case GL_STACK_UNDERFLOW:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_STACK_UNDERFLOW\n\t\t"
					"An attempt has been made to perform an operation that would cause an internal stack to underflow.";
				break;
			case GL_STACK_OVERFLOW:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << ":" << "GL_STACK_OVERFLOW\n\t\t"
					"An attempt has been made to perform an operation that would cause an internal stack to overflow.";
				break;
			default:
				log_console->errorStream() << "OpenGL error : " << file << ":" << line << "\n\t\t"
					"Unknown error !";
		}

		if(error != GL_NO_ERROR && abort)
			exit(EXIT_FAILURE);
	}

	bool isExtensionSupported(const char *extList, const char *extension)
	{
		const char *start;
		const char *where, *terminator;

		where = strchr(extension, ' ');
		if (where || *extension == '\0')
			return false;

		for (start=extList;;) {
			where = strstr(start, extension);

			if (!where)
				break;

			terminator = where + strlen(extension);

			if ( where == start || *(where - 1) == ' ' )
				if ( *terminator == ' ' || *terminator == '\0' )
					return true;

			start = terminator;
		}

		return false;
	}

	typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
	void createOpenGLContext(Display **display, GLXContext *ctx, Window *win, Colormap *cmap, GLXContext shareList) {

		if(shareList == 0)
			log_console->infoStream()  << "Creating an openGL context !";
		else
			log_console->infoStream()  << "Creating a shared openGL/openGL context !";

		static int visual_attribs[] =
		{
			GLX_X_RENDERABLE    , True,
			GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
			GLX_RENDER_TYPE     , GLX_RGBA_BIT,
			GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
			GLX_RED_SIZE        , 8,
			GLX_GREEN_SIZE      , 8,
			GLX_BLUE_SIZE       , 8,
			GLX_ALPHA_SIZE      , 8,
			GLX_DEPTH_SIZE      , 24,
			GLX_STENCIL_SIZE    , 8,
			GLX_DOUBLEBUFFER    , True,
			None
		};

		(*display) = XOpenDisplay(NULL);
		if (!(*display)) {
			log_console->errorStream()  << "Failed to open X (*display), aborting !";
			exit(EXIT_FAILURE);
		}

		int glx_major, glx_minor;
		if (!glXQueryVersion((*display), &glx_major, &glx_minor) 
				|| ((glx_major == 1) && (glx_minor< 3)) 
				|| ( glx_major < 1 ) ) {
			log_console->errorStream() << "Invalid GLX version : " << glx_major << "." << glx_minor << " (1.3 min)";
			exit(EXIT_FAILURE);
		}

		int fbcount;
		GLXFBConfig* fbc = glXChooseFBConfig((*display), DefaultScreen((*display)), visual_attribs, &fbcount);
		if (!fbc) {
			log_console->errorStream() << "Failed to retrieve a framebuffer config !";
			exit(EXIT_FAILURE);
		}
		log_console->infoStream() <<  "Found " << fbcount << " matching framebuffers configs !";

		log_console->infoStream() <<  "Getting XVisualInfos...";
		int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
		int i;
		for (i=0; i<fbcount; ++i)
		{
			XVisualInfo *vi = glXGetVisualFromFBConfig((*display), fbc[i]);
			if (vi) {
				int samp_buf, samples;
				glXGetFBConfigAttrib((*display), fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf);
				glXGetFBConfigAttrib((*display), fbc[i], GLX_SAMPLES       , &samples );

				log_console->infoStream() <<  "\t\tMatching fbconfig " << i << " visual ID 0x" << vi->visualid << ": "
					<< "SAMPLE_BUFFERS = " << samp_buf << " SAMPLES = " << samples;

				if (best_fbc < 0 || (samp_buf && samples > best_num_samp))
					best_fbc = i, best_num_samp = samples;
				if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp)
					worst_fbc = i, worst_num_samp = samples;
			}
			XFree(vi);
		}

		GLXFBConfig bestFbc = fbc[best_fbc];
		XFree(fbc);

		// Get a visual
		XVisualInfo *vi = glXGetVisualFromFBConfig( (*display), bestFbc );
		log_console->infoStream() <<  "Chosen visual ID = 0x" <<  vi->visualid;
		log_console->infoStream() <<  "Creating colormap...";
		XSetWindowAttributes swa;
		swa.colormap = (*cmap) = XCreateColormap( (*display),
				RootWindow( (*display), vi->screen ), 
				vi->visual, AllocNone );
		swa.background_pixmap = None ;
		swa.border_pixel      = 0;
		swa.event_mask        = StructureNotifyMask;

		log_console->infoStream() <<  "Creating window...";
		(*win) = XCreateWindow( (*display), RootWindow( (*display), vi->screen ), 
				0, 0, 100, 100, 0, vi->depth, InputOutput, 
				vi->visual, 
				CWBorderPixel|CWColormap|CWEventMask, &swa );
		if (!(*win)) {
			log_console->infoStream() <<  "Failed to create window !";
			exit(EXIT_FAILURE);
		}

		XFree(vi);
		XStoreName((*display), (*win), "GL 3.0 window");

		//log_console->infoStream() <<  "Mapping window...";
		//XMapWindow((*display), (*win));

		const char *glxExts = glXQueryExtensionsString((*display), DefaultScreen((*display)));
		glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
		glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) 
			glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );

		//change error handler to handle fail
		contextError = false;
		int (*oldHandler)(Display*, XErrorEvent*) = XSetErrorHandler(&contextErrorHandler);

		(*ctx) = 0;
		if (!utils::isExtensionSupported(glxExts, "GLX_ARB_create_context") || !glXCreateContextAttribsARB)
		{
			log_console->infoStream() 
				<<  "glXCreateContextAttribsARB() not found... "
				<< "Using old-style GLX context";
			(*ctx) = glXCreateNewContext( (*display), bestFbc, GLX_RGBA_TYPE, shareList, True );
		}
		else
		{
			int context_attribs[] =
			{
				GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
				GLX_CONTEXT_MINOR_VERSION_ARB, 1,
				None
			};

			log_console->infoStream() <<  "Creating context...";
			(*ctx) = glXCreateContextAttribsARB((*display), bestFbc, shareList,
					True, context_attribs);

			//sync errors
			XSync((*display), False);
			if (!contextError && (*ctx))
				log_console->infoStream() <<  "Created GL 3.1 context !";
			else
			{
				context_attribs[1] = 1;
				context_attribs[3] = 0;
				contextError = false;

				log_console->infoStream() <<  "Failed to create GL 3.0 context"
					<< 	" ... using old-style GLX context !" ;
				(*ctx) = glXCreateContextAttribsARB( (*display), bestFbc, shareList, 
						True, context_attribs );
			}
		}

		//sync errors
		XSync( (*display), False );

		//restore error handler
		XSetErrorHandler(oldHandler);

		if (contextError || !(*ctx)){
			log_console->infoStream() <<  "Failed to create an OpenGL context !";
			exit(EXIT_FAILURE);
		}

		// Verifying that context is a direct context
		if ( ! glXIsDirect ( (*display), (*ctx) ) )
			log_console->infoStream() <<  "Indirect GLX rendering context obtained !";
		else
			log_console->infoStream() <<  "Direct GLX rendering context obtained !";

		//EXAMPLE :
		//glXMakeCurrent( (*display), (*win), (*ctx) );
		//glClearColor( 0, 0.5, 1, 1 );
		//glClear( GL_COLOR_BUFFER_BIT );
		//glXSwapBuffers ( (*display), (*win) );

		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//glClearColor ( 1, 0.5, 0, 1 );
		//glClear ( GL_COLOR_BUFFER_BIT );
		//glXSwapBuffers ( (*display), (*win) );

		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	int contextErrorHandler(Display *dpy, XErrorEvent *ev)
	{
		log_console->warnStream() << "OpenGL context error " << ev->error_code << " !";
		contextError = true;
		return 0;
	}
    
    size_t externalTypeToBytes(GLenum externalType) {
        switch(externalType) {
            case(GL_BYTE):
            case(GL_UNSIGNED_BYTE):
            case(GL_UNSIGNED_BYTE_3_3_2):
            case(GL_UNSIGNED_BYTE_2_3_3_REV):
                return sizeof(GLbyte);

            case(GL_SHORT):
            case(GL_UNSIGNED_SHORT):
            case(GL_UNSIGNED_SHORT_5_6_5):
            case(GL_UNSIGNED_SHORT_5_6_5_REV):
            case(GL_UNSIGNED_SHORT_4_4_4_4):
            case(GL_UNSIGNED_SHORT_4_4_4_4_REV):
            case(GL_UNSIGNED_SHORT_5_5_5_1):
            case(GL_UNSIGNED_SHORT_1_5_5_5_REV):
                return sizeof(GLshort);

            case(GL_INT):
            case(GL_FLOAT):
            case(GL_UNSIGNED_INT):
            case(GL_UNSIGNED_INT_8_8_8_8):
            case(GL_UNSIGNED_INT_8_8_8_8_REV):
            case(GL_UNSIGNED_INT_10_10_10_2):
            case(GL_UNSIGNED_INT_2_10_10_10_REV):
                return sizeof(GLint);

            default:
                printf("External type not supported !\n");
                exit(1);
        }
    }

    unsigned int externalFormatToChannelNumber(GLenum externalFormat) {

        switch(externalFormat) {
            case(GL_DEPTH_COMPONENT):
            case(GL_DEPTH_STENCIL):
            case(GL_RED):
            case(GL_RED_INTEGER):
                return 1;

            case(GL_RG):
            case(GL_RG_INTEGER):
                return 2;

            case(GL_RGB):
            case(GL_RGB_INTEGER):
                return 3;

            case(GL_RGBA):
            case(GL_RGBA_INTEGER):
                return 4;

            default:
                printf("External format not supported !\n");
                exit(1);
        }
    }

    GLenum internalFormatToValidExternalFormat(unsigned int internalFormat) {

        switch(internalFormat) {
            case(GL_DEPTH_COMPONENT16):
            case(GL_DEPTH_COMPONENT24):
            case(GL_DEPTH_COMPONENT32F):
                return GL_DEPTH_COMPONENT;

            case(GL_DEPTH24_STENCIL8):
            case(GL_DEPTH32F_STENCIL8):
                return GL_DEPTH_STENCIL;

            case(GL_R16F):
            case(GL_R32F):
            case(GL_R8):
            case(GL_R8_SNORM):
                return GL_RED;

            case(GL_R16I):
            case(GL_R16UI):
            case(GL_R32I):
            case(GL_R32UI):
            case(GL_R8I):
            case(GL_R8UI):
                return GL_RED_INTEGER;

            case(GL_RG16F):
            case(GL_RG32F):
            case(GL_RG8):
            case(GL_RG8_SNORM):
                return GL_RG;

            case(GL_RG16I):
            case(GL_RG16UI):
            case(GL_RG32I):
            case(GL_RG32UI):
            case(GL_RG8I):
            case(GL_RG8UI):
                return GL_RG_INTEGER;

            case(GL_R11F_G11F_B10F):
            case(GL_RGB16F):
            case(GL_RGB32F):
            case(GL_RGB565):
            case(GL_RGB8):
            case(GL_RGB8_SNORM):
            case(GL_RGB9_E5):
            case(GL_SRGB8):
                return GL_RGB;

            case(GL_RGB16I):
            case(GL_RGB16UI):
            case(GL_RGB32I):
            case(GL_RGB32UI):
            case(GL_RGB8I):
            case(GL_RGB8UI):
                return GL_RGB_INTEGER;

            case(GL_RGB10_A2):
            case(GL_RGB5_A1):
            case(GL_RGBA16F):
            case(GL_RGBA32F):
            case(GL_RGBA4):
            case(GL_RGBA8):
            case(GL_RGBA8_SNORM):
            case(GL_SRGB8_ALPHA8):
                return GL_RGBA;

            case(GL_RGB10_A2UI):
            case(GL_RGBA16I):
            case(GL_RGBA16UI):
            case(GL_RGBA32I):
            case(GL_RGBA32UI):
            case(GL_RGBA8I):
            case(GL_RGBA8UI):
                return GL_RGBA_INTEGER;

            default: 
                printf("Internal type not supported !\n");
                exit(1);
        }
    }

    GLenum internalFormatToValidExternalType(unsigned int internalFormat) {

        switch(internalFormat) {

            case(GL_R8I):
            case(GL_R8_SNORM):
            case(GL_RG8I):
            case(GL_RG8_SNORM):
            case(GL_RGB8I):
            case(GL_RGB8_SNORM):
            case(GL_RGBA8I):
            case(GL_RGBA8_SNORM):
                return GL_BYTE;

            case(GL_DEPTH_COMPONENT32F):
            case(GL_R32F):
            case(GL_RG32F):
            case(GL_RGB32F):
            case(GL_RGBA32F):
                return GL_FLOAT;

            case(GL_DEPTH32F_STENCIL8):
                return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;

            case(GL_R16F):
            case(GL_RG16F):
            case(GL_RGB16F):
            case(GL_RGBA16F):
                return GL_HALF_FLOAT;

            case(GL_R32I):
            case(GL_RG32I):
            case(GL_RGB32I):
            case(GL_RGBA32I):
                return GL_INT;

            case(GL_R16I):
            case(GL_RG16I):
            case(GL_RGB16I):
            case(GL_RGBA16I):
                return GL_SHORT;

            case(GL_R8):
            case(GL_R8UI):
            case(GL_RG8):
            case(GL_RG8UI):
            case(GL_RGB565):
            case(GL_RGB5_A1):
            case(GL_RGB8):
            case(GL_RGB8UI):
            case(GL_RGBA4):
            case(GL_RGBA8):
            case(GL_RGBA8UI):
            case(GL_SRGB8_ALPHA8):
            case(GL_SRGB8):
                return GL_UNSIGNED_BYTE;

            case(GL_DEPTH_COMPONENT24):
            case(GL_R32UI):
            case(GL_RG32UI):
            case(GL_RGB32UI):
            case(GL_RGBA32UI):
                return GL_UNSIGNED_INT;

            case(GL_R11F_G11F_B10F):
                return GL_UNSIGNED_INT_10F_11F_11F_REV;

            case(GL_RGB10_A2):
            case(GL_RGB10_A2UI):
                return GL_UNSIGNED_INT_2_10_10_10_REV;

            case(GL_DEPTH24_STENCIL8):
                return GL_UNSIGNED_INT_24_8;

            case(GL_RGB9_E5):
                return GL_UNSIGNED_INT_5_9_9_9_REV;

            case(GL_DEPTH_COMPONENT16):
            case(GL_R16UI):
            case(GL_RG16UI):
            case(GL_RGB16UI):
            case(GL_RGBA16UI):
                return GL_UNSIGNED_SHORT;

            default: 
                assert(false);

        }
    }


    const std::string toStringInternalFormat(unsigned int internalFormat) {
        switch(internalFormat) {
            case(GL_R8I):
                return "GL_R8I";
            case(GL_R8_SNORM):
                return "GL_R8_SNORM";
            case(GL_RG8I):
                return "GL_RG8I";
            case(GL_RG8_SNORM):
                return "GL_RG8_SNORM";
            case(GL_RGB8I):
                return "GL_RGB8I";
            case(GL_RGB8_SNORM):
                return "GL_RGB8_SNORM";
            case(GL_RGBA8I):
                return "GL_RGBA8I";
            case(GL_RGBA8_SNORM):
                return "GL_RGBA8_SNORM";
            case(GL_DEPTH_COMPONENT32F):
                return "GL_DEPTH_COMPONENT32F";
            case(GL_R32F):
                return "GL_R32F";
            case(GL_RG32F):
                return "GL_RG32F";
            case(GL_RGB32F):
                return "GL_RGB32F";
            case(GL_RGBA32F):
                return "GL_RGBA32F";
            case(GL_DEPTH32F_STENCIL8):
                return "GL_DEPTH32F_STENCIL8";
            case(GL_R16F):
                return "GL_R16F";
            case(GL_RG16F):
                return "GL_RG16F";
            case(GL_RGB16F):
                return "GL_RGB16F";
            case(GL_RGBA16F):
                return "GL_RGBA16F";
            case(GL_R32I):
                return "GL_R32I";
            case(GL_RG32I):
                return "GL_RG32I";
            case(GL_RGB32I):
                return "GL_RGB32I";
            case(GL_RGBA32I):
                return "GL_RGBA32I";
            case(GL_R16I):
                return "GL_R16I";
            case(GL_RG16I):
                return "GL_RG16I";
            case(GL_RGB16I):
                return "GL_RGB16I";
            case(GL_RGBA16I):
                return "GL_RGBA16I";
            case(GL_R8):
                return "GL_R8";
            case(GL_R8UI):
                return "GL_R8UI";
            case(GL_RG8):
                return "GL_RG8";
            case(GL_RG8UI):
                return "GL_RG8UI";
            case(GL_RGB565):
                return "GL_RGB565";
            case(GL_RGB5_A1):
                return "GL_RGB5_A1";
            case(GL_RGB8):
                return "GL_RGB8";
            case(GL_RGB8UI):
                return "GL_RGB8UI";
            case(GL_RGBA4):
                return "GL_RGBA4";
            case(GL_RGBA8):
                return "GL_RGBA8";
            case(GL_RGBA8UI):
                return "GL_RGBA8UI";
            case(GL_SRGB8_ALPHA8):
                return "GL_SRGB8_ALPHA8";
            case(GL_SRGB8):
                return "GL_SRGB8";
            case(GL_DEPTH_COMPONENT24):
                return "GL_DEPTH_COMPONENT24";
            case(GL_R32UI):
                return "GL_R32UI";
            case(GL_RG32UI):
                return "GL_RG32UI";
            case(GL_RGB32UI):
                return "GL_RGB32UI";
            case(GL_RGBA32UI):
                return "GL_RGBA32UI";
            case(GL_R11F_G11F_B10F):
                return "GL_R11F_G11F_B10F";
            case(GL_RGB10_A2):
                return "GL_RGB10_A2";
            case(GL_RGB10_A2UI):
                return "GL_RGB10_A2UI";
            case(GL_DEPTH24_STENCIL8):
                return "GL_DEPTH24_STENCIL8";
            case(GL_RGB9_E5):
                return "GL_RGB9_E5";
            case(GL_DEPTH_COMPONENT16):
                return "GL_DEPTH_COMPONENT16";
            case(GL_R16UI):
                return "GL_R16UI";
            case(GL_RG16UI):
                return "GL_RG16UI";
            case(GL_RGB16UI):
                return "GL_RGB16UI";
            case(GL_RGBA16UI):
                return "GL_RGBA16UI";

            case(GL_LUMINANCE8):
                return "GL_LUMINANCE8";
            case(GL_LUMINANCE16):
                return "GL_LUMINANCE16";
            case(GL_LUMINANCE16F_ARB):
                return "GL_LUMINANCE16F_ARB";
            case(GL_LUMINANCE32F_ARB):
                return "GL_LUMINANCE32F_ARB";
            case(GL_LUMINANCE8UI_EXT):
                return "GL_LUMINANCE8UI_EXT";
            case(GL_LUMINANCE16UI_EXT):
                return "GL_LUMINANCE16UI_EXT";
            case(GL_LUMINANCE32UI_EXT):
                return "GL_LUMINANCE32UI_EXT";
            case(GL_LUMINANCE8I_EXT):
                return "GL_LUMINANCE8I_EXT";
            case(GL_LUMINANCE16I_EXT):
                return "GL_LUMINANCE16I_EXT";
            case(GL_LUMINANCE32I_EXT):
                return "GL_LUMINANCE32I_EXT";

            case(GL_ALPHA8):
                return "GL_ALPHA8";
            case(GL_ALPHA16):
                return "GL_ALPHA16";
            case(GL_ALPHA16F_ARB):
                return "GL_ALPHA16F_ARB";
            case(GL_ALPHA32F_ARB):
                return "GL_ALPHA32F_ARB";
            case(GL_ALPHA8UI_EXT):
                return "GL_ALPHA8UI_EXT";
            case(GL_ALPHA16UI_EXT):
                return "GL_ALPHA16UI_EXT";
            case(GL_ALPHA32UI_EXT):
                return "GL_ALPHA32UI_EXT";
            case(GL_ALPHA8I_EXT):
                return "GL_ALPHA8I_EXT";
            case(GL_ALPHA16I_EXT):
                return "GL_ALPHA16I_EXT";
            case(GL_ALPHA32I_EXT):
                return "GL_ALPHA32I_EXT";

                //case(GL_LUMINANCE_ALPHA8):
                //return "GL_LUMINANCE_ALPHA8";
                //case(GL_LUMINANCE_ALPHA16):
                //return "GL_LUMINANCE_ALPHA16";
            case(GL_LUMINANCE_ALPHA16F_ARB):
                return "GL_LUMINANCE_ALPHA16F_ARB";
            case(GL_LUMINANCE_ALPHA32F_ARB):
                return "GL_LUMINANCE_ALPHA32F_ARB";
            case(GL_LUMINANCE_ALPHA8UI_EXT):
                return "GL_LUMINANCE_ALPHA8UI_EXT";
            case(GL_LUMINANCE_ALPHA16UI_EXT):
                return "GL_LUMINANCE_ALPHA16UI_EXT";
            case(GL_LUMINANCE_ALPHA32UI_EXT):
                return "GL_LUMINANCE_ALPHA32UI_EXT";
            case(GL_LUMINANCE_ALPHA8I_EXT):
                return "GL_LUMINANCE_ALPHA8I_EXT";
            case(GL_LUMINANCE_ALPHA16I_EXT):
                return "GL_LUMINANCE_ALPHA16I_EXT";
            case(GL_LUMINANCE_ALPHA32I_EXT):
                return "GL_LUMINANCE_ALPHA32I_EXT";

            case(GL_INTENSITY8):
                return "GL_INTENSITY8";
            case(GL_INTENSITY16):
                return "GL_INTENSITY16";
            case(GL_INTENSITY16F_ARB):
                return "GL_INTENSITY16F_ARB";
            case(GL_INTENSITY32F_ARB):
                return "GL_INTENSITY32F_ARB";
            case(GL_INTENSITY8UI_EXT):
                return "GL_INTENSITY8UI_EXT";
            case(GL_INTENSITY16UI_EXT):
                return "GL_INTENSITY16UI_EXT";
            case(GL_INTENSITY32UI_EXT):
                return "GL_INTENSITY32UI_EXT";
            case(GL_INTENSITY8I_EXT):
                return "GL_INTENSITY8I_EXT";
            case(GL_INTENSITY16I_EXT):
                return "GL_INTENSITY16I_EXT";
            case(GL_INTENSITY32I_EXT):
                return "GL_INTENSITY32I_EXT";

            default:
                return "UNKNOWN_INTERNAL_TYPE";
        }
    }
    const std::string toStringExternalFormat(GLenum externalFormat) {
        switch(externalFormat) {
            case(GL_RED):
                return "GL_RED";
            case(GL_RED_INTEGER):
                return "GL_RED_INTEGER";
            case(GL_RG):
                return "GL_RG";
            case(GL_RG_INTEGER):
                return "GL_RG_INTEGER";
            case(GL_RGB):
                return "GL_RGB";
            case(GL_RGB_INTEGER):
                return "GL_RGB_INTEGER";
            case(GL_RGBA):
                return "GL_RGBA";
            case(GL_RGBA_INTEGER):
                return "GL_RGBA_INTEGER";
            case(GL_DEPTH_COMPONENT):
                return "GL_DEPTH_COMPONENT";
            case(GL_DEPTH_STENCIL):
                return "GL_DEPTH_STENCIL";
            case(GL_LUMINANCE_ALPHA):
                return "GL_LUMINANCE_ALPHA";
            case(GL_LUMINANCE):
                return "GL_LUMINANCE";
            case(GL_ALPHA):
                return "GL_ALPHA";
            default:
                return "UNKNOWN_EXTERNAL_FORMAT";
        }
    }
    const std::string toStringExternalType(GLenum externalType) {

        switch(externalType) {
            case(GL_UNSIGNED_BYTE):
                return "GL_UNSIGNED_BYTE";
            case(GL_BYTE):
                return "GL_BYTE";
            case(GL_UNSIGNED_SHORT):
                return "GL_UNSIGNED_SHORT";
            case(GL_SHORT):
                return "GL_SHORT";
            case(GL_UNSIGNED_INT):
                return "GL_UNSIGNED_INT";
            case(GL_INT):
                return "GL_INT";
            case(GL_HALF_FLOAT):
                return "GL_HALF_FLOAT";
            case(GL_FLOAT):
                return "GL_FLOAT";
            case(GL_UNSIGNED_SHORT_5_6_5):
                return "GL_UNSIGNED_SHORT_5_6_5";
            case(GL_UNSIGNED_SHORT_4_4_4_4):
                return "GL_UNSIGNED_SHORT_4_4_4_4";
            case(GL_UNSIGNED_SHORT_5_5_5_1):
                return "GL_UNSIGNED_SHORT_5_5_5_1";
            case(GL_UNSIGNED_INT_2_10_10_10_REV):
                return "GL_UNSIGNED_INT_2_10_10_10_REV";
            case(GL_UNSIGNED_INT_10F_11F_11F_REV):
                return "GL_UNSIGNED_INT_10F_11F_11F_REV";
            case(GL_UNSIGNED_INT_5_9_9_9_REV):
                return "GL_UNSIGNED_INT_5_9_9_9_REV";
            case(GL_UNSIGNED_INT_24_8):
                return "GL_UNSIGNED_INT_24_8";
            case(GL_FLOAT_32_UNSIGNED_INT_24_8_REV):
                return "GL_FLOAT_32_UNSIGNED_INT_24_8_REV";
            default:
                return "UNKNOWN_EXTERNAL_TYPE";
        }
    }

    const std::string toStringTextureTarget(GLenum textureTarget) {

        switch(textureTarget) {
            case(GL_TEXTURE_1D):
                return "GL_TEXTURE_1D";
            case(GL_TEXTURE_2D):
                return "GL_TEXTURE_2D";
            case(GL_TEXTURE_3D):
                return "GL_TEXTURE_3D";
            case(GL_TEXTURE_1D_ARRAY):
                return "GL_TEXTURE_1D_ARRAY";
            case(GL_TEXTURE_2D_ARRAY):
                return "GL_TEXTURE_2D_ARRAY";
            case(GL_TEXTURE_RECTANGLE):
                return "GL_TEXTURE_RECTANGLE";
            case(GL_TEXTURE_CUBE_MAP):
                return "GL_TEXTURE_CUBE_MAP";
            case(GL_TEXTURE_CUBE_MAP_ARRAY):
                return "GL_TEXTURE_CUBE_MAP_ARRAY";
            case(GL_TEXTURE_BUFFER):
                return "GL_TEXTURE_BUFFER";
            case(GL_TEXTURE_2D_MULTISAMPLE):
                return "GL_TEXTURE_2D_MULTISAMPLE";
            case(GL_TEXTURE_2D_MULTISAMPLE_ARRAY):
                return "GL_TEXTURE_2D_MULTISAMPLE_ARRAY";
            default:
                    return "UNKNOWN_TEXTURE_TARGET";
        }
    }
	
    void checkFrameBufferStatus() {
        using log4cpp::log_console;

		switch(glCheckFramebufferStatus(GL_FRAMEBUFFER))  {
			case GL_FRAMEBUFFER_COMPLETE:
				log_console->infoStream() << "Framebuffer complete !";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				log_console->errorStream() << "Framebuffer incomplete layer targets !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				log_console->errorStream() << "Framebuffer incomplete attachement !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
				log_console->errorStream() << "Framebuffer missing attachment !";
				exit(1);
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				log_console->errorStream() << "Framebuffer unsupported !";
				exit(1);
				break;
			default:
				log_console->errorStream() << "Something went wrong when configuring the framebuffer !";
				exit(1);
		}
	}
}

#endif
