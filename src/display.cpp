#include "utils/headers.hpp"
#include "utils/logs/log.hpp"
#include "utils/opengl/program/program.hpp"
#include "utils/globals.hpp"
#include "utils/opengl/texture/texture.hpp"
#include "rendering/renderable/renderRoot.hpp"
#include "utils/random/rand.h"

#include <qapplication.h>
#include <QWidget>
#include <ctime>

using namespace std;
using namespace log4cpp;

int main(int argc, char** argv) {

#ifdef GUI_ENABLED
        //random
        srand(time(NULL));

        //logs
        log4cpp::initLogs();

        //cuda
        //CudaUtils::logCudaDevices(log_console);

        log_console->infoStream() << "[Rand Init] ";
        log_console->infoStream() << "[Logs Init] ";

        // glut initialisation (mandatory) 
        glutInit(&argc, argv);
        log_console->infoStream() << "[Glut Init] ";

        // Read command lines arguments.
        QApplication application(argc,argv);
        log_console->infoStream() << "[Qt Init] ";

        // Instantiate the viewer (mandatory)
        Viewer *viewer = new Viewer();
        viewer->setWindowTitle("Flocking boids");
        viewer->show();
        Globals::viewer = viewer;

        //glew initialisation (mandatory)
        log_console->infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

        //global vars
        Globals::init();
        Globals::print(std::cout);
        Globals::check();

        //texture manager
        Texture::init();

        log_console->infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
        //FIN INIT//
        //
        RenderRoot *root = new RenderRoot(); 
#endif
}
