/*
 * Main GL viewer class, based on the QGLViewer library.
 * 
 * QGLViewer is a Qt-based viewer, which provides an OpenGL rendering
 * context and GUI event management.
 * The documentation of the QGLViewer library can be found at the following 
 * url: http://www.libqglviewer.com
 */

#ifndef _VIEWER_
#define _VIEWER_

#ifdef GUI_ENABLED

#include "headers.hpp"
#include <list>

class Renderable;

class Viewer : public QGLViewer
{
	public :
	        
		Viewer();
		virtual ~Viewer();
		void addRenderable(Renderable *r);

/* Scene methods */
	protected :
		/// List of the scene objects, to render, animate, ...
        std::list<Renderable *> renderableList;
		
		/// Create the scene and initializes rendering parameters
		virtual void init();
		
		/// Draw every objects of the scene
		virtual void draw();
		
		/// Animate every objects of the scene
		virtual void animate();


/* Viewing parameters */
	protected :
		bool toogleWireframe;
		bool toogleLight;
        bool toggleRecord;

		/// Handle keyboard events specifically
		virtual void keyPressEvent(QKeyEvent *e);

		/// Handle keyboard events specifically
		virtual void mouseMoveEvent(QMouseEvent *e);
		
		/// Draw every objects of the scene
		virtual std::string help() const;
};

#endif
#endif
