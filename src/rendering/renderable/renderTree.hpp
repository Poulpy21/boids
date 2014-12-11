
#ifndef RENDERTREE_H
#define RENDERTREE_H

#include "headers.hpp"
#ifdef USE_GUI

#include "renderable.hpp"

#include <string>
#include <map>
#include <iostream>

// Depth first drawing tree
// Childs are drawn according to their priority //TODO
// Father node draw can be done before and/or after all its children
// The same possibilities are available for animate()
// Note : Your class should inherit RenderTree instead of Renderable
// Warning : Renderable init func is deprecated

class RenderTree : public Renderable {

    public:
        virtual ~RenderTree();

        //draw the whole tree with initial modelMatrix set to 4x4 Identity
        //(renderable draw wrapper)
        void draw(const float *currentTransformationMatrix = consts::identity4);

        const float *getRelativeModelMatrix() const;
        void setRelativeModelMatrix(const float *matrix);

        void addChild(std::string key, RenderTree *child);
        void removeChild(std::string key);

        void desactivateChild(std::string childName);
        void activateChild(std::string childName);

        void move(float x, float y, float z);
        void move(Vec<float> v);
        void orientate(qglviewer::Quaternion rot, float scale);

        void translate(float x, float y, float z);
        void translate(Vec<float> v);
        void scale(float alpha);
        void scale(float alpha, float beta, float gamma);
        void scale(Vec<float> v);
        void rotate(qglviewer::Quaternion rot);
        void pushMatrix(const float *matrix); //matrice 4x4

        void moveChild(std::string childName, float x, float y, float z);
        void moveChild(std::string childName, Vec<float> v);
        void orientateChild(std::string childName, qglviewer::Quaternion rot, float scale);
        void translateChild(std::string childName, float x, float y, float z);
        void translateChild(std::string childName, Vec<float> v);
        void scaleChild(std::string childName, float alpha);
        void scaleChild(std::string childName, float alpha, float beta, float gamma);
        void scaleChild(std::string childName, Vec<float> v);
        void rotateChild(std::string childName, qglviewer::Quaternion rot);
        void pushMatrixToChild(std::string childName, const float *matrix); //matrice 4x4

    protected:
        RenderTree(bool active = true);

        //render current node before all children
        virtual void drawDownwards(const float *currentTransformationMatrix = consts::identity4) = 0;
        //render current node after all children
        virtual void drawUpwards(const float *currentTransformationMatrix = consts::identity4) {};
        //NOTE: Changed const-qualifier, const was to restrictive for drawing.

        //same thing with animations
        virtual void animateDownwards() {};
        virtual void animateUpwards() {};

        virtual void keyPressEvent(QKeyEvent*) {};
        virtual void mouseMoveEvent(QMouseEvent*) {};

        float *relativeModelMatrix;

    private:
        //QGLViewer overrides (via renderable)
        void draw();
        void animate();
        void keyPressEvent(QKeyEvent*, Viewer&);
        void mouseMoveEvent(QMouseEvent*, Viewer&);
        ///////////////////////////////////////

        bool active;
        std::map<std::string, RenderTree*> children;
};

#endif
#endif /* end of include guard: RENDERTREE_H */

