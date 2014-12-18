
#ifndef RENDERROOT_H
#define RENDERROOT_H

#include "headers.hpp"
#ifdef GUI_ENABLED

#include "renderTree.hpp"

class RenderRoot : public RenderTree {

    public:
        explicit RenderRoot(bool active = true) : RenderTree(active) {};

        ~RenderRoot() {};

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
};

#endif
#endif /* end of include guard: RENDERROOT_H */
