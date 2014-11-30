
#ifndef RENDERROOT_H
#define RENDERROOT_H

#include "headers.hpp"
#include "renderTree.hpp"

class RenderRoot : public RenderTree {
	
	public:
		explicit RenderRoot(bool active = true) : RenderTree(active) {};

		~RenderRoot() {};
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
};

#endif /* end of include guard: RENDERROOT_H */
