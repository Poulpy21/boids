
#ifndef NEIGHBORIDS_H
#define NEIGHBORIDS_H

#include <list>
#include "triState.hpp"
#include "relativePosition.hpp"
#include "vec3.hpp"

struct NeighborIds {
   
    NeighborIds(unsigned int globalCellId) { init(globalCellId); }
    virtual ~NeighborIds() {}
    
    virtual void init(unsigned int globalCellId) {};
    virtual void update(unsigned int borderId, std::list<unsigned int> ids) {}

    virtual std::list<unsigned int> getNeighborCellIds(const RelativePosition &relPos) {
        return neighborIds[toBorderId(relPos)];
    }
    
    unsigned int toBorderId(const RelativePosition &relPos) {
        return relPos.x + relPos.y*3u + relPos.z*9u;
    }
   
protected:
    std::list<unsigned int> neighborIds[27u];
};

#endif /* end of include guard: NEIGHBORIDS_H */
