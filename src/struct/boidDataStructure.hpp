
#ifndef BOIDDATASTRUCTURE_H
#define BOIDDATASTRUCTURE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include "vec3.hpp"
#include <vector>

template <typename T>
class BoidDataStructure {

public:
    BoidDataStructure(const BoundingBox<3u, T> &domain) : domain(domain) {}
    ~BoidDataStructure() {}

    BoundingBox<3u,T> getDomain() const {
        return domain;
    }

protected:
    const BoundingBox<3u, T> domain;

public:
    
    virtual void init(T* agent_h, unsigned int agentCount) = 0;

    virtual unsigned int getTotalAgentCount() const = 0;
    virtual unsigned int getAgentCount(unsigned int cellId) const = 0;
    virtual T*           getAgentArray(unsigned int cellId) const = 0;

    virtual unsigned int getCellId(const Vec3<T> &pos) const = 0;
    virtual std::vector<unsigned int> getNeighborCellIds(unsigned int cellId) const = 0;
};

#endif /* end of include guard: BOIDDATASTRUCTURE_H */
