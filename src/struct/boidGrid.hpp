

#ifndef BOID_GRID_H
#define BOID_GRID_H

#include "headers.hpp"
#include "boidDataStructure.hpp"
#include "PinnedCPUResource.hpp"
#include "vec3.hpp"
#include <vector>

#ifdef THRUST_ENABLED
template <typename T>
class BoidGrid;

template <typename T>
__HOST__ void initBoidGridThrustArrays(const BoidGrid<T> &boidGrid, T* agents_h, unsigned int nAgents);
#endif

template <typename T>
class BoidGrid : public BoidDataStructure<T> {

    public:
        BoidGrid(const BoundingBox<3u, Real> &domain, Real minRadius);
        BoidGrid(const BoidGrid<T> &other);
        BoidGrid<T>& operator=(const BoidGrid<T> &other);
        ~BoidGrid();
    
        void init(T* agent_h, unsigned int agentCount);

        unsigned int          getTotalAgentCount() const override;
        unsigned int          getAgentCount(unsigned int cellId)    const override;
        T*                    getAgentArray(unsigned int cellId)    const override;
        PinnedCPUResource<T>  getAgentResource(unsigned int cellId) const;

        unsigned int getCellId    (const Vec3<T> &pos) const override;
        std::vector<unsigned int> getNeighborCellIds(unsigned int cellId) const override;

        T getMinRadius() const;
        T getDomainWidth() const;
        T getDomainLength() const;
        T getDomainHeight() const;
        unsigned int getWidth() const;
        unsigned int getHeight() const;
        unsigned int getLength() const;
        unsigned int getCellsCount() const;
        Vec3<unsigned int> getBoxSize() const;

        std::string toString() const;

    private:

        unsigned int makeId(unsigned int x, unsigned int y, unsigned int z) const;
        Vec3<T> relativePos(const Vec3<T> &pos) const;

    private:
        const T minRadius;

        const T domainWidth, domainLength, domainHeight;
        const unsigned int width, length, height;
        const Vec3<unsigned int> boxSize;

        const unsigned int nCells;

        unsigned int agentCount;

        std::vector<PinnedCPUResource<T> > agents_h;
};

template <typename T>
std::ostream& operator<<(std::ostream &os, const BoidGrid<T> &grid) {
    os << grid.toString();
    return os;
}
        
template <typename T>
std::string BoidGrid<T>::toString() const {
    std::stringstream ss;

    ss << "BoidGrid<";
#ifndef __CUDACC__
    utils::templatePrettyPrint<T>(ss);
#endif
    ss << ">";

    ss << "\n\tMin radius : " << getMinRadius();
    ss << "\n\tDomain: min " << this->domain.min << "\t max " << this->domain.max;
    ss << "\n\tBox   : " << getBoxSize();
    ss << "\n\tCells : " << getCellsCount();

    return ss.str();
}
        
template <typename T>
void BoidGrid<T>::init(T* agent_h, unsigned int agentCount) {
#ifdef THRUST_ENABLED
        initBoidGridThrustArrays<T>(*this, agent_h, agentCount);
#endif
}

template <typename T>
T BoidGrid<T>::getMinRadius() const {
    return minRadius;
}
template <typename T>
T BoidGrid<T>::getDomainWidth() const {
    return domainWidth;
}
template <typename T>
T BoidGrid<T>::getDomainLength() const {
    return domainLength;
}
template <typename T>
T BoidGrid<T>::getDomainHeight() const {
    return domainHeight;
}
template <typename T>
unsigned int BoidGrid<T>::getWidth() const {
    return width;
}
template <typename T>
unsigned int BoidGrid<T>::getHeight() const {
    return height;
}
template <typename T>
unsigned int BoidGrid<T>::getLength() const {
    return length;
}
template <typename T>
unsigned int BoidGrid<T>::getCellsCount() const {
    return nCells;
}
template <typename T>
Vec3<unsigned int> BoidGrid<T>::getBoxSize() const {
    return boxSize;
}


template <typename T>
BoidGrid<T>::BoidGrid(const BoundingBox<3u, Real> &domain, Real minRadius) :
    BoidDataStructure<T>(domain),
    minRadius(minRadius),
    domainWidth (domain.max[0] - domain.min[0]),
    domainLength(domain.max[1] - domain.min[1]),
    domainHeight(domain.max[2] - domain.min[2]),
    width (std::max(1,static_cast<int>(ceil(domainWidth /minRadius)))),
    length(std::max(1,static_cast<int>(ceil(domainLength/minRadius)))),
    height(std::max(1,static_cast<int>(ceil(domainHeight/minRadius)))),
    boxSize(width, length, height),
    nCells(width*length*height),
    agentCount(0u)
{
}

template <typename T>
BoidGrid<T>::~BoidGrid() {
}

template <typename T>
BoidGrid<T>::BoidGrid(const BoidGrid<T> &other) :
    BoidDataStructure<T>(other),
    minRadius(0),
    domainWidth (0),
    domainLength(0),
    domainHeight(0),
    width (0),
    length(0),
    height(0),
    boxSize(0),
    nCells(0),
    agentCount(0) {
    throw std::logic_error("Cannot copy a BoidGrid.");
}

template <typename T>
BoidGrid<T>& BoidGrid<T>::operator=(const BoidGrid<T> &other) {
    throw std::logic_error("Cannot copy a BoidGrid.");
}

template <typename T>
unsigned int BoidGrid<T>::getTotalAgentCount() const {
    return agentCount;
}

template <typename T>
unsigned int BoidGrid<T>::getAgentCount(unsigned int cellId) const {
    return agents_h[cellId].size();
}

template <typename T>
T* BoidGrid<T>::getAgentArray(unsigned int cellId) const {
    return agents_h[cellId].data();
}

template <typename T>
PinnedCPUResource<T> BoidGrid<T>::getAgentResource(unsigned int cellId) const {
    return agents_h[cellId];
}

template <typename T>
unsigned int BoidGrid<T>::getCellId(const Vec3<T> &pos) const {
    Vec3<float> relPos = relativePos(pos);
    return makeId(relPos.x * boxSize.x, relPos.y * boxSize.y, relPos.z * boxSize.z);
}

template <typename T>
std::vector<unsigned int> BoidGrid<T>::getNeighborCellIds(unsigned int cellId) const {
    std::vector<unsigned int> neighbors;
    return neighbors;
}

template <typename T>
unsigned int BoidGrid<T>::makeId(unsigned int x, unsigned int y, unsigned int z) const {
    return (width*length*z + width*y + x);
}

template <typename T>
Vec3<T> BoidGrid<T>::relativePos(const Vec3<T> &pos) const {
    return Vec3<T>(
            (pos.x - this->domain.min[0])/domainWidth,
            (pos.y - this->domain.min[1])/domainLength,
            (pos.z - this->domain.min[2])/domainHeight
            );
}
        

#endif /* end of include guard: GRID_H */
