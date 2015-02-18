

#ifndef BOID_GRID_H
#define BOID_GRID_H

#include "headers.hpp"
#include "boidDataStructure.hpp"
#include "boidMemoryView.hpp"
#include "PinnedCPUResource.hpp"
#include "vec3.hpp"
#include <vector>

#ifdef CUDA_ENABLED
#include "GPUResource.hpp"
#endif

#ifdef THRUST_ENABLED
template <typename T>
class BoidGrid;

template <typename T>
__HOST__ void initBoidGridThrustArrays(const BoidGrid<T> &boidGrid, 
        BoidMemoryView<T> &agents_h, BoidMemoryView<T> &agents_d, 
        unsigned int nAgents);
#endif

template <typename T>
class BoidGrid : public BoidDataStructure<T> {

    public:
        BoidGrid(const BoundingBox<3u, Real> &domain, Real maxRadius);
        BoidGrid(const BoidGrid<T> &other);
        BoidGrid<T>& operator=(const BoidGrid<T> &other);
        ~BoidGrid();
    
        //initialize structure with given host boids
        //All boids should initially be contained in the domain
        void init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) override;
    
        //compute one step and return boids that went out the domain 
        //if keepBoidsInDomain is set to true, no boids are returned
        BoidMemoryView<T> computeStep(bool keepBoidsInDomain) override;

        unsigned int          getTotalAgentCount() const override;
        unsigned int          getAgentCount(unsigned int cellId)    const override;
        T*                    getAgentArray(unsigned int cellId)    const override;
        PinnedCPUResource<T>  getAgentResource(unsigned int cellId) const;

        unsigned int getCellId    (const Vec3<T> &pos) const override;
        std::vector<unsigned int> getNeighborCellIds(unsigned int cellId) const override;

        T getMaxRadius() const;
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
        const T maxRadius;

        const T domainWidth, domainLength, domainHeight;
        const unsigned int width, length, height;
        const Vec3<unsigned int> boxSize;

        const unsigned int nCells;

        unsigned int agentCount;

        PinnedCPUResource<T> agents_h;
        BoidMemoryView<T> agents_view_h;

#ifdef CUDA_ENABLED
        GPUResource<T> agents_d;
        BoidMemoryView<T> agents_view_d;
#endif
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

    ss << "\n\tMax radius : " << getMaxRadius();
    ss << "\n\tDomain: min " << this->domain.min << "\t max " << this->domain.max;
    ss << "\n\tBox   : " << getBoxSize();
    ss << "\n\tCells : " << getCellsCount();

    return ss.str();
}
        
template <typename T>
void BoidGrid<T>::init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) {
    
    log4cpp::log_console->infoStream() << "Initializing Boid Grid with " << agentCount << " boids !";

    agents_h = PinnedCPUResource<T>(agent_h.data(), 10u*agentCount, false);
    agents_view_h = agent_h;

#ifdef CUDA_ENABLED
    this->agentCount = agentCount;
   
    agents_d = GPUResource<T>(0, 10u*agentCount);
    agents_d.allocate();
    agents_view_d = BoidMemoryView<T>(agents_d.data(), agentCount);

    initBoidGridThrustArrays<T>(*this, agents_view_h, agents_view_d, agentCount);
#endif
}
        
//compute one step and return boids that went out the domain 
//if keepBoidsInDomain is set to true, no boids are returned
template <typename T>
BoidMemoryView<T> BoidGrid<T>::computeStep(bool keepBoidsInDomain) {
    BoidMemoryView<T> outofDomainBoids_h;
    return outofDomainBoids_h;
}

template <typename T>
T BoidGrid<T>::getMaxRadius() const {
    return maxRadius;
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
BoidGrid<T>::BoidGrid(const BoundingBox<3u, Real> &domain, Real maxRadius) :
    BoidDataStructure<T>(domain),
    maxRadius(maxRadius),
    domainWidth (domain.max[0] - domain.min[0]),
    domainLength(domain.max[1] - domain.min[1]),
    domainHeight(domain.max[2] - domain.min[2]),
    width (std::max(1,static_cast<int>(ceil(domainWidth /maxRadius)))),
    length(std::max(1,static_cast<int>(ceil(domainLength/maxRadius)))),
    height(std::max(1,static_cast<int>(ceil(domainHeight/maxRadius)))),
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
    maxRadius(0),
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
    //return agents_h[cellId].size();
    //TODO
    return 0;
}

template <typename T>
T* BoidGrid<T>::getAgentArray(unsigned int cellId) const {
    //return agents_h[cellId].data();
    //TODO
    return 0;
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
