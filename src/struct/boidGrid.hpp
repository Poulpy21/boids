

#ifndef BOID_GRID_H
#define BOID_GRID_H

#include "headers.hpp"
#include "localBoidDataStructure.hpp"
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
__HOST__ void initBoidGridThrustArrays(BoidGrid<T> &boidGrid);

template <typename T>
__HOST__ BoidMemoryView<T> computeThrustStep(BoidGrid<T> &boidGrid);

#endif

template <typename T>
class BoidGrid : public LocalBoidDataStructure<T> {

    public:
        BoidGrid(unsigned int globalId,
                const BoundingBox<3u, Real> &globalDomain,
                const BoundingBox<3u, Real> &localDomain,
                bool keepBoidsInGlobalDomain,
                Real maxRadius);
        BoidGrid(const BoidGrid<T> &other);
        BoidGrid<T>& operator=(const BoidGrid<T> &other);
        ~BoidGrid();
      
    //Interface
    public:
        void init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) override;
        void feed(const BoidMemoryView<T> &agent_h, unsigned int agentCount) override; 
        BoidMemoryView<T> computeLocalStep() override; 

        unsigned int getLocalAgentCount(unsigned int localCellId) const override;
        bool isLocalCellAtCorner(unsigned int localCellId) const override;

    protected:
        unsigned int getLocalCellId(const Vec3<T> &pos) const override;
        NeighborIds& getGlobalNeighborCellIds(unsigned int globalCellId) const override;
        BoidMemoryView<T> getLocalHostAgentsArray(unsigned int localCellId) const override;
  

    //Implementation specific
    public:
        BoidMemoryView<T>& getBoidHostMemoryView();

        T getMaxRadius() const;
        T getDomainWidth() const;
        T getDomainHeight() const;
        T getDomainLength() const;
        unsigned int getWidth() const;
        unsigned int getHeight() const;
        unsigned int getLength() const;
        unsigned int getCellsCount() const;
        Vec3<unsigned int> getBoxSize() const;

        std::string toString() const;

    protected:
        Vec3<T> relativePos(const Vec3<T> &pos) const;
        unsigned int makeLocalId(unsigned int x, unsigned int y, unsigned int z) const;

    protected:
        const T maxRadius;

        const T domainWidth, domainHeight, domainLength;
        const unsigned int width, height, length;
        const Vec3<unsigned int> boxSize;

        const unsigned int nCells;


#ifdef CUDA_ENABLED
    protected:
        GPUResource<T> agents_d;
        BoidMemoryView<T> agents_view_d;

        GPUResource<unsigned int> uniqueIds_d;
        GPUResource<unsigned int> count_d;
        GPUResource<unsigned int> offsets_d;
        GPUResource<int> validIds_d;

    public:
        BoidMemoryView<T>& getBoidDeviceMemoryView();
        GPUResource<T>& getDeviceBoids();
        GPUResource<unsigned int>& getDeviceUniqueIds();
        GPUResource<unsigned int>& getDeviceCount();
        GPUResource<unsigned int>& getDeviceOffsets();
        GPUResource<int>& getDeviceValidIds();
#endif

};

        
template <typename T>
void BoidGrid<T>::init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) {
    this->agents_h = PinnedCPUResource<T>(agent_h.data(), BoidMemoryView<T>::N*agentCount, false);
    this->agents_view_h = agent_h;

#ifdef CUDA_ENABLED
    this->agentCount = agentCount;
   
    agents_d = GPUResource<T>(0, BoidMemoryView<T>::N*agentCount);
    agents_d.allocate();
    agents_view_d = BoidMemoryView<T>(agents_d.data(), agentCount);

    initBoidGridThrustArrays<T>(*this);
#else
    NOT_IMPLEMENTED_YET;
#endif
}
        
template <typename T>
void BoidGrid<T>::feed(const BoidMemoryView<T> &agent_h, unsigned int agentCount) {
#ifdef CUDA_ENABLED
    NOT_IMPLEMENTED_YET;
#else
    NOT_IMPLEMENTED_YET;
#endif
}
        
//compute one step and return boids that went out the domain 
//if keepBoidsInDomain is set to true, no boids are returned
template <typename T>
BoidMemoryView<T> BoidGrid<T>::computeLocalStep() {
#ifdef CUDA_ENABLED
    return computeThrustStep(*this);
#else
    NOT_IMPLEMENTED_YET;
#endif
}

template <typename T>
BoidGrid<T>::BoidGrid(unsigned int globalId,
                const BoundingBox<3u, Real> &localDomain,
                const BoundingBox<3u, Real> &globalDomain,
                bool keepBoidsInGlobalDomain,
                Real maxRadius) :
    LocalBoidDataStructure<T>(globalId, localDomain, globalDomain, keepBoidsInGlobalDomain),
    maxRadius(maxRadius),
    domainWidth (localDomain.max[0] - localDomain.min[0]),
    domainHeight(localDomain.max[1] - localDomain.min[1]),
    domainLength(localDomain.max[2] - localDomain.min[2]),
    width (std::max(1,static_cast<int>(ceil(domainWidth /maxRadius)))),
    height(std::max(1,static_cast<int>(ceil(domainHeight/maxRadius)))),
    length(std::max(1,static_cast<int>(ceil(domainLength/maxRadius)))),
    boxSize(width, height, length),
    nCells(width*height*length)
#ifdef CUDA_ENABLED 
        ,agents_d(0,0),
        agents_view_d(),
        uniqueIds_d(0,0),
        count_d(0,0),
        offsets_d(0,0),
        validIds_d(0,0)
#endif
{}

template <typename T>
BoidGrid<T>::~BoidGrid() {
}

template <typename T>
BoidGrid<T>::BoidGrid(const BoidGrid<T> &other) :
    LocalBoidDataStructure<T>(other),
    maxRadius(0),
    domainWidth (0),
    domainHeight(0),
    domainLength(0),
    width (0),
    height(0),
    length(0),
    boxSize(0),
    nCells(0) {
        throw std::logic_error("Cannot copy a BoidGrid.");
    }

template <typename T>
BoidGrid<T>& BoidGrid<T>::operator=(const BoidGrid<T> &other) {
    throw std::logic_error("Cannot copy a BoidGrid.");
}

template <typename T>
unsigned int BoidGrid<T>::getLocalAgentCount(unsigned int cellId) const {
    NOT_IMPLEMENTED_YET;
}
        
template <typename T>
bool BoidGrid<T>::isLocalCellAtCorner(unsigned int localCellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T>
BoidMemoryView<T> BoidGrid<T>::getLocalHostAgentsArray(unsigned int cellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T>
unsigned int BoidGrid<T>::getLocalCellId(const Vec3<T> &pos) const {
    Vec3<float> relPos = relativePos(pos);
    return makeLocalId(relPos.x * boxSize.x, relPos.y * boxSize.y, relPos.z * boxSize.z);
}
        
template <typename T>
NeighborIds& BoidGrid<T>::getGlobalNeighborCellIds(unsigned int globalCellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T>
unsigned int BoidGrid<T>::makeLocalId(unsigned int x, unsigned int y, unsigned int z) const {
    return (width*height*z + width*y + x);
}

template <typename T>
Vec3<T> BoidGrid<T>::relativePos(const Vec3<T> &pos) const {
    return Vec3<T>(
            (pos.x - this->localDomain.min[0])/domainWidth,
            (pos.y - this->localDomain.min[1])/domainHeight,
            (pos.z - this->localDomain.min[2])/domainLength
            );
}

template <typename T>
BoidMemoryView<T>& BoidGrid<T>::getBoidHostMemoryView() {
    return this->agents_view_h;
}
        
#ifdef CUDA_ENABLED
template <typename T>
GPUResource<unsigned int>& BoidGrid<T>::getDeviceOffsets() {
    return this->offsets_d;
}

template <typename T>
GPUResource<unsigned int>& BoidGrid<T>::getDeviceUniqueIds() {
    return this->uniqueIds_d;
}
    
template <typename T>
GPUResource<int>& BoidGrid<T>::getDeviceValidIds() {
    return this->validIds_d;
}
        
template <typename T>
GPUResource<unsigned int>& BoidGrid<T>::getDeviceCount() {
    return this->count_d;
}
       
template <typename T>
BoidMemoryView<T>& BoidGrid<T>::getBoidDeviceMemoryView() {
    return agents_view_d;
}
template <typename T>
GPUResource<T>& BoidGrid<T>::getDeviceBoids() {
    return this->agents_d;
}
#endif

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
    ss << "\n\tGlobal domain id: " << this->getGlobalId();
    ss << "\n\tGlobal domain: min " << this->globalDomain.min << "\t max " << this->globalDomain.max;
    ss << "\n\tLocal domain: min " << this->localDomain.min << "\t max " << this->localDomain.max;
    ss << "\n\tBox   : " << getBoxSize();
    ss << "\n\tCells : " << getCellsCount();
    ss << "\n\tMax radius : " << getMaxRadius();

    return ss.str();
}
        

#endif /* end of include guard: GRID_H */
