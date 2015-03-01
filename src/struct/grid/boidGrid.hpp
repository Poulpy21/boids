

#ifndef BOID_GRID_H
#define BOID_GRID_H

#include "headers.hpp"
#include "localBoidDataStructure.hpp"
#include "boidMemoryView.hpp"
#include "CPUResource.hpp"
#include "UnpagedCPUResource.hpp"
#include "PinnedCPUResource.hpp"
#include "vec3.hpp"

#include <vector>

#if __cplusplus >= 201103L
#include <type_traits>
#endif

#ifdef CUDA_ENABLED
#include "GPUResource.hpp"
#endif

#ifdef THRUST_ENABLED
template <typename T, typename HostMemoryType>
class BoidGrid;

template <typename T, typename HostMemoryType>
__HOST__ void initBoidGridThrustArrays(BoidGrid<T,HostMemoryType> &boidGrid);

template <typename T, typename HostMemoryType>
__HOST__ BoidMemoryView<T> computeThrustStep(BoidGrid<T,HostMemoryType> &boidGrid);

#endif

template <typename T, typename HostMemoryType>
class BoidGrid : public LocalBoidDataStructure<T,HostMemoryType> {
            
#if __cplusplus >= 201103L
    static_assert(std::is_base_of<CPUResource<T>,HostMemoryType>(), "HostMemoryType should inherit CPUResource<T> !");
#endif

    public:
        BoidGrid(unsigned int globalId,
                const BoundingBox<3u, Real> &globalDomain,
                const BoundingBox<3u, Real> &localDomain,
                bool keepBoidsInGlobalDomain,
                Real maxRadius,
                const unsigned int deviceId);
        BoidGrid(const BoidGrid<T,HostMemoryType> &other);
        BoidGrid<T,HostMemoryType>& operator=(const BoidGrid<T,HostMemoryType> &other);
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
        int getDeviceId() const;
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
        const unsigned int deviceId;

        const T domainWidth, domainHeight, domainLength;
        const unsigned int width, height, length;
        const Vec3<unsigned int> boxSize;

        const unsigned int nCells;


#ifdef CUDA_ENABLED
    protected:
        GPUResource<T> agents_d;
        BoidMemoryView<T> agents_view_d;
        GPUResource<T> means_d;

        GPUResource<unsigned int> uniqueIds_d;
        GPUResource<unsigned int> count_d;
        GPUResource<unsigned int> offsets_d;
        GPUResource<int> validIds_d;

    public:
        BoidMemoryView<T>& getBoidDeviceMemoryView();
        GPUResource<T>& getDeviceBoids();
        GPUResource<T>& getMeanBoids();
        GPUResource<unsigned int>& getDeviceUniqueIds();
        GPUResource<unsigned int>& getDeviceCount();
        GPUResource<unsigned int>& getDeviceOffsets();
        GPUResource<int>& getDeviceValidIds();
#endif

};

        
template <typename T, typename HostMemoryType>
void BoidGrid<T,HostMemoryType>::init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) {
    this->agents_h = HostMemoryType(agent_h.data(), BoidMemoryView<T>::N*agentCount, false);
    this->agents_view_h = agent_h;

#ifdef CUDA_ENABLED
    this->agentCount = agentCount;

    agents_d = GPUResource<T>(deviceId, BoidMemoryView<T>::N*agentCount, BoidMemoryView<T>::N*agentCount);
    agents_d.allocate();
    agents_view_d = BoidMemoryView<T>(agents_d.data(), agentCount);

    initBoidGridThrustArrays<T>(*this);
#else
    NOT_IMPLEMENTED_YET;
#endif
}
        
template <typename T, typename HostMemoryType>
void BoidGrid<T,HostMemoryType>::feed(const BoidMemoryView<T> &agent_h, unsigned int agentCount) {
#ifdef CUDA_ENABLED
    NOT_IMPLEMENTED_YET;
#else
    NOT_IMPLEMENTED_YET;
#endif
}
        
//compute one step and return boids that went out the domain 
//if keepBoidsInDomain is set to true, no boids are returned
template <typename T, typename HostMemoryType>
BoidMemoryView<T> BoidGrid<T,HostMemoryType>::computeLocalStep() {
#ifdef CUDA_ENABLED
    return computeThrustStep(*this);
#else
    NOT_IMPLEMENTED_YET;
#endif
}

template <typename T, typename HostMemoryType>
BoidGrid<T,HostMemoryType>::BoidGrid(unsigned int globalId,
                const BoundingBox<3u, Real> &localDomain,
                const BoundingBox<3u, Real> &globalDomain,
                bool keepBoidsInGlobalDomain,
                Real maxRadius,
                unsigned int deviceId_) :
    LocalBoidDataStructure<T,HostMemoryType>(globalId, localDomain, globalDomain, keepBoidsInGlobalDomain),
    maxRadius(maxRadius),
    deviceId(deviceId_),
    domainWidth (localDomain.max[0] - localDomain.min[0]),
    domainHeight(localDomain.max[1] - localDomain.min[1]),
    domainLength(localDomain.max[2] - localDomain.min[2]),
    width (std::max(1,static_cast<int>(ceil(domainWidth /maxRadius)))),
    height(std::max(1,static_cast<int>(ceil(domainHeight/maxRadius)))),
    length(std::max(1,static_cast<int>(ceil(domainLength/maxRadius)))),
    boxSize(width, height, length),
    nCells(width*height*length)
#ifdef CUDA_ENABLED 
        ,agents_d(deviceId,0,0),
        agents_view_d(),
        means_d(deviceId,0,0),
        uniqueIds_d(deviceId,0,0),
        count_d(deviceId,0,0),
        offsets_d(deviceId,0,0),
        validIds_d(deviceId,0,0)
#endif
{
}

template <typename T, typename HostMemoryType>
BoidGrid<T,HostMemoryType>::~BoidGrid() {
}

template <typename T, typename HostMemoryType>
BoidGrid<T,HostMemoryType>::BoidGrid(const BoidGrid<T,HostMemoryType> &other) :
    LocalBoidDataStructure<T,HostMemoryType>(other),
    maxRadius(0),
    deviceId(0),
    domainWidth (0),
    domainHeight(0),
    domainLength(0),
    width (0),
    height(0),
    length(0),
    boxSize(0),
    nCells(0)
#ifdef CUDA_ENABLED 
        ,agents_d(deviceId,0,0),
        agents_view_d(),
        means_d(deviceId,0,0),
        uniqueIds_d(deviceId,0,0),
        count_d(deviceId,0,0),
        offsets_d(deviceId,0,0),
        validIds_d(deviceId,0,0)
#endif
{
        throw std::logic_error("Cannot copy a BoidGrid.");
    }

template <typename T, typename HostMemoryType>
BoidGrid<T,HostMemoryType>& BoidGrid<T,HostMemoryType>::operator=(const BoidGrid<T,HostMemoryType> &other) {
    throw std::logic_error("Cannot copy a BoidGrid.");
}

template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getLocalAgentCount(unsigned int cellId) const {
    NOT_IMPLEMENTED_YET;
}
        
template <typename T, typename HostMemoryType>
bool BoidGrid<T,HostMemoryType>::isLocalCellAtCorner(unsigned int localCellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T, typename HostMemoryType>
BoidMemoryView<T> BoidGrid<T,HostMemoryType>::getLocalHostAgentsArray(unsigned int cellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getLocalCellId(const Vec3<T> &pos) const {
    Vec3<float> relPos = relativePos(pos);
    return makeLocalId(relPos.x * boxSize.x, relPos.y * boxSize.y, relPos.z * boxSize.z);
}
        
template <typename T, typename HostMemoryType>
NeighborIds& BoidGrid<T,HostMemoryType>::getGlobalNeighborCellIds(unsigned int globalCellId) const {
    NOT_IMPLEMENTED_YET;
}

template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::makeLocalId(unsigned int x, unsigned int y, unsigned int z) const {
    return (width*height*z + width*y + x);
}

template <typename T, typename HostMemoryType>
Vec3<T> BoidGrid<T,HostMemoryType>::relativePos(const Vec3<T> &pos) const {
    return Vec3<T>(
            (pos.x - this->localDomain.min[0])/domainWidth,
            (pos.y - this->localDomain.min[1])/domainHeight,
            (pos.z - this->localDomain.min[2])/domainLength
            );
}

template <typename T, typename HostMemoryType>
BoidMemoryView<T>& BoidGrid<T,HostMemoryType>::getBoidHostMemoryView() {
    return this->agents_view_h;
}
        
#ifdef CUDA_ENABLED
template <typename T, typename HostMemoryType>
GPUResource<unsigned int>& BoidGrid<T,HostMemoryType>::getDeviceOffsets() {
    return this->offsets_d;
}

template <typename T, typename HostMemoryType>
GPUResource<unsigned int>& BoidGrid<T,HostMemoryType>::getDeviceUniqueIds() {
    return this->uniqueIds_d;
}
    
template <typename T, typename HostMemoryType>
GPUResource<int>& BoidGrid<T,HostMemoryType>::getDeviceValidIds() {
    return this->validIds_d;
}
        
template <typename T, typename HostMemoryType>
GPUResource<unsigned int>& BoidGrid<T,HostMemoryType>::getDeviceCount() {
    return this->count_d;
}
       
template <typename T, typename HostMemoryType>
BoidMemoryView<T>& BoidGrid<T,HostMemoryType>::getBoidDeviceMemoryView() {
    return agents_view_d;
}
template <typename T, typename HostMemoryType>
GPUResource<T>& BoidGrid<T,HostMemoryType>::getDeviceBoids() {
    return this->agents_d;
}
template <typename T, typename HostMemoryType>
GPUResource<T>& BoidGrid<T,HostMemoryType>::getMeanBoids() {
    return this->means_d;
}
#endif

template <typename T, typename HostMemoryType>
T BoidGrid<T,HostMemoryType>::getMaxRadius() const {
    return maxRadius;
}
template <typename T, typename HostMemoryType>
T BoidGrid<T,HostMemoryType>::getDomainWidth() const {
    return domainWidth;
}
template <typename T, typename HostMemoryType>
T BoidGrid<T,HostMemoryType>::getDomainLength() const {
    return domainLength;
}
template <typename T, typename HostMemoryType>
T BoidGrid<T,HostMemoryType>::getDomainHeight() const {
    return domainHeight;
}
template <typename T, typename HostMemoryType>
int BoidGrid<T,HostMemoryType>::getDeviceId() const {
    return deviceId;
}
template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getWidth() const {
    return width;
}
template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getHeight() const {
    return height;
}
template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getLength() const {
    return length;
}
template <typename T, typename HostMemoryType>
unsigned int BoidGrid<T,HostMemoryType>::getCellsCount() const {
    return nCells;
}
template <typename T, typename HostMemoryType>
Vec3<unsigned int> BoidGrid<T,HostMemoryType>::getBoxSize() const {
    return boxSize;
}

template <typename T, typename HostMemoryType>
std::ostream& operator<<(std::ostream &os, const BoidGrid<T,HostMemoryType> &grid) {
    os << grid.toString();
    return os;
}
        
template <typename T, typename HostMemoryType>
std::string BoidGrid<T,HostMemoryType>::toString() const {
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
    ss << "\n\tKeep boids in local domain : " << this->keepBoidsInLocalDomain;

    return ss.str();
}
        

#endif /* end of include guard: GRID_H */
