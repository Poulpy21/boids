
#ifndef BOIDDATASTRUCTURE_H
#define BOIDDATASTRUCTURE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include "PinnedCPUResource.hpp"
#include "boidMemoryView.hpp"
#include "neighborIds.hpp"
#include "vec3.hpp"
#include <vector>

#if __cplusplus >= 201103L
#include <type_traits>
#endif

template <typename T, typename HostMemoryType>
class LocalBoidDataStructure {

#if __cplusplus >= 201103L
    static_assert(std::is_base_of<CPUResource<T>,HostMemoryType>(), "HostMemoryType should inherit CPUResource<T> !");
#endif


    public:
        virtual ~LocalBoidDataStructure() {}

        unsigned int getGlobalId() const {
            return globalDomainId;
        }

        BoundingBox<3u,T> getLocalDomain() const {
            return localDomain;
        }

        BoundingBox<3u,T> getGlobalDomain() const {
            return globalDomain;
        }

        BoidMemoryView<T> getHostAgentsArray() const {
            return agents_h;
        }
        
        unsigned int getTotalLocalAgentCount() const {
            return agentCount;
        }

        //Interface
    public:
        //initialize structure with given host boids
        //All boids should initially be contained in the domain
        virtual void init(const BoidMemoryView<T> &agent_h, unsigned int agentCount) = 0;

        //insert boids in the structure with given host boids
        //All boids given in input should be contained in the domain
        virtual void feed(const BoidMemoryView<T> &agent_h, unsigned int agentCount) = 0; 

        //compute one step and return boids that went out the domain 
        //if keepBoidsInGlobalDomain is set to true, no boids are returned
        virtual BoidMemoryView<T> computeLocalStep() = 0; 

        //return the total number of boids contained in th structure
        virtual unsigned int getLocalAgentCount(unsigned int localCellId) const = 0;

        //return true if the structure is in the local bounding box corners
        virtual bool isLocalCellAtCorner(unsigned int localCellId) const = 0;

    protected:
        //return local cellId with given position (should reside in localDomain)
        virtual unsigned int getLocalCellId(const Vec3<T> &pos) const = 0;

        //return neighbors cells in the computing grid
        virtual NeighborIds& getGlobalNeighborCellIds(unsigned int globalCellId) const = 0;

        //return local memory view
        virtual BoidMemoryView<T> getLocalHostAgentsArray(unsigned int localCellId) const = 0;

        //Implementation specific
    protected:
        LocalBoidDataStructure(unsigned int globalDomainId,
                const BoundingBox<3u,T> &localDomain, 
                const BoundingBox<3u,T> &globalDomain,
                bool keepBoidsInGlobalDomain) :
            globalDomainId(globalDomainId),
            localDomain(localDomain),
            globalDomain(globalDomain),
            keepBoidsInGlobalDomain(keepBoidsInGlobalDomain),
            keepBoidsInLocalDomain(localDomain == globalDomain) {
            }

        const unsigned int globalDomainId;
        const BoundingBox<3u, T> localDomain, globalDomain;
        const bool keepBoidsInGlobalDomain, keepBoidsInLocalDomain;

    protected:
        unsigned int agentCount;
        HostMemoryType agents_h;
        BoidMemoryView<T> agents_view_h;
};

#endif /* end of include guard: BOIDDATASTRUCTURE_H */
