
#ifndef GLOBALBOIDDATASTRUCTURE_H
#define GLOBALBOIDDATASTRUCTURE_H

#include "headers.hpp"
#include "boundingBox.hpp"
#include "boidMemoryView.hpp"
#include "vec3.hpp"
#include <vector>

template <typename T>
class GlobalBoidDataStructure {

    public:
        GlobalBoidDataStructure(const BoundingBox<3u,T> &globalDomain,
                bool keepBoidsInGlobalDomain = true) :
            globalDomain(globalDomain),
            _keepBoidsInGlobalDomain(keepBoidsInGlobalDomain) {
            }

        virtual ~GlobalBoidDataStructure() {}

        BoundingBox<3u,T> getGlobalDomain() const {
            return globalDomain;
        }

        bool keepBoidsInGlobalDomain() {
            return _keepBoidsInGlobalDomain;
        }

    protected:
        const BoundingBox<3u, T> globalDomain;
        bool _keepBoidsInGlobalDomain;

    public:

        //initialize structure with given host boids
        //All boids should initially be contained in the global domain
        virtual void init(unsigned int totalAgentCount, const BoundingBox<3u,T> &spawnZone) = 0;

        //insert boids in the global structure with given host boids
        //All boids given in input should be contained in the global domain
        virtual void feed(unsigned int addedAgentCount, const BoundingBox<3u,T> &spawnZone) = 0; 

        //compute one step and remove boids that went out the domain 
        //if keepBoidsInGlobalDomain is set to true, no boids are removed
        virtual void computeGlobalStep() = 0; 

        //return the total number of boids contained in th structure
        virtual unsigned int getTotalAgentCount() const = 0;
        virtual unsigned int getGlobalAgentCount(unsigned int globalCellId) const = 0;

        //return neighbors cells in the computing grid
        virtual std::vector<unsigned int> getGlobalNeighborCellIds(unsigned int globalCellId) const = 0;

        //return true if the structure is in the global domain corners
        virtual bool isAtCorner(unsigned int globalCellId) = 0;

    protected:
        virtual unsigned int getGlobalCellId(const Vec3<T> &pos) const = 0;
};


#endif /* end of include guard: GLOBALBOIDDATASTRUCTURE_H */
