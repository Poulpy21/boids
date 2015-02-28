
#ifndef COMPUTEGRID_H
#define COMPUTEGRID_H

#include "headers.hpp"
#include "globalBoidDataStructure.hpp"
#include "boidMemoryView.hpp"
#include "PinnedCPUResource.hpp"
#include "vec3.hpp"
#include <vector>

#ifdef CUDA_ENABLED
#include "GPUResource.hpp"
#endif

template <typename T>
class ComputeGrid {
       
    public:
        ComputeGrid(const BoundingBox<3u,T> &globalDomain,
                const Options &opt, 
                unsigned int nWorkers, unsigned int workerId, unsigned int masterWorkerId) :
            globalDomain(globalDomain),
            opt(opt),
            nWorkers(nWorkers), workerId(workerId), masterWorkerId(masterWorkerId),
            cellIdWorker(nWorkers) {

                //Get max radius
                Real maxRadius = std::max<Real>(opt.rCohesion, std::max<Real>(opt.rAlignment, opt.rSeparation));

                //Split the domain in nWorkers
                Real w,h,l;
                dim = Vec3<unsigned int>(1u,1u,1u);
                w = globalDomain.size()[0];
                h = globalDomain.size()[1];
                l = globalDomain.size()[2];
                
                unsigned int nDomains = 1;
                while(nDomains < nWorkers) {
                    if(l >= h && l >= w) {
                        l /= Real(2);
                        dim.z <<= 1;
                    }
                    else if(h >= l && h >= w) {
                        h /= Real(2);
                        dim.y <<= 1;
                    }
                    else {
                        w /= Real(2);
                        dim.x <<= 1;
                    }

                    nDomains <<= 1;
                }

                assert(l > maxRadius && w > maxRadius && l > maxRadius);
                if(nDomains != nWorkers)
                    NOT_IMPLEMENTED_YET;

                subdomainSize.x = w;
                subdomainSize.y = h;
                subdomainSize.z = l;

                if(workerId == masterWorkerId)
                    log_console->infoStream() << "Splitted the simulation domain in " << nDomains << " !";
        }

    public:
        BoundingBox<3u,T> getSubdomain(int rank) const {
            Vec<3u,T> min = this->globalDomain.min + Vec3<T>(getSubdomainOffset(rank))*subdomainSize;
            return BoundingBox<3u,T>(min, min+subdomainSize);
        }

        unsigned int makeRank(const Vec3<unsigned int> &subdomainOffset) const {
            return subdomainOffset.x + (subdomainOffset.y + dim.y*subdomainOffset.z)*dim.x;
        }

        Vec3<unsigned int> getSubdomainOffset(int rank) const {
            return Vec3<unsigned int>(rank % dim.x, (rank/dim.x) % dim.y, rank / (dim.x * dim.y) );
        }
    
    protected:
    
        BoundingBox<3u,T> globalDomain;
        const Options opt;
        const unsigned int nWorkers, workerId, masterWorkerId;
        std::vector<int> cellIdWorker;
        Vec3<T> subdomainSize;
        Vec3<unsigned int> dim;
        unsigned int nTotalAgents;
};



#endif /* end of include guard: COMPUTEGRID_H */
