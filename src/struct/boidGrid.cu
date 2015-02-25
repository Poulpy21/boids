
#ifdef __CUDACC__

#include "headers.hpp"
#include "kernel.hpp"
#include "boidGrid.hpp"
#include "vectorMemoryView.hpp"
#include "boidMemoryView.hpp"
#include "thrustVectorMemoryView.hpp"
#include "thrustBoidMemoryView.hpp"
#include "kernel_utilities.cuh"
  
#ifdef THRUST_ENABLED
   
namespace kernel {

    template <typename T> 
        Domain<T> makeCudaDomain(const BoundingBox<3u,T> &bbox, const Vec<3u,unsigned int> &size) {
            return Domain<T>(
                    bbox.min[0], bbox.min[1], bbox.min[2],
                    bbox.max[0], bbox.max[1], bbox.max[2],
                    size[0],     size[1],     size[2]);
        }

    template <typename T>
        struct ComputeCellFunctor
        {
            ComputeCellFunctor(const Domain<T> &localDomain) : _localDomain(localDomain) {}

            Domain<T> _localDomain;

            template <typename Tuple>
                __host__ __device__ void operator()(Tuple t) //X Y Z C
                {
                    thrust::get<3>(t) = _localDomain.getCellId(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t)); 
                }
        };

    namespace boidgrid {

        template <typename T>
            __launch_bounds__(MAX_THREAD_PER_BLOCK)
            __global__ void computeForces(
                    T                   *const __restrict__ boidData,              // with allocated id : x y z vx vy vz id
                    int                 *const __restrict__ outOfDomain,           // direction of output 0 -- 26 for the 27 cases (0==stayDomain)
                    T            const  *const __restrict__ meanBoidData,          // local structure means
                    T            const  *const __restrict__ meanNeighborBoidData,  // local structure neighbors (not in local domain)
                    unsigned int const  *const __restrict__ uniqueCellIds,
                    unsigned int const  *const __restrict__ uniqueCellCount, 
                    unsigned int const  *const __restrict__ uniqueCellOffsets,
                    int          const  *const __restrict__ validCells,
                    const Domain<T> localDomain,  //to compute local  id
                    const Domain<T> globalDomain, //to compute global id
                    unsigned int const nAgents, 
                    unsigned int const nUniqueIds,
                    unsigned int const nCells,
                    bool         const keepInLocalDomain) {

                typedef typename MakeCudaVec<T,3>::type vec3; //either float3 or double3

                const unsigned int boidId = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

                if(boidId >= nAgents)
                    return;

                //Reconstruct memory views 
                BoidMemoryView<T>       const boids(boidData, nAgents);
                ConstBoidMemoryView<T>  const localMeanBoids   (meanBoidData, nUniqueIds);
                ConstBoidMemoryView<T>  const neighborMeanBoids(meanNeighborBoidData, 3u*3u*3u); //TODO

                //Get infos
                unsigned int const  myCellId         = boids.id[boidId];
                unsigned int const  validCellOffset  = validCells[myCellId];
                unsigned int const  localAgentsCount = uniqueCellCount[validCellOffset];
                unsigned int const  boidArrayOffset  = uniqueCellOffsets[validCellOffset];

                vec3 myPosition;
                myPosition.x = boids.x[boidId];
                myPosition.y = boids.y[boidId];
                myPosition.z = boids.z[boidId];

                //Compute forces
                unsigned int countSeparation=0u, countCohesion=0u, countAlignment=0u;
                vec3 forceSeparation = {}, forceCohesion = {}, forceAlignment = {};

                //Compute "internal forces"
                {
                    vec3 neighborPosition;
                    for (unsigned int i = 0; i < localAgentsCount; i++) {
                        unsigned int offset = boidArrayOffset + i;
                        if(offset != boidId) {
                            neighborPosition.x = boids.x[offset];
                            neighborPosition.y = boids.y[offset];
                            neighborPosition.z = boids.z[offset];
                            T dist = distance<T>(myPosition, neighborPosition);

                            if(dist < kernel::rSeparation) {
                                forceSeparation.x -= (myPosition.x - neighborPosition.x)/dist;
                                forceSeparation.y -= (myPosition.y - neighborPosition.y)/dist;
                                forceSeparation.z -= (myPosition.z - neighborPosition.z)/dist;
                                countSeparation++;
                            }
                            if(dist < kernel::rCohesion) {
                                forceCohesion.x += neighborPosition.x;
                                forceCohesion.y += neighborPosition.y;
                                forceCohesion.z += neighborPosition.z;
                                countCohesion++;
                            }
                            if(dist < kernel::rAlignment) {
                                forceAlignment.x += boids.vx[offset];
                                forceAlignment.y += boids.vy[offset];
                                forceAlignment.z += boids.vz[offset];
                                countAlignment++;
                            }
                        }
                    }
                }

                //Compute "external forces" -- SOOO MUCH BRANCHING
                {
                    unsigned int targetCellId;
                    const uint3 &dim = localDomain.dim;
                    unsigned int idx, idy, idz;
                    int iix, iiy, iiz;

                    //Extract local domain offsets of current cell
                    idx = myCellId % dim.x; 
                    idy = (myCellId/dim.x) % dim.y; 
                    idz = myCellId/(dim.x*dim.y); 
                    
                    for (int k = -1; k <= 1; k++) {
                        iiz = idz + k;
                        for (int j = -1; j <= 1; j++) {
                            iiy = idy + j;
                            for (int i = -1; i <= 1; i++) {
                                iix = idx + i;
                               
                                //Compute target neighbor cell id
                                targetCellId = localDomain.makeId(idx+i, idy+j, idz+k);

                                //Skip my cell
                                if(k == 0 && j == 0 && i == 0)
                                    continue;

                                //Handle global domain borders
                                //Already done on the host side, seen as classical local domain border

                                //Handle local domain borders
                                if(iix < 0 || iiy < 0 || iiz < 0 
                                        || iix >= dim.x || iiy >= dim.y || iiz >= dim.z) {
                                    ; //TODO
                                }

                                //Handle internal external interactions, checks if there is any boids in the target cell
                                else if (validCells[targetCellId] != -1) {
                                    vec3 neighborMeanPosition, neighborMeanVelocity;
                                    unsigned int cellUniqueIdOffset, count;
                                    
                                    cellUniqueIdOffset = validCells     [targetCellId];
                                    count              = uniqueCellCount[targetCellId];
                                    neighborMeanPosition.x = localMeanBoids.x [cellUniqueIdOffset];
                                    neighborMeanPosition.y = localMeanBoids.y [cellUniqueIdOffset];
                                    neighborMeanPosition.z = localMeanBoids.z [cellUniqueIdOffset];
                                    neighborMeanVelocity.x = localMeanBoids.vx[cellUniqueIdOffset];
                                    neighborMeanVelocity.y = localMeanBoids.vy[cellUniqueIdOffset];
                                    neighborMeanVelocity.z = localMeanBoids.vz[cellUniqueIdOffset];
                        
                                    T dist = distance<T>(myPosition, neighborMeanPosition);

                                    if(dist < kernel::rSeparation) {
                                        forceSeparation.x -= count*(myPosition.x - neighborMeanPosition.x)/dist;
                                        forceSeparation.y -= count*(myPosition.y - neighborMeanPosition.y)/dist;
                                        forceSeparation.z -= count*(myPosition.z - neighborMeanPosition.z)/dist;
                                        countSeparation++;
                                    }
                                    if(dist < kernel::rCohesion) {
                                        forceCohesion.x += count*neighborMeanPosition.x;
                                        forceCohesion.y += count*neighborMeanPosition.y;
                                        forceCohesion.z += count*neighborMeanPosition.z;
                                        countCohesion++;
                                    }
                                    if(dist < kernel::rAlignment) {
                                        forceAlignment.x += count*neighborMeanVelocity.x;
                                        forceAlignment.y += count*neighborMeanVelocity.y;
                                        forceAlignment.z += count*neighborMeanVelocity.z;
                                        countAlignment++;
                                    }
                                }
                            }
                        }
                    }
                }

                //Update forces
                vec3 force = {};

                if(countSeparation > 0) {
                    force.x += kernel::wSeparation*forceSeparation.x/countSeparation;
                    force.y += kernel::wSeparation*forceSeparation.y/countSeparation;
                    force.z += kernel::wSeparation*forceSeparation.z/countSeparation;
                }
                if(countCohesion > 0) {
                    force.x += kernel::wCohesion*forceCohesion.x/countCohesion;
                    force.y += kernel::wCohesion*forceCohesion.y/countCohesion;
                    force.z += kernel::wCohesion*forceCohesion.z/countCohesion;
                }
                if(countCohesion > 0) {
                    force.x += kernel::wAlignment*forceAlignment.x/countAlignment;
                    force.y += kernel::wAlignment*forceAlignment.y/countAlignment;
                    force.z += kernel::wAlignment*forceAlignment.z/countAlignment;
                }

                //Integrate in time
                vec3 myVelocity;
                myVelocity.x = boids.vx[boidId] + force.x;
                myVelocity.y = boids.vy[boidId] + force.y;
                myVelocity.z = boids.vz[boidId] + force.z;

                T speed = kernel::norm<T>(myVelocity);

                if(speed > kernel::maxVelocity) {
                    myVelocity.x *= kernel::maxVelocity/speed;
                    myVelocity.y *= kernel::maxVelocity/speed;
                    myVelocity.z *= kernel::maxVelocity/speed;
                }

                myPosition.x += kernel::dt * myVelocity.x;
                myPosition.y += kernel::dt * myVelocity.y;
                myPosition.z += kernel::dt * myVelocity.z;
        
                if(keepInLocalDomain) { //Here in theory localDomain == globalDomain, or you are weird.
                    //Clamp positions to global domain
                    localDomain.moduloDomain(myPosition);
                }
                else { 
                    //Clamp positions to global domain
                    globalDomain.moduloDomain(myPosition);
                
                    //Handle boids that went outside of local domain 
                    outOfDomain[boidId] = localDomain.isInDomain(myPosition); //0 if it stays inside the local domain
                }

                //Compute new cell id
                unsigned int myNewCellId = localDomain.getCellId(myPosition);

                //Write back data to memory
                boids.x[boidId]  = myPosition.x;
                boids.y[boidId]  = myPosition.y;
                boids.z[boidId]  = myPosition.z;
                boids.vx[boidId] = myVelocity.x;
                boids.vy[boidId] = myVelocity.y;
                boids.vz[boidId] = myVelocity.z;
                boids.id[boidId] = myNewCellId;
            }


        template <typename T>
            void computeForcesKernel(
                    T                   *const boidData,              // with allocated id : x y z vx vy vz id
                    int                 *const outOfDomain,           // direction of output 0 -- 26 for the 27 cases (0==stayDomain)
                    T            const  *const meanBoidData,          // no id !
                    T            const  *const meanNeighborBoidData,  // no id !
                    unsigned int const  *const uniqueCellIds,
                    unsigned int const  *const uniqueCellCount, 
                    unsigned int const  *const uniqueCellOffsets,
                    int          const  *const validCells,
                    const Domain<T> localDomain,  //to compute local  id
                    const Domain<T> globalDomain, //to compute global id
                    unsigned int const nAgents, 
                    unsigned int const nUniqueIds,
                    unsigned int const nCells) {

                float nAgents_f = nAgents;

                dim3 dimBlock(MAX_THREAD_PER_BLOCK);
                dim3 dimGrid((unsigned int)ceil(nAgents_f/MAX_THREAD_PER_BLOCK) % 65535, ceil(nAgents_f/(MAX_THREAD_PER_BLOCK*65535.0f)));

                log4cpp::log_console->infoStream() << "[KERNEL::BoidGrid::computeForces] <<<" 
                    << utils::toStringDim(dimBlock) << ", " 
                    << utils::toStringDim(dimGrid)
                    << ">>>";

                computeForces<T><<<dimGrid,dimBlock>>>(
                        boidData,
                        outOfDomain,
                        meanBoidData, 
                        meanNeighborBoidData,
                        uniqueCellIds,
                        uniqueCellCount, 
                        uniqueCellOffsets,
                        validCells,
                        localDomain,
                        globalDomain,
                        nAgents, 
                        nUniqueIds,
                        nCells,
                        true);

                CHECK_KERNEL_EXECUTION();
            }
    }
    
}




/*template <typename T>*/
/*using thrust::device_vector<T>::iterator = deviceIterator;*/
typedef thrust::device_vector<Real>::iterator  deviceIterator_real;
typedef thrust::device_vector<unsigned int>::iterator  deviceIterator_ui;

template <typename T>
__host__ void initBoidGridThrustArrays(BoidGrid<T> &boidGrid) {

    unsigned int nAgents = boidGrid.getTotalLocalAgentCount();
    unsigned int nCells = boidGrid.getCellsCount();

    ThrustBoidMemoryView<T> agents_thrust_d(boidGrid.getBoidDeviceMemoryView());
    thrust::device_vector<unsigned int> cellIds(nAgents);

    //copy  X Y Z  VX VY VZ  data to device
    for(unsigned int i = 0; i < BoidMemoryView<T>::N-1; i++) {
        CHECK_THRUST_ERRORS(thrust::copy(boidGrid.getBoidHostMemoryView()[i], boidGrid.getBoidHostMemoryView()[i] + nAgents, agents_thrust_d[i]));
    }

    //compute cell Id for each boid
    CHECK_THRUST_ERRORS(
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(agents_thrust_d.x, agents_thrust_d.y, agents_thrust_d.z, agents_thrust_d.id)),
                thrust::make_zip_iterator(thrust::make_tuple(
                        agents_thrust_d.x + nAgents, 
                        agents_thrust_d.y + nAgents,
                        agents_thrust_d.z + nAgents,
                        agents_thrust_d.id + nAgents)),
                kernel::ComputeCellFunctor<T>(kernel::makeCudaDomain<T>(boidGrid.getLocalDomain(), boidGrid.getBoxSize())))
            );

    //find the permutation to sort everyone according to the cellIds
    thrust::device_vector<unsigned int> keys(nAgents);
    CHECK_THRUST_ERRORS(thrust::sequence(keys.begin(), keys.end()));
    CHECK_THRUST_ERRORS(thrust::stable_sort_by_key(agents_thrust_d.id, agents_thrust_d.id + nAgents, keys.begin()));

    //find the cells that contains at least one agent
    //and find coresponding array offsets to be copied from the cells
    thrust::device_vector<unsigned int> uniqueIds(agents_thrust_d.id, agents_thrust_d.id + nAgents);
    thrust::device_vector<unsigned int> offsets(nAgents);
    CHECK_THRUST_ERRORS(thrust::sequence(offsets.begin(), offsets.end()));

    thrust::pair<deviceIterator_ui, deviceIterator_ui> end =
        thrust::unique_by_key(uniqueIds.begin(), uniqueIds.end(), offsets.begin());

    unsigned int nUniqueIds = thrust::distance(uniqueIds.begin(), end.first);
    uniqueIds.resize(nUniqueIds);
    offsets.resize(nUniqueIds);

    //count number of boids per key using computed offsets
    thrust::device_vector<unsigned int> count(nUniqueIds);
    CHECK_THRUST_ERRORS(thrust::transform(offsets.begin()+1, offsets.end(), offsets.begin(), count.begin(), thrust::minus<unsigned int>()));
    count[nUniqueIds-1] = nAgents - offsets[nUniqueIds-1];

    //keep filled cells for neighborlookup
    thrust::device_vector<int> validIds(nCells);
    CHECK_THRUST_ERRORS(thrust::fill(validIds.begin(), validIds.end(), -1));
    CHECK_THRUST_ERRORS(
            thrust::scatter(
                thrust::make_counting_iterator<int>(0), 
                thrust::make_counting_iterator<int>(nUniqueIds),
                uniqueIds.begin(), validIds.begin())
            );

    //sort the boids with precomputed permutation
    thrust::device_vector<T> buffer((BoidMemoryView<T>::N-1)*nAgents);
    BoidMemoryView<T> buf_view(buffer.data().get(), nAgents);
    ThrustBoidMemoryView<T> buffer_view(buf_view);

    for(unsigned int i = 0u; i < BoidMemoryView<T>::N-1; i++) {
        CHECK_THRUST_ERRORS(
                thrust::copy(
                    thrust::make_permutation_iterator(agents_thrust_d[i], keys.begin()),
                    thrust::make_permutation_iterator(agents_thrust_d[i], keys.end()),
                    buffer_view[i])
                );
        CHECK_THRUST_ERRORS(thrust::copy(buffer_view[i], buffer_view[i]+nAgents, agents_thrust_d[i]));
    }

    //allocate and store additional data
    GPUResource<int> &validIds_d = boidGrid.getDeviceValidIds(); 
    GPUResource<unsigned int> &uniqueIds_d = boidGrid.getDeviceUniqueIds(); 
    GPUResource<unsigned int> &offsets_d = boidGrid.getDeviceOffsets(); 
    GPUResource<unsigned int> &count_d = boidGrid.getDeviceCount(); 

    uniqueIds_d.setSize(nUniqueIds);
    uniqueIds_d.allocate();
    offsets_d.setSize(nUniqueIds);
    offsets_d.allocate();
    count_d.setSize(nUniqueIds);
    count_d.allocate();
    validIds_d.setSize(nCells);
    validIds_d.allocate();

    CHECK_THRUST_ERRORS(thrust::copy(uniqueIds.begin(), uniqueIds.end(), uniqueIds_d.wrap()));
    CHECK_THRUST_ERRORS(thrust::copy(offsets.begin(), offsets.end(), offsets_d.wrap()));
    CHECK_THRUST_ERRORS(thrust::copy(count.begin(), count.end(), count_d.wrap()));
    CHECK_THRUST_ERRORS(thrust::copy(validIds.begin(), validIds.end(), validIds_d.wrap()));

    //DEBUG
    //std::cout << "Boid IDs:\t";
    //for(int i = 0; i < nAgents; i++)
    //std::cout << agents_thrust_d.id[i] << " ";
    //std::cout << std::endl;

    //std::cout << "Unique IDs:\t";
    //for(int i = 0; i < uniqueIds.size(); i++)
    //std::cout << uniqueIds[i] << " ";
    //std::cout << std::endl;

    //std::cout << "Offsets:\t";
    //for(int i = 0; i < uniqueIds.size(); i++)
    //std::cout << offsets[i] << " ";
    //std::cout << std::endl;

    //std::cout << "Count:\t\t";
    //for(int i = 0; i < uniqueIds.size(); i++)
    //std::cout << count[i] << " ";
    //std::cout << std::endl;

    //std::cout << "Valid Ids:\t";
    //for(int i = 0; i < validIds.size(); i++)
    //std::cout << validIds[i] << " ";
    //std::cout << std::endl;
}

template <typename T>
__host__ BoidMemoryView<T> computeThrustStep(BoidGrid<T> &boidGrid) {

    unsigned int nAgents    = boidGrid.getTotalLocalAgentCount();
    unsigned int nUniqueIds = boidGrid.getDeviceUniqueIds().size();
    unsigned int nCells     = boidGrid.getCellsCount();

    GPUResource<T>            &agents_d    = boidGrid.getDeviceBoids(); 
    GPUResource<unsigned int> &count_d     = boidGrid.getDeviceCount(); 
    GPUResource<unsigned int> &offsets_d   = boidGrid.getDeviceOffsets(); 
    GPUResource<unsigned int> &uniqueIds_d = boidGrid.getDeviceUniqueIds(); 
    GPUResource<int>          &validIds_d  = boidGrid.getDeviceValidIds(); 

    ThrustBoidMemoryView<T> agents_thrust_d(boidGrid.getBoidDeviceMemoryView());

    // Sort and find unique ids

    // Compute mean positions (only for filled cells)
    thrust::device_vector<T>  means(6*nUniqueIds);
    ThrustBoidMemoryView<T>   means_v(means, nUniqueIds);
    {
        thrust::device_vector<unsigned int> buffKeys(nUniqueIds);

        for (unsigned int i = 0; i < 6u; i++) {
            CHECK_THRUST_ERRORS(
                    thrust::reduce_by_key(agents_thrust_d.id, agents_thrust_d.id + nAgents,
                        agents_thrust_d[i], buffKeys.begin(), means_v[i], 
                        thrust::equal_to<unsigned int>(), thrust::plus<float>())
                    );
            CHECK_THRUST_ERRORS(
                    thrust::transform(means_v[i], means_v[i] + nUniqueIds,
                        count_d.wrap(), means_v[i], thrust::divides<float>())
                    );
        }
    }

    // Get globals neighbors mean position and velocity
    int* outOfDomain = 0;
    T* meanNeighborBoidData = 0;
    
    // Call kernel
    kernel::boidgrid::computeForcesKernel(agents_d.data(), outOfDomain, means_v.data(), meanNeighborBoidData,
            uniqueIds_d.data(), count_d.data(),
            offsets_d.data(), validIds_d.data(),
            kernel::makeCudaDomain<T>(boidGrid.getLocalDomain(),  boidGrid.getBoxSize()),
            kernel::makeCudaDomain<T>(boidGrid.getGlobalDomain(), globalDomainSize),
            nAgents, nUniqueIds, nCells);

    // Check for bad elements

    //check for boids that went outside the domain
    
    std::cout << "Unique IDs:\t";
    for(int i = 0; i < nUniqueIds; i++)
        std::cout << thrust::device_ptr<unsigned int>(uniqueIds_d.data())[i] << " ";
    std::cout << std::endl;

    BoidMemoryView<T> outOfDomainBoids;
    return outOfDomainBoids;
}


// full specializations
template __host__ void initBoidGridThrustArrays<float> (BoidGrid<float > &boidGrid);
template __host__ void initBoidGridThrustArrays<double>(BoidGrid<double> &boidGrid);

template __host__ BoidMemoryView<float> computeThrustStep<float>(BoidGrid<float> &boidGrid);
template __host__ BoidMemoryView<double> computeThrustStep<double>(BoidGrid<double> &boidGrid);

#endif

#endif
