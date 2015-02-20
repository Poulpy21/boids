
#ifdef __CUDACC__

#include "headers.hpp"
#include "kernel.hpp"
#include "boidGrid.hpp"
#include "vectorMemoryView.hpp"
#include "boidMemoryView.hpp"
#include "thrustVectorMemoryView.hpp"
#include "thrustBoidMemoryView.hpp"

#ifdef THRUST_ENABLED

namespace kernel {
    namespace boidgrid {


        //Rebuild memory views (use shared memory broadcast capabilities)
        //__shared__ ConstVectorMemoryView<T>      boids;
        //__shared__ Const:w
        //__syncthreads();
        //if(threadIdx.x == 0u) {
        //boids   = ConstVectorMemoryView<T>     (boidData            , nAgents   );
        //meanPos = ConstVectorConstMemoryView<T>(meanBoidPositionData, nUniqueIds);
        //}
        //__syncthreads();

        template <typename T>
            __launch_bounds__(MAX_THREAD_PER_BLOCK)
            __global__ void computeForces(
                    T                   *const __restrict__ boidData,
                    T            const  *const __restrict__ meanBoidPositionData, 
                    unsigned int const  *const __restrict__ uniqueCellIds,
                    unsigned int const  *const __restrict__ uniqueCellCount, 
                    unsigned int const  *const __restrict__ uniqueCellOffsets,
                    int          const  *const __restrict__ validCells,
                    unsigned int const nAgents, 
                    unsigned int const nUniqueIds,
                    unsigned int const nCells) {

                const unsigned int boidId = blockIdx.y*65535ul*512ul + blockIdx.x*512ul + threadIdx.x;

                if(boidId >= nAgents)
                    return;
                
                //Reconstruct memory views 
                ConstBoidMemoryView<T>   const boids(boidData, nAgents);
                ConstVectorMemoryView<T> const meanPos(meanBoidPositionData, nUniqueIds);

                //Compute "internal forces"
                unsigned int const  myCellId         = boids.id[boidId];
                unsigned int const  validCellOffset  = validCells[myCellId];
                unsigned int const  localAgentsCount = uniqueCellCount[validCellOffset];
                unsigned int const  boidArrayOffset  = uniqueCellOffsets[validCellOffset];

                //unsigned int counter = 0u;
                //for (unsigned int i = 0; i < localAgentsCount; i++) {
                    //counter++;
                //}

                printf("cellId %i  validCellOffset %i  localAgentsCount %i  boidArrayOffset %i \n", 
                        myCellId, validCellOffset, localAgentsCount, boidArrayOffset);

            }

        //Compute "external forces"
        //for (int i = 0; i < nMeanBoids; i++) {
        //otherBoidPosition = boidList.getPosition(i);
        //T dist = (thisBoidPosition - otherBoidPosition).norm();
        //T weight = meanBoidWeights[i];
        //if (dist < opt->rSeparation) {
        //forceSeparation -= weight * (thisBoidPosition - otherBoidPosition).normalized();
        //countSeparation += weight;
        //}
        //if (dist < opt->rCohesion) {
        //forceCohesion += weight * thisBoidPosition;
        //countCohesion += weight;
        //}
        //if (dist < opt->rAlignment) {
        //forceAlignment += weight * boidList.getVelocity(i);
        //countAlignment += weight;
        //}
        //}

        //Update direction
        //Vector direction( opt->wSeparation * ( countSeparation>0 ? forceSeparation/static_cast<T>(countSeparation) : forceSeparation) +
        //opt->wCohesion   * ( countCohesion  >0 ? forceCohesion  /static_cast<T>(countCohesion)   : forceCohesion  ) +
        //opt->wAlignment  * ( countAlignment >0 ? forceAlignment /static_cast<T>(countAlignment)  : forceAlignment ));
        //boidList.setDirection(id, direction);
        //}

    template <typename T>
        void computeForcesKernel(
                T                   *const __restrict__ boidData,
                T            const  *const __restrict__ meanBoidPositionData, 
                unsigned int const  *const __restrict__ uniqueCellIds,
                unsigned int const  *const __restrict__ uniqueCellCount, 
                unsigned int const  *const __restrict__ uniqueCellOffsets,
                int          const  *const __restrict__ validCells,
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
                    meanBoidPositionData, 
                    uniqueCellIds,
                    uniqueCellCount, 
                    uniqueCellOffsets,
                    validCells,
                    nAgents, 
                    nUniqueIds,
                    nCells);

            CHECK_KERNEL_EXECUTION();
        }



    //__global__ void applyForces(Real *boidData, const int nBoids, const struct Options *opt) {

    //int id = blockIdx.x*blockDim.x + threadIdx.x;
    //if (id >= nBoids)
    //return;

    //Rebuild AgentData
    //AgentData boidList(boidData, nBoids);

    //Update velocity
    //Vector velocity = boidList.getVelocity(id) + boidList.getDirection(id);
    //Real speed = velocity.norm();
    //velocity = (speed > opt->maxVel ? velocity*opt->maxVel/speed : velocity);
    //boidList.setVelocity(id, velocity);

    //Update position
    //Vector pos = boidList.getPosition(id) + opt->dt * boidList.getVelocity(id);

    //Make sure the boid stays inside the domain
    //Real modX = fmod(pos.x, opt->domainSize);
    //Real modY = fmod(pos.y, opt->domainSize);
    //Real modZ = fmod(pos.z, opt->domainSize);
    //pos.x = modX > 0 ? modX : modX + opt->domainSize;
    //pos.y = modY > 0 ? modY : modY + opt->domainSize;
    //pos.z = modZ > 0 ? modZ : modZ + opt->domainSize;
    //boidList.setPosition(id, pos);
    //}

    //void applyForcesKernel(Real*boidData, const int nBoids, const struct Options *opt) {
    //dim3 gridDim(1024,1,1); // TODO: max threads/block in globals.hpp using cudaUtils
    //dim3 blockDim(ceil((float)nBoids/1024),1,1); 

    //applyForces<<<gridDim,blockDim,0,0>>>(boidData, nBoids, opt);

    //cudaDeviceSynchronize();
    //checkKernelExecution();
    //}


    //template <typename T1, typename T2>
    //struct AffectLeftOperandIfEqual : public thrust::binary_function<T1, T2, T1> {
    //__device__ T1 operator()(const T1 &a, const T2 &b) const {
    //if(static_cast<T2>(a) == b)
    //return a;
    //else
    //return T1(-1);
    //}
    //};

}
}

template <typename T>
struct ComputeCellFunctor
{
    unsigned int width, length, height;
    T xmin, ymin, zmin, xmax, ymax, zmax, radius;

    ComputeCellFunctor(const BoidGrid<T> &boidGrid) :
        width(boidGrid.getWidth()), length(boidGrid.getLength()), height(boidGrid.getHeight()),
        xmin(boidGrid.getLocalDomain().min[0]), ymin(boidGrid.getLocalDomain().min[1]), zmin(boidGrid.getLocalDomain().min[2]),
        xmax(boidGrid.getLocalDomain().max[0]), ymax(boidGrid.getLocalDomain().max[2]), zmax(boidGrid.getLocalDomain().max[2]),
        radius(boidGrid.getMaxRadius()) {
        }

    template <typename Tuple>
        __host__ __device__ void operator()(Tuple t) //X Y Z C
        {
            thrust::get<3>(t) = makeId(
                    static_cast<unsigned int>(floor(relativeX(thrust::get<0>(t)) * width)), 
                    static_cast<unsigned int>(floor(relativeY(thrust::get<1>(t)) * length)), 
                    static_cast<unsigned int>(floor(relativeZ(thrust::get<2>(t)) * height))
                    );
        }

    __host__ __device__ T relativeX(T x) { return (x - xmin)/(xmax - xmin);}
    __host__ __device__ T relativeY(T y) { return (y - ymin)/(ymax - ymin);}
    __host__ __device__ T relativeZ(T z) { return (z - zmin)/(zmax - zmin);}
    __host__ __device__ unsigned int makeId(unsigned int x, unsigned int y, unsigned int z) { return (width*length*z + width*y + x); }
};

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

    //copy  X Y Z  VX VY VZ  AX AY AZ  data to device
    for(unsigned int i = 0; i < 9u; i++)
        thrust::copy(boidGrid.getBoidHostMemoryView()[i], boidGrid.getBoidHostMemoryView()[i] + nAgents, agents_thrust_d[i]);

    //compute cell Id for each boid
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(agents_thrust_d.x, agents_thrust_d.y, agents_thrust_d.z, agents_thrust_d.id)),
            thrust::make_zip_iterator(thrust::make_tuple(
                    agents_thrust_d.x + nAgents, 
                    agents_thrust_d.y + nAgents,
                    agents_thrust_d.z + nAgents,
                    agents_thrust_d.id + nAgents)),
            ComputeCellFunctor<T>(boidGrid));

    //find the permutation to sort everyone according to the cellIds
    thrust::device_vector<unsigned int> keys(nAgents);
    thrust::sequence(keys.begin(), keys.end());
    thrust::stable_sort_by_key(agents_thrust_d.id, agents_thrust_d.id + nAgents, keys.begin());

    //find the cells that contains at least one agent
    //and find coresponding array offsets to be copied from the cells
    thrust::device_vector<unsigned int> uniqueIds(agents_thrust_d.id, agents_thrust_d.id + nAgents);
    thrust::device_vector<unsigned int> offsets(nAgents);
    thrust::sequence(offsets.begin(), offsets.end());

    thrust::pair<deviceIterator_ui, deviceIterator_ui> end =
        thrust::unique_by_key(
                uniqueIds.begin(),
                uniqueIds.end(),
                offsets.begin());

    unsigned int nUniqueIds = thrust::distance(uniqueIds.begin(), end.first);
    uniqueIds.resize(nUniqueIds);
    offsets.resize(nUniqueIds);

    //count number of boids per key using computed offsets
    thrust::device_vector<unsigned int> count(nUniqueIds);
    thrust::transform(offsets.begin()+1, offsets.end(), offsets.begin(), count.begin(), thrust::minus<unsigned int>());
    count[nUniqueIds-1] = nAgents - offsets[nUniqueIds-1];

    //keep filled cells for neighborlookup
    thrust::device_vector<int> validIds(nCells);
    thrust::fill(validIds.begin(), validIds.end(), -1);
    thrust::scatter(
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(nUniqueIds),
            uniqueIds.begin(), validIds.begin());

    //sort the boids with precomputed permutation
    thrust::device_vector<T> buffer(9u*nAgents);
    BoidMemoryView<T> buf_view(buffer.data().get(), nAgents);
    ThrustBoidMemoryView<T> buffer_view(buf_view);

    for(unsigned int i = 0u; i < 9u; i++) {
        thrust::copy(
                thrust::make_permutation_iterator(agents_thrust_d[i], keys.begin()),
                thrust::make_permutation_iterator(agents_thrust_d[i], keys.end()),
                buffer_view[i]);
        thrust::copy(buffer_view[i], buffer_view[i]+nAgents, agents_thrust_d[i]);
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

    thrust::copy(uniqueIds.begin(), uniqueIds.end(), uniqueIds_d.wrap());
    thrust::copy(offsets.begin(), offsets.end(), offsets_d.wrap());
    thrust::copy(count.begin(), count.end(), count_d.wrap());
    thrust::copy(validIds.begin(), validIds.end(), validIds_d.wrap());

    std::cout << "Boid IDs:\t";
    for(int i = 0; i < nAgents; i++)
    std::cout << agents_thrust_d.id[i] << " ";
    std::cout << std::endl;

    std::cout << "Unique IDs:\t";
    for(int i = 0; i < uniqueIds.size(); i++)
    std::cout << uniqueIds[i] << " ";
    std::cout << std::endl;

    std::cout << "Offsets:\t";
    for(int i = 0; i < uniqueIds.size(); i++)
    std::cout << offsets[i] << " ";
    std::cout << std::endl;

    std::cout << "Count:\t\t";
    for(int i = 0; i < uniqueIds.size(); i++)
    std::cout << count[i] << " ";
    std::cout << std::endl;

    std::cout << "Valid Ids:\t";
    for(int i = 0; i < validIds.size(); i++)
    std::cout << validIds[i] << " ";
    std::cout << std::endl;
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

    // Compute mean positions (only for filled cells)
    thrust::device_vector<T>  means(3*nUniqueIds);
    ThrustVectorMemoryView<T> means_v(means);
    {
        thrust::device_vector<unsigned int> buffKeys(nUniqueIds);

        for (unsigned int i = 0; i < 3u; i++) {
            thrust::reduce_by_key(agents_thrust_d.id, agents_thrust_d.id + nAgents,
                    agents_thrust_d[i], buffKeys.begin(), means_v[i], 
                    thrust::equal_to<unsigned int>(), thrust::plus<float>());

            thrust::transform(means_v[i], means_v[i] + nUniqueIds,
                    count_d.wrap(), means_v[i], thrust::divides<float>());
        }
    }

    //Compute forces
    kernel::boidgrid::computeForcesKernel(agents_d.data(), means_v.data(),
            uniqueIds_d.data(), count_d.data(),
            offsets_d.data(), validIds_d.data(),
            nAgents, nUniqueIds, nCells);


    //apply forces
    //applyForcesKernel(d_agents, agents.size(), d_opt);

    //check for boids that went outside the domain

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
