
#ifdef __CUDACC__

#include "headers.hpp"
#include "boidGrid.hpp"
#include "boidMemoryView.hpp"
#include "thrustBoidMemoryView.hpp"

#ifdef THRUST_ENABLED

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

    uniqueIds.resize(thrust::distance(uniqueIds.begin(), end.first));
    offsets.resize(thrust::distance(uniqueIds.begin(), end.first));

    //sort the boids with precomputed permutation
    thrust::device_vector<T> buffer(9u*nAgents);
    BoidMemoryView<T> buf_view(buffer.data().get(), nAgents);
    ThrustBoidMemoryView<T> buffer_view(buf_view);

    for(unsigned int i = 0u; i < 9u; i++) {
        thrust::copy(thrust::make_permutation_iterator(agents_thrust_d[i], keys.begin()),
                thrust::make_permutation_iterator(agents_thrust_d[i], keys.end()),
                buffer_view[i]);
        thrust::copy(buffer_view[i], buffer_view[i]+nAgents, agents_thrust_d[i]);
    }

    //allocate and store additional data
    unsigned int uniqueIdsSize = uniqueIds.size();
    GPUResource<bool> &validIds_d = boidGrid.getDeviceValidIds(); 
    GPUResource<unsigned int> &uniqueIds_d = boidGrid.getDeviceUniqueIds(); 
    GPUResource<unsigned int> &offsets_d = boidGrid.getDeviceOffsets(); 

    uniqueIds_d.setSize(uniqueIdsSize);
    uniqueIds_d.allocate();
    offsets_d.setSize(uniqueIdsSize);
    offsets_d.allocate();
    validIds_d.setSize(nAgents);
    validIds_d.allocate();
        
    thrust::copy(uniqueIds.begin(), uniqueIds.end(), uniqueIds_d.wrap());
    thrust::copy(offsets.begin(), offsets.end(), offsets_d.wrap());
    //thrust::copy(validIds.begin(), validIds.end(), validIds_d.wrap());
   
    for(int i = 0; i < uniqueIds.size(); i++)
        /*std::cout << uniqueIds[i] << " " << offsets[i] << std::endl;;*/
    std::cout << std::endl;
}

template <typename T>
__host__ BoidMemoryView<T> computeThrustStep(BoidGrid<T> &boidGrid) {
    
    BoidMemoryView<T> outOfDomainBoids;

    ThrustBoidMemoryView<T> agents_thrust_d(boidGrid.getBoidDeviceMemoryView());

    return outOfDomainBoids;
}


// full specializations
template __host__ void initBoidGridThrustArrays<float> (BoidGrid<float > &boidGrid);
template __host__ void initBoidGridThrustArrays<double>(BoidGrid<double> &boidGrid);

template __host__ BoidMemoryView<float> computeThrustStep<float>(BoidGrid<float> &boidGrid);
template __host__ BoidMemoryView<double> computeThrustStep<double>(BoidGrid<double> &boidGrid);

#endif

#endif
