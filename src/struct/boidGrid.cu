
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
        xmin(boidGrid.getDomain().min[0]), ymin(boidGrid.getDomain().min[1]), zmin(boidGrid.getDomain().min[2]),
        xmax(boidGrid.getDomain().max[0]), ymax(boidGrid.getDomain().max[2]), zmax(boidGrid.getDomain().max[2]),
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
__host__ void initBoidGridThrustArrays(const BoidGrid<T> &boidGrid, 
        BoidMemoryView<T> &agents_h, BoidMemoryView<T> &agents_d, 
        unsigned int nAgents) {
   
    ThrustBoidMemoryView<T> agents_thrust_d(agents_d);
    thrust::device_vector<unsigned int> cellIds(nAgents);

    //copy X Y Z data to device
    thrust::copy(agents_h.x, agents_h.x + nAgents, agents_thrust_d.x);
    thrust::copy(agents_h.y, agents_h.y + nAgents, agents_thrust_d.y);
    thrust::copy(agents_h.z, agents_h.z + nAgents, agents_thrust_d.z);

    //compute cell Id for each boid
    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(agents_thrust_d.x, agents_thrust_d.y, agents_thrust_d.z, cellIds.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(agents_thrust_d.x, agents_thrust_d.y, agents_thrust_d.z, cellIds.end())),
            ComputeCellFunctor<T>(boidGrid));

    //find the permutation to sort everyone according to the cellIds
    thrust::device_vector<unsigned int> keys(nAgents);
    thrust::sequence(keys.begin(), keys.end());
    thrust::stable_sort_by_key(cellIds.begin(), cellIds.end(), keys.begin());

    //find the cells that contains at least one agent
    //and find coresponding array offsets to be copied from the cells
    thrust::device_vector<unsigned int> uniqueIds(cellIds);
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


    /*for(int i = 0; i < uniqueIds.size(); i++)*/
        /*std::cout << uniqueIds[i] << " " << offsets[i] << std::endl;;*/
    /*std::cout << std::endl;*/

}


template __host__ void initBoidGridThrustArrays<float>(const BoidGrid<float> &boidGrid, BoidMemoryView<float> &agents_h, 
        BoidMemoryView<float> &agents_d, unsigned int nAgents);
/*template __host__ void initBoidGridThrustArrays<double>(const BoidGrid<double> &boidGrid, BoidMemoryView<double> *agents_h, BoidMemoryView<double> *agents_d, unsigned int nAgents);*/

#endif

#endif
