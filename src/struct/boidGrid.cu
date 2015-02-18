
#ifdef __CUDACC__

#include "headers.hpp"
#include "boidGrid.hpp"
#include "boidMemoryView.hpp"

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
        radius(boidGrid.getMinRadius()) {
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

template <typename T>
__host__ void initBoidGridThrustArrays(const BoidGrid<T> &boidGrid, T* agents_h, unsigned int nAgents) {
    
    BoidMemoryView<T> view(agents_h, nAgents);

    thrust::device_vector<T> x(nAgents);
    thrust::device_vector<T> y(nAgents);
    thrust::device_vector<T> z(nAgents);
    thrust::device_vector<T> cellIds(nAgents);

    thrust::copy(view.x, view.x + nAgents, x.begin());
    thrust::copy(view.y, view.y + nAgents, y.begin());
    thrust::copy(view.z, view.z + nAgents, z.begin());

    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), cellIds.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), cellIds.end())),
            ComputeCellFunctor<T>(boidGrid));


    // print the output
    for(int i = 0; i < 100; i++)
        std::cout << x[i] << ", " << y[i] << ", " << z[i] << " => " << cellIds[i] << std::endl;
}


template __host__ void initBoidGridThrustArrays<float>(const BoidGrid<float> &boidGrid, float* agents_h, unsigned int nAgents);
/*template __host__ void initBoidGridThrustArrays<double>(const BoidGrid<double> &boidGrid, double* agents_h, unsigned int nAgents);*/

#endif


#endif
