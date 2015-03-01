

#ifdef __CUDACC__
#ifdef THRUST_ENABLED

#ifndef THRUSTBOIDMEMORYVIEW_H
#define THRUSTBOIDMEMORYVIEW_H

#include "headers.hpp"
#include "boidMemoryView.hpp"

template <typename T> 
struct ThrustBoidMemoryView {

public:
    thrust::device_ptr<T> &x,  &y,  &z;
    thrust::device_ptr<T> &vx,  &vy, &vz;
    thrust::device_ptr<T> &fx,  &fy, &fz;
    thrust::device_ptr<unsigned int> &id;
    
    static const unsigned int N = 10;
private:
    size_t nAgents;
    thrust::device_ptr<T> ptrs[N-1];
    thrust::device_ptr<unsigned int> cellId;


public:
    ThrustBoidMemoryView() :
        nAgents(0),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        fx(*(ptrs+6)), fy(*(ptrs+7)), fz(*(ptrs+8)),
        id(cellId), cellId() {
            for (unsigned int i = 0; i < N-1; i++) {
                ptrs[i] = thrust::device_ptr<T>(nullptr);
            }
    }
    
    ThrustBoidMemoryView(T* a, size_t nAgents) :
        nAgents(nAgents),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        fx(*(ptrs+6)), fy(*(ptrs+7)), fz(*(ptrs+8)),
        id(cellId), cellId() {
            
            for (unsigned int i = 0; i < N-1; i++) {
                ptrs[i] = thrust::device_ptr<T>(a + i*nAgents);
            }
            cellId = thrust::device_ptr<unsigned int>(reinterpret_cast<unsigned int*>(a + (N-2)*nAgents));
        }
    
    ThrustBoidMemoryView(const BoidMemoryView<T> &memView) :
        nAgents(memView.size()),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        fx(*(ptrs+6)), fy(*(ptrs+7)), fz(*(ptrs+8)),
        id(cellId), cellId() {
            
            for (unsigned int i = 0; i < N-1; i++) {
                ptrs[i] = thrust::device_ptr<T>(memView[i]);
            }
            cellId = thrust::device_ptr<unsigned int>(memView.id);
        }
        
    ThrustBoidMemoryView(thrust::device_vector<T> v, size_t nAgents) :
        nAgents(nAgents), 
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        fx(*(ptrs+6)), fy(*(ptrs+7)), fz(*(ptrs+8)),
        id(cellId), cellId() {
            for (unsigned int i = 0; i < N-1; i++) {
                ptrs[i] = thrust::device_ptr<T>(thrust::raw_pointer_cast(v.data()) + i*nAgents);
            }
            cellId = thrust::device_ptr<unsigned int>(reinterpret_cast<unsigned int*>(thrust::raw_pointer_cast(v.data())) + (N-1)*nAgents);
        }


    ThrustBoidMemoryView<T>& operator=(const ThrustBoidMemoryView<T> &other) {
        for (unsigned int i = 0; i < N-1; i++) {
            this->ptrs[i] = other[i];
        }
        this->cellId = other.cellId;

        return *this;
    }
    
    T* data() const {
        return thrust::raw_pointer_cast(ptrs[0]);
    }

    thrust::device_ptr<T> operator[](unsigned int i) const {
        return ptrs[i];
    }

    
};
#endif /* end of include guard: THRUSTBOIDMEMORYVIEW_H */
#endif 
#endif
