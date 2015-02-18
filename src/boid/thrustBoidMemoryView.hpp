

#ifdef __CUDACC__
#ifdef THRUST_ENABLED

#ifndef THRUSTBOIDMEMORYVIEW_H
#define THRUSTBOIDMEMORYVIEW_H

#include "headers.hpp"
#include "boidMemoryView.hpp"

template <typename T> 
struct ThrustBoidMemoryView {

public:
    size_t nAgents;
    thrust::device_ptr<T> & x,  &y,  &z;
    thrust::device_ptr<T> &vx,  &vy, &vz;
    thrust::device_ptr<T> &ax,  &ay, &az;
    thrust::device_ptr<unsigned int> &id;
private:
    thrust::device_ptr<T> ptrs[9];
    thrust::device_ptr<unsigned int> cellId;


public:
    ThrustBoidMemoryView() :
        nAgents(0),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        ax(*(ptrs+6)), ay(*(ptrs+7)), az(*(ptrs+8)),
        id(cellId), cellId() {
            for (unsigned int i = 0; i < 9; i++) {
                ptrs[i] = thrust::device_ptr<T>(nullptr);
            }
    }
    
    ThrustBoidMemoryView(T* a, size_t nAgents) :
        nAgents(nAgents),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        ax(*(ptrs+6)), ay(*(ptrs+7)), az(*(ptrs+8)),
        id(cellId), cellId() {
            
            for (unsigned int i = 0; i < 9; i++) {
                ptrs[i] = thrust::device_ptr<T>(a + i*nAgents);
            }
            cellId = thrust::device_ptr<unsigned int>(a + 9*nAgents);
        }
    
    ThrustBoidMemoryView(const BoidMemoryView<T> &memView) :
        nAgents(nAgents),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        ax(*(ptrs+6)), ay(*(ptrs+7)), az(*(ptrs+8)),
        id(cellId), cellId() {
            
            for (unsigned int i = 0; i < 9; i++) {
                ptrs[i] = thrust::device_ptr<T>(memView[i]);
            }
            cellId = thrust::device_ptr<unsigned int>(memView.id);
        }

    ThrustBoidMemoryView<T>& operator=(const ThrustBoidMemoryView<T> &other) {
        for (unsigned int i = 0; i < 9; i++) {
            this->ptrs[i] = other[i];
        }
        this->cellId = other.cellId;

        return *this;
    }

    thrust::device_ptr<T> operator[](unsigned int i) const {
        return ptrs[i];
    }

    
};
#endif /* end of include guard: THRUSTBOIDMEMORYVIEW_H */
#endif 
#endif
