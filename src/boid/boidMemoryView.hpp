
#ifndef BOIDMEMORYVIEW_H
#define BOIDMEMORYVIEW_H

#include "headers.hpp"
#include "vectorMemoryView.hpp"
#include "constBoidMemoryView.hpp"

template <typename T> 
struct BoidMemoryView {

    private:
        size_t _size;
        VectorMemoryView<T> position, velocity, force;
        unsigned int *cellId;

    public:
        T *&x, *&y, *&z;
        T *&vx,  *&vy,  *&vz;
        T *&fx,  *&fy,  *&fz; 
        unsigned int *&id;
        
        static const unsigned int N = 10;

    public:
        __HOST__ __DEVICE__ BoidMemoryView() :
            _size(0),
            position(), velocity(), force(), 
            cellId(nullptr), 
            x(position.x),  y(position.y),  z(position.z), 
            vx(velocity.x), vy(velocity.y), vz(velocity.z), 
            fx(force.x), fy(force.y), fz(force.z),
            id(cellId) {
            }

        __HOST__ __DEVICE__ BoidMemoryView(T* a, size_t nAgents) :
            _size(nAgents), 
            position(a,_size), velocity(a+3*_size,_size), force(a+6*_size,_size), 
            cellId(reinterpret_cast<unsigned int*>(a+(N-1)*_size)),
            x(position.x),  y(position.y),  z(position.z), 
            vx(velocity.x), vy(velocity.y), vz(velocity.z), 
            fx(force.x), fy(force.y), fz(force.z),
            id(cellId) {
            }
        
        __HOST__ __DEVICE__ BoidMemoryView<T>& operator=(const BoidMemoryView<T> &other) {
            _size = other.size();
            position = other.pos();
            velocity = other.vel();
            force = other.forc();
            this->cellId = other.cellId;
            return *this;
        }

        __HOST__ __DEVICE__ T* data() const {
            return position.data();
        }

        __HOST__ __DEVICE__ unsigned int size() const {
            return _size;
        }

        __HOST__ __DEVICE__ VectorMemoryView<T> pos() const {
            return position;
        }

        __HOST__ __DEVICE__ VectorMemoryView<T> vel() const {
            return velocity;
        }

        __HOST__ __DEVICE__ VectorMemoryView<T> forc() const {
            return force;
        }

        __HOST__ __DEVICE__ T* operator[](unsigned int i) const {
            unsigned int k = i/3;
            unsigned int r = i%3;

            switch(k) {
                case 0: return position[r];
                case 1: return velocity[r];
                case 2: return force[r];
                default: return nullptr;
            }
        }

};

#endif /* end of include guard: BOIDMEMORYVIEW_H */
