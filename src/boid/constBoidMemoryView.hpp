
#ifndef CONSTBOIDMEMORYVIEW_H
#define CONSTBOIDMEMORYVIEW_H

#include "headers.hpp"
#include "constVectorMemoryView.hpp"

template <typename T> 
struct ConstBoidMemoryView {

    private:
        size_t _size;
        ConstVectorMemoryView<T> position, velocity;//, acceleration;
        unsigned int const *cellId;

    public:
        T const *&x, *&y, *&z;
        T const *&vx,  *&vy,  *&vz;
        //T const *&ax,  *&ay,  *&az;
        unsigned int const *&id;
        
        static const unsigned int N = 7;

    public:
        __HOST__ __DEVICE__ ConstBoidMemoryView() :
            _size(0),
            position(), velocity(), //acceleration(), 
            cellId(nullptr), 
            x(position.x),  y(position.y),  z(position.z), 
            vx(velocity.x), vy(velocity.y), vz(velocity.z), 
            //ax(acceleration.x), ay(acceleration.y), az(acceleration.z),
            id(cellId) {
            }

        __HOST__ __DEVICE__ ConstBoidMemoryView(T const *const a, size_t nAgents) :
            _size(nAgents), 
            position(a,_size), velocity(a+3*_size,_size), //acceleration(a+6*_size,_size), 
            cellId(reinterpret_cast<unsigned int const *>(a+(N-1)*_size)),
            x(position.x),  y(position.y),  z(position.z), 
            vx(velocity.x), vy(velocity.y), vz(velocity.z), 
            //ax(acceleration.x), ay(acceleration.y), az(acceleration.z),
            id(cellId) {
            }

        __HOST__ __DEVICE__ ConstBoidMemoryView<T>& operator=(const ConstBoidMemoryView<T> &other) {
            _size = other.size();
            position = other.pos();
            velocity = other.vel();
            //acceleration = other.acc();
            this->cellId = other.cellId;
            return *this;
        }

        __HOST__ __DEVICE__ T const * data() const {
            return position.data();
        }

        __HOST__ __DEVICE__ unsigned int size() const {
            return _size;
        }

        __HOST__ __DEVICE__ ConstVectorMemoryView<T> pos() const {
            return position;
        }

        __HOST__ __DEVICE__ ConstVectorMemoryView<T> vel() const {
            return velocity;
        }

        //__HOST__ __DEVICE__ ConstVectorMemoryView<T> acc() const {
            //return acceleration;
        //}

        __HOST__ __DEVICE__ T const * operator[](unsigned int i) const {
            unsigned int k = i/3;
            unsigned int r = i%3;

            switch(k) {
                case 0: return position[r];
                case 1: return velocity[r];
                //case 2: return acceleration[r];
                default: return nullptr;
            }
        }

};

#endif /* end of include guard: BOIDMEMORYVIEW_H */
