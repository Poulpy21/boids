
#ifndef VECTORMEMORYVIEW_H
#define VECTORMEMORYVIEW_H

#include "constVectorMemoryView.hpp"

template <typename T> 
struct VectorMemoryView {

    public:
        size_t _size;
        T *&x,  *&y,  *&z;
    private:
        T* ptrs[3];

    public:
        __HOST__ __DEVICE__ VectorMemoryView() :
            _size(0), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {
                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = nullptr;
                }
            }

        __HOST__ __DEVICE__ VectorMemoryView(T* v, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = v + i*_size;
                }
            }

        __HOST__ __DEVICE__ VectorMemoryView<T>& operator=(const VectorMemoryView<T> &other) {
            for (unsigned int i = 0; i < 3; i++) {
                this->ptrs[i] = other[i];
            }
            return *this;
        }

        __HOST__ __DEVICE__ T* data() const {
            return ptrs[0];
        }

        __HOST__ __DEVICE__ unsigned int size() const {
            return _size;
        }

        __HOST__ __DEVICE__ T* operator[](unsigned int i) const {
            return ptrs[i];
        }
};

#endif /* end of include guard: VECTORMEMORYVIEW_H */