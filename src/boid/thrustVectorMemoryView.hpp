

#ifdef __CUDACC__
#ifdef THRUST_ENABLED

#ifndef THRUSTVECTORMEMORYVIEW_H
#define THRUSTVECTORMEMORYVIEW_H

#include "headers.hpp"
#include "vectorMemoryView.hpp"

template <typename T> 
struct ThrustVectorMemoryView {

    public:
        size_t _size;
        thrust::device_ptr<T> &x,  &y,  &z;
    private:
        thrust::device_ptr<T> ptrs[3];
        thrust::device_ptr<unsigned int> cellId;


    public:
        ThrustVectorMemoryView() :
            _size(0), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {
                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = thrust::device_ptr<T>(nullptr);
                }
            }

        ThrustVectorMemoryView(T* a, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {
                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = thrust::device_ptr<T>(a + i*_size);
                }
            }
        
        ThrustVectorMemoryView(thrust::device_ptr<T> &ptr, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = ptr + i*_size;
                }
            }
        
        ThrustVectorMemoryView(thrust::device_vector<T> &v, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = v.data() + i*_size;
                }
            }


        ThrustVectorMemoryView(const VectorMemoryView<T> &memView) :
            _size(memView.size()), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = thrust::device_ptr<T>(memView[i]);
                }
            }

        ThrustVectorMemoryView<T>& operator=(const ThrustVectorMemoryView<T> &other) {
            for (unsigned int i = 0; i < 3; i++) {
                this->ptrs[i] = other[i];
            }
            return *this;
        }

        thrust::device_ptr<T> operator[](unsigned int i) const {
            return ptrs[i];
        }

        unsigned int size() const {
            return _size;
        }
        
        T* data() const {
            return ptrs[0].get();
        }
};
#endif /* end of include guard: THRUSTVECTORMEMORYVIEW_H */
#endif 
#endif
