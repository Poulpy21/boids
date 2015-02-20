
#ifndef CONSTVECTORMEMORYVIEW_H
#define CONSTVECTORMEMORYVIEW_H

template <typename T> 
struct ConstVectorMemoryView {

    public:
        size_t _size;
        T const * &x,  *&y,  *&z;
    private:
        T const * ptrs[3];

    public:
        __HOST__ __DEVICE__ ConstVectorMemoryView() :
            _size(0), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {
                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = nullptr;
                }
            }

        __HOST__ __DEVICE__ ConstVectorMemoryView(T const *const v, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = v + i*_size;
                }
            }

        __HOST__ __DEVICE__ ConstVectorMemoryView<T>& operator=(const ConstVectorMemoryView<T> &other) {
            for (unsigned int i = 0; i < 3; i++) {
                this->ptrs[i] = other[i];
            }
            return *this;
        }

        __HOST__ __DEVICE__ T const * data() const {
            return ptrs[0];
        }

        __HOST__ __DEVICE__ unsigned int size() const {
            return _size;
        }

        __HOST__ __DEVICE__ T const * operator[](unsigned int i) const {
            return ptrs[i];
        }
};

#endif /* end of include guard: VECTORMEMORYVIEW_H */
