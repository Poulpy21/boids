
#ifndef VECTORMEMORYVIEW_H
#define VECTORMEMORYVIEW_H

template <typename T> 
struct VectorMemoryView {

    public:
        size_t _size;
        T *&x,  *&y,  *&z;
    private:
        T* ptrs[3];

    public:
        VectorMemoryView() :
            _size(0), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {
                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = nullptr;
                }
            }

        VectorMemoryView(T* v, size_t size) :
            _size(size), x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)) {

                for (unsigned int i = 0; i < 3; i++) {
                    ptrs[i] = v + i*_size;
                }
            }

        VectorMemoryView<T>& operator=(const VectorMemoryView<T> &other) {
            for (unsigned int i = 0; i < 3; i++) {
                this->ptrs[i] = other[i];
            }
            return *this;
        }

        T* data() const {
            return ptrs[0];
        }

        unsigned int size() const {
            return _size;
        }

        T* operator[](unsigned int i) const {
            return ptrs[i];
        }
};

#endif /* end of include guard: VECTORMEMORYVIEW_H */
