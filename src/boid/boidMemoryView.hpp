
#ifndef BOIDMEMORYVIEW_H
#define BOIDMEMORYVIEW_H

template <typename T> 
struct BoidMemoryView {

public:
    size_t nAgents;
    T *&x,  *&y,  *&z;
    T *&vx, *&vy, *&vz;
    T *&ax, *&ay, *&az;
    unsigned int *&id;
private:
    T* ptrs[9];
    unsigned int *cellId;


public:
    BoidMemoryView() :
        nAgents(0),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        ax(*(ptrs+6)), ay(*(ptrs+7)), az(*(ptrs+8)),
        id(cellId), cellId(nullptr) {
            for (unsigned int i = 0; i < 9; i++) {
                ptrs[i] = nullptr;
            }
    }
    
    BoidMemoryView(T* a, size_t nAgents) :
        nAgents(nAgents),
         x(*(ptrs+0)),  y(*(ptrs+1)),  z(*(ptrs+2)), 
        vx(*(ptrs+3)), vy(*(ptrs+4)), vz(*(ptrs+5)), 
        ax(*(ptrs+6)), ay(*(ptrs+7)), az(*(ptrs+8)),
        id(cellId), cellId(nullptr) {
            
            for (unsigned int i = 0; i < 9; i++) {
                ptrs[i] = a + i*nAgents;
            }
            cellId = reinterpret_cast<unsigned int*>(a + 9*nAgents);
        }

    BoidMemoryView<T>& operator=(const BoidMemoryView<T> &other) {
        for (unsigned int i = 0; i < 9; i++) {
            this->ptrs[i] = other[i];
        }
        this->cellId = other.cellId;

        return *this;
    }

    T* data() const {
        return ptrs[0];
    }
    
    unsigned int size() const {
        return nAgents;
    }

    T* operator[](unsigned int i) const {
        return ptrs[i];
    }

};

#endif /* end of include guard: BOIDMEMORYVIEW_H */
