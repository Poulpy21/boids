
#ifndef BOIDMEMORYVIEW_H
#define BOIDMEMORYVIEW_H

template <typename T> 
struct BoidMemoryView {
    const T *x,  *y,  *z;
    const T *vx, *vy, *vz;
    const T *ax, *ay, *az;
    
    BoidMemoryView(T* a, size_t nAgents) :
        x(a+0*nAgents),  y(a+1*nAgents),  z(a+2*nAgents), 
        vx(a+3*nAgents), vy(a+4*nAgents), vz(a+5*nAgents), 
        ax(a+6*nAgents), ay(a+7*nAgents), az(a+8*nAgents) {
        }
};

#endif /* end of include guard: BOIDMEMORYVIEW_H */
