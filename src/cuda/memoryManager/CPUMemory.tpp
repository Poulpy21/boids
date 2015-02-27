
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudaUtils.hpp"
#include "log.hpp"
#include "utils.hpp"

template <typename T>
T* CPUMemory::malloc(unsigned long nData, bool pinnedMemory) {

    pthread_mutex_lock(&_mtx);
    T *data;
    {
        assert(CPUMemory::_memoryLeft >= nData * sizeof(T));

        if(_verbose) {
            if(nData == 0) {
                log_console->warn("Trying to allocate a 0 size block !");
            }
            else {
                log_console->infoStream() << "\tAllocating " << utils::toStringMemory(nData*sizeof(T)) << " on host !";  
            }
        }

        if(pinnedMemory) {
            CHECK_CUDA_ERRORS(cudaMallocHost((void **) &data, nData * sizeof(T)));
        }
        else {
            try
            {
                data = new T[nData];		
            }
            catch (std::bad_alloc &e)
            {
                log_console->infoStream() << "\tAllocating " << utils::toStringMemory(nData*sizeof(T)) << " on host !";  
                log_console->errorStream() << "\tAllocation failed...";
                exit(EXIT_FAILURE);
            }
        }

        CPUMemory::_memoryLeft -= nData*sizeof(T);
    }
    pthread_mutex_unlock(&_mtx);

    return data;
}

template <typename T>
void CPUMemory::free(T* data, unsigned long nData, bool pinnedMemory) {

    pthread_mutex_lock(&_mtx);
    {
        if(_verbose) {
            if(nData == 0) {
                log4cpp::log_console->warn("Trying to free a 0 size block !");
            }
            else {
                log4cpp::log_console->infoStream() << "\tFreeing " << utils::toStringMemory(nData*sizeof(T)) << " on host !";  
            }
        }

        if(pinnedMemory) { 
            CHECK_CUDA_ERRORS(cudaFreeHost(data));
        }
        else {
            delete [] data;
        }

        CPUMemory::_memoryLeft += nData*sizeof(T);

    }
    pthread_mutex_unlock(&_mtx);
}

template <typename T>
bool CPUMemory::canAllocate(unsigned long nData) {
    return (CPUMemory::_memoryLeft >= nData * sizeof(T));
}


