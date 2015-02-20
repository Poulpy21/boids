

#include "GPUMemory.hpp"
#include <cassert>

const unsigned long * GPUMemory::_memorySize = nullptr;
const unsigned long * GPUMemory::_memoryRuntime = nullptr;
unsigned long * GPUMemory::_memoryLeft = nullptr;
int GPUMemory::_nDevice = 0;
bool GPUMemory::_verbose = true;

GPUMemory::GPUMemory() {
}

void GPUMemory::init() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	GPUMemory::_nDevice = nDevices;
	GPUMemory::_memorySize = new unsigned long[nDevices];
	GPUMemory::_memoryLeft = new unsigned long[nDevices];
	GPUMemory::_memoryRuntime = new unsigned long[nDevices];

	cudaDeviceProp prop;
	for (int i = 0; i < nDevices; i++) {
		size_t avail, total;
		cudaSetDevice(i);
		cudaGetDeviceProperties(&prop, i);
		cudaMemGetInfo( &avail, &total );
		const_cast<unsigned long *>(GPUMemory::_memorySize)[i] = total;
		const_cast<unsigned long *>(GPUMemory::_memoryRuntime)[i] = _GPU_MIN_RESERVED_MEMORY;
		GPUMemory::_memoryLeft[i] = avail - _GPU_MIN_RESERVED_MEMORY;
	}
}

unsigned long GPUMemory::memorySize(int deviceId) {
	assert(_nDevice > deviceId);
	return GPUMemory::_memorySize[deviceId];
}

unsigned long GPUMemory::memoryLeft(int deviceId) {
	size_t avail;
	size_t total;
	cudaSetDevice(deviceId);
	cudaMemGetInfo( &avail, &total );
	return avail - _GPU_MIN_RESERVED_MEMORY;
}

		
void GPUMemory::display(std::ostream &out) {
	out << ":: GPU VRAM Status ::" << std::endl; 
	for (int i = 0; i < _nDevice; i++) {
		out << "\t GPU " << i
			<< "\t Total : " << utils::toStringMemory(GPUMemory::_memorySize[i]) 
			<< "\t Reserved : " << utils::toStringMemory(GPUMemory::_memoryRuntime[i])
			<< "\t Used : " << utils::toStringMemory(GPUMemory::_memorySize[i] - GPUMemory::_memoryLeft[i]) 
			<< "\t " << 100*(float)(GPUMemory::_memorySize[i] - GPUMemory::_memoryLeft[i])/GPUMemory::_memorySize[i] << "%"
			<< std::endl; 
	}

}
		
unsigned long GPUMemory::getMinAvailableMemoryOnDevices() {

	unsigned long minMemAvailable = memoryLeft(0);
	for (int i = 1; i < _nDevice; i++) {
		minMemAvailable = (GPUMemory::memoryLeft(i) < minMemAvailable ? GPUMemory::memoryLeft(i) : minMemAvailable);
	}

	return minMemAvailable;
}

void GPUMemory::setVerbose(bool verbose) {
	GPUMemory::_verbose = verbose;
}
