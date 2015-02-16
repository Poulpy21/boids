

#ifndef UNPAGEDCPURESOURCE_H
#define UNPAGEDCPURESOURCE_H

#include "CPUResource.hpp"

template <typename T>
class UnpagedCPUResource : public CPUResource<T> {
	
public:
	UnpagedCPUResource(unsigned long size = 0);
	UnpagedCPUResource(T *data, unsigned int size, bool owner = false);
	~UnpagedCPUResource();

	const std::string getResourceType() const;
	
	void free();
	void allocate();
};

#include "UnpagedCPUResource.tpp"

#endif /* end of include guard: PAGEDCPURESOURCE_H */
