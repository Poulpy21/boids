#ifndef GPURESSOURCE_H
#define GPURESSOURCE_H

#include "headers.hpp"
#include "GPUMemory.hpp"

template <typename T>
class GPUResource {
public:
	
	GPUResource(int device, unsigned long size = 0);
	GPUResource(const GPUResource<T> &original);
	explicit GPUResource(T *data, int deviceId, unsigned int size, bool owner);
	~GPUResource();

	void setData(T* data, int deviceId, unsigned int size, bool isOwner);
	void free();
	void allocate();
	void reallocate(unsigned long size);

	void setSize(unsigned long size);

	T* data() const;
	int deviceId() const;
	unsigned long size() const;
	unsigned long bytes() const;

	bool isOwner() const;
	bool isGPUResource() const;
	
	const std::string getResourceType() const;

#ifdef THRUST_ENABLED
    thrust::device_ptr<T> wrap();
#endif

protected:
	T* _data;
	int _deviceId;
	unsigned long _size;
	bool _isOwner;
	bool _isGPUResource;

};

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource);

#include "GPUResource.tpp"

#endif /* end of include guard: GPURESOURCE_H */
