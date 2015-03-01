#ifndef GPURESSOURCE_H
#define GPURESSOURCE_H

#include "headers.hpp"
#include "GPUMemory.hpp"

#define NEXT_POW_2(x) (1ul << (static_cast<unsigned int>(ceil(log2(static_cast<double>(x))))))

template <typename T>
class GPUResource {
public:
	
	GPUResource(int device, unsigned long size, unsigned long realSize);
	GPUResource(const GPUResource<T> &original);
	~GPUResource();

	explicit GPUResource(T *data, int deviceId, unsigned long size, unsigned long realSize, bool owner);

	void setData(T* data, int deviceId, unsigned long size, unsigned long realSize, bool isOwner);
	void free();
	void allocate();
	void reallocate(unsigned long size, unsigned long realSize);

	void setSize(unsigned long size, unsigned long realSize);

	T* data() const;
	int deviceId() const;
	unsigned long size() const;
	unsigned long realSize() const;
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
	unsigned long _size, _realSize;
    bool _isAlignedOnPowerOfTwo;
	bool _isOwner;
	bool _isGPUResource;

};

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource);

#include "GPUResource.tpp"

#endif /* end of include guard: GPURESOURCE_H */
