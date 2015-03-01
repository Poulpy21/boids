
#include <cassert>
#include <iostream>

template <typename T>
GPUResource<T>::GPUResource(int device, unsigned long size, unsigned long realSize) :
    _data(0), _deviceId(device), _size(size), _realSize(realSize), _isOwner(false), _isGPUResource(false) {
    assert(realSize >= size);
}
	
template <typename T>
GPUResource<T>::GPUResource(const GPUResource<T> &original) :
_data(original.data()), _deviceId(original.deviceId()), _size(original.size()), _realSize(original.realSize()), 
_isOwner(false), _isGPUResource(original.isGPUResource()) {
}

template <typename T>
GPUResource<T>::GPUResource(T *data, int deviceId, unsigned long size, unsigned long realSize, bool owner) :
_data(data), _deviceId(deviceId), _size(size), _realSize(realSize), _isOwner(owner), _isGPUResource(true) {
	assert((data == 0 && size == 0 && realSize == 0) || (data != 0 && size != 0 && realSize != 0));
    assert(realSize >= size);
}

template <typename T>
GPUResource<T>::~GPUResource() {
	this->free();
}

template <typename T>
T* GPUResource<T>::data() const {
	return _data;
}

template <typename T>
unsigned long GPUResource<T>::size() const {
	return _size;
}
	
template <typename T>
unsigned long GPUResource<T>::realSize() const {
    return _realSize;
}

template <typename T>
unsigned long GPUResource<T>::bytes() const {
	return _size * sizeof(T);
}

template <typename T>
bool GPUResource<T>::isOwner() const {
	return _isOwner;
}

template <typename T>
bool GPUResource<T>::isGPUResource() const {
	return _isGPUResource;
}

template <typename T>
int GPUResource<T>::deviceId() const {
	return _deviceId;
}

template <typename T>
const std::string GPUResource<T>::getResourceType() const {
	const std::string str("Device array");
	return str;
}

template <typename T>
void GPUResource<T>::setData(T* data, int deviceId, unsigned long size, unsigned long realSize, bool isOwner) {
	assert((data == 0 && size == 0 && realSize == 0) || (data != 0 && size != 0 && realSize != 0));
    assert(realSize >= size);
	assert(_isOwner != true);

	_data = data;
	_deviceId = deviceId;
	_size = size;
    _realSize = realSize;
	_isOwner = isOwner;
	_isGPUResource = true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const GPUResource<T> &resource) {
	out << "::GPURessource::" << std::endl;
	out << "\t Is GPU Ressource : " << resource.isGPUResource() << std::endl;
	out << "\t Device ID : " << resource.deviceId() << std::endl;
	out << "\t Ressource type : " << resource.getResourceType() << std::endl;
	out << "\t Data : " << typeid(T).name() << std::endl;
	out << "\t Size : " << resource.size() << std::endl;
	out << "\t Bytes : " << resource.bytes() << std::endl;

	return out;
}
	
template <typename T>
void GPUResource<T>::free() {

	if(_isGPUResource && _isOwner) {
		GPUMemory::free<T>(_data, _realSize, _deviceId);
	}

	_data = 0;
	_deviceId = 0;
	_size = 0;
	_isOwner = false;
	_isGPUResource = false;
}

template <typename T>
void GPUResource<T>::allocate() {
	assert(!(_isGPUResource && _isOwner));
	_data = GPUMemory::malloc<T>(_realSize, _deviceId);

	_isOwner = true;
	_isGPUResource = true;
}
	
template <typename T>
void GPUResource<T>::setSize(unsigned long size, unsigned long realsize) {
    assert(realsize >= size);
	this->_size = size;
    this->_realSize = realsize;
}

template <typename T>
void GPUResource<T>::reallocate(unsigned long size, unsigned long realsize) {
    assert(realsize >= size);
    if(size <= _realSize && realsize == _realSize) {
        _size = size; 
    }
    else {
        this->free();
        this->setSize(size, realsize);
        this->allocate();
    }
}
	

#ifdef THRUST_ENABLED
template <typename T>
    thrust::device_ptr<T> GPUResource<T>::wrap() {
        return thrust::device_ptr<T>(this->data());
    }
#endif

