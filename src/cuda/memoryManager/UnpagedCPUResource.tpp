
template <typename T>
UnpagedCPUResource<T>::UnpagedCPUResource(unsigned long size) : CPUResource<T>(size)
{
}

template <typename T>
UnpagedCPUResource<T>::UnpagedCPUResource(T *data, unsigned int size, bool owner) :
CPUResource<T>(data, size, owner)
{
}

template <typename T>
UnpagedCPUResource<T>::~UnpagedCPUResource() {
	if(this->_isCPUResource && this->_isOwner) {
		CPUMemory::free<T>(this->_data, this->_size, false);
	}
}
			
template <typename T>
const std::string UnpagedCPUResource<T>::getResourceType() const {
	return std::string("Unpaged CPU Memory");
}

template <typename T>
void UnpagedCPUResource<T>::free() {
	
	if(this->_isCPUResource && this->_isOwner) {
		delete [] this->_data;
	}

	this->_data = 0;
	this->_size = 0;
	this->_isOwner = false;
	this->_isCPUResource = false;
}

template <typename T>
void UnpagedCPUResource<T>::allocate() {
	this->_data = CPUMemory::malloc<T>(this->_size, false);
	this->_isCPUResource = true;
	this->_isOwner = true;
}
