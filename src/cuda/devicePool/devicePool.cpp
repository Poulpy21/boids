
#include "devicePool.hpp"

#ifdef CUDA_ENABLED 

unsigned int DevicePool::nDevice = 0u;
unsigned int DevicePool::nCores = 0u;
std::vector<Device> DevicePool::devices;

DevicePool::DevicePool() {
}

DevicePool::~DevicePool() {
}
        
void DevicePool::init() {
    int nDev;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&nDev));

    nDevice = static_cast<unsigned int>(nDev);
    nCores = 0u;
    
    for (unsigned int i = 0; i < nDevice; i++) {
        devices.push_back(Device(i));
        nCores += devices[i].coresCount;
    }
}

void DevicePool::display(std::ostream &os) {
    os << DevicePool::toString();
}
        
std::string DevicePool::toString() {
    std::stringstream ss;
    ss << ":: Device Pool ::" << std::endl;
    for(auto &dev : devices) {
        ss <<  "\t" << dev.pciBusID << ":" << dev.pciDeviceID << ":" << dev.pciDomainID
            << "\t" << dev.name 
            << "\t" << dev.totalGlobalMem/(1024*1024) << "MB"
            << "\t" << 2.0*dev.memoryClockRate*(dev.memoryBusWidth/8)/1.0e6 << "GB/s"
            << "\t" << dev.multiProcessorCount << "x" << dev.coresPerSM
            << "="  << dev.coresCount << " CUDA Cores"
            << std::endl;
    }
    ss << "\nTOTAL : " << nDevice << " devices with " << nCores << " CUDA cores." << std::endl;

    ss << "\t=> ";
    if(nCores < 500)
        ss << "Even TX have more computing power... Please upgrade your potato !";
    else if(nCores < 1000)
        ss << "Nice card. Nop just kidding, upgrade your GTX330M please.";
    else 
        ss << "You have enough power. Let the barbecue begin !";
    ss << std::endl;

    return ss.str();
}

#endif
