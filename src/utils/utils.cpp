
#include <string>
#include <sstream>
#include <cmath>

namespace utils {

	std::string toStringMemory(unsigned long bytes) {
		std::stringstream ss;

		const char prefix[] = {' ', 'K', 'M', 'G', 'T', 'P'};
		unsigned long val = 1;
		for (int i = 0; i < 6; i++) {
			if(bytes < 1024*val) {
				ss << round(100*(float)bytes/val)/100.0 << prefix[i] << 'B';
				break;
			}
			val *= 1024;
		}

		const std::string str(ss.str());
		return str;
	}

#ifdef CUDA_ENABLED
    std::string toStringDim(const dim3 &dim) {
        std::stringstream ss;
        ss << "(" << dim.x << "," << dim.y << "," << dim.z << ")";
        return ss.str();
    }
#endif

}
