
#include <string>
#include <sstream>
#include <cmath>
#include "defines.hpp"

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
    template <typename I> bool areEqual(I a, I b);

    template <>
        bool areEqual<float>(float a, float b) {
            float epsilon = 1.19209e-07;
            return (std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b)));
        }

    template <>
        bool areEqual<double>(double a, double b) {
            double epsilon = 2.22045e-16;
            return (std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b)));
        }

    template <typename I> I modulo(I a, I b);

    template <>
        float modulo<float>(float a, float b) {
            return fmodf(a, b);
        }
    
    template <>
        double modulo<double>(double a, double b) {
            return fmod(a, b);
        }
#endif

}

