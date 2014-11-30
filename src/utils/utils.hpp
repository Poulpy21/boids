
#ifndef UTILS_H
#define UTILS_H

#include "headers.hpp"

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>

namespace utils {

	template<typename A, typename B>
		std::pair<B,A> flip_pair(const std::pair<A,B> &p)
		{
			return std::pair<B,A>(p.second, p.first);
		}

	template<typename A, typename B>
		std::map<B,A> flip_map(const std::map<A,B> &src)
		{
			std::map<B,A> dst;
			std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()), 
					flip_pair<A,B>);
			return dst;
		}

	template<typename A, typename B>
		std::vector<std::pair<B,A> > mapToReversePairVector(const std::map<A,B> &src) {
			std::vector<std::pair<B,A> > dst;

			auto it = src.begin();
			while(it != src.end()) {
				dst.push_back(std::pair<B,A>(it->second, it->first));
				++it;
			}

			return dst;
		}
    
    template <typename F>
    bool areEqual(const F a, const F b) {
        static_assert(std::is_floating_point<F>::value, "F must be floating point !");
        return (std::abs(a - b) <= 10*std::numeric_limits<F>::epsilon() * std::max(std::abs(a), std::abs(b)));
    }

	const std::string toStringMemory(unsigned long bytes);
}


#endif /* end of include guard: UTILS_H */
