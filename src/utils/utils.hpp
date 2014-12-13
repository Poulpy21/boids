
#ifndef UTILS_H
#define UTILS_H

#include "headers.hpp"

#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <cmath>

namespace utils {
    
    // power of two
    template <typename T>
    constexpr bool is_power_of_two(T x)
    {
        static_assert(std::is_integral<T>::value,"Only integrals types should call a power of two check !");
        return x && ((x & (x-1)) == 0);
    }

    template <typename T>
    constexpr T get_power_of_two(T x)
    {
        static_assert(utils::is_power_of_two<T>(x), "x is not a power of two !");
        T xx(x);
        T p(0);
        while(xx > 1) {
            xx >> 1;
            p++;
        }
        return p;
    }

    // map inversion
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
   
    
    //integral and floating point equality (SFINAE)
    template <typename I, typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
    inline bool areEqual(I a, I b) {
        return a == b;
    }

    template <typename F, typename std::enable_if<std::is_floating_point<F>::value>::type* = nullptr>
    inline bool areEqual(F a, F b) {
        return (std::abs(a - b) <= std::numeric_limits<F>::epsilon() * std::max(std::abs(a), std::abs(b)));
    }
    
    template <typename I, typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
    inline I modulo(I a, I b) {
        return a % b;
    }

    template <typename F, typename std::enable_if<std::is_floating_point<F>::value>::type* = nullptr>
    inline F modulo(const F a, const F b) {
        return F(fmod(a, b));
    }

	const std::string toStringMemory(unsigned long bytes);
}


#endif /* end of include guard: UTILS_H */
