
#ifndef UTILS_H
#define UTILS_H

#include "headers.hpp"
#include <cxxabi.h>

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
        static_assert(x != T(0), "x is zero !");

        T xx(x);
        T p(0);
        while(xx) {
            xx >> 1;
            p++;
        }
        return p-1;
    }
    
    template <typename T>
    constexpr T compute_power_of_two(T x)
    {
        return T(1) << x;
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
  
    template <typename T>
        std::string templatePrettyPrint(std::string &str = std::string("")) {
            int status;
            char * demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
            if(status == 0)
                str += demangled;
            else
                str += typeid(T).name();
            free(demangled);
            return str;
        }
    
    //termination function
    template <typename T>
    std::string genericTemplatePrettyPrint(std::string &str = std::string("")) {
            templatePrettyPrint<T>(str);
            return str;
    }

    template <typename T1, typename T2, typename... Args>
        std::string genericTemplatePrettyPrint(std::string &str = std::string("")) {      
            templatePrettyPrint<T1>(str);
            str += ",";
            genericTemplatePrettyPrint<T2, Args...>(str);
            return str;
        }
    
    template <typename T1, typename T2, typename... Args>
        std::string templatePrettyPrint(std::string &str = std::string("")) {
            str += "<";
            genericTemplatePrettyPrint<T1,T2,Args...>(str);
            str += " >";
            return str;
        }
    
}


#endif /* end of include guard: UTILS_H */
