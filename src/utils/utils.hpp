
#ifndef UTILS_H
#define UTILS_H

#include "headers.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <sstream>

#ifndef __CUDACC__
#include <vector>
#include <type_traits>
#include <map>
#include <cxxabi.h>
#endif

namespace utils {
    
    std::string toStringMemory(unsigned long bytes);

#ifdef CUDA_ENABLED
    std::string toStringDim(const dim3 &dim);
#endif

    
    // power of two
    template <typename T>
    __HOST__ __DEVICE__ constexpr bool is_power_of_two(T x)
    {
#ifndef __CUDACC__
        static_assert(std::is_integral<T>::value,"Only integrals types should call a power of two check !");
#endif
        return x && ((x & (x-1)) == 0);
    }

    template <typename T>
    __HOST__ __DEVICE__ constexpr T get_power_of_two(T x)
    {
#ifndef __CUDACC__
        static_assert(x != T(0), "x is zero !");
#endif

        T xx(x);
        T p(0);
        while(xx) {
            xx >> 1;
            p++;
        }
        return p-1;
    }
    
    template <typename T>
    __HOST__ __DEVICE__ constexpr T compute_power_of_two(T x)
    {
        return T(1) << x;
    }

#ifndef __CUDACC__
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
#endif
   
    
//integral and floating point equality (SFINAE)
#ifndef CUDA_ENABLED
    template <typename I, typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
     bool areEqual(I a, I b) {
        return a == b;
    }
    
    template <typename F, typename std::enable_if<std::is_floating_point<F>::value>::type* = nullptr>
     bool areEqual(F a, F b) {
        return (std::abs(a - b) <= std::numeric_limits<F>::epsilon() * std::max(std::abs(a), std::abs(b)));
    }
    
    template <typename I, typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
     modulo(I a, I b) {
        return a % b;
    }

    template <typename F, typename std::enable_if<std::is_floating_point<F>::value>::type* = nullptr>
     F modulo(const F a, const F b) {
        return F(fmod(a, b));
    }
#else /* else (CUDA ENABLED) */
    template <typename I>
    __HOST__ __DEVICE__  bool areEqual(I a, I b) { return a == b; }
    template <>
    __HOST__ __DEVICE__  bool areEqual(float a, float b) { 
#ifdef __CUDA_ARCH__
        const float epsilon = 1.19209e-07f;
        return (fabsf(a - b) <= epsilon * fmaxf(fabsf(a), fabsf(b)));
#else
        return (std::abs(a - b) <= std::numeric_limits<float>::epsilon() * std::max(std::abs(a), std::abs(b)));
#endif
    }
    template <>
    __HOST__ __DEVICE__  bool areEqual(double a, double b) { 
#ifdef __CUDA_ARCH__
        const float epsilon = 1.19209e-07f;
        return (fabsf(a - b) <= epsilon * fmaxf(fabsf(a), fabsf(b)));
#else
        return (std::abs(a - b) <= std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b)));
#endif
    }
    
    template <typename I>
    __HOST__ __DEVICE__  I modulo(I a, I b) { return a % b; }
    template <>
    __HOST__ __DEVICE__  float modulo(float a, float b) {
#ifdef __CUDA_ARCH__
        return fmodf(a,b);
#else 
        return fmod(a,b);
#endif
    }
    template <>
    __HOST__ __DEVICE__  double modulo(double a, double b) {
#ifdef __CUDA_ARCH__
        return static_cast<double>(fmodf(a,b));
#else 
        return static_cast<double>(fmod(a,b));
#endif
    }
#endif /* CUDA ENABLED */

#ifndef __CUDACC__ //recursive template :(

    template <typename T>
        void templatePrettyPrint(std::ostream &os) {
            int status;
            char * demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
            if(status == 0)
                os << demangled;
            else
                os << typeid(T).name();
            free(demangled);
        }

    //termination function
    template <typename T>
        void genericTemplatePrettyPrint(std::ostream &os) {
            templatePrettyPrint<T>(os);
        }

    template <typename T1, typename T2, typename... Args>
        void genericTemplatePrettyPrint(std::ostream &os) {      
            templatePrettyPrint<T1>(os);
            os << ",";
            genericTemplatePrettyPrint<T2, Args...>(os);
        }

    template <typename T1, typename T2, typename... Args>
        void templatePrettyPrint(std::ostream &os) {
            os << "<";
            genericTemplatePrettyPrint<T1,T2,Args...>(os);
            os << " >";
        }
#endif
    
}

#endif /* end of include guard: UTILS_H */
