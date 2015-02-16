#include "headers.hpp"
#include <ostream>

#ifndef CPUMEMORY_H
#define CPUMEMORY_H

class CPUMemory {
	public:
		static void init();

		static unsigned long memorySize(); //bytes
		static unsigned long memoryLeft(); //bytes

		static bool canAllocateBytes(unsigned long nBytes);

		template <typename T>
		static bool canAllocate(unsigned long nData);

		template <typename T>
		static T *malloc(unsigned long nData, bool pinnedMemory=false);

		template <typename T>
		static void free(T *data, unsigned long nData, bool force=false);

		static void display(std::ostream &out);

		static void setVerbose(bool verbose);
			

	private:
		CPUMemory();
		static const volatile unsigned long _memorySize;
		static const volatile unsigned long _memoryRuntime;
		static unsigned long _memoryLeft;
		static bool _verbose;
};

#include "CPUMemory.tpp"

#endif /* end of include guard: CPUMEMORY_H */
