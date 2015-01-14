#ifndef TYPES
#define TYPES

#include "headers.hpp"

// Forward declaration
class Agent;

#ifndef __CUDACC__
using Real = double;
using Container = std::vector<Agent>;
#else
#define Real double
#endif

#endif
