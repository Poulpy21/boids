#ifndef TYPES
#define TYPES

#include "headers.hpp"

// Forward declaration
class Agent;

#ifndef __CUDACC__
using Real = float;
using Container = std::vector<Agent>;
#else
#define Real float
#endif

#endif
