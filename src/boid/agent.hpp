#ifndef AGENT_HXX
#define AGENT_HXX

#include "headers.hpp"
#include "types.hpp"
#include "vector.hpp"

typedef enum {
  prey,
  predator,
  active,
  wall
} AgentType;

class Agent{
  public :
    Vector position;
    Vector velocity;
    Vector direction;

    Agent(const Vector &pos, const Vector &vel, const Vector &dir);

    Vector separation(Container &agent_list, size_t index, double dist);
    Vector cohesion(Container &agent_list, size_t index, double dist);
    Vector alignment(Container &agent_list, size_t index, double dist);
    size_t find_closest(Container &agent_list, size_t index);
};

#endif
