#ifndef AGENT_HXX
#define AGENT_HXX

#include "utils/headers.hpp"
#include "types.hpp"
#include "vec3.hpp"

typedef enum {
    prey,
    predator,
    active,
    wall
} AgentType;

class Agent{
    public :
        Vec3<Real> position;
        Vec3<Real> velocity;
        Vec3<Real> direction;

        Agent();
        Agent(const Vec3<Real> &pos, const Vec3<Real> &vel, const Vec3<Real> &dir);

        Vec3<Real> separation(Container &agent_list, size_t index, double dist);
        Vec3<Real> cohesion(Container &agent_list, size_t index, double dist);
        Vec3<Real> alignment(Container &agent_list, size_t index, double dist);
        size_t find_closest(Container &agent_list, size_t index);
};

#endif
