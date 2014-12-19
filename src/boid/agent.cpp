#include "agent.hpp"

Agent::Agent() : position(), velocity(), direction()
{}

Agent::Agent(const Vec3<Real> &pos, const Vec3<Real> &vel, const Vec3<Real> &dir) : 
    position(pos), velocity(vel), direction(dir)
{}


Vec3<Real> Agent::separation(Container &agent_list, size_t index, double rad) {
    Vec3<Real> force;
    int count = 0;
    for(size_t i = 0; i < agent_list.size(); i++) {
        double dist = (this->position - agent_list[i].position).norm();
        if (i != index && dist < rad) {
            force -= (this->position - agent_list[i].position).normalized();
            ++count;
        }
    }
    return ( count>0 ? force/count : force);
}

Vec3<Real> Agent::cohesion(Container &agent_list, size_t index, double rad) {
    Vec<3u,Real> force;

    int count = 0;
    for(size_t i = 0; i < agent_list.size(); i++) {
        double dist = (this->position - agent_list[i].position).norm();
        if (i != index && dist < rad){
            force += agent_list[i].position;
            ++count;
        }
    }
    return ( count>0 ? force/count : force);
}

Vec3<Real> Agent::alignment(Container&agent_list, size_t index, double rad) {
    Vec<3u,Real> force;

    int count = 0;
    for(size_t i = 0; i < agent_list.size(); i++) {
        double dist = (this->position - agent_list[i].position).norm();
        if (i != index && dist < rad) {
            force += agent_list[i].velocity;
            ++count;
        }
    }
    return ( count>0 ? force/count : force);
}

size_t Agent::find_closest(Container &agent_list, size_t index) {
    size_t closest_agent = index;
    double min_dist = 1000;

    double dist;

    for(size_t i = 0; i < agent_list.size(); i++) {
        if (i != index) {
            dist= (this->position - agent_list[i].position).norm();

            if(dist < min_dist) {
                min_dist = dist;
                closest_agent = i;
            }
        }
    }
    return closest_agent;
}
