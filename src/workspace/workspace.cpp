#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

#include "../boid/agent.hpp"
#include "vec3.hpp"
#include "workspace.hpp"

Workspace::Workspace(ArgumentParser &parser) :
    dt(0.05), maxU(2.0)
{

    na = static_cast<unsigned int>(parser("agents").asInt());

    wCohesion = parser("wc").asDouble();
    wAlignment = parser("wa").asDouble();
    wSeparation = parser("ws").asDouble();

    rCohesion = parser("rc").asDouble();
    rAlignment = parser("ra").asDouble();
    rSeparation = parser("rs").asDouble();

    this->init();
}

Workspace::Workspace(size_t nAgents,
        Real wc, Real wa, Real ws,
        Real rc, Real ra, Real rs) :
    na(nAgents), dt(.05), time(0),
    wCohesion(wc), wAlignment(wa), wSeparation(ws),
    rCohesion(rc), rAlignment(ra), rSeparation(rs),
    maxU(2.)
{ 
    this->init();
}

void  Workspace::init() {
    domainsize = 1.0;

    // Random generator seed
    srand48(std::time(0));

    // Initialize agents
    // This loop may be quite expensive due to random number generation
    for(size_t j = 0; j < na; j++){
        // Create random position
        Vec3<Real> position(drand48(), drand48(), drand48());

        // Create random velocity
        agents.push_back(Agent(position, Vec3<Real>(), Vec3<Real>()));
    }
}

void Workspace::move()
{
    Vec3<Real> s,c,a;

    for(size_t k = 0; k< na; k++){
        s = agents[k].separation(agents, k, rSeparation);
        c = agents[k].cohesion(agents, k, rCohesion);
        a = agents[k].alignment(agents, k, rAlignment);

        agents[k].direction = wCohesion*c + wAlignment*a + wSeparation*s;
    }

    // Integration in time using euler method
    for(size_t k = 0; k< na; k++){
        agents[k].velocity += agents[k].direction;

        double speed = agents[k].velocity.norm();
        if (speed > maxU) {
            agents[k].velocity *= maxU/speed;
        }
        agents[k].position += dt*agents[k].velocity;

        agents[k].position.x= fmod(agents[k].position.x,domainsize);
        agents[k].position.y= fmod(agents[k].position.y,domainsize);
        agents[k].position.z= fmod(agents[k].position.z,domainsize);
    }
}

void Workspace::simulate(int nsteps) {
    // store initial positions
    save(0);

    // perform nsteps time steps of the simulation
    int step = 0;
    while (step++ < nsteps) {
        this->move();
        // store every 20 steps
        /*if (step%20 == 0)*/ save(step);
    }
}

void Workspace::save(int stepid) {
    std::ofstream myfile;
    myfile.open("data/boids.xyz", stepid==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << na << std::endl;
    for (size_t p=0; p<na; p++)
        myfile << "B " << agents[p].position << std::endl;

    myfile.close();
}
