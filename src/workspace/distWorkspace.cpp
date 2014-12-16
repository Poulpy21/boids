#include "utils/headers.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "distWorkspace.hpp"
#include "messenger.hpp"

DistWorkspace::DistWorkspace(MPI_Comm comm, int root) : 
    agents(), comm(comm), rootID(root)
{
    MPI_Comm_rank(comm, &myID);
    init();
}

void DistWorkspace::init() {
    mess.broadcastOptions(comm, &opt, rootID);
    //TODO init boids
}

void DistWorkspace::update() {
    // Compute mean boid
    // Get mean boids from the neighborhood and send ours (approximation)
    // Compute and apply forces to local boids
    // Exchange boids that cross domain boundaries
}

void DistWorkspace::save(int stepId) {
    std::ofstream myfile;

    std::stringstream ss;
    ss << "boids" << std::setw(3) << std::setfill('0') << myID << ".xyz";
    myfile.open(ss.str(), stepId==0 ? std::ios::out : std::ios::app);

    myfile << std::endl;
    myfile << agents.size() << std::endl;
    for (Agent a : agents)
        myfile << "B " << a.position;

    myfile.close();

    //TODO zlib?
}

