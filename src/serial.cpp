#include "agent.hpp"
#include "types.hpp"
#include "parser.hpp"
#include "workspace.hpp"

// Main class for running the parallel flocking sim
int main(int argc, char **argv) {
    // Create parser
    ArgumentParser parser;

    // Add options to parser
    parser.addOption("agents", 640);
    parser.addOption("steps", 500);
    //parser.addOption("wc", 12);
    //parser.addOption("wa", 15);
    //parser.addOption("ws", 35);
    parser.addOption("wc", 5);
    parser.addOption("wa", 5);
    parser.addOption("ws", 7);

    //parser.addOption("rc", 0.11);
    //parser.addOption("ra", 0.15);
    //parser.addOption("rs", 0.01);
    parser.addOption("rc", 0.9);
    parser.addOption("ra", 0.5);
    parser.addOption("rs", 0.1);

    // Parse command line arguments
    parser.setOptions(argc, argv);

    // Create workspace
    Workspace workspace(parser);

    // Launch simulation
    int nSteps = parser("steps").asInt();
    workspace.simulate(nSteps);

    return 0;
}

