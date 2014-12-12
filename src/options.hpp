#include "utils/types.hpp"
#include "parser.hpp"

#ifndef OPTIONS_HPP
#define OPTIONS_HPP

struct Options {
    unsigned long int nAgents;
    unsigned long int nSteps;
    double wCohesion;
    double wAlignment;
    double wSeparation;
    double rCohesion;
    double rAlignment;
    double rSeparation;

    Options() : nAgents(0l), nSteps(0l), 
        wCohesion(0.0), wAlignment(0.0), wSeparation(0.0),
        rCohesion(0.0), rAlignment(0.0), rSeparation(0.0) {};

    Options(ArgumentParser &parser) {
        nAgents     = static_cast<unsigned long int>(parser("agents").asInt());
        nSteps      = static_cast<unsigned long int>(parser("steps").asInt());
        wCohesion   = parser("wc").asDouble();
        wAlignment  = parser("wa").asDouble();
        wSeparation = parser("ws").asDouble();
        rCohesion   = parser("rc").asDouble();
        rAlignment  = parser("ra").asDouble();
        rSeparation = parser("rs").asDouble();
    }

    friend std::ostream& operator<<(std::ostream &stream, const Options &opt) {
        return stream << opt.nAgents << " "
                      << opt.nSteps << " "
                      << opt.wCohesion << " "
                      << opt.wAlignment << " "
                      << opt.wSeparation << " "
                      << opt.rCohesion << " "
                      << opt.rAlignment << " "
                      << opt.rSeparation;
    }
};

#endif
