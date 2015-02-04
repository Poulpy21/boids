#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#ifndef __CUDACC__
#include "utils/types.hpp"
#include "parser.hpp"
#endif


struct Options {
    unsigned long int nAgents;
    unsigned long int nSteps;
    Real wCohesion;
    Real wAlignment;
    Real wSeparation;
    Real rCohesion;
    Real rAlignment;
    Real rSeparation;
    Real dt;
    Real maxVel;
    Real domainSize;

#ifndef __CUDACC__
    Options() : nAgents(0l), nSteps(0l), 
        wCohesion(0.0), wAlignment(0.0), wSeparation(0.0),
        rCohesion(0.0), rAlignment(0.0), rSeparation(0.0),
        dt(0.0), maxVel(0.0), domainSize(1.0) {};

    Options(ArgumentParser &parser) {
        nAgents     = static_cast<unsigned long int>(parser("agents").asInt());
        nSteps      = static_cast<unsigned long int>(parser("steps").asInt());
        wCohesion   = parser("wc").asDouble();
        wAlignment  = parser("wa").asDouble();
        wSeparation = parser("ws").asDouble();
        rCohesion   = parser("rc").asDouble();
        rAlignment  = parser("ra").asDouble();
        rSeparation = parser("rs").asDouble();
        dt          = parser("dt").asDouble();
        maxVel      = parser("mv").asDouble();
        domainSize  = parser("size").asDouble(1.0);
    }

    friend std::ostream& operator<<(std::ostream &stream, const Options &opt) {
        return stream <<"The simulation will be executed with the following parameters " << std::endl
                      << "[ nAgents : "     << opt.nAgents      << " ] "
                      << "[ nSteps : "      << opt.nSteps       << " ]"
                      << "[ wCohesion : "   << opt.wCohesion    << " ]"
                      << "[ wAlignement : " << opt.wAlignment   << " ]"
                      << "[ wSeparation : " << opt.wSeparation  << " ]"
                      << "[ rCohesion : "   << opt.rCohesion    << " ]"
                      << "[ rAlignment : "  << opt.rAlignment   << " ]"
                      << "[ rSeparation : " << opt.rSeparation  << " ]"
                      << "[ dt : "          << opt.dt           << " ]"
                      << "[ maxVelocity : " << opt.maxVel       << " ]";
    }
#endif
};

#endif
