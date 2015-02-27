#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "utils/types.hpp"
#include "parser.hpp"

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

    static const unsigned int nData = 11u;

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

    Options(double *data) {
        assert(sizeof(double) == sizeof(unsigned long int));
        unsigned int i = 0;
        nAgents = static_cast<unsigned long int>(data[i++]);
        nSteps = static_cast<unsigned long int>(data[i++]);
        wCohesion = data[i++];
        wAlignment = data[i++];
        wSeparation = data[i++];
        rCohesion = data[i++];
        rAlignment = data[i++];
        rSeparation = data[i++];
        dt = data[i++];
        maxVel = data[i++];
        domainSize = data[i++];
    }

    double* serialize() {
        assert(sizeof(double) == sizeof(unsigned long int));
        unsigned int i = 0;
        double *data = new double[Options::nData];
        data[i++] = static_cast<double>(nAgents);
        data[i++] = static_cast<double>(nSteps);
        data[i++] = wCohesion;
        data[i++] = wAlignment;
        data[i++] = wSeparation;
        data[i++] = rCohesion;
        data[i++] = rAlignment;
        data[i++] = rSeparation;
        data[i++] = dt;
        data[i++] = maxVel;
        data[i++] = domainSize;
        return data;
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


};

#endif
