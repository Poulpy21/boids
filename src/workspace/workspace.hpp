/*
*/
#ifndef  WORKSPACE
#define  WORKSPACE

#include "parser.hpp"
#include "types.hpp"


class Workspace
{
protected:
  Container agents;
  unsigned int na;

  Real dt;
  int time;
  Real wCohesion, wAlignment, wSeparation;
  Real rCohesion, rAlignment, rSeparation;
  Real maxU;

  Real tUpload, tDownload, tCohesion, tAlignment, tSeparation;

  Real domainsize;
  void init();
public:
  Workspace(ArgumentParser &parser);

  Workspace(size_t nAgents,
  Real wc, Real wa, Real ws,
  Real rc, Real ra, Real rs);

  void move();
  void simulate(int nsteps);
  void save(int stepid);
};

#endif
