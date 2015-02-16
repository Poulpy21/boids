
#ifndef RAND_H
#define RAND_H

#include <cstdlib>
#include <time.h>  

namespace Random {
                void init();

                float randf();
                unsigned long randl();

                int randi(int LO, int HI);
                float randf(float LO, float HI);
};

#endif /* end of include guard: RAND_H */
