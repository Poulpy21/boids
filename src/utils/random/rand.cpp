

#include "rand.hpp"

namespace Random {
    void init() {
        srand (time(NULL));
    }

    float randf() {
        return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    float randf(float LO, float HI) {
        return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }

    int randi(int LO, int HI) {
        return LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }

    unsigned long randl()
    {
        if (sizeof(int) < sizeof(long))
            return static_cast<unsigned long>(static_cast<unsigned long>(rand()) << (sizeof(int) * 8u)) | rand();

        return rand();
    }
}
