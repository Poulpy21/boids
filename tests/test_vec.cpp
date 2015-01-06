
#include "headers.hpp"
#include "vec3.hpp"
#include <cassert>

int main(int argc, char **argv) {

    using std::cout;
    using std::endl;
    
    Vec3<float> v0;
    Vec3<float> v1(1.0f,2.0f,3.0f);
    Vec3<float> v2(3.0f,2.0f,1.0f);
    
    cout << "v0: " << v0 << endl;
    cout << "v1: " << v1 << endl;
    cout << "v2: " << v2 << endl;

    assert(v0 + v0 == 2.0f*v0);
    assert(v1 + v1 == v1*2.0f);

    assert(v1 + v2 == v0+4.0f);
    assert(v2 + v1 == 4.0f+v0);

    assert(v0 + v1 + v2 == v2 + v1 + v0);
    //assert((-v0) == -1000000.0f*v0);
    //assert((-v1) == -1.0f*v1);

    cout << v1 << endl;
    cout << "(" << v1.x << "," << v1.y << "," << v1.z << ")" << endl;
    
    v1.x = 10.0f;
    v1.y = 20.0f;
    v1.z = 30.0f;
    cout << "(" << v1.x << "," << v1.y << "," << v1.z << ")" << endl;
    
    std::swap(v1.x, v1.z);
    cout << "(" << v1.x << "," << v1.y << "," << v1.z << ")" << endl;


    {
        Vec3<float>   v1(1.0f,1.0f,1.0f);
        Vec3<float>   v2(v1);
        Vec<3u,float> v3(v1);

        v2.x = 2.0f;
        v2.y = 2.0f;
        v2.z = 2.0f;

        v3[0] = 3.0f;
        v3[1] = 3.0f;
        v3[2] = 3.0f;

        cout << v1 << " || " << v2 << " || " << v3 << endl;
    }
}

