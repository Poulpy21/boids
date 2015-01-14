#version 330 

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

out vec4 out_colour;

uniform int level = 0;

vec4 levelToColor(int level);
void main (void)
{	
	out_colour = levelToColor(level);
}

vec4 levelToColor(int level) {
    
    int levelm = level % 8;
    vec3 outColour;

    if(level == 0)
        outColour = vec3(0.0,0.0,0.0);
    else if(levelm == 1)
        outColour = vec3(0.0,0.0,1.0);
    else if(levelm == 2)
        outColour = vec3(0.0,1.0,1.0);
    else if(levelm == 3)
        outColour = vec3(0.0,1.0,0.0);
    else if(levelm == 4)
        outColour = vec3(1.0,1.0,0.0);
    else if(levelm == 5)
        outColour = vec3(1.0,0.0,0.0);
    else if(levelm == 6)
        outColour = vec3(1.0,0.0,1.0);
    else 
        outColour = vec3(1.0,1.0,1.0);

    return vec4(outColour, 1.0);
}
