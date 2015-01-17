#version 330 

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

out vec4 out_colour;

uniform int level = 0;
uniform float fillrate = 0;

vec4 levelToColor(int level);
vec4 interpLevel(int level, float alpha);
void main (void)
{	
	/*out_colour = interpLevel(level, fillrate);*/
    out_colour = levelToColor(level);
}

vec4 levelToColor(int level) {
    int levelm = level % 8;
    
    float r = float(levelm >= 4);
    float g = float((levelm == 2 || levelm == 3 || levelm == 4 || levelm == 7));
    float b = float((levelm == 1 || levelm == 2 || levelm == 6 || levelm == 7));

    return vec4(r,g,b,1.0);
}

vec4 interpLevel(int level, float alpha) {
    return mix(levelToColor(level), levelToColor(level+1), alpha);
}
