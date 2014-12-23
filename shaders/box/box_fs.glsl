#version 130 

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

out vec4 out_colour;

void main (void)
{	
	out_colour = vec4(0.88,0.66,0.37,1.0);
}
