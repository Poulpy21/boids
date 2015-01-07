#version 330

in vec3 vertex_position;

out VS_FS_VERTEX {
	vec3 pos;
} vertex_out;

uniform mat4 modelMatrix = mat4(1,0,0,0,
				0,1,0,0,
				0,0,1,0,
				0,0,0,1);

layout(std140) uniform projectionView {
    mat4 projectionMatrix;
    mat4 viewMatrix;
    vec3 cameraPos;
    vec3 cameraDir;
    vec3 cameraUp;
    vec3 cameraRight;
};

void main (void)
{	
	vertex_out.pos = vertex_position;
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertex_position,1);
}

