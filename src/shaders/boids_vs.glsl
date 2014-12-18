#version 330

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
uniform floatboidSize = 1.0;

in vec3 vertexPosition;
//in int boidType;

//out int fBoidType;

void main(void)
{
    //fBoidType = boidType;
    gl_PointSize = boidSize;
    gl_Position = projectionMatrix * vec4(cameraPos + vertexPosition);
}
