#version 330

uniform vec3 inColor;
/*uniform sampler2D boidTexture0;*/
//uniform sampler2D boidTexture1;
//uniform sampler2D boidTexture2;

//in fBoidType;

out vec4 outColor;

in float test;

void main (void)
{	
	//outColor = vec4(1.0);
    //if (fBoidType == 0) {
        //outColor = texture(boidTexture0, gl_PointCoord);
    /*} else if (fBoidType == 1) {
        outColor = texture(boidTexture0, gl_PointCoord);
    } else if (fBoidType == 2) {
        outColor = texture(boidTexture0, gl_PointCoord);
    }*/
    if (distance(gl_PointCoord,vec2(0.5,0.5))<0.5)
        outColor = vec4(inColor,1.0);
    else
        discard;
}
