#version 330

//uniform sampler2D boidTexture0;
//uniform sampler2D boidTexture1;
//uniform sampler2D boidTexture2;

//in fBoidType;

out vec4 outColor;

void main (void)
{	
	outColor = vec4(1.0);
    /*if (fBoidType == 0) {
        outColor = texture(boidTexture0, gl_PointCoord);
    } else if (fBoidType == 1) {
        outColor = texture(boidTexture0, gl_PointCoord);
    } else if (fBoidType == 2) {
        outColor = texture(boidTexture0, gl_PointCoord);
    }*/
}
