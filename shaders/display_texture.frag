#version 460

in vec2 TexCoord;

uniform sampler2D tex;

// simply output the color of the texture as fragment color
out vec4 FragColor;

void main()
{
    FragColor = texture(tex, TexCoord);
}