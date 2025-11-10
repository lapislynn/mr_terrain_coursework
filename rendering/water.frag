#version 330 core
out vec4 FragColor;
in vec3 FragPos;
uniform float time;


void main()
{
    vec4 color = vec4(0.11, 0.63, 0.84, 0.6);
    FragColor = color;
}
