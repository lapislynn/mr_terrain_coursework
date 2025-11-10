#version 330 core
layout (location = 0) in vec3 aPos;     // Input attribute: vertex position
layout (location = 1) in vec3 aNormal;  // Input attribute: vertex normal.

out vec3 FragPos;
out vec3 Normal;
out float y;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;
    y = aPos.y*20.0 + 0.25;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}