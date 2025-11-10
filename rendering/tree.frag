#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;

uniform sampler2D texture_diffuse1;
uniform vec3 aLightDir;

void main()
{
    vec4 color = texture(texture_diffuse1, TexCoords);
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-aLightDir);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 colorfinal = (0.4 + diff) * color * 0.8;
    colorfinal.w = 1.0;
    FragColor = colorfinal;
}