#version 330 core
out vec4 FragColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;    
    float shininess;
}; 

struct Light {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;  
in vec3 Normal;
in float y;  // for terrain texture
  
uniform vec3 viewPos;
uniform Material material;
uniform Light light;


void main()
{
    vec3 color1 = vec3(0.44, 0.51, 0.34);  // terrain green
    vec3 color2 = vec3(0.58, 0.50, 0.40);  // terrain brown
    float factor = clamp(y, 0.0, 1.0);
    vec3 color = mix(color1, color2, factor);

    // ambient
    // vec3 ambient = light.ambient * material.ambient;
    vec3 ambient = light.ambient * color * 0.2;
    
    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * color * 0.8);
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);  
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}