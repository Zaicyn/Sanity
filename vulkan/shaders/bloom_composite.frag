#version 450

layout(set = 0, binding = 0) uniform sampler2D hdrScene;
layout(set = 0, binding = 1) uniform sampler2D bloomBlur;

layout(push_constant) uniform PushConstants {
    float bloomStrength;
    float exposure;
    float gamma;
} push;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

// ACES filmic tone mapping
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdrColor = texture(hdrScene, fragTexCoord).rgb;
    vec3 bloom = texture(bloomBlur, fragTexCoord).rgb;

    // Add bloom
    vec3 combined = hdrColor + bloom * push.bloomStrength;

    // Apply exposure
    combined *= push.exposure;

    // Tone mapping (ACES)
    vec3 mapped = aces(combined);

    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / push.gamma));

    outColor = vec4(mapped, 1.0);
}
