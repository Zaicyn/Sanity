#version 450

layout(set = 0, binding = 0) uniform sampler2D inputImage;

layout(push_constant) uniform PushConstants {
    vec2 direction;  // (1,0) for horizontal, (0,1) for vertical
    float texelSize;
} push;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

// 9-tap gaussian kernel
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 texOffset = push.direction * push.texelSize;

    // Center sample
    vec3 result = texture(inputImage, fragTexCoord).rgb * weights[0];

    // Blur samples
    for (int i = 1; i < 5; i++) {
        result += texture(inputImage, fragTexCoord + texOffset * float(i)).rgb * weights[i];
        result += texture(inputImage, fragTexCoord - texOffset * float(i)).rgb * weights[i];
    }

    outColor = vec4(result, 1.0);
}
