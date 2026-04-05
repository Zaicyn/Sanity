#version 450

layout(set = 0, binding = 0) uniform sampler2D hdrInput;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outBright;

void main() {
    vec3 color = texture(hdrInput, fragTexCoord).rgb;

    // Calculate luminance
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Extract bright pixels (threshold at 1.0 for HDR values)
    float brightness = max(0.0, luma - 1.0);

    // Soft knee for smooth transition
    float knee = 0.5;
    float soft = brightness - 1.0 + knee;
    soft = clamp(soft / (2.0 * knee), 0.0, 1.0);
    soft = soft * soft;
    float contribution = max(soft, brightness - 1.0) / max(brightness, 0.0001);

    outBright = vec4(color * contribution, 1.0);
}
