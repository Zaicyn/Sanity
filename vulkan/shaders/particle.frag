#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in float fragResidual;
layout(location = 2) in float fragElongation;

layout(location = 0) out vec4 outColor;

void main() {
    // Minimal test - just output the color directly
    outColor = vec4(fragColor.rgb, 1.0);
}
