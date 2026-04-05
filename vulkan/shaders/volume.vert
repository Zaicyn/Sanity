#version 450

// Fullscreen triangle vertex shader for volume raymarching
// Outputs ray direction from camera through each pixel

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float time;
    float avgScale;
    float avgResidual;
    float heartbeat;
    float padding;
} ubo;

// Fullscreen triangle vertices (no vertex buffer needed)
// Uses gl_VertexIndex to generate a triangle that covers the screen
vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

layout(location = 0) out vec3 fragRayOrigin;
layout(location = 1) out vec3 fragRayDir;
layout(location = 2) out vec2 fragUV;

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);

    // UV for the fragment (0,0 to 1,1)
    fragUV = pos * 0.5 + 0.5;

    // Reconstruct ray direction from clip space
    // We need the inverse of viewProj to unproject
    // For now, use camera position and compute ray direction per-fragment
    fragRayOrigin = ubo.cameraPos;

    // Ray direction will be computed in fragment shader using inverse viewProj
    // Pass clip-space coordinates
    fragRayDir = vec3(pos, 1.0);
}
