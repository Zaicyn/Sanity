#version 450

// Fullscreen triangle - no vertex buffer needed
// Draws a single triangle that covers the screen
// Vertex IDs: 0, 1, 2 -> positions and UVs calculated from ID

layout(location = 0) out vec2 fragTexCoord;

void main() {
    // Generate fullscreen triangle vertices from gl_VertexIndex
    // Triangle covers [-1,-1] to [3,3] in clip space
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 texCoords[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragTexCoord = texCoords[gl_VertexIndex];
}
