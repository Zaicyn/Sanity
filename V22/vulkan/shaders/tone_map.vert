// tone_map.vert
// Fullscreen triangle — no VBO needed, generates positions from gl_VertexIndex.
// Draw with vkCmdDraw(cmd, 3, 1, 0, 0).

#version 450

layout(location = 0) out vec2 uv;

void main() {
    // Three vertices covering NDC [-1,1]² with one overdraw triangle
    // Vertex 0: (-1, -1)  UV (0, 0)
    // Vertex 1: ( 3, -1)  UV (2, 0)
    // Vertex 2: (-1,  3)  UV (0, 2)
    vec2 pos = vec2(
        float((gl_VertexIndex & 1) << 2) - 1.0,
        float((gl_VertexIndex & 2) << 1) - 1.0
    );
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
