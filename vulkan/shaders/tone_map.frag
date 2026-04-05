// tone_map.frag
// Fullscreen quad pass. Reads R32_UINT density buffer, outputs HDR color.
//
// Pipeline: one triangle covering NDC [-1,1]² (no vertex buffer needed —
// generate positions in the vertex shader from gl_VertexIndex).
// See tone_map.vert below.

#version 450

layout(location = 0) in  vec2 uv;        // [0,1]², from vertex shader
layout(location = 0) out vec4 frag_color;

// Density buffer written by harmonic_sample.comp
layout(set = 0, binding = 0) uniform utexture2D density_buf;
layout(set = 0, binding = 1) uniform sampler   samp;         // nearest, clamp

layout(push_constant) uniform TonePush {
    float peak_density;   // normalisation: max expected count per pixel
    float exposure;       // linear pre-scale before log compression
    float gamma;          // output gamma (2.2 for sRGB, 1.0 for linear)
    int   colormap;       // 0=thermal, 1=plasma, 2=blue-white, 3=hopf
} tp;

// ----------------------------------------------------------------------------
// Colormaps — all analytic, no texture lookups
// t in [0, 1], returns linear RGB
// ----------------------------------------------------------------------------

vec3 colormap_thermal(float t) {
    // Black → deep red → orange → yellow → white
    return clamp(vec3(
        t * 3.0,
        t * 3.0 - 1.0,
        t * 3.0 - 2.0
    ), 0.0, 1.0);
}

vec3 colormap_plasma(float t) {
    // Viridis-adjacent: purple → magenta → orange → yellow
    const vec3 c0 = vec3(0.050, 0.030, 0.527);
    const vec3 c1 = vec3(0.796, 0.196, 0.388);
    const vec3 c2 = vec3(0.993, 0.906, 0.144);
    float s = clamp(t, 0.0, 1.0);
    return s < 0.5
        ? mix(c0, c1, s * 2.0)
        : mix(c1, c2, s * 2.0 - 1.0);
}

vec3 colormap_blue_white(float t) {
    // Deep space: black → navy → cyan → white
    return clamp(vec3(
        t * t,
        t * t * t,
        t
    ), 0.0, 1.0);
}

vec3 colormap_hopf(float t) {
    // Custom: black → deep violet → hopf-blue → shell-gold
    // Chosen to make the 8-shell structure visually distinct
    const vec3 c0 = vec3(0.00, 0.00, 0.00);  // void
    const vec3 c1 = vec3(0.18, 0.02, 0.42);  // inner shells, deep violet
    const vec3 c2 = vec3(0.05, 0.45, 0.90);  // mid shells, hopf-blue
    const vec3 c3 = vec3(0.98, 0.82, 0.30);  // outer shell / jets, gold
    float s = clamp(t, 0.0, 1.0);
    if (s < 0.33)      return mix(c0, c1, s * 3.0);
    else if (s < 0.66) return mix(c1, c2, (s - 0.33) * 3.0);
    else               return mix(c2, c3, (s - 0.66) * 3.0);
}

// ----------------------------------------------------------------------------
// Log-compression tone map — preserves structure across many decades of density
// Same idea as astronomical image processing (arcsinh stretch)
// ----------------------------------------------------------------------------
float tone_compress(float raw_density) {
    float d = raw_density * tp.exposure / max(tp.peak_density, 1.0);
    return log(1.0 + d) / log(1.0 + tp.exposure);  // log stretch [0,1]
}

void main() {
    // Sample density buffer (nearest — no filtering on integer counts)
    uint  raw = texture(usampler2D(density_buf, samp), uv).r;
    float t   = tone_compress(float(raw));

    vec3 rgb;
    if      (tp.colormap == 0) rgb = colormap_thermal(t);
    else if (tp.colormap == 1) rgb = colormap_plasma(t);
    else if (tp.colormap == 2) rgb = colormap_blue_white(t);
    else                       rgb = colormap_hopf(t);      // default

    // Gamma encode for display
    rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / tp.gamma));

    frag_color = vec4(rgb, 1.0);
}


// ============================================================================
// tone_map.vert — paste into separate file or use specialisation constants
// Generates a fullscreen triangle from gl_VertexIndex, no VBO needed.
// Draw with vkCmdDraw(cmd, 3, 1, 0, 0).
// ============================================================================
//
// #version 450
// layout(location = 0) out vec2 uv;
// void main() {
//     // Three vertices covering NDC [-1,1]² with one overdraw triangle
//     vec2 pos = vec2(
//         float((gl_VertexIndex & 1) << 2) - 1.0,
//         float((gl_VertexIndex & 2) << 1) - 1.0
//     );
//     uv = pos * 0.5 + 0.5;
//     gl_Position = vec4(pos, 0.0, 1.0);
// }
