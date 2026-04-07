// render_color.cuh — Device-side color helpers for rendering
// ===========================================================
// Contains:
//   - vec3: simple device-only 3-component vector
//   - mix(): linear interpolation
//   - blackbody(): Kuhlman approximation of Planckian locus (temp → RGB)
#pragma once

#include <cuda_runtime.h>

struct vec3 {
    float x, y, z;
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

__device__ vec3 mix(vec3 a, vec3 b, float t) {
    return vec3(
        a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
    );
}

// ============================================================================
// BLACKBODY RADIATION (Kuhlman approximation of Planckian Locus)
// Maps temperature (Kelvin) to RGB via Stefan-Boltzmann + Wien's law
// ============================================================================
__device__ float3 blackbody(float temp) {
    // Clamp temperature to visible range (1000K = deep red, 40000K = electric blue)
    temp = fminf(fmaxf(temp, 1000.0f), 40000.0f) / 100.0f;
    float3 color;

    // Red component
    if (temp <= 66.0f) {
        color.x = 255.0f;
    } else {
        color.x = temp - 60.0f;
        color.x = 329.698727446f * powf(color.x, -0.1332047592f);
    }

    // Green component
    if (temp <= 66.0f) {
        color.y = temp;
        color.y = 99.4708025861f * logf(color.y) - 161.1195681661f;
    } else {
        color.y = temp - 60.0f;
        color.y = 288.1221695283f * powf(color.y, -0.0755148492f);
    }

    // Blue component
    if (temp >= 66.0f) {
        color.z = 255.0f;
    } else if (temp <= 19.0f) {
        color.z = 0.0f;
    } else {
        color.z = temp - 10.0f;
        color.z = 138.5177312231f * logf(color.z) - 305.0447927307f;
    }

    // Normalize to [0,1] and clamp
    return make_float3(
        fminf(fmaxf(color.x / 255.0f, 0.0f), 1.0f),
                       fminf(fmaxf(color.y / 255.0f, 0.0f), 1.0f),
                       fminf(fmaxf(color.z / 255.0f, 0.0f), 1.0f)
    );
}
