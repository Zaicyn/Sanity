/*
 * V21 VERTEX PACK — SoA particles → AoS ParticleVertex for Vulkan
 * =================================================================
 *
 * Packs V21's CPU particle arrays into V20's ParticleVertex format
 * for rendering. This is the ONLY bridge between physics and rendering.
 *
 * One loop. No grid. No accumulator. No binning.
 * The GPU's additive blending IS the harmonic sum.
 *
 * License: Public domain / CC0
 */

#ifndef V21_VERTEX_PACK_H
#define V21_VERTEX_PACK_H

#include <stdint.h>
#include <math.h>

/* Must match V20's ParticleVertex (vk_types.h) — 44 bytes */
typedef struct {
    float position[3];
    float pump_scale;
    float pump_residual;
    float temp;              /* topo_dim → blackbody temperature */
    float velocity[3];
    float elongation;
} v21_packed_vertex_t;

/* Topo dim → temperature for blackbody coloring (matches V20 render_fill.cuh) */
static const float V21_DIM_TEMP[] = {1.5f, 3.0f, 5.5f, 8.5f, 15.0f};

static inline int v21_topo_dim(uint8_t state) {
    int d = 0;
    for (int a = 0; a < 4; a++) {
        uint8_t bits = (state >> (a * 2)) & 0x03;
        if (bits == 1 || bits == 2) d++;
    }
    return d;
}

/*
 * Pack N particles from SoA arrays into AoS vertex buffer.
 * Returns number of active particles written.
 *
 * The output buffer must hold at least N v21_packed_vertex_t entries.
 */
static inline int v21_pack_vertices(
    v21_packed_vertex_t* out,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const float* pump_scale, const float* pump_residual,
    const uint8_t* flags, const uint8_t* topo_state,
    int N)
{
    int written = 0;
    for (int i = 0; i < N; i++) {
        if (!(flags[i] & 0x01)) {
            /* Inactive: park far away (GPU culls) */
            out[i].position[0] = 0.0f;
            out[i].position[1] = 1e9f;
            out[i].position[2] = 0.0f;
            out[i].pump_scale = 0.0f;
            out[i].pump_residual = 1.0f;
            out[i].temp = 0.0f;
            out[i].velocity[0] = 0.0f;
            out[i].velocity[1] = 0.0f;
            out[i].velocity[2] = 0.0f;
            out[i].elongation = 0.0f;
            continue;
        }

        out[i].position[0] = pos_x[i];
        out[i].position[1] = pos_y[i];
        out[i].position[2] = pos_z[i];
        out[i].pump_scale = pump_scale[i];
        out[i].pump_residual = pump_residual[i];

        int dim = v21_topo_dim(topo_state[i]);
        out[i].temp = V21_DIM_TEMP[dim];

        out[i].velocity[0] = vel_x[i];
        out[i].velocity[1] = vel_y[i];
        out[i].velocity[2] = vel_z[i];

        float vel_mag = sqrtf(vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i]);
        out[i].elongation = 1.0f + vel_mag * 0.01f;

        written++;
    }
    return written;
}

#endif /* V21_VERTEX_PACK_H */
