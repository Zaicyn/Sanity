/*
 * V21 ACCUMULATOR — Collapse particles into coherent Gaussian wave packets
 * ========================================================================
 *
 * Linear pass over particles, bins into spatial cells, computes:
 *   - Mean position (density-weighted centroid)
 *   - Density ρ (particle count)
 *   - Phase coherence R = |⟨e^{iθ}⟩|
 *   - Mean topo_dim (eigenstate classification)
 *   - Mean pump_scale (activity level)
 *   - Mean frequency ω (orbital velocity / radius)
 *
 * Output: sparse list of Splat structs for the renderer.
 *
 * License: Public domain / CC0
 */

#ifndef V21_ACCUMULATE_H
#define V21_ACCUMULATE_H

#include "v21_types.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * SPLAT — one Gaussian wave packet for the renderer
 * ======================================================================== */

typedef struct {
    float x, y, z;          /* Position (centroid) */
    float density;           /* ρ: particle count in cell */
    float coherence;         /* R: Kuramoto order parameter [0,1] */
    float amplitude;         /* R × √ρ × pump_scale */
    float sigma;             /* Gaussian width (set by renderer based on distance) */
    float frequency;         /* Mean ω (orbital velocity / radius) */
    float phase;             /* Mean θ (circular mean) */
    float pump_scale;        /* Mean pump activity */
    int   topo_dim;          /* Dominant eigenstate dimension */
    int   particle_count;    /* Raw count */
} v21_splat_t;

/* ========================================================================
 * CELL ACCUMULATOR — intermediate per-cell data
 * ======================================================================== */

typedef struct {
    float sum_x, sum_y, sum_z;    /* Position sums */
    float sum_vx, sum_vy, sum_vz; /* Velocity sums */
    float phase_sin, phase_cos;   /* Phase accumulation */
    float sum_pump_scale;
    int   sum_topo_dim;
    int   count;
} v21_cell_acc_t;

/* ========================================================================
 * ACCUMULATE — collapse particles into splats
 *
 * grid_dim: spatial grid resolution (e.g., 32 for 32³ cells)
 * grid_extent: half-size of simulation domain (e.g., 250)
 * min_density: minimum particles per cell to emit a splat (e.g., 2)
 * min_coherence: minimum R to emit (e.g., 0.1)
 * ======================================================================== */

static inline int v21_accumulate(
    /* Particle SoA arrays */
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const float* theta, const float* pump_scale,
    const uint8_t* flags, const uint8_t* topo_state,
    int N,
    /* Grid parameters */
    int grid_dim, float grid_extent,
    /* Thresholds */
    float min_density, float min_coherence,
    /* Output */
    v21_splat_t* out_splats, int max_splats)
{
    int total_cells = grid_dim * grid_dim * grid_dim;
    float cell_size = (grid_extent * 2.0f) / (float)grid_dim;
    float inv_cell = 1.0f / cell_size;

    /* Allocate cell accumulators */
    v21_cell_acc_t* cells = (v21_cell_acc_t*)calloc(total_cells, sizeof(v21_cell_acc_t));
    if (!cells) return 0;

    /* Pass 1: Scatter particles into cells */
    for (int i = 0; i < N; i++) {
        if (!(flags[i] & 0x01)) continue;  /* Skip inactive */

        float px = pos_x[i], py = pos_y[i], pz = pos_z[i];

        int cx = (int)((px + grid_extent) * inv_cell);
        int cy = (int)((py + grid_extent) * inv_cell);
        int cz = (int)((pz + grid_extent) * inv_cell);

        if (cx < 0) cx = 0; if (cx >= grid_dim) cx = grid_dim - 1;
        if (cy < 0) cy = 0; if (cy >= grid_dim) cy = grid_dim - 1;
        if (cz < 0) cz = 0; if (cz >= grid_dim) cz = grid_dim - 1;

        int cell = cx + cy * grid_dim + cz * grid_dim * grid_dim;

        cells[cell].sum_x += px;
        cells[cell].sum_y += py;
        cells[cell].sum_z += pz;
        cells[cell].sum_vx += vel_x[i];
        cells[cell].sum_vy += vel_y[i];
        cells[cell].sum_vz += vel_z[i];
        cells[cell].phase_sin += sinf(theta[i]);
        cells[cell].phase_cos += cosf(theta[i]);
        cells[cell].sum_pump_scale += pump_scale[i];
        cells[cell].count++;

        /* Topo dim */
        uint8_t ts = topo_state[i];
        int dim = 0;
        for (int a = 0; a < 4; a++) {
            uint8_t bits = (ts >> (a * 2)) & 0x03;
            if (bits == 1 || bits == 2) dim++;
        }
        cells[cell].sum_topo_dim += dim;
    }

    /* Pass 2: Emit splats from qualifying cells */
    int splat_count = 0;

    for (int c = 0; c < total_cells && splat_count < max_splats; c++) {
        int count = cells[c].count;
        if (count < (int)min_density) continue;

        float rho = (float)count;
        float inv_rho = 1.0f / rho;

        /* Phase coherence R = |⟨e^{iθ}⟩| / ρ */
        float ps = cells[c].phase_sin;
        float pc = cells[c].phase_cos;
        float R = sqrtf(ps * ps + pc * pc) * inv_rho;

        if (R < min_coherence) continue;

        v21_splat_t* s = &out_splats[splat_count++];

        /* Centroid */
        s->x = cells[c].sum_x * inv_rho;
        s->y = cells[c].sum_y * inv_rho;
        s->z = cells[c].sum_z * inv_rho;

        s->density = rho;
        s->coherence = R;
        s->particle_count = count;

        /* Mean pump scale */
        s->pump_scale = cells[c].sum_pump_scale * inv_rho;

        /* Amplitude = R × √ρ × pump_scale */
        s->amplitude = R * sqrtf(rho) * s->pump_scale;

        /* Phase (circular mean) */
        s->phase = atan2f(ps * inv_rho, pc * inv_rho);

        /* Frequency from mean velocity / radius */
        float mvx = cells[c].sum_vx * inv_rho;
        float mvy = cells[c].sum_vy * inv_rho;
        float mvz = cells[c].sum_vz * inv_rho;
        float v_mean = sqrtf(mvx*mvx + mvy*mvy + mvz*mvz);
        float r_pos = sqrtf(s->x*s->x + s->y*s->y + s->z*s->z);
        s->frequency = (r_pos > 1.0f) ? v_mean / r_pos : v_mean;

        /* Dominant topo_dim (mean, rounded) */
        s->topo_dim = (cells[c].sum_topo_dim + count/2) / count;

        /* Sigma set later by renderer (depends on distance to camera) */
        s->sigma = 0.0f;
    }

    free(cells);
    return splat_count;
}

#ifdef __cplusplus
}
#endif

#endif /* V21_ACCUMULATE_H */
