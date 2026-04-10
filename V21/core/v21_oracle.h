/*
 * V21 ORACLE — CPU Validation Observer
 * =====================================
 *
 * Watches the particle state after each physics_step() and checks that
 * 5 aggregate metrics evolve within physical bounds. Dense validation at
 * the core (every frame), sparse sampling at the shells (every 100 frames).
 *
 * No shadow simulation. No heap allocation. One loop computes all metrics.
 * The oracle is an observer, not a parallel engine.
 *
 * License: Public domain / CC0
 */

#ifndef V21_ORACLE_H
#define V21_ORACLE_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * CONFIGURATION
 * ======================================================================== */

#define ORACLE_CORE_RADIUS      30.0f
#define ORACLE_NUM_SHELLS       8
#define ORACLE_SHELL_MAX_R      200.0f
#define ORACLE_DENSE_INTERVAL   1       /* Every frame for core */
#define ORACLE_SPARSE_INTERVAL  100     /* Every 100 frames for shells */

/* Base tolerances (scaled by √frame at runtime) */
#define ORACLE_TOL_Q_RATE       2.0f    /* Max Q change per check */
#define ORACLE_TOL_R            1e-3f
#define ORACLE_TOL_VEL          1e-2f
#define ORACLE_TOL_DENSITY      5e-3f   /* L2 norm of histogram shift */
#define ORACLE_TOL_TOPO         1e-3f   /* L2 norm of distribution shift */

/* ========================================================================
 * TOPO HELPERS (pure C, no CUDA dependency)
 * ======================================================================== */

static inline int oracle_topo_get_axis(uint8_t state, int axis) {
    int bits = (state >> (axis * 2)) & 0x03;
    return (bits == 1) ? 1 : (bits == 2) ? -1 : 0;
}

static inline int oracle_topo_compute_Q(uint8_t state) {
    int s[4];
    for (int a = 0; a < 4; a++) s[a] = oracle_topo_get_axis(state, a);
    return s[0]*s[1]*s[2] - s[0]*s[1]*s[3] + s[0]*s[2]*s[3] - s[1]*s[2]*s[3];
}

static inline int oracle_topo_dim(uint8_t state) {
    int d = 0;
    for (int a = 0; a < 4; a++)
        if (oracle_topo_get_axis(state, a) != 0) d++;
    return d;
}

/* ========================================================================
 * ORACLE STATE
 * ======================================================================== */

typedef struct {
    int   initialized;
    int   total_passes;
    int   total_fails;

    /* Previous-frame metrics */
    float prev_Q;
    float prev_R;
    float prev_avg_vel;
    float prev_density[ORACLE_NUM_SHELLS];
    float prev_topo[5];  /* dim 0-4 */

    /* Cumulative stats */
    int   frames_checked;
    int   dense_checks;
    int   sparse_checks;
} v21_oracle_t;

static inline void v21_oracle_init(v21_oracle_t* orc) {
    orc->initialized = 0;
    orc->total_passes = 0;
    orc->total_fails = 0;
    orc->prev_Q = 0;
    orc->prev_R = 0;
    orc->prev_avg_vel = 0;
    orc->frames_checked = 0;
    orc->dense_checks = 0;
    orc->sparse_checks = 0;
    for (int i = 0; i < ORACLE_NUM_SHELLS; i++) orc->prev_density[i] = 0;
    for (int i = 0; i < 5; i++) orc->prev_topo[i] = 0;
}

/* ========================================================================
 * MAIN CHECK — called after physics_step() each frame
 *
 * One loop computes all 5 metrics simultaneously over the sampled particles.
 * Dense mode: every frame, particles with r_cyl < CORE_RADIUS
 * Sparse mode: every SPARSE_INTERVAL frames, stratified across all shells
 * ======================================================================== */

static inline void v21_oracle_check(
    v21_oracle_t*   orc,
    const float*    pos_x, const float* pos_y, const float* pos_z,
    const float*    vel_x, const float* vel_y, const float* vel_z,
    const float*    theta,
    const float*    pump_scale,
    const uint8_t*  flags,
    const uint8_t*  topo_state,
    int N, int frame)
{
    /* Decide mode */
    int do_dense  = (frame % ORACLE_DENSE_INTERVAL == 0);
    int do_sparse = (frame % ORACLE_SPARSE_INTERVAL == 0);
    if (!do_dense && !do_sparse) return;

    const char* mode = do_sparse ? "SPARSE" : "DENSE";
    float r_max = do_sparse ? ORACLE_SHELL_MAX_R : ORACLE_CORE_RADIUS;
    int stride = do_sparse ? 10 : 1;  /* Sample every 10th particle in sparse mode */

    /* Accumulators */
    float sum_Q = 0;
    double sum_sin = 0, sum_cos = 0;
    double sum_vel = 0;
    float density_hist[ORACLE_NUM_SHELLS];
    float topo_dist[5];
    for (int i = 0; i < ORACLE_NUM_SHELLS; i++) density_hist[i] = 0;
    for (int i = 0; i < 5; i++) topo_dist[i] = 0;
    int n_sampled = 0;

    /* Shell boundaries (quadratic spacing — finer near core) */
    float shell_bounds[ORACLE_NUM_SHELLS + 1];
    for (int s = 0; s <= ORACLE_NUM_SHELLS; s++) {
        float t = (float)s / (float)ORACLE_NUM_SHELLS;
        shell_bounds[s] = ORACLE_SHELL_MAX_R * t * t;
    }

    /* ONE LOOP — all 5 metrics simultaneously */
    for (int i = 0; i < N; i += stride) {
        if (!(flags[i] & 0x01)) continue;  /* Skip inactive */

        float px = pos_x[i], pz = pos_z[i];
        float r_cyl = sqrtf(px * px + pz * pz);

        /* In dense mode, only sample core particles */
        if (!do_sparse && r_cyl > ORACLE_CORE_RADIUS) continue;
        if (r_cyl > r_max) continue;

        n_sampled++;

        /* Metric 1: Q */
        sum_Q += oracle_topo_compute_Q(topo_state[i]);

        /* Metric 2: Kuramoto R */
        sum_sin += sinf(theta[i]);
        sum_cos += cosf(theta[i]);

        /* Metric 3: Avg velocity */
        float vx = vel_x[i], vy = vel_y[i], vz = vel_z[i];
        sum_vel += sqrtf(vx*vx + vy*vy + vz*vz);

        /* Metric 4: Radial density histogram */
        for (int s = 0; s < ORACLE_NUM_SHELLS; s++) {
            if (r_cyl >= shell_bounds[s] && r_cyl < shell_bounds[s+1]) {
                density_hist[s] += 1.0f;
                break;
            }
        }

        /* Metric 5: Topo_dim distribution */
        int dim = oracle_topo_dim(topo_state[i]);
        if (dim >= 0 && dim < 5) topo_dist[dim] += 1.0f;
    }

    if (n_sampled < 2) return;  /* Not enough particles to validate */

    /* Finalize metrics */
    float inv_n = 1.0f / (float)n_sampled;
    float cur_Q = sum_Q;
    float cur_R = (float)(sqrtf(sum_sin*sum_sin + sum_cos*sum_cos) * inv_n);
    float cur_vel = (float)(sum_vel * inv_n);

    /* Normalize histograms */
    for (int i = 0; i < ORACLE_NUM_SHELLS; i++) density_hist[i] *= inv_n;
    for (int i = 0; i < 5; i++) topo_dist[i] *= inv_n;

    /* Tolerance scaling: base × √(frame + 1) */
    float scale = sqrtf((float)(frame + 1));

    /* Compare against previous (skip first frame) */
    int pass_Q = 1, pass_R = 1, pass_vel = 1, pass_dens = 1, pass_topo = 1;

    if (orc->initialized) {
        float dQ = fabsf(cur_Q - orc->prev_Q);
        float dR = fabsf(cur_R - orc->prev_R);
        float dV = fabsf(cur_vel - orc->prev_avg_vel);

        if (dQ > ORACLE_TOL_Q_RATE * scale) pass_Q = 0;
        if (dR > ORACLE_TOL_R * scale) pass_R = 0;
        if (dV > ORACLE_TOL_VEL * scale) pass_vel = 0;

        /* L2 norm of histogram shifts */
        float dens_l2 = 0, topo_l2 = 0;
        for (int i = 0; i < ORACLE_NUM_SHELLS; i++) {
            float d = density_hist[i] - orc->prev_density[i];
            dens_l2 += d * d;
        }
        for (int i = 0; i < 5; i++) {
            float d = topo_dist[i] - orc->prev_topo[i];
            topo_l2 += d * d;
        }
        dens_l2 = sqrtf(dens_l2);
        topo_l2 = sqrtf(topo_l2);

        if (dens_l2 > ORACLE_TOL_DENSITY * scale) pass_dens = 0;
        if (topo_l2 > ORACLE_TOL_TOPO * scale) pass_topo = 0;
    }

    int all_pass = pass_Q && pass_R && pass_vel && pass_dens && pass_topo;

    /* Log — only on failure (zero print overhead in normal operation) */
    if (!all_pass) {
        printf("[ORACLE] FAIL frame=%d mode=%-6s Q=%.0f R=%.4f vel=%.4f | %s %s %s %s %s\n",
               frame, mode, cur_Q, cur_R, cur_vel,
               pass_Q ? "ok" : "FAIL(Q)",
               pass_R ? "ok" : "FAIL(R)",
               pass_vel ? "ok" : "FAIL(vel)",
               pass_dens ? "ok" : "FAIL(dens)",
               pass_topo ? "PASS" : "FAIL(topo)");
    }

    /* Update state */
    orc->prev_Q = cur_Q;
    orc->prev_R = cur_R;
    orc->prev_avg_vel = cur_vel;
    for (int i = 0; i < ORACLE_NUM_SHELLS; i++) orc->prev_density[i] = density_hist[i];
    for (int i = 0; i < 5; i++) orc->prev_topo[i] = topo_dist[i];

    if (all_pass) orc->total_passes++; else orc->total_fails++;
    orc->initialized = 1;
    orc->frames_checked++;
    if (do_sparse) orc->sparse_checks++;
    else orc->dense_checks++;
}

/* ========================================================================
 * SUMMARY — call at shutdown
 * ======================================================================== */

static inline void v21_oracle_summary(const v21_oracle_t* orc) {
    printf("[ORACLE] Summary: %d checks (%d dense, %d sparse), %d pass, %d fail (%.1f%% pass rate)\n",
           orc->frames_checked, orc->dense_checks, orc->sparse_checks,
           orc->total_passes, orc->total_fails,
           orc->frames_checked > 0 ?
               100.0f * orc->total_passes / orc->frames_checked : 0.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* V21_ORACLE_H */
