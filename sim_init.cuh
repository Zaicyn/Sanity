// sim_init.cuh — Simulation Initialization Functions
// ===================================================
// Extracts initialization code from blackhole_v20.cu into reusable functions.
// Each function populates a subsystem of the SimulationContext.
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstdio>
#include "sim_context.h"
#include "disk.cuh"

// ============================================================================
// initParticles — Host-side particle setup + GPU allocation + upload
// ============================================================================
// Creates the GPUDisk struct on device, initializes N particles with
// position, velocity, pump state, Kuramoto phase, and hopfion topo_state.
// Stores the device pointer in ctx.particles.buf_disk.
//
// Returns the GPUDisk* for backward compatibility with existing code.

inline GPUDisk* initParticles(SimulationContext& ctx, int N, unsigned int seed) {
    std::vector<float> h_px(N), h_py(N), h_pz(N);
    std::vector<float> h_vx(N), h_vy(N), h_vz(N);
    std::vector<int> h_state(N);
    std::vector<float> h_scale(N);
    std::vector<int> h_coherent(N);
    std::vector<uint8_t> h_seam(N);
    std::vector<float> h_residual(N), h_work(N);
    std::vector<float> h_history(N);
    std::vector<float> h_theta(N);
    std::vector<float> h_omega_nat(N);
    std::vector<uint8_t> h_flags(N);
    std::vector<uint8_t> h_topo(N);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> runif(0.0f, 1.0f);
    std::uniform_real_distribution<float> rphase(0, TWO_PI);
    std::normal_distribution<float> rnorm(0.0f, 1.0f);

    // Host-side copy of shell radii for init (device __constant__ not readable on host)
    static const float h_shell_radii[8] = {
        6.0f, 9.7f, 15.7f, 25.4f, 41.1f, 66.5f, 107.5f, 174.0f
    };

    float box_half = DISK_OUTER_R;
    float box_height = box_half * 0.3f;

    for (int i = 0; i < N; i++) {
        float x, y, z;

        if (g_shell_init) {
            // Shell-aware initialization: distribute across 8 shells weighted by 1/r
            float weight_sum = 0.0f;
            for (int s = 0; s < 8; s++) weight_sum += 1.0f / h_shell_radii[s];
            float pick = runif(rng) * weight_sum;
            float accum = 0.0f;
            int shell = 7;
            for (int s = 0; s < 8; s++) {
                accum += 1.0f / h_shell_radii[s];
                if (pick <= accum) { shell = s; break; }
            }
            float r_shell = h_shell_radii[shell];
            float r_jitter = r_shell + rnorm(rng) * 2.0f;
            if (r_jitter < ISCO_R * 0.8f) r_jitter = ISCO_R * 0.8f;

            float phi = rphase(rng);
            x = r_jitter * cosf(phi);
            z = r_jitter * sinf(phi);
            y = rnorm(rng) * r_shell * 0.2f;
        } else {
            // Uniform box initialization
            x = (runif(rng) * 2.0f - 1.0f) * box_half;
            y = (runif(rng) * 2.0f - 1.0f) * box_height;
            z = (runif(rng) * 2.0f - 1.0f) * box_half;

            float r = sqrtf(x*x + y*y + z*z);
            if (r < SCHW_R * 3.0f) {
                float scale = (SCHW_R * 3.0f) / r;
                x *= scale; y *= scale; z *= scale;
            }
        }

        h_px[i] = x;
        h_py[i] = y;
        h_pz[i] = z;

        // Full Keplerian tangential velocity
        float r_xz = sqrtf(x*x + z*z);
        if (r_xz > 0.1f) {
            float rot_sign = g_retrograde_init ? -1.0f : 1.0f;
            float v_rot = rot_sign * sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            float thermal = 0.02f;
            h_vx[i] = -v_rot * (z / r_xz) + rnorm(rng) * thermal;
            h_vy[i] = rnorm(rng) * 0.005f;
            h_vz[i] = v_rot * (x / r_xz) + rnorm(rng) * thermal;
        } else {
            h_vx[i] = rnorm(rng) * 0.05f;
            h_vy[i] = rnorm(rng) * 0.05f;
            h_vz[i] = rnorm(rng) * 0.05f;
        }

        // Pump state: IDLE
        h_state[i] = 0;
        h_scale[i] = 1.0f;
        h_coherent[i] = 0;
        h_seam[i] = 0x00;
        h_residual[i] = 0.0f;
        h_work[i] = 0.0f;
        h_history[i] = 1.0f;

        // Kuramoto: uniform random phase, Gaussian natural frequency
        h_theta[i] = rphase(rng);
        h_omega_nat[i] = g_omega_base + g_omega_spread * rnorm(rng);

        h_flags[i] = PFLAG_ACTIVE;

        // Hopfion: random single-axis-active state (Q=0)
        int topo_axis = (int)(runif(rng) * 4.0f) % 4;
        int topo_sign = (runif(rng) < 0.5f) ? 1 : -1;
        h_topo[i] = topo_set_axis(0, topo_axis, topo_sign);
    }

    // Verification
    int active_count = 0;
    for (int i = 0; i < N; i++) if (h_flags[i] & PFLAG_ACTIVE) active_count++;
    printf("[lattice] %d particles initialized in %.0f×%.0f×%.0f box (active=%d)\n",
           N, box_half*2, box_height*2, box_half*2, active_count);
    printf("[kuramoto-init] theta uniform[0,2π), omega_nat ~ N(%.2f, %.2f)\n",
           g_omega_base, g_omega_spread);

    int topo_dist[8] = {};
    int topo_Q_sum = 0;
    for (int i = 0; i < N; i++) {
        for (int a = 0; a < 4; a++) {
            int s = topo_get_axis(h_topo[i], a);
            if (s == 1) topo_dist[a * 2]++;
            else if (s == -1) topo_dist[a * 2 + 1]++;
        }
        topo_Q_sum += topo_compute_Q(h_topo[i]);
    }
    printf("[hopfion-init] topo_state distribution (axis+/axis-): "
           "e1(%d/%d) e2(%d/%d) e3(%d/%d) e4(%d/%d) Q_sum=%d\n",
           topo_dist[0], topo_dist[1], topo_dist[2], topo_dist[3],
           topo_dist[4], topo_dist[5], topo_dist[6], topo_dist[7], topo_Q_sum);

    // GPU allocation and upload
    GPUDisk* d_disk;
    cudaMalloc(&d_disk, sizeof(GPUDisk));
    cudaMemset(d_disk, 0, sizeof(GPUDisk));

    #define UPLOAD(field, hvec) \
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, field), hvec.data(), N*sizeof(hvec[0]), cudaMemcpyHostToDevice)

    UPLOAD(pos_x, h_px);    UPLOAD(pos_y, h_py);    UPLOAD(pos_z, h_pz);
    UPLOAD(vel_x, h_vx);    UPLOAD(vel_y, h_vy);    UPLOAD(vel_z, h_vz);
    UPLOAD(pump_state, h_state);
    UPLOAD(pump_scale, h_scale);
    UPLOAD(pump_coherent, h_coherent);
    UPLOAD(pump_seam, h_seam);
    UPLOAD(pump_residual, h_residual);
    UPLOAD(pump_work, h_work);
    UPLOAD(pump_history, h_history);
    UPLOAD(theta, h_theta);
    UPLOAD(omega_nat, h_omega_nat);
    UPLOAD(flags, h_flags);
    UPLOAD(topo_state, h_topo);
    #undef UPLOAD

    // Wire into context
    ctx.particles.buf_disk = d_disk;
    ctx.particles.N_current = N;

    return d_disk;
}
