// passive_advection.cuh — Passive Keplerian advection kernel
// ============================================================
//
// Part of Tree Architecture Step 2. See docs/active_flag_audit.md
// and /home/zaiken/sanity/math.md Part V for full context.
//
// PURPOSE
// -------
// The `advectPassiveParticles` kernel is a cheap alternative to
// `siphonDiskKernel` for particles that sit near one of the 8 stable
// resonance shells and don't need full siphon pump physics. It
// advances the particle along its current orbital radius at the
// Keplerian rate, applies vertical damping, and advances the Kuramoto
// phase — that's it. No pump state machine, no grid interaction, no
// ejection handling.
//
// STEP 2 BEHAVIOR
// ---------------
// In Step 2 the kernel is wired into the main loop in commit 2c, but
// the `in_active_region[]` buffer is initialized to all-0xFF in commit
// 2d's bootstrap. The kernel's third early-return triggers on every
// particle, so in Step 2 it does ZERO writes. The kernel's existence
// is purely plumbing for Step 3; Step 2 verifies that adding the
// launch doesn't perturb the baseline trajectory.
//
// STEP 2 INVARIANT (for memory race safety)
// -----------------------------------------
// Both this kernel and `siphonDiskKernel` in physics.cu write to
// `disk->pos_x/y/z` and `disk->theta`. In Step 2 they are trivially
// non-racy because this kernel early-returns on every particle. In
// Step 3, their early-return conditions must be mutually exclusive:
// siphon writes iff `in_active_region[i] != 0`, passive writes iff
// `in_active_region[i] == 0`. See R3 in the plan file.
//
// WHAT IS INTENTIONALLY NOT UPDATED
// ---------------------------------
// - `vel_x` / `vel_z` tangential component: the angular phase advance
//   doesn't rewrite the full tangential velocity. The mask kernel
//   (active_region.cuh) snaps vel_x/vel_z to Keplerian on passive→active
//   promotion. Step 6 added radial velocity damping (2% per frame) to
//   bleed off residual oscillation energy and help particles settle.
// - `pump_state`, `pump_work`, `pump_history`, `pump_scale`,
//   `pump_coherent`, `pump_seam`, `jet_phase`, `flags`: owned by
//   `siphonDiskKernel`.

#pragma once

#include "disk.cuh"
#include "forces.cuh"
#include "cuda_lut.cuh"

// The passive kernel's time-constant for pump_residual decay. This is
// a frame-independent exponential time-constant in simulation time
// units — at dt = 1/60 * 2 ≈ 0.033 simulation-time per frame, a tau
// of 5.0 gives ~0.66% decay per frame, matching the "slow memory fade"
// intent from the upstream design discussion.
#ifndef PASSIVE_RESIDUAL_TAU
#define PASSIVE_RESIDUAL_TAU  5.0f
#endif

// Step 5: residual injection rate from shell deviation. Particles
// off-shell accumulate pump_residual proportional to their distance
// from the nearest resonance shell. This breaks the "all passive forever"
// feedback loop and makes CORNER_THRESHOLD meaningful. Step 6 runtime
// candidate — tune empirically via sweep.
#ifndef RESIDUAL_INJECT_RATE
#define RESIDUAL_INJECT_RATE  0.1f
#endif

// Step 6: radial drift rate toward nearest resonance shell. Passive
// particles gently settle onto the nearest shell in d_shell_radii[8].
// At rate 0.005 and dt ≈ 0.033, a particle 1 unit off-shell moves
// ~0.017% per frame → settling timescale ~200 frames. Slow enough
// not to override siphon physics when a particle re-enters active.
#ifndef PASSIVE_DRIFT_RATE
#define PASSIVE_DRIFT_RATE  0.005f
#endif

// Safety clamp — the passive kernel only handles particles inside the
// eigenspectrum cascade range. Outside of this the siphon path owns
// them (including apply_boundary_recycle at r > ION_KICK_OUTER_R).
#ifndef PASSIVE_R_MIN
#define PASSIVE_R_MIN  (ISCO_R * 0.5f)   // 3.0f
#endif
#ifndef PASSIVE_R_MAX
#define PASSIVE_R_MAX  200.0f
#endif

// ============================================================================
// advectPassiveParticles
// ============================================================================
// Advance particles outside any active region along their current
// shell at the Keplerian rate. Early-returns for:
//   - dead particles (PFLAG_ACTIVE == 0)
//   - ejected particles (PFLAG_EJECTED != 0) — owned by siphon Aizawa jet
//   - active-region particles (in_active_region[i] != 0) — owned by siphon
//   - particles outside the shell cascade range — owned by siphon
//     boundary recycle
__global__ void advectPassiveParticles(
    GPUDisk* disk,
    const uint8_t* __restrict__ in_active_region,
    int N,
    float dt,
    float residual_tau)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Early returns: respect ownership invariants with siphonDiskKernel.
    uint8_t flags = disk->flags[i];
    if ((flags & PFLAG_ACTIVE) == 0) return;          // dead
    if ((flags & PFLAG_EJECTED) != 0) return;         // in jet
    if (in_active_region[i] != 0) return;             // siphon owns active regions

    // Load position and velocity.
    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // 3D radius.
    float r3d_sq = px * px + py * py + pz * pz;
    float r3d = sqrtf(r3d_sq);

    // Safety clamp — outside the cascade range, let siphon own the particle.
    if (r3d < PASSIVE_R_MIN || r3d > PASSIVE_R_MAX) return;

    // Compute local orbital frame from angular momentum.
    float Lx = py * vz - pz * vy;
    float Ly = pz * vx - px * vz;
    float Lz = px * vy - py * vx;
    float inv_L = rsqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
    float lx = Lx * inv_L;  // orbital normal
    float ly = Ly * inv_L;
    float lz = Lz * inv_L;

    float inv_r3d = 1.0f / (r3d + 1e-8f);
    float rx = px * inv_r3d;  // radial
    float ry = py * inv_r3d;
    float rz = pz * inv_r3d;

    // Tangential: t = L_hat × r_hat
    float tx = ly * rz - lz * ry;
    float ty = lz * rx - lx * rz;
    float tz = lx * ry - ly * rx;
    float inv_t = rsqrtf(tx*tx + ty*ty + tz*tz + 1e-8f);
    tx *= inv_t; ty *= inv_t; tz *= inv_t;

    // Step 6: radial drift toward hardcoded shells — REMOVED
    // Shells should emerge from the Viviani field harmonics, not be forced.
    // The field topology creates natural density concentrations at resonant
    // radii — those ARE the shells.

    // Keplerian angular velocity at current 3D radius.
    float r3 = r3d * r3d * r3d;
    float omega_kep = sqrtf(BH_MASS / r3);
    float dtheta = omega_kep * dt;

    // Rodrigues rotation: advance position around L_hat by dtheta.
    // r_new = r·cos(dθ) + (L_hat × r)·sin(dθ) + L_hat·(L_hat·r)·(1−cos(dθ))
    // For small dθ (typical ~0.002), cos≈1 and sin≈dθ, but we use exact
    // for correctness at large radii where omega is higher.
    float cos_dt = cuda_lut_cos(dtheta);
    float sin_dt = cuda_lut_sin(dtheta);

    // L_hat × r
    float crx = ly * pz - lz * py;
    float cry = lz * px - lx * pz;
    float crz = lx * py - ly * px;

    // L_hat · r (should be ~0 for orbits in the plane, but nonzero for tilted)
    float ldotr = lx * px + ly * py + lz * pz;

    float px_new = px * cos_dt + crx * sin_dt + lx * ldotr * (1.0f - cos_dt);
    float py_new = py * cos_dt + cry * sin_dt + ly * ldotr * (1.0f - cos_dt);
    float pz_new = pz * cos_dt + crz * sin_dt + lz * ldotr * (1.0f - cos_dt);

    // Orbital-plane damping (damp velocity perpendicular to orbital plane)
    float v_normal = vx * lx + vy * ly + vz * lz;
    float damping = 0.02f;
    vx -= damping * v_normal * lx;
    vy -= damping * v_normal * ly;
    vz -= damping * v_normal * lz;

    // Write Keplerian-correct tangential velocity using local frame.
    float v_kep = omega_kep * r3d;
    disk->vel_x[i] = v_kep * tx;
    disk->vel_y[i] = v_kep * ty;
    disk->vel_z[i] = v_kep * tz;

    // Write back position.
    disk->pos_x[i] = px_new;
    disk->pos_y[i] = py_new;
    disk->pos_z[i] = pz_new;

    // Advance Kuramoto phase with modular wrap.
    float theta = disk->theta[i] + disk->omega_nat[i] * dt;
    // Wrap to [0, 2*PI) without fmodf (cheaper branch on GPUs).
    if (theta >= TWO_PI) theta -= TWO_PI;
    if (theta < 0.0f)    theta += TWO_PI;
    disk->theta[i] = theta;

    // Slow decay of pump residual — the "memory fade" of the branch.
    // Clamped to avoid negative residuals if dt > residual_tau.
    float decay = 1.0f - dt / residual_tau;
    if (decay < 0.0f) decay = 0.0f;
    disk->pump_residual[i] *= decay;
}

// ============================================================================
// Inline unit test (optional, guarded off by default)
// ============================================================================
// Enable with -DTEST_PASSIVE_ADVECTION=1 at build time, or a local
// #define at the top of blackhole_v20.cu before the include. The test
// early-exits main() after printing pass/fail.
#if defined(TEST_PASSIVE_ADVECTION) && TEST_PASSIVE_ADVECTION

#include <cstdio>
#include <cmath>

// A minimal GPUDisk test fixture using only the first 4 particles.
// All other fields are zero-initialized by cudaMemset.
static inline bool run_passive_advection_test() {
    constexpr int NT = 4;
    GPUDisk* d_disk = nullptr;
    uint8_t* d_in_region = nullptr;
    cudaMalloc(&d_disk, sizeof(GPUDisk));
    cudaMalloc(&d_in_region, NT * sizeof(uint8_t));
    cudaMemset(d_disk, 0, sizeof(GPUDisk));
    cudaMemset(d_in_region, 0, NT * sizeof(uint8_t));

    // Initialize 4 particles at r = {10, 25, 50, 100} on the +x axis.
    float h_px[NT] = {10.0f, 25.0f, 50.0f, 100.0f};
    float h_py[NT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float h_pz[NT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float h_vy[NT] = {0.01f, 0.01f, 0.01f, 0.01f};
    float h_theta[NT] = {0.0f, 0.0f, 0.0f, 0.0f};
    float h_omega[NT] = {0.1f, 0.1f, 0.1f, 0.1f};
    float h_res[NT]   = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t h_flags[NT] = {PFLAG_ACTIVE, PFLAG_ACTIVE, PFLAG_ACTIVE, PFLAG_ACTIVE};

    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, pos_x), h_px,  NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, pos_y), h_py,  NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, pos_z), h_pz,  NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, vel_y), h_vy,  NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, theta), h_theta, NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, omega_nat), h_omega, NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, pump_residual), h_res, NT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, flags), h_flags, NT*sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Advance for 100 steps at dt matching main loop (1/60 * 2).
    const float dt = (1.0f / 60.0f) * 2.0f;
    const int STEPS = 100;
    for (int s = 0; s < STEPS; s++) {
        advectPassiveParticles<<<1, NT>>>(d_disk, d_in_region, NT, dt, PASSIVE_RESIDUAL_TAU);
    }
    cudaDeviceSynchronize();

    // Read results back.
    float out_px[NT], out_py[NT], out_pz[NT], out_theta[NT], out_res[NT];
    cudaMemcpy(out_px, (char*)d_disk + offsetof(GPUDisk, pos_x), NT*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_py, (char*)d_disk + offsetof(GPUDisk, pos_y), NT*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_pz, (char*)d_disk + offsetof(GPUDisk, pos_z), NT*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_theta, (char*)d_disk + offsetof(GPUDisk, theta), NT*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_res, (char*)d_disk + offsetof(GPUDisk, pump_residual), NT*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < NT; i++) {
        float r_out = sqrtf(out_px[i]*out_px[i] + out_pz[i]*out_pz[i]);
        float r_in = h_px[i];
        float dr = fabsf(r_out - r_in);
        if (dr > 0.01f) {
            fprintf(stderr, "[passive-test] FAIL: particle %d radial drift %.6f (initial r=%.2f, final r=%.6f)\n",
                    i, dr, r_in, r_out);
            ok = false;
        }
        float expected_theta_lower = 0.1f * STEPS * dt * 0.5f;
        // Wrap-aware: theta lives in [0, 2*PI).
        if (out_theta[i] < expected_theta_lower && out_theta[i] < TWO_PI - expected_theta_lower) {
            // Accept both pre-wrap and post-wrap cases.
            // For omega_nat=0.1, 100 steps of dt=0.033 gives ~0.333 total, well under 2*PI.
            fprintf(stderr, "[passive-test] FAIL: particle %d theta=%.6f, expected >= %.6f\n",
                    i, out_theta[i], expected_theta_lower);
            ok = false;
        }
        float expected_res_upper = expf(-STEPS * dt / PASSIVE_RESIDUAL_TAU) * 1.1f;
        if (out_res[i] > expected_res_upper) {
            fprintf(stderr, "[passive-test] FAIL: particle %d residual=%.6f, expected <= %.6f\n",
                    i, out_res[i], expected_res_upper);
            ok = false;
        }
        if (fabsf(out_py[i]) > 0.5f) {
            fprintf(stderr, "[passive-test] FAIL: particle %d py=%.6f (vertical damping amplified)\n",
                    i, out_py[i]);
            ok = false;
        }
    }

    cudaFree(d_disk);
    cudaFree(d_in_region);

    if (ok) {
        fprintf(stderr, "[passive-test] PASS: all 4 test particles\n");
    }
    return ok;
}

#endif  // TEST_PASSIVE_ADVECTION
