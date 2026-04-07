// diagnostics.cuh — Stress Counters, Pump Metrics, Sampling Kernels
// =================================================================
// Contains:
//   - StressCounters struct (physics diagnostics)
//   - PumpMetrics struct (smoothed pump bridge state)
//   - SampleMetrics struct (output of stratified sampling)
//   - sampleReductionKernel: 128-sample O(1) bridge metrics
//   - generateStratifiedSamples: host function for radial+angular binning
//
// Dependencies: disk.cuh, sun_trace.cuh (d_shell_radii), cuda_lut.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ============================================================================
// Stress Counters
// ============================================================================

struct StressCounters {
    unsigned int ejected_count;
    unsigned int active_count;
    float avg_temp;
    float avg_residual;
    float avg_scale;
    float total_work;
    unsigned int high_stress_count;  // Particles above 0.95 stress (dissolution zone)

    // === ENERGY CONSERVATION DIAGNOSTIC ===
    // Track total system energy to verify spawning doesn't create free energy
    // E_total = E_kinetic + E_pump where:
    //   E_kinetic = Σ 0.5 * v²
    //   E_pump = Σ (0.1 * pump_scale + 0.05 * pump_history)
    float total_kinetic_energy;      // Σ 0.5 * v²
    float total_pump_energy;         // Σ (α * scale + β * history)
    float energy_per_particle;       // E_total / N (should stay ~constant)

    // === AZIMUTHAL EJECTION DIAGNOSTIC ===
    // Track which azimuthal sectors particles are ejected from
    // For m=3 arms: bin 0 = arm 1, bin 1 = arm 2, bin 2 = arm 3
    // For inter-arm escape: bins should be roughly equal
    // For arm boundary escape: bins should peak at sector boundaries
    unsigned int ejection_bins[16];  // 16 angular bins (22.5° each)

    // === RESIDENCE TIME DIAGNOSTIC (Test A) ===
    // Track accumulated time particles spend in arms vs gaps
    // If discrete topology works: arm_time >> gap_time
    float arm_residence_time;   // Total frames spent in arm regions
    float gap_residence_time;   // Total frames spent in gap regions
    unsigned int arm_particle_count;  // Active particles currently in arms
    unsigned int gap_particle_count;  // Active particles currently in gaps

    // === BEAT FREQUENCY CROSSOVER TEST ===
    // Test: ω_orb(r) vs ω_pump crossover at r≈185
    // If beat frequency model is correct:
    //   - Inner zone (r < 185): ω_orb > ω_pump → clustering at ~120° (m=3)
    //   - Outer zone (r > 185): ω_orb < ω_pump → different or no clustering
    // Critical radius: r³ = BH_MASS / ω_pump² = 100 / 0.125² ≈ 6400 → r ≈ 18.6 × ISCO
    #define BEAT_CROSSOVER_RADIUS 185.0f  // Where ω_orb = ω_pump (at ISCO=6, this is ~31×ISCO)
    unsigned int inner_ejection_bins[16];  // r < BEAT_CROSSOVER_RADIUS
    unsigned int outer_ejection_bins[16];  // r >= BEAT_CROSSOVER_RADIUS
    unsigned int inner_ejection_total;
    unsigned int outer_ejection_total;

    // === HIGH-STRESS FIELD SPATIAL DISTRIBUTION ===
    // Track pump_residual > 0.7 across the full disk (not just ejection zone)
    // This reveals the spatial structure of the pump instability field
    // 4 radial bins × 16 angular bins = 64 cells
    // Radial bins: [ISCO, 2×ISCO), [2×ISCO, 4×ISCO), [4×ISCO, 8×ISCO), [8×ISCO, 16×ISCO)
    // If beat frequency model is correct: m-mode should grade outward as ω_orb drops
    #define STRESS_RADIAL_BINS 4
    #define STRESS_ANGULAR_BINS 16
    #define STRESS_THRESHOLD 0.7f
    unsigned int stress_field[STRESS_RADIAL_BINS][STRESS_ANGULAR_BINS];
    unsigned int stress_radial_totals[STRESS_RADIAL_BINS];  // Total high-stress per radial bin

    // === TIME-RESOLVED SEAM DRIFT TRACKING ===
    // Track circular mean of high-stress particle angles (innermost bin only)
    // sin_sum and cos_sum allow computing weighted circular mean: atan2(sin_sum, cos_sum)
    // This reveals whether the m=3 asymmetry (seam gap) is stationary or precessing
    float stress_sin_sum;   // Σ sin(3θ) for m=3 phase tracking
    float stress_cos_sum;   // Σ cos(3θ) for m=3 phase tracking
    unsigned int stress_sample_count;  // Number of samples in this window
};

// Bridge uniforms for raymarcher - smoothed for visual stability
struct PumpMetrics {
    float avg_scale;      // Smoothed average pump scale
    float avg_residual;   // Smoothed average residual (entropy leak)
    float total_work;     // Smoothed total work (luminance driver)
    float heartbeat;      // Oscillating component for pulsing effects
};

// ============================================================================
// Stratified Sampling for O(1) Bridge Metrics
// ============================================================================
// Instead of reducing all 3.5M particles (O(N)), we sample 128 particles
// distributed evenly across radial and angular bins. This gives us
// statistically accurate global averages at O(1) cost.

#define SAMPLE_COUNT 128
#define RADIAL_BINS  8
#define ANGULAR_BINS 16  // 8 * 16 = 128 samples

// Lightweight output for sampled reduction (just what raymarcher needs)
struct SampleMetrics {
    // Global averages (keep for heartbeat/photon ring)
    float avg_scale;
    float avg_residual;
    float total_work;

    // Hopfion shell structure (EM confinement layers)
    float shell_radii[8];   // Radial boundaries (from 8 radial bins)
    float shell_n[8];       // Refractive index per shell (n = 1 + k*pump_scale)
    int num_shells;         // Number of active shells

    // Photon ring observable (EHT comparison)
    float photon_ring_radius;  // Radius of peak lensing (Einstein ring)
};

// Fast reduction kernel that only processes sampled particles
__global__ void sampleReductionKernel(
    const GPUDisk* disk,
    const int* sample_indices,  // Pre-computed indices (128 = 8 radial × 16 angular)
int N_samples,
SampleMetrics* out,
bool use_hopfion_topology)  // NEW: control experiment parameter
{
    __shared__ float s_scale;
    __shared__ float s_residual;
    __shared__ float s_work;
    __shared__ int s_count;

    // Shared memory for shell binning (8 radial bins)
    __shared__ float shell_scale_sum[8];
    __shared__ int shell_count[8];
    __shared__ float shell_r_sum[8];

    if (threadIdx.x == 0) {
        s_scale = 0.0f;
        s_residual = 0.0f;
        s_work = 0.0f;
        s_count = 0;

        for (int i = 0; i < 8; i++) {
            shell_scale_sum[i] = 0.0f;
            shell_count[i] = 0;
            shell_r_sum[i] = 0.0f;
        }
    }
    __syncthreads();

    // Samples are stratified: 8 radial bins × 16 angular bins = 128 samples
    // Process samples and bin by radius
    for (int i = threadIdx.x; i < N_samples; i += blockDim.x) {
        int idx = sample_indices[i];
        if (particle_active(disk, idx)) {
            // Global averages
            atomicAdd(&s_scale, disk->pump_scale[idx]);
            atomicAdd(&s_residual, fabsf(disk->pump_residual[idx]));
            atomicAdd(&s_work, disk->pump_work[idx]);
            atomicAdd(&s_count, 1);

            // Get particle position
            float px = disk->pos_x[idx];
            float pz = disk->pos_z[idx];
            float r_cyl = sqrtf(px*px + pz*pz);
            float pump_scale = disk->pump_scale[idx];

            // Determine which radial bin this sample belongs to
            // Samples are ordered: first 16 are innermost radial bin, etc.
            int radial_bin = i / 16;  // Integer division: 0-15→0, 16-31→1, etc.
            if (radial_bin < 8) {
                atomicAdd(&shell_scale_sum[radial_bin], pump_scale);
                atomicAdd(&shell_r_sum[radial_bin], r_cyl);
                atomicAdd(&shell_count[radial_bin], 1);
            }
        }
    }
    __syncthreads();

    // Thread 0 computes final shell profile
    if (threadIdx.x == 0 && s_count > 0) {
        // Global averages
        out->avg_scale = s_scale / (float)s_count;
        out->avg_residual = s_residual / (float)s_count;
        out->total_work = s_work;

        // Compute shell profile with BASELINE TENSION
        // Physics: Skyrmion threads have resting tension + pump modulation
        // This creates stable ring (DC component) + subtle breathing (AC component)
        // Target: ~10% instability ratio to match EHT observations
        const float baseline_stress = 0.09f;     // Static thread tension (DC)
        const float k_refraction = 0.001f;       // Pump modulation strength (AC) - weak coupling
        int num_shells = 0;

        // === CONTROL EXPERIMENT: Hopfion Topology vs Smooth Gradient ===
        if (use_hopfion_topology) {
            // MODE 1: DISCRETE HOPFION SHELLS (original)
            // Discrete shell boundaries with topological twist
            for (int i = 0; i < 8; i++) {
                if (shell_count[i] > 0) {
                    // Average radius for this shell
                    float avg_r = shell_r_sum[i] / (float)shell_count[i];

                    // Average pump_scale for this shell
                    float avg_pump_scale = shell_scale_sum[i] / (float)shell_count[i];

                    // Refractive index with baseline + saturating modulation
                    float saturation_scale = baseline_stress / k_refraction;
                    float saturated_pump = saturation_scale * tanhf(avg_pump_scale / saturation_scale);
                    float n = 1.0f + baseline_stress + k_refraction * saturated_pump;

                    out->shell_radii[num_shells] = avg_r;
                    out->shell_n[num_shells] = n;
                    num_shells++;
                }
            }
        } else {
            // MODE 2: SMOOTH EXPONENTIAL GRADIENT (no topology)
            // Continuous n(r) = 1 + A * exp(-r/L) profile
            // Same baseline stress and coupling, but NO discrete boundaries

            // Collect all sample data to compute smooth profile parameters
            float r_min = 1e10f;
            float r_max = 0.0f;
            float pump_sum = 0.0f;
            float r_sum = 0.0f;
            int total_count = 0;

            for (int i = 0; i < 8; i++) {
                if (shell_count[i] > 0) {
                    float avg_r = shell_r_sum[i] / (float)shell_count[i];
                    float avg_pump = shell_scale_sum[i] / (float)shell_count[i];

                    if (avg_r < r_min) r_min = avg_r;
                    if (avg_r > r_max) r_max = avg_r;

                    pump_sum += avg_pump * shell_count[i];
                    r_sum += avg_r * shell_count[i];
                    total_count += shell_count[i];
                }
            }

            // Fit exponential: pump_scale(r) ≈ A * exp(-r/L)
            // Use innermost and outermost points to estimate decay length
            float avg_pump_global = (total_count > 0) ? pump_sum / total_count : 1.0f;
            float avg_r_global = (total_count > 0) ? r_sum / total_count : 50.0f;

            // Decay length from characteristic radius
            float L_decay = avg_r_global;  // Exponential scale length

            // Generate smooth shell samples at same radii as discrete mode
            for (int i = 0; i < 8; i++) {
                if (shell_count[i] > 0) {
                    float r = shell_r_sum[i] / (float)shell_count[i];

                    // Smooth exponential profile (no discrete boundaries)
                    float pump_smooth = avg_pump_global * expf(-r / L_decay);

                    // Same saturation formula, but applied to smooth field
                    float saturation_scale = baseline_stress / k_refraction;
                    float saturated_pump = saturation_scale * tanhf(pump_smooth / saturation_scale);
                    float n = 1.0f + baseline_stress + k_refraction * saturated_pump;

                    out->shell_radii[num_shells] = r;
                    out->shell_n[num_shells] = n;
                    num_shells++;
                }
            }
        }

        out->num_shells = num_shells;

        // === PHOTON RING RADIUS CALCULATION ===
        // Compute where deflection angle is maximum (Einstein ring)
        // For thin lens: deflection α(r) = 2 * ∫ (dn/dr) / r dr
        // Maximum α occurs where gradient is steepest (innermost shell)
        //
        // Physical interpretation: The photon ring forms where light rays
        // are bent ~90 degrees, creating the characteristic Einstein ring.
        // This is the observable EHT measures, not internal stress.

        if (num_shells > 0) {
            // Innermost shell has steepest gradient → peak deflection
            // Photon ring radius is approximately at innermost shell boundary
            out->photon_ring_radius = out->shell_radii[num_shells - 1];

            // For more accuracy, find radius of maximum dn/dr
            // But for now, innermost shell is good approximation
        } else {
            out->photon_ring_radius = 0.0f;
        }
    }
}

// Host function to generate stratified sample indices at initialization
// Computes r and phi on-demand from position arrays
void generateStratifiedSamples(
    int* h_indices,
    const float* h_pos_x,
    const float* h_pos_z,
    int N,
    float r_min,
    float r_max)
{
    // Bin particles by radius and angle, pick one from each bin
    float r_step = (r_max - r_min) / RADIAL_BINS;
    float phi_step = TWO_PI / ANGULAR_BINS;

    // For each bin, find the first particle that falls in it
    int sample_idx = 0;
    for (int r_bin = 0; r_bin < RADIAL_BINS && sample_idx < SAMPLE_COUNT; r_bin++) {
        float r_lo = r_min + r_bin * r_step;
        float r_hi = r_lo + r_step;

        for (int phi_bin = 0; phi_bin < ANGULAR_BINS && sample_idx < SAMPLE_COUNT; phi_bin++) {
            float phi_lo = phi_bin * phi_step;
            float phi_hi = phi_lo + phi_step;

            // Find first particle in this bin
            for (int i = 0; i < N; i++) {
                // Compute r and phi on-demand from position
                float px = h_pos_x[i];
                float pz = h_pos_z[i];
                float r = sqrtf(px * px + pz * pz);
                float phi = atan2f(pz, px);
                if (phi < 0) phi += TWO_PI;

                if (r >= r_lo && r < r_hi && phi >= phi_lo && phi < phi_hi) {
                    h_indices[sample_idx++] = i;
                    break;
                }
            }
        }
    }

    // Fill remaining slots with evenly spaced particles (fallback)
    while (sample_idx < SAMPLE_COUNT) {
        h_indices[sample_idx] = (sample_idx * N) / SAMPLE_COUNT;
        sample_idx++;
    }

    printf("[sampling] Generated %d stratified sample indices\n", SAMPLE_COUNT);
}

// End of diagnostics.cuh
