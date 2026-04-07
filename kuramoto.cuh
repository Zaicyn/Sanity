// kuramoto.cuh — Kuramoto Order Parameter Reduction Kernels
// ==========================================================
// - computeRcell: per-cell R = |⟨e^{iθ}⟩|
// - reduceRcellRadialProfile: bins R_cell by radial distance
// - reduceKuramotoR: two-stage block reduction for global R
// - reducePhaseHistogram: bins theta values, collects omega per bin
//
// Dependencies: cuda_lut.cuh, vram_config.cuh (d_grid_* constants)
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// PER-CELL KURAMOTO ORDER PARAMETER
// ============================================================================
// Computes R_cell = |⟨e^{iθ}⟩| per grid cell from the accumulated
// phase_sin/phase_cos/density fields. Unlike the global R, this exposes
// spatial structure: chimera states (coherent + incoherent coexisting),
// traveling coherence packets, radial coherence waves, and localized
// destabilization events. Nearly free — one sqrt + two mul + one div per cell.

__global__ void computeRcell(
    const float* __restrict__ density,
    const float* __restrict__ phase_sin,
    const float* __restrict__ phase_cos,
    float* __restrict__ R_cell,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float rho = density[idx];
    if (rho < 1e-6f) {
        R_cell[idx] = 0.0f;
        return;
    }
    float inv_rho = 1.0f / rho;
    float s = phase_sin[idx] * inv_rho;
    float c = phase_cos[idx] * inv_rho;
    R_cell[idx] = sqrtf(s * s + c * c);
}

// Radial profile reduction: bins cells by distance from grid center.
// Finite-sample bias: R_cell for low-density cells is dominated by 1/√ρ
// noise, not real coherence. We correct by subtracting the expected noise
// floor (√(1/ρ) for ρ particles) and clamping to [0, 1]. Only cells with
// ρ ≥ MIN_RHO contribute to avoid noise domination entirely.
__global__ void reduceRcellRadialProfile(
    const float* __restrict__ R_cell,
    const float* __restrict__ density,
    int grid_dim,
    int n_bins,
    float grid_cell_size,
    float* __restrict__ bin_R_sum,       // [n_bins] sum of bias-corrected R * density
    float* __restrict__ bin_weight_sum,  // [n_bins] sum of density (dense cells only)
    float* __restrict__ bin_cell_count   // [n_bins] count of dense cells (for diagnostics)
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = grid_dim * grid_dim * grid_dim;
    if (cell >= n_cells) return;

    float rho = density[cell];
    const float MIN_RHO = 10.0f;  // Reject noise-dominated cells
    if (rho < MIN_RHO) return;

    // Cell coords (grid-center origin)
    int cx = cell % grid_dim;
    int cy = (cell / grid_dim) % grid_dim;
    int cz = cell / (grid_dim * grid_dim);
    float fx = ((float)cx - 0.5f * (float)grid_dim) * grid_cell_size;
    float fy = ((float)cy - 0.5f * (float)grid_dim) * grid_cell_size;
    float fz = ((float)cz - 0.5f * (float)grid_dim) * grid_cell_size;
    float r = sqrtf(fx * fx + fy * fy + fz * fz);

    float r_max = 0.5f * grid_dim * grid_cell_size;
    int bin = (int)(r / r_max * n_bins);
    if (bin < 0) bin = 0;
    if (bin >= n_bins) bin = n_bins - 1;

    // Bias-corrected R: subtract noise floor 1/√ρ, clamp ≥ 0
    float R = R_cell[cell];
    float noise_floor = rsqrtf(rho);
    float R_corrected = R - noise_floor;
    if (R_corrected < 0.0f) R_corrected = 0.0f;

    atomicAdd(&bin_R_sum[bin], R_corrected * rho);
    atomicAdd(&bin_weight_sum[bin], rho);
    atomicAdd(&bin_cell_count[bin], 1.0f);
}

// ============================================================================
// KURAMOTO ORDER PARAMETER REDUCTION
// ============================================================================
// Computes R = |⟨e^{iθ}⟩| over all active particles as the global Kuramoto
// order parameter. R ≈ 0 means incoherent; R ≈ 1 means fully synchronized.
// Classical Kuramoto predicts a sharp transition at K_c ≈ 2σ/π for Gaussian
// natural frequency distribution.
//
// Two-stage reduction: each block computes block-level partial sums, then
// a final host-side combine (block count is small).

__global__ void reduceKuramotoR(
    const GPUDisk* __restrict__ disk,
    uint32_t N,
    float* __restrict__ block_sin_sum,
    float* __restrict__ block_cos_sum,
    int* __restrict__ block_count
) {
    __shared__ float s_sin[256];
    __shared__ float s_cos[256];
    __shared__ int s_cnt[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float my_sin = 0.0f, my_cos = 0.0f;
    int my_cnt = 0;
    if (i < (int)N && particle_active(disk, i) && !particle_ejected(disk, i)) {
        float theta = disk->theta[i];
        my_sin = cuda_lut_sin(theta);
        my_cos = cuda_lut_cos(theta);
        my_cnt = 1;
    }

    s_sin[tid] = my_sin;
    s_cos[tid] = my_cos;
    s_cnt[tid] = my_cnt;
    __syncthreads();

    // Tree reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sin[tid] += s_sin[tid + stride];
            s_cos[tid] += s_cos[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sin_sum[blockIdx.x] = s_sin[0];
        block_cos_sum[blockIdx.x] = s_cos[0];
        block_count[blockIdx.x] = s_cnt[0];
    }
}

// ============================================================================
// PHASE HISTOGRAM
// ============================================================================
// Bins particle θ values into PHASE_HIST_BINS equal-width bins over [0, 2π).
// A multi-peak histogram confirms multi-domain clustering; a single-peak or
// smooth unimodal distribution would indicate the R_cell > R_global gap
// comes from a different mechanism (e.g. spatial phase waves). Cheap:
// one block-reduce, one small DtoH copy.

#define PHASE_HIST_BINS 32

__global__ void reducePhaseHistogram(
    const GPUDisk* __restrict__ disk,
    uint32_t N,
    int* __restrict__ bin_counts,       // [PHASE_HIST_BINS]
    float* __restrict__ bin_omega_sum,  // [PHASE_HIST_BINS] Σ ω per bin
    float* __restrict__ bin_omega_sq    // [PHASE_HIST_BINS] Σ ω² per bin
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)N) return;
    if (!particle_active(disk, i) || particle_ejected(disk, i)) return;

    float theta = disk->theta[i];
    float t = theta * (float)PHASE_HIST_BINS / 6.28318530718f;
    int bin = (int)t;
    if (bin < 0) bin = 0;
    if (bin >= PHASE_HIST_BINS) bin = PHASE_HIST_BINS - 1;

    float omega = disk->omega_nat[i];
    atomicAdd(&bin_counts[bin], 1);
    atomicAdd(&bin_omega_sum[bin], omega);
    atomicAdd(&bin_omega_sq[bin], omega * omega);
}

// ============================================================================
// ACTIVE PARTICLE COMPACTION KERNELS
// ============================================================================
// Separate static (shell) particles from active (moving) particles.
// Static particles get baked once; active particles scatter/gather every frame.

// Compute activity mask: active if moving fast OR changed cell

// End of kuramoto.cuh
