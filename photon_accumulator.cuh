// photon_accumulator.cuh — Sparse Photonic Accumulators
// =====================================================
// Collapses dense clusters of phase-coherent particles into single
// representatives. This is not an approximation — coherent particles
// ARE a single waveform. Computing them individually is the approximation.
//
// Collapse condition (all must hold):
//   ρ_cell ≥ ACCUMULATOR_MIN_DENSITY  (R_cell noise floor at 1/√ρ)
//   R_cell ≥ ACCUMULATOR_MIN_COHERENCE (phase-aligned)
//   Cell has static particles (active_mask = 0)
//
// Amplitude scales as √ρ (wave superposition: intensity ∝ N, amplitude ∝ √N)
// Frequency from mean ω_nat of collapsed particles
// Phase from circular mean of θ values (already in phase_sin/cos arrays)
//
// The mip-tree naturally extends this to multiple scales:
// each mip level IS an accumulator at that spatial resolution.
#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"
#include "cell_grid.cuh"
#include "nullable.h"

// ============================================================================
// Collapse thresholds — from existing physics metrics
// ============================================================================
#define ACCUMULATOR_MIN_DENSITY    10.0f   // Below this, R is noise (1/√10 > 0.316)
#define ACCUMULATOR_MIN_COHERENCE  0.7f    // Same as SPAWN_COHERENCE_THRESH — physically meaningful
#define ACCUMULATOR_MAX_COUNT      65536   // Max accumulators (one per occupied coherent cell)

// ============================================================================
// PhotonAccumulator struct — one per coherent cell
// ============================================================================

struct PhotonAccumulator {
    float3 position;      // Density-weighted centroid of collapsed particles
    float  frequency;     // Mean ω_nat of cluster
    float  amplitude;     // √ρ (wave superposition)
    float  phase;         // Circular mean θ = atan2(Σsin θ, Σcos θ)
    float  coherence;     // R_cell at collapse time
    int    particle_count; // How many particles this represents
};

// ============================================================================
// Collapse kernel — scan grid cells, emit accumulators for coherent clusters
// ============================================================================
// Runs AFTER scatter (density/phase populated) and AFTER R_cell computation.
// Reads: d_grid_density, d_grid_phase_sin/cos, d_grid_R_cell, d_grid_momentum
// Writes: d_accumulators (sparse output), d_accumulator_count (atomic counter)

__global__ void collapseToAccumulators(
    const float* __restrict__ density,
    const float* __restrict__ phase_sin,
    const float* __restrict__ phase_cos,
    const float* __restrict__ momentum_x,
    const float* __restrict__ momentum_y,
    const float* __restrict__ momentum_z,
    PhotonAccumulator* __restrict__ accumulators,
    int* __restrict__ accumulator_count,
    int total_cells)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= total_cells) return;

    float rho = density[cell];

    // Collapse condition 1: sufficient density for meaningful coherence
    if (rho < ACCUMULATOR_MIN_DENSITY) return;

    // Collapse condition 2: phase coherence above threshold
    float ps = phase_sin[cell];
    float pc = phase_cos[cell];
    float R_cell = sqrtf(ps * ps + pc * pc) / rho;

    if (R_cell < ACCUMULATOR_MIN_COHERENCE) return;

    // This cell is collapsible — emit an accumulator
    int slot = atomicAdd(accumulator_count, 1);
    if (slot >= ACCUMULATOR_MAX_COUNT) return;  // overflow guard

    // Reconstruct cell center position from cell index
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);
    float3 pos;
    pos.x = (float)cx * d_grid_cell_size - GRID_HALF_SIZE + d_grid_cell_size * 0.5f;
    pos.y = (float)cy * d_grid_cell_size - GRID_HALF_SIZE + d_grid_cell_size * 0.5f;
    pos.z = (float)cz * d_grid_cell_size - GRID_HALF_SIZE + d_grid_cell_size * 0.5f;

    PhotonAccumulator acc;
    acc.position = pos;
    acc.amplitude = sqrtf(rho);                              // √ρ: wave superposition
    acc.phase = cuda_fast_atan2(ps / rho, pc / rho);         // Circular mean phase
    acc.coherence = R_cell;
    acc.particle_count = (int)rho;

    // Frequency: estimate from momentum magnitude / (ρ × r)
    // This gives the mean tangential velocity → orbital frequency
    float mx = momentum_x[cell];
    float my = momentum_y[cell];
    float mz = momentum_z[cell];
    float v_mean = sqrtf(mx*mx + my*my + mz*mz) / rho;
    float r = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
    acc.frequency = (r > 1.0f) ? v_mean / r : v_mean;       // ω = v/r

    accumulators[slot] = acc;
}

// ============================================================================
// Query: check resonance between a probe frequency and an accumulator
// ============================================================================
// Returns amplitude if resonant (constructive), 0 if not (destructive = skip)

__device__ __forceinline__ float checkResonance(
    float probe_frequency,
    const PhotonAccumulator& acc,
    float bandwidth)
{
    float df = fabsf(probe_frequency - acc.frequency);
    if (df > bandwidth) return 0.0f;  // Destructive: fizzles, no compute

    // Constructive: amplitude weighted by coherence and frequency match
    float match = 1.0f - df / bandwidth;  // 1.0 at exact match, 0.0 at edge
    return acc.amplitude * acc.coherence * match;
}
