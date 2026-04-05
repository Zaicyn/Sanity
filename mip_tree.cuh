// mip_tree.cuh — Hierarchical Grid Coherence (Mip-Tree)
// =====================================================
//
// Replaces Morton-sorted octree with deterministic grid hierarchy.
// No sorting, no atomics (except final reductions), perfect GPU occupancy.
//
// Architecture:
//   Level 0: 128³ grid (existing physics grid)
//   Level 1:  64³ (2×2×2 reduction)
//   Level 2:  32³
//   Level 3:  16³
//   Level 4:   8³
//   Level 5:   4³
//   Level 6:   2³
//   Level 7:   1³ (global mode)
//
// Each level stores:
//   - density: sum of child densities
//   - phase: circular mean of child phases
//   - momentum: sum of child momenta
//
// Usage:
//   1. mip_tree_init() at startup
//   2. mip_tree_build_up() after grid scatter
//   3. mip_tree_propagate_down() for coherence coupling
//   4. mip_tree_cleanup() at shutdown
//
// Philosophy: "Everything correct stays untouched forever."
// Grid → mip is pure forward passes, no sorting, no dependencies.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

#define MIP_MAX_LEVELS  8    // Maximum levels (for static array sizing)

// Per-cell coherence data (lightweight for mip operations)
struct MipCell {
    float density;      // Particle count / cell volume
    float phase;        // S3 phase angle [0, 2π)
    float momentum_x;   // Sum of particle momenta
    float momentum_y;
    float momentum_z;
    float coherence;    // Phase coherence magnitude [0, 1]
};

// ============================================================================
// Global State
// ============================================================================

struct MipTree {
    // Device buffers for each level
    MipCell* d_levels[MIP_MAX_LEVELS];

    // Level dimensions (computed at init based on base_dim)
    int dims[MIP_MAX_LEVELS];   // e.g., 96, 48, 24, 12, 6, 3, 1 for base_dim=96
    int cells[MIP_MAX_LEVELS];  // dims[i]³
    int num_levels;             // Actual number of levels used

    // Configuration
    int base_dim;         // Level 0 dimension (from grid)
    float alpha_down;     // Coupling strength for downward propagation
    bool initialized;
};

extern MipTree g_mip_tree;

// ============================================================================
// Host Functions
// ============================================================================

inline bool mip_tree_init(int base_dim, float coupling_alpha = 0.1f) {
    MipTree& mt = g_mip_tree;

    // Compute level dimensions based on base_dim
    // For base_dim=96: 96 → 48 → 24 → 12 → 6 → 3 → 1 (7 levels)
    // For base_dim=128: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 (8 levels)
    int dim = base_dim;
    size_t total_bytes = 0;
    int num_levels = 0;

    while (dim >= 1 && num_levels < MIP_MAX_LEVELS) {
        mt.dims[num_levels] = dim;
        mt.cells[num_levels] = dim * dim * dim;
        total_bytes += mt.cells[num_levels] * sizeof(MipCell);
        num_levels++;
        dim >>= 1;
    }

    mt.num_levels = num_levels;
    mt.base_dim = base_dim;

    printf("[mip] Initializing %d-level hierarchy (base=%d³): %.1f MB total\n",
           num_levels, base_dim, total_bytes / (1024.0f * 1024.0f));

    // Allocate all levels
    for (int i = 0; i < num_levels; i++) {
        cudaError_t err = cudaMalloc(&mt.d_levels[i], mt.cells[i] * sizeof(MipCell));
        if (err != cudaSuccess) {
            fprintf(stderr, "[mip] Failed to allocate level %d (%d cells): %s\n",
                    i, mt.cells[i], cudaGetErrorString(err));
            // Cleanup partial allocations
            for (int j = 0; j < i; j++) {
                cudaFree(mt.d_levels[j]);
                mt.d_levels[j] = nullptr;
            }
            return false;
        }
        cudaMemset(mt.d_levels[i], 0, mt.cells[i] * sizeof(MipCell));
    }

    // Initialize unused levels to nullptr
    for (int i = num_levels; i < MIP_MAX_LEVELS; i++) {
        mt.d_levels[i] = nullptr;
        mt.dims[i] = 0;
        mt.cells[i] = 0;
    }

    mt.alpha_down = coupling_alpha;
    mt.initialized = true;

    printf("[mip] Hierarchy ready: L0=%d³, L%d=%d³ (coarsest)\n",
           mt.dims[0], num_levels-1, mt.dims[num_levels-1]);

    return true;
}

inline void mip_tree_cleanup() {
    MipTree& mt = g_mip_tree;
    for (int i = 0; i < MIP_MAX_LEVELS; i++) {
        if (mt.d_levels[i]) {
            cudaFree(mt.d_levels[i]);
            mt.d_levels[i] = nullptr;
        }
    }
    mt.initialized = false;
}

// ============================================================================
// Device Kernels — Up-Sweep (Build Hierarchy)
// ============================================================================

// Reduce 2×2×2 fine cells into 1 coarse cell
// Density: sum
// Phase: circular mean (atan2 of sin/cos sums)
// Momentum: sum
// Coherence: magnitude of phase vector / count
__global__ void mipKernel_reduce(
    const MipCell* __restrict__ fine,
    MipCell* __restrict__ coarse,
    int fine_dim
) {
    int coarse_dim = fine_dim >> 1;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cz = blockIdx.z * blockDim.z + threadIdx.z;

    if (cx >= coarse_dim || cy >= coarse_dim || cz >= coarse_dim) return;

    // Accumulate 8 children
    float density_sum = 0.0f;
    float sin_sum = 0.0f;
    float cos_sum = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;
    int valid_count = 0;

    #pragma unroll
    for (int dz = 0; dz < 2; dz++) {
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int fx = cx * 2 + dx;
                int fy = cy * 2 + dy;
                int fz = cz * 2 + dz;
                int fine_idx = fx + fy * fine_dim + fz * fine_dim * fine_dim;

                const MipCell& child = fine[fine_idx];
                density_sum += child.density;

                if (child.density > 0.0f) {
                    // Weight phase contribution by density
                    sin_sum += child.density * sinf(child.phase);
                    cos_sum += child.density * cosf(child.phase);
                    valid_count++;
                }

                mom_x += child.momentum_x;
                mom_y += child.momentum_y;
                mom_z += child.momentum_z;
            }
        }
    }

    // Write coarse cell
    int coarse_idx = cx + cy * coarse_dim + cz * coarse_dim * coarse_dim;
    MipCell& out = coarse[coarse_idx];

    out.density = density_sum;
    out.momentum_x = mom_x;
    out.momentum_y = mom_y;
    out.momentum_z = mom_z;

    // Circular mean of phase
    if (valid_count > 0 && (sin_sum != 0.0f || cos_sum != 0.0f)) {
        out.phase = atan2f(sin_sum, cos_sum);
        if (out.phase < 0.0f) out.phase += 2.0f * 3.14159265f;
        // Coherence = |phase vector| / total weight
        float phase_mag = sqrtf(sin_sum * sin_sum + cos_sum * cos_sum);
        out.coherence = phase_mag / density_sum;
    } else {
        out.phase = 0.0f;
        out.coherence = 0.0f;
    }
}

// ============================================================================
// Device Kernels — Down-Sweep (Propagate Coherence)
// ============================================================================

// Add parent influence to fine cells
// Fine phase += alpha * (parent_phase - fine_phase) * parent_coherence
// This creates scale coupling without overwriting local structure
__global__ void mipKernel_propagate(
    const MipCell* __restrict__ coarse,
    MipCell* __restrict__ fine,
    int fine_dim,
    float alpha
) {
    int coarse_dim = fine_dim >> 1;
    int fx = blockIdx.x * blockDim.x + threadIdx.x;
    int fy = blockIdx.y * blockDim.y + threadIdx.y;
    int fz = blockIdx.z * blockDim.z + threadIdx.z;

    if (fx >= fine_dim || fy >= fine_dim || fz >= fine_dim) return;

    // Find parent cell
    int px = fx >> 1;
    int py = fy >> 1;
    int pz = fz >> 1;
    int parent_idx = px + py * coarse_dim + pz * coarse_dim * coarse_dim;

    const MipCell& parent = coarse[parent_idx];

    int fine_idx = fx + fy * fine_dim + fz * fine_dim * fine_dim;
    MipCell& child = fine[fine_idx];

    if (child.density > 0.0f && parent.coherence > 0.01f) {
        // Phase coupling: attract toward parent phase, weighted by parent coherence
        float phase_diff = parent.phase - child.phase;

        // Wrap to [-π, π]
        while (phase_diff > 3.14159265f) phase_diff -= 2.0f * 3.14159265f;
        while (phase_diff < -3.14159265f) phase_diff += 2.0f * 3.14159265f;

        child.phase += alpha * phase_diff * parent.coherence;

        // Wrap result to [0, 2π)
        while (child.phase >= 2.0f * 3.14159265f) child.phase -= 2.0f * 3.14159265f;
        while (child.phase < 0.0f) child.phase += 2.0f * 3.14159265f;
    }
}

// ============================================================================
// Host Functions — Build & Propagate
// ============================================================================

// Build hierarchy from Level 0 (call after grid scatter)
inline void mip_tree_build_up() {
    MipTree& mt = g_mip_tree;
    if (!mt.initialized || mt.num_levels < 2) return;

    // Reduce: L0 → L1 → L2 → ... → Lmax
    for (int level = 0; level < mt.num_levels - 1; level++) {
        int fine_dim = mt.dims[level];
        int coarse_dim = mt.dims[level + 1];

        // Skip if coarse dimension would be 0
        if (coarse_dim < 1) break;

        // 3D grid of thread blocks
        dim3 block(8, 8, 8);
        dim3 grid(
            (coarse_dim + block.x - 1) / block.x,
            (coarse_dim + block.y - 1) / block.y,
            (coarse_dim + block.z - 1) / block.z
        );

        mipKernel_reduce<<<grid, block>>>(
            mt.d_levels[level],
            mt.d_levels[level + 1],
            fine_dim
        );
    }
}

// Propagate coherence back down (call after build_up)
inline void mip_tree_propagate_down(float alpha_override = -1.0f) {
    MipTree& mt = g_mip_tree;
    if (!mt.initialized || mt.num_levels < 2) return;

    float alpha = (alpha_override >= 0.0f) ? alpha_override : mt.alpha_down;

    // Propagate: Lmax → ... → L1 → L0
    for (int level = mt.num_levels - 1; level > 0; level--) {
        int fine_dim = mt.dims[level - 1];

        dim3 block(8, 8, 8);
        dim3 grid(
            (fine_dim + block.x - 1) / block.x,
            (fine_dim + block.y - 1) / block.y,
            (fine_dim + block.z - 1) / block.z
        );

        mipKernel_propagate<<<grid, block>>>(
            mt.d_levels[level],
            mt.d_levels[level - 1],
            fine_dim,
            alpha
        );
    }
}

// ============================================================================
// Transfer Functions — Grid ↔ Mip Level 0
// ============================================================================

// Copy density/momentum from physics grid to mip level 0
// (Grid format: separate arrays; Mip format: struct array)
// Uses sin/cos buffers to reconstruct phase via atan2
__global__ void mipKernel_fromGrid(
    MipCell* __restrict__ mip_level0,
    const float* __restrict__ grid_density,
    const float* __restrict__ grid_mom_x,
    const float* __restrict__ grid_mom_y,
    const float* __restrict__ grid_mom_z,
    const float* __restrict__ grid_phase_sin,  // Can be nullptr
    const float* __restrict__ grid_phase_cos,  // Can be nullptr
    int dim
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim || y >= dim || z >= dim) return;

    int idx = x + y * dim + z * dim * dim;

    MipCell& cell = mip_level0[idx];
    cell.density = grid_density[idx];
    cell.momentum_x = grid_mom_x ? grid_mom_x[idx] : 0.0f;
    cell.momentum_y = grid_mom_y ? grid_mom_y[idx] : 0.0f;
    cell.momentum_z = grid_mom_z ? grid_mom_z[idx] : 0.0f;

    // Reconstruct phase from sin/cos (circular representation)
    if (grid_phase_sin && grid_phase_cos) {
        float s = grid_phase_sin[idx];
        float c = grid_phase_cos[idx];
        cell.phase = atan2f(s, c);
        if (cell.phase < 0.0f) cell.phase += 2.0f * 3.14159265f;
        // Coherence from magnitude of sin/cos vector
        float mag = sqrtf(s * s + c * c);
        cell.coherence = (cell.density > 0.0f) ? fminf(mag, 1.0f) : 0.0f;
    } else {
        cell.phase = 0.0f;
        cell.coherence = (cell.density > 0.0f) ? 1.0f : 0.0f;
    }
}

// Copy phase back from mip level 0 to grid (for rendering/physics)
__global__ void mipKernel_toGrid(
    const MipCell* __restrict__ mip_level0,
    float* __restrict__ grid_phase,
    int dim
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim || y >= dim || z >= dim) return;

    int idx = x + y * dim + z * dim * dim;
    grid_phase[idx] = mip_level0[idx].phase;
}

// Host wrappers
inline void mip_tree_from_grid(
    const float* d_density,
    const float* d_mom_x,
    const float* d_mom_y,
    const float* d_mom_z,
    const float* d_phase_sin,
    const float* d_phase_cos
) {
    MipTree& mt = g_mip_tree;
    if (!mt.initialized) return;

    int dim = mt.dims[0];
    dim3 block(8, 8, 8);
    dim3 grid(
        (dim + block.x - 1) / block.x,
        (dim + block.y - 1) / block.y,
        (dim + block.z - 1) / block.z
    );

    mipKernel_fromGrid<<<grid, block>>>(
        mt.d_levels[0],
        d_density,
        d_mom_x, d_mom_y, d_mom_z,
        d_phase_sin, d_phase_cos,
        dim
    );
}

inline void mip_tree_to_grid(float* d_phase) {
    MipTree& mt = g_mip_tree;
    if (!mt.initialized) return;

    int dim = mt.dims[0];
    dim3 block(8, 8, 8);
    dim3 grid(
        (dim + block.x - 1) / block.x,
        (dim + block.y - 1) / block.y,
        (dim + block.z - 1) / block.z
    );

    mipKernel_toGrid<<<grid, block>>>(
        mt.d_levels[0],
        d_phase,
        dim
    );
}

// ============================================================================
// Diagnostics
// ============================================================================

inline void mip_tree_print_stats() {
    MipTree& mt = g_mip_tree;
    if (!mt.initialized) {
        printf("[mip] Not initialized\n");
        return;
    }

    printf("[mip] Hierarchy (%d levels): ", mt.num_levels);
    for (int i = 0; i < mt.num_levels; i++) {
        printf("L%d=%d³ ", i, mt.dims[i]);
    }
    printf("| alpha=%.3f\n", mt.alpha_down);
}

// Global instance definition
MipTree g_mip_tree = {};

// End of mip_tree.cuh
