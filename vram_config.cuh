// vram_config.cuh — Dynamic VRAM Management
// ============================================
// Queries GPU memory at startup and configures:
//   - Grid dimension (128³/96³/64³) based on VRAM tier
//   - Safe particle cap (warp-aligned, 82% VRAM usage)
//   - Octree particle cap (separate budget for CUB temp)
//   - Device-side grid constants via cudaMemcpyToSymbol
//
// Call initVRAMConfig() once at startup before any cudaMalloc.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "disk.cuh"  // MAX_DISK_PTS, GPUDisk

// ============================================================================
// Runtime globals (set once at startup by initVRAMConfig())
// ============================================================================

static int g_runtime_particle_cap = 3000000;  // Safe default, overwritten by VRAM query
static int g_grid_dim = 128;                  // Grid dimension (128/96/64)
static int g_grid_cells = 128*128*128;        // Total grid cells
static float g_grid_cell_size = 500.0f/128;   // Cell size in world units
static bool g_vram_initialized = false;       // Guard for one-time init

// Octree VRAM management — separate cap for morton buffers + thrust temp
// Geometry: 24 = crystallized (LAMBDA_OCTREE), 128 = max inference
static int g_octree_particle_cap = 500000;    // Safe default, recalculated from VRAM

// Legacy compile-time override (if user wants to force a value)
#ifndef RUNTIME_PARTICLE_CAP
#define RUNTIME_PARTICLE_CAP  g_runtime_particle_cap  // Use runtime value
#endif

// ============================================================================
// DEVICE-SIDE GRID CONSTANTS (set from host via cudaMemcpyToSymbol)
// ============================================================================
// These replace compile-time GRID_DIM in device functions for dynamic scaling.
// Use d_grid_* in device code, g_grid_* in host code.

__constant__ int d_grid_dim;           // Grid dimension (128/96/64)
__constant__ int d_grid_cells;         // Total cells (dim³)
__constant__ float d_grid_cell_size;   // Cell size (500.0 / dim)
__constant__ int d_grid_stride_y;      // Y stride (dim)
__constant__ int d_grid_stride_z;      // Z stride (dim²)

// ============================================================================
// VRAM CONFIGURATION (call once at startup, before any cudaMalloc)
// ============================================================================

inline void initVRAMConfig() {
    if (g_vram_initialized) return;

    // Query GPU memory
    size_t free_mem = 0, total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        printf("[VRAM] Warning: cudaMemGetInfo failed (%s), using defaults\n",
               cudaGetErrorString(err));
        free_mem = 3ULL * 1024 * 1024 * 1024;  // Assume 3GB if query fails
        total_mem = free_mem;
    }

    // Reserve 18% headroom (82% usage cap per user request)
    size_t usable_mem = (size_t)(free_mem * 0.82);
    printf("[VRAM] Total: %.1f MB, Free: %.1f MB, Usable (82%%): %.1f MB\n",
           total_mem / 1e6, free_mem / 1e6, usable_mem / 1e6);

    // Scale grid size based on VRAM tier
    if (usable_mem >= 3500ULL * 1024 * 1024) {
        g_grid_dim = 128;  // 96 MB grid, for 6GB+ GPUs
    } else if (usable_mem >= 2000ULL * 1024 * 1024) {
        g_grid_dim = 96;   // 40.5 MB grid, for 3-6GB GPUs
    } else {
        g_grid_dim = 64;   // 12 MB grid, for <3GB GPUs
    }
    g_grid_cells = g_grid_dim * g_grid_dim * g_grid_dim;
    g_grid_cell_size = 500.0f / g_grid_dim;

    printf("[VRAM] Grid size: %d³ = %d cells (%.1f MB)\n",
           g_grid_dim, g_grid_cells, g_grid_cells * 12 * 4 / 1e6);

    // Determine whether the octree subsystem will be allocated at all.
    // The morton/octree_nodes/leaf_* buffers are only needed if either the
    // rebuild path or the render path is active. With both off (the default),
    // the analytic tree build, the stochastic rebuild, and the V3 render
    // traversal are all no-ops, and we can reclaim ~320 MB of morton/xor/ids
    // plus ~88 MB of octree_nodes + leaf buffers.
    extern bool g_octree_rebuild;
    extern bool g_octree_render;
    const bool octree_needed = g_octree_rebuild || g_octree_render;

    // Calculate fixed overhead (with scaled grid)
    size_t grid_overhead = (size_t)g_grid_cells * 12 * sizeof(float);
    size_t octree_overhead = octree_needed ? (88ULL * 1024 * 1024) : 0;  // 88 MB fixed, skipped when octree disabled
    size_t sparse_overhead = 12ULL * 1024 * 1024;   // 12 MB (fixed)
    size_t misc_overhead = 5ULL * 1024 * 1024;      // 5 MB (fixed)
    size_t fixed_overhead = grid_overhead + octree_overhead + sparse_overhead + misc_overhead;

    // Calculate safe particle cap. Derive per-particle bytes from the struct
    // so a future field addition/removal updates the calculator automatically.
    //   GPUDisk:  sizeof(GPUDisk) / MAX_DISK_PTS  (struct size + alignment)
    //   Auxiliary (always): cell index (4 bytes — used by grid physics)
    //   Auxiliary (octree-only, 16 bytes): morton (8) + xor (4) + ids (4)
    const size_t gpudisk_bytes = sizeof(GPUDisk) / (size_t)MAX_DISK_PTS;
    const size_t aux_bytes = octree_needed ? 20 : 4;  // 20 with octree, 4 without
    size_t bytes_per_particle = gpudisk_bytes + aux_bytes;
    int max_particles = (int)((usable_mem - fixed_overhead) / bytes_per_particle);

    // Round down to warp-aligned boundary (V8 style: 32 threads)
    max_particles = (max_particles / 32) * 32;

    // Clamp to absolute maximum
    if (max_particles > MAX_DISK_PTS) max_particles = MAX_DISK_PTS;
    if (max_particles < 32) max_particles = 32;  // Sanity floor

    g_runtime_particle_cap = max_particles;

    printf("[VRAM] Safe particle cap: %d (%.1f MB for particles%s)\n",
           g_runtime_particle_cap, (float)g_runtime_particle_cap * bytes_per_particle / 1e6,
           octree_needed ? "" : ", octree disabled");

    // Octree particle capacity — only relevant if octree is actually enabled.
    // When disabled, g_octree_particle_cap is set to g_runtime_particle_cap so
    // any code path that reads it has a sensible upper bound, but no buffers
    // will actually be sized by it.
    if (octree_needed) {
        // Thrust radix_sort needs ~24 bytes temporary per element
        // Morton buffers: 8 (keys) + 4 (xor) + 4 (ids) = 16 bytes/particle
        // Total: 16 + 24 = 40 bytes/particle for octree rebuild
        size_t gpudisk_overhead = (size_t)g_runtime_particle_cap * gpudisk_bytes;
        size_t octree_budget = usable_mem - grid_overhead - octree_overhead - sparse_overhead - misc_overhead - gpudisk_overhead;
        size_t bytes_per_particle_octree = 40;  // morton + thrust temp
        int octree_cap = (int)(octree_budget / bytes_per_particle_octree);
        octree_cap = (octree_cap / 32) * 32;
        if (octree_cap > g_runtime_particle_cap) octree_cap = g_runtime_particle_cap;
        if (octree_cap < 32) octree_cap = 32;
        g_octree_particle_cap = octree_cap;
        printf("[VRAM] Octree cap: %d particles (%.1f MB budget, thrust temp: %.1f MB)\n",
               g_octree_particle_cap, octree_budget / 1e6,
               (float)g_octree_particle_cap * 24 / 1e6);
    } else {
        g_octree_particle_cap = g_runtime_particle_cap;
        printf("[VRAM] Octree subsystem DISABLED — no morton/node/leaf buffers allocated\n");
    }

    // Copy grid constants to device
    int stride_y = g_grid_dim;
    int stride_z = g_grid_dim * g_grid_dim;
    cudaMemcpyToSymbol(d_grid_dim, &g_grid_dim, sizeof(int));
    cudaMemcpyToSymbol(d_grid_cells, &g_grid_cells, sizeof(int));
    cudaMemcpyToSymbol(d_grid_cell_size, &g_grid_cell_size, sizeof(float));
    cudaMemcpyToSymbol(d_grid_stride_y, &stride_y, sizeof(int));
    cudaMemcpyToSymbol(d_grid_stride_z, &stride_z, sizeof(int));

    g_vram_initialized = true;
}

// ============================================================================
// OCTREE VRAM CHECK — Test if rebuild will fit before attempting
// ============================================================================
// The octree is the crystallization (24 = LAMBDA_OCTREE, finished stone layer).
// CUB radix sort allocates temporary buffers during execution.
// This function checks if current VRAM can handle the sort for N particles.

inline bool canOctreeFit(int N_current) {
    // Always query current free VRAM — pre-calculated cap doesn't account for runtime usage
    size_t free_now = 0, total = 0;
    cudaError_t err = cudaMemGetInfo(&free_now, &total);
    if (err != cudaSuccess) {
        // Can't query — be conservative, skip rebuild
        return false;
    }

    // CUB radix sort needs temporary buffers:
    // - Double-buffer for keys: N * 8 * 2 bytes
    // - Double-buffer for values: N * 4 * 2 bytes
    // Total: ~N * 24 bytes conservatively
    size_t cub_temp = (size_t)N_current * 24;

    // Need 25% headroom for fragmentation and other CUDA operations
    size_t required = (size_t)(cub_temp / 0.75);

    bool fits = (free_now >= required);

    // Debug logging on tight conditions (only log first time we're close)
    static bool logged_tight = false;
    if (!fits && !logged_tight) {
        printf("[octree VRAM] N=%d needs %.1f MB, only %.1f MB free — will skip\n",
               N_current, required / 1e6, free_now / 1e6);
        logged_tight = true;
    }

    return fits;
}
