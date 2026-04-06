// blackhole_v20.cu — V20 Siphon Pump Integration
// ================================================
//
// Realtime Hopfion Lattice Black Hole + Siphon Pump State Machine
//
// This version integrates:
//   - squaragon.h: O(1) cuboctahedral primitive
//   - siphon_pump.h: 12↔16 dimensional siphon state machine
//   - Original V8 Aizawa/Viviani physics
//
// The disk particles now operate as individual siphon pumps:
//   - Each particle has its own 12→16→12 cycle
//   - Seam phase bits controlled by local stress
//   - Scale cascades propagate through the disk
//   - Ejection = pump overload (residual exceeds threshold)
//
// Physics mapping:
//   - ISCO region: pump running at full coupling (SEAM_FULL)
//   - Outer disk: pump in reduced coupling (SEAM_UP_ONLY or SEAM_DOWN_ONLY)
//   - Ejected jets: pump overflow, scale cascade triggered
//   - Accretion: pump intake, 16→12 downstroke dominates
//
// Compile:
//   nvcc -O3 -arch=sm_75 -std=c++17 blackhole_v20.cu -lglfw -lGLEW -lGL -o blackhole_v20
//
// Controls:
//   Left drag  — orbit camera
//   Scroll     — zoom
//   R          — reset simulation
//   Space      — pause/resume
//   C          — toggle color scheme (topology vs intensity)
//   1/2/3/4    — set seam bits (0=closed, 1=up, 2=down, 3=full)
//   ESC        — quit

#include "squaragon.h"
#include "siphon_pump.h"

// CUDA LUT for fast trigonometry (quarter-sector sine table in constant memory)
// Define CUDA_LUT_IMPLEMENTATION before including to get the init function
#define CUDA_LUT_IMPLEMENTATION
#include "cuda_lut.cuh"

// ============================================================================
// Modular Physics Headers (Math.md compliant)
// ============================================================================
// These headers factor the physics into auditable components:
//   - disk.cuh:        GPUDisk struct, constants, inline compute
//   - harmonic.cuh:    heartbeat (cos θ cos 3θ), coherence filter
//   - forces.cuh:      Viviani field, angular momentum, ion kick
//   - siphon_pump.cuh: 8-state pump machine, ejection
//   - aizawa.cuh:      phase-breathing attractor for jets
//   - topology.cuh:    spiral arm structure
//
// The monolithic siphonDiskKernel below can be replaced with physics.cu
// by defining USE_MODULAR_PHYSICS before compilation.
#include "disk.cuh"
#include "harmonic.cuh"
#include "forces.cuh"
#include "siphon_pump.cuh"
#include "aizawa.cuh"
// topology.cuh moved after device constant definitions (needs d_NUM_ARMS etc.)
#include "sun_trace.cuh"
#include "passive_advection.cuh"  // Tree Architecture Step 2: passive Keplerian advection kernel
#include "active_region.cuh"      // Tree Architecture Step 2: ActiveRegion struct + in-region mask kernel
#include "cuda_primitives.cuh"
#include "octree.cuh"
#include "cell_grid.cuh"
#include "render_fill.cuh"
#include "validator/frame_export.cuh"
#include "topology_recorder.cuh"
#include "mip_tree.cuh"

// Global instance of topology ring buffer (defined in topology_recorder.cuh as extern)
TopologyRecorder g_topo_recorder = {};

// ============================================================================
// VALIDATION CONTEXT — Global state for key handler access
// ============================================================================
// The key callback can't access main()'s local variables, so we maintain
// a global context that gets updated each frame in the main loop.
struct ValidationContext {
    GPUDisk* d_disk = nullptr;
    int N_current = 0;
    float sim_time = 0.0f;
    float heartbeat = 1.0f;
    float avg_scale = 0.0f;
    float avg_residual = 0.0f;
    int export_frame_id = 0;
};
static ValidationContext g_validation_ctx;

// Vulkan + CUDA interop mode (single app, zero-copy)
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "vulkan/vk_types.h"
#include "vulkan/vk_cuda_interop.h"
#include "vulkan/vk_attractor.h"

// Global attractor pipeline (used by vk_buffer.cpp via extern)
AttractorPipeline g_attractor;

namespace vk {
    void initWindow(VulkanContext& ctx);
    void initVulkan(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createDescriptorSetLayout(VulkanContext& ctx);
    void createGraphicsPipeline(VulkanContext& ctx);
    void createSwapchain(VulkanContext& ctx);
    void createFramebuffers(VulkanContext& ctx);
    void createCommandPool(VulkanContext& ctx);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);
    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx);
    void createDepthResources(VulkanContext& ctx);
    void createVolumeDescriptorSetLayout(VulkanContext& ctx);
    void createVolumePipeline(VulkanContext& ctx);
    void createVolumeUniformBuffers(VulkanContext& ctx);
    void createVolumeDescriptorSets(VulkanContext& ctx);
    void createAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);
    void destroyAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);
    void drawFrame(VulkanContext& ctx);
    void cleanup(VulkanContext& ctx);
    void updateUniformBuffer(VulkanContext& ctx, uint32_t currentImage);
    void recordCommandBuffer(VulkanContext& ctx, VkCommandBuffer commandBuffer, uint32_t imageIndex);
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

// Native GPU primitives — no external dependencies (V8 philosophy)
// CUB/thrust removed: "everything correct stays untouched forever"

// ============================================================================
// Configuration — Rendering-Specific (physics constants in disk.cuh)
// ============================================================================

#define WIDTH         1280
#define HEIGHT        720

// ============================================================================
// DYNAMIC VRAM MANAGEMENT — Runtime particle cap & grid scaling
// ============================================================================
// V8 philosophy: "Everything correct stays untouched forever."
// V8 corollary: Size allocations to 82% of available VRAM, warp-aligned.
//
// Grid scaling by VRAM tier:
//   6GB+: 128³ = 96 MB grid, ~5.5M particle cap
//   3-6GB: 96³ = 40.5 MB grid, ~2.5M particle cap
//   <3GB: 64³ = 12 MB grid, ~1.2M particle cap
//
// All caps rounded down to warp boundary (32 threads).

// Runtime globals (set once at startup by initVRAMConfig())
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

// Tree Architecture Step 2: compile-time guard for passive advection dispatch.
// Flipped ON in commit 2d. Still zero-behavior-change because the
// all-encompassing bootstrap ActiveRegion makes in_active_region[i] == 1 for
// every alive particle, so the passive kernel early-returns on every particle
// and siphonDiskKernel runs unchanged on every particle.
#ifndef ENABLE_PASSIVE_ADVECTION
#define ENABLE_PASSIVE_ADVECTION 1
#endif

// NOTE: Physics constants (BH_MASS, ISCO_R, PUMP_*, ION_KICK_*, etc.)
// are now defined in disk.cuh and the modular physics headers.
// This avoids duplication and ensures single source of truth.

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

// ============================================================================
// Vec3 / Mat4 (minimal, no GLM)
// ============================================================================

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float a=0, float b=0, float c=0): x(a), y(b), z(c) {}
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    __host__ __device__ float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    __host__ __device__ float len() const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ Vec3 norm() const { float l = len(); return l > 1e-7f ? *this * (1/l) : Vec3(0,0,1); }
};

struct Mat4 {
    float m[16];
    __host__ __device__ Mat4() { memset(m, 0, sizeof(m)); }

    static Mat4 identity() {
        Mat4 r; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r;
    }

    static Mat4 perspective(float fovY, float aspect, float near_, float far_) {
        Mat4 r;
        float f = 1.0f / tanf(fovY * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (far_ + near_) / (near_ - far_);
        r.m[11] = -1;
        r.m[14] = 2 * far_ * near_ / (near_ - far_);
        return r;
    }

    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f = (center - eye).norm();
        Vec3 r = f.cross(up).norm();
        Vec3 u = r.cross(f);
        Mat4 res;
        res.m[0]=r.x;  res.m[4]=r.y;  res.m[8]=r.z;   res.m[12]=-(r.x*eye.x+r.y*eye.y+r.z*eye.z);
        res.m[1]=u.x;  res.m[5]=u.y;  res.m[9]=u.z;   res.m[13]=-(u.x*eye.x+u.y*eye.y+u.z*eye.z);
        res.m[2]=-f.x; res.m[6]=-f.y; res.m[10]=-f.z; res.m[14]= (f.x*eye.x+f.y*eye.y+f.z*eye.z);
        res.m[15]=1;
        return res;
    }

    static Mat4 mul(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int c = 0; c < 4; c++) for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k*4+rr] * b.m[c*4+k];
            r.m[c*4+rr] = s;
        }
        return r;
    }
};

// ============================================================================
// GPU Disk with Siphon Pump State
// ============================================================================

// NOTE: GPUDisk struct now defined in disk.cuh (modular physics headers)
// The struct definition has been moved to avoid duplication.

// ============================================================================
// Octree Node Structure - Hybrid Analytic/Stochastic Spatial Tree
// ============================================================================
// Morton encoding for memory layout, XOR corner for O(1) neighbor lookup.
// Levels 0-5: ANALYTIC (frozen at init, field-derived energy)
// Levels 6-13: STOCHASTIC (rebuilt from particle positions)
// BOUNDARY nodes at level 5-6 interface for smoothstep blending
// OctreeNode struct and defines now in octree.cuh

// CellGrid struct and grid constants now in cell_grid.cuh

// Runtime values — HOST code uses g_grid_*, DEVICE code uses d_grid_*
// The macros below provide host-side compatibility with existing code
#define GRID_DIM        g_grid_dim
#define GRID_CELLS      g_grid_cells
#define GRID_CELL_SIZE  g_grid_cell_size
#define GRID_STRIDE_Y   g_grid_dim                        // Y stride = dim
#define GRID_STRIDE_Z   (g_grid_dim * g_grid_dim)         // Z stride = dim²

// IMPORTANT: In __device__ and __global__ code, use d_grid_* directly!
// The macros above expand to host variables which won't work in kernels.

// Convert cell index to tile index
// Uses device constants for runtime-scaled grids
__device__ __forceinline__ uint32_t cellToTile(uint32_t cell) {
    int tiles_per_dim = d_grid_dim / TILE_DIM;
    uint32_t cx = cell % d_grid_dim;
    uint32_t cy = (cell / d_grid_stride_y) % d_grid_dim;
    uint32_t cz = cell / d_grid_stride_z;
    uint32_t tx = cx / TILE_DIM;
    uint32_t ty = cy / TILE_DIM;
    uint32_t tz = cz / TILE_DIM;
    return tx + ty * tiles_per_dim + tz * tiles_per_dim * tiles_per_dim;
}

// Convert tile index to first cell in tile
// Uses device constants for runtime-scaled grids
__device__ __forceinline__ uint32_t tileToFirstCell(uint32_t tile) {
    int tiles_per_dim = d_grid_dim / TILE_DIM;
    uint32_t tx = tile % tiles_per_dim;
    uint32_t ty = (tile / tiles_per_dim) % tiles_per_dim;
    uint32_t tz = tile / (tiles_per_dim * tiles_per_dim);
    return (tx * TILE_DIM) + (ty * TILE_DIM) * d_grid_stride_y + (tz * TILE_DIM) * d_grid_stride_z;
}

// ============================================================================
// Device constants
// ============================================================================
// Define guard so physics.cu doesn't redefine these when included
#define PHYSICS_CONSTANTS_DEFINED

__device__ __constant__ float d_PI = 3.14159265358979f;
__device__ __constant__ float d_TWO_PI = 6.28318530717959f;
__device__ __constant__ float d_ISCO = 6.0f;
__device__ __constant__ float d_BH_MASS = 1.0f;
__device__ __constant__ float d_SCHW_R = 2.0f;
__device__ __constant__ float d_DISK_THICKNESS = 0.8f;
__device__ __constant__ float d_PHI = 1.6180339887498948f;
__device__ __constant__ float d_SCALE_RATIO = 1.6875f;
__device__ __constant__ float d_BIAS = 0.75f;
__device__ __constant__ float d_PHI_EXCESS = 0.09017f;

// Spiral arm topology parameters (Deepseek's experiment)
__device__ __constant__ int d_NUM_ARMS = 3;
__device__ __constant__ float d_ARM_WIDTH_DEG = 45.0f;
__device__ __constant__ float d_ARM_TRAP_STRENGTH = 0.15f;
__device__ __constant__ bool d_USE_ARM_TOPOLOGY = true;
__device__ __constant__ float d_ARM_BOOST_OVERRIDE = 0.0f;  // Test C: Override discrete boost

// Include topology.cuh here AFTER arm constants are defined
#include "topology.cuh"

// Natural growth: dynamic particle count tracking
// System can grow from seed population via spawning in coherent regions
__device__ unsigned int d_current_particle_count = 0;  // Current active particle count
__device__ unsigned int d_spawn_count = 0;             // Particles spawned this frame

// ============================================================================
// Derived Particle Properties — Now in disk.cuh
// ============================================================================
// compute_disk_r, compute_disk_phi, compute_temp, compute_in_disk
// are defined in disk.cuh (modular physics headers).
// Saves 13 bytes/particle × 10M = 130 MB VRAM and ~37 GB/s bandwidth.

// Morton Key / Octree Device Functions now in octree.cuh:
// expandBits21, morton64, xorCorner, fieldEnergy

// ============================================================================
// Cell Grid Device Functions — O(1) position-to-cell mapping (DNA layer)
// ============================================================================
// These replace Morton sorting + binary search with direct arithmetic.
// Forward-only: no sorting, no tree traversal, no binary search.

// O(1) position → cell index (replaces morton64 + sort + binary search)
// Uses d_grid_* device constants for runtime-scaled grids
__device__ __forceinline__ uint32_t cellIndexFromPos(float px, float py, float pz) {
    // Map [-250, 250] to [0, dim-1] integer coordinates
    uint32_t cx = (uint32_t)fminf(fmaxf((px + GRID_HALF_SIZE) / d_grid_cell_size, 0.f), (float)(d_grid_dim - 1));
    uint32_t cy = (uint32_t)fminf(fmaxf((py + GRID_HALF_SIZE) / d_grid_cell_size, 0.f), (float)(d_grid_dim - 1));
    uint32_t cz = (uint32_t)fminf(fmaxf((pz + GRID_HALF_SIZE) / d_grid_cell_size, 0.f), (float)(d_grid_dim - 1));
    return cx + cy * d_grid_stride_y + cz * d_grid_stride_z;
}

// Extract cell coordinates from linear index
__device__ __forceinline__ void cellCoords(uint32_t cell, uint32_t* cx, uint32_t* cy, uint32_t* cz) {
    *cx = cell % d_grid_dim;
    *cy = (cell / d_grid_stride_y) % d_grid_dim;
    *cz = cell / d_grid_stride_z;
}

// O(1) neighbor cell index — direct arithmetic (replaces binary search neighbor lookup)
// Returns cell index or UINT32_MAX if out of bounds
__device__ __forceinline__ uint32_t neighborCellIndex(uint32_t cell, int dx, int dy, int dz) {
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);

    // Check bounds
    int nx = (int)cx + dx;
    int ny = (int)cy + dy;
    int nz = (int)cz + dz;

    if (nx < 0 || nx >= d_grid_dim || ny < 0 || ny >= d_grid_dim || nz < 0 || nz >= d_grid_dim) {
        return UINT32_MAX;  // Out of bounds
    }

    return (uint32_t)nx + (uint32_t)ny * d_grid_stride_y + (uint32_t)nz * d_grid_stride_z;
}

// XOR Neighbor Lookup functions now in octree.cuh:
// mortonNeighbor, getNeighborKeys, findLeafByHash, findLeafByMorton

// ============================================================================
// Siphon Pump Kernel - Modular Physics Implementation
// ============================================================================
// Uses modular functions from .cuh headers via physics.cu
// ============================================================================
#include "physics.cu"

// ============================================================================
// Natural Growth Spawning Kernel - ENERGY-CONSERVING Star Formation
// ============================================================================
// Models gravitational collapse in coherent gas. When pump_history is high
// (indicating sustained pumping activity), the region is gravitationally bound
// and can form new stars.
//
// ENERGY CONSERVATION (per GPT's audit):
// Total energy E = T + S where:
//   T = 0.5 * m * v² (kinetic)
//   S = α * pump_scale + β * pump_history (pump/phase energy)
//
// When spawning:
//   E_parent_before = T_parent + S_parent
//   E_child = fraction of parent energy (split ratio)
//   E_parent_after = E_parent_before - E_child * (1 + tax)
//
// Parent velocity reduced: v_new = v_old * sqrt(1 - ε)
// Parent pump_scale reduced: scale_new = scale_old * (1 - ε)
// where ε = E_child / E_parent (including tax)

__global__ void spawnParticlesKernel(
    GPUDisk* disk,
    int N_current,           // Current active particle count
    int N_max,               // Maximum allowed particles
    unsigned int* spawn_idx, // Atomic counter for spawn slot allocation
    unsigned int* spawn_success, // V8-style: counts SUCCESSFUL spawns only
    float time,
    unsigned int seed,       // Per-frame random seed
    const uint8_t* __restrict__ in_active_region  // Step 3: skip passive parents
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_current || !particle_active(disk, i)) return;
    // Step 3d: passive parents don't spawn. Their pump_history is frozen
    // (siphon doesn't update them), so a stale high history could trigger
    // spawning from a settled particle. Also avoids a vel write race:
    // spawn reduces parent vel on spawn_stream while passive kernel writes
    // vel_y on default stream.
    if (in_active_region && !in_active_region[i]) return;

    // Only coherent particles can spawn (sustained pumping = gravitationally bound)
    float history = disk->pump_history[i];
    if (history < SPAWN_COHERENCE_THRESH) return;

    // Spawn probability scales with pump_scale (high D = more accretion = more spawn)
    float scale = disk->pump_scale[i];
    float spawn_prob = SPAWN_PROB_BASE * (1.0f + scale * SPAWN_SCALE_BOOST);

    // Simple LCG random for spawn decision (per-particle, per-frame)
    unsigned int rng = seed ^ (i * 1664525u + 1013904223u);
    rng = rng * 1664525u + 1013904223u;
    float rand_val = (float)(rng & 0xFFFFFF) / 16777216.0f;

    if (rand_val > spawn_prob) return;

    // === ENERGY BUDGET CHECK ===
    // Compute parent's current energy
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];
    float v2_parent = vx*vx + vy*vy + vz*vz;
    float T_parent = 0.5f * v2_parent;  // Kinetic (m=1)

    // Pump energy: α * scale + β * history (using α=0.1, β=0.05)
    float S_parent = 0.1f * scale + 0.05f * history;
    float E_parent = T_parent + S_parent;

    // Child gets 20% of parent's energy (after tax)
    float split_ratio = 0.2f;
    float E_child_raw = E_parent * split_ratio;
    float E_removed = E_child_raw * (1.0f + SPAWN_ENERGY_TAX);  // Include 10% tax

    // Check if parent has enough energy to spawn
    // Parent must retain at least SPAWN_MIN_PARENT_KE of original KE
    float E_parent_after = E_parent - E_removed;
    if (E_parent_after < E_parent * SPAWN_MIN_PARENT_KE) return;

    // === V8-STYLE SLOT ALLOCATION ===
    // Pre-check capacity before atomic increment (reduces contention)
    int available = N_max - N_current;
    if (available <= 0) return;

    // Allocate spawn slot atomically
    unsigned int slot = atomicAdd(spawn_idx, 1);

    // V8 pattern: reject if slot exceeds available capacity
    // This prevents phantom particles from being counted
    if (slot >= (unsigned int)available) return;

    int new_idx = N_current + slot;

    // === ENERGY-CONSERVING VELOCITY SPLIT ===
    // Parent loses kinetic energy: v_new = v_old * sqrt(1 - ε_kinetic)
    // where ε_kinetic = fraction of KE going to child
    float ke_fraction = (E_removed * 0.7f) / fmaxf(T_parent, 0.001f);  // 70% from KE
    ke_fraction = fminf(ke_fraction, 0.5f);  // Cap at 50% KE loss
    float vel_factor = sqrtf(1.0f - ke_fraction);

    // Reduce parent velocity (ENERGY COST)
    disk->vel_x[i] *= vel_factor;
    disk->vel_y[i] *= vel_factor;
    disk->vel_z[i] *= vel_factor;

    // Reduce parent pump_scale (ENERGY COST from pump energy)
    float pump_fraction = (E_removed * 0.3f) / fmaxf(S_parent, 0.001f);  // 30% from pump
    pump_fraction = fminf(pump_fraction, 0.3f);  // Cap at 30% pump loss
    disk->pump_scale[i] *= (1.0f - pump_fraction);
    disk->pump_history[i] *= (1.0f - pump_fraction * 0.5f);  // History decays slower

    // Generate small position offset (so offspring doesn't overlap parent)
    rng = rng * 1664525u + 1013904223u;
    float dx = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    rng = rng * 1664525u + 1013904223u;
    float dy = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.5f;
    rng = rng * 1664525u + 1013904223u;
    float dz = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;

    // Initialize offspring position
    disk->pos_x[new_idx] = disk->pos_x[i] + dx;
    disk->pos_y[new_idx] = disk->pos_y[i] + dy;
    disk->pos_z[new_idx] = disk->pos_z[i] + dz;

    // Child velocity: fraction of parent's (reduced) velocity with perturbation
    // Child gets sqrt(E_child_kinetic) worth of velocity
    float child_vel_scale = sqrtf(E_child_raw * 0.7f / fmaxf(T_parent, 0.001f));
    child_vel_scale = fminf(child_vel_scale, 0.5f);  // Cap at 50% of parent velocity

    rng = rng * 1664525u + 1013904223u;
    float vx_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;
    rng = rng * 1664525u + 1013904223u;
    float vy_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;
    rng = rng * 1664525u + 1013904223u;
    float vz_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;

    // Child velocity from parent's ORIGINAL velocity (before reduction)
    disk->vel_x[new_idx] = vx * child_vel_scale + vx_pert;
    disk->vel_y[new_idx] = vy * child_vel_scale + vy_pert;
    disk->vel_z[new_idx] = vz * child_vel_scale + vz_pert;

    // NOTE: disk_r, disk_phi, temp, in_disk no longer stored — computed on-demand

    // Child pump state: fraction of parent's energy goes to pump
    float child_pump_scale = scale * sqrtf(E_child_raw * 0.3f / fmaxf(S_parent, 0.001f));
    child_pump_scale = fminf(child_pump_scale, scale * 0.5f);  // Cap at 50% of parent scale

    disk->pump_state[new_idx] = 0;  // IDLE
    disk->pump_scale[new_idx] = fmaxf(child_pump_scale, 0.5f);  // Min 0.5
    disk->pump_coherent[new_idx] = 0;
    disk->pump_seam[new_idx] = 0;
    disk->pump_residual[new_idx] = 0.0f;
    disk->pump_work[new_idx] = 0.0f;
    disk->pump_history[new_idx] = history * 0.3f;  // Inherit 30% of parent's history
    disk->jet_phase[new_idx] = disk->jet_phase[i];  // Inherit parent's phase coherence

    // Kuramoto state: child inherits parent's theta + random offset, same ω
    // (keeps children loosely phase-coupled to parent, lets drift decorrelate)
    rng = rng * 1664525u + 1013904223u;
    float theta_offset = ((float)(rng & 0xFFFF) / 65536.0f) * 6.28318f;
    float parent_theta = disk->theta[i];
    disk->theta[new_idx] = fmodf(parent_theta + theta_offset, 6.28318f);
    disk->omega_nat[new_idx] = disk->omega_nat[i];

    // Activate the new particle
    disk->flags[new_idx] = PFLAG_ACTIVE;  // active, not ejected

    // V8-style: only count SUCCESSFUL spawns (after all writes complete)
    // This ensures spawn_success == actual initialized particles
    atomicAdd(spawn_success, 1);
}

// ============================================================================
// Instance Fill Kernel
// ============================================================================
// Helper functions for color interpolation
// ============================================================================
struct vec3 {
    float x, y, z;
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

__device__ vec3 mix(vec3 a, vec3 b, float t) {
    return vec3(
        a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
    );
}

// ============================================================================
// BLACKBODY RADIATION (Kuhlman approximation of Planckian Locus)
// Maps temperature (Kelvin) to RGB via Stefan-Boltzmann + Wien's law
// ============================================================================
__device__ float3 blackbody(float temp) {
    // Clamp temperature to visible range (1000K = deep red, 40000K = electric blue)
    temp = fminf(fmaxf(temp, 1000.0f), 40000.0f) / 100.0f;
    float3 color;

    // Red component
    if (temp <= 66.0f) {
        color.x = 255.0f;
    } else {
        color.x = temp - 60.0f;
        color.x = 329.698727446f * powf(color.x, -0.1332047592f);
    }

    // Green component
    if (temp <= 66.0f) {
        color.y = temp;
        color.y = 99.4708025861f * logf(color.y) - 161.1195681661f;
    } else {
        color.y = temp - 60.0f;
        color.y = 288.1221695283f * powf(color.y, -0.0755148492f);
    }

    // Blue component
    if (temp >= 66.0f) {
        color.z = 255.0f;
    } else if (temp <= 19.0f) {
        color.z = 0.0f;
    } else {
        color.z = temp - 10.0f;
        color.z = 138.5177312231f * logf(color.z) - 305.0447927307f;
    }

    // Normalize to [0,1] and clamp
    return make_float3(
        fminf(fmaxf(color.x / 255.0f, 0.0f), 1.0f),
                       fminf(fmaxf(color.y / 255.0f, 0.0f), 1.0f),
                       fminf(fmaxf(color.z / 255.0f, 0.0f), 1.0f)
    );
}

// ============================================================================
// Vulkan Particle Buffer Fill Kernel (Legacy - simple version)
// ============================================================================
// Packs GPUDisk arrays into ParticleVertex format for Vulkan rendering
// This kernel writes directly to a Vulkan-visible buffer via CUDA interop
#ifdef VULKAN_INTEROP
__global__ void fillVulkanParticleBuffer(
    ParticleVertex* output,  // Shared buffer (Vulkan-visible)
    const GPUDisk* disk,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Pack data into Vulkan vertex format
    // Inactive particles get zeroed (will be culled by alpha=0 in shader)
    if (particle_active(disk, i)) {
        float px = disk->pos_x[i];
        float py = disk->pos_y[i];
        float pz = disk->pos_z[i];
        output[i].position[0] = px;
        output[i].position[1] = py;
        output[i].position[2] = pz;
        output[i].pump_scale = disk->pump_scale[i];
        output[i].pump_residual = disk->pump_residual[i];
        // Compute temp on-demand (saves 4 bytes/particle storage)
        float r_cyl = compute_disk_r(px, pz);
        output[i].temp = compute_temp(r_cyl);
        output[i].velocity[0] = disk->vel_x[i];
        output[i].velocity[1] = disk->vel_y[i];
        output[i].velocity[2] = disk->vel_z[i];
        // Elongation from velocity magnitude (proxy for shear)
        float vel_mag = sqrtf(disk->vel_x[i]*disk->vel_x[i] +
                              disk->vel_y[i]*disk->vel_y[i] +
                              disk->vel_z[i]*disk->vel_z[i]);
        output[i].elongation = 1.0f + vel_mag * 0.01f;
    } else {
        // Zero out inactive particles
        output[i].position[0] = 0.0f;
        output[i].position[1] = 1e9f;  // Far away, culled
        output[i].position[2] = 0.0f;
        output[i].pump_scale = 0.0f;
        output[i].pump_residual = 1.0f;  // Max residual = invisible
        output[i].temp = 0.0f;
        output[i].velocity[0] = 0.0f;
        output[i].velocity[1] = 0.0f;
        output[i].velocity[2] = 0.0f;
        output[i].elongation = 0.0f;
    }
}

// VulkanSunTrace struct now in render_fill.cuh

// ============================================================================
// Phase-Primary Fill Kernel — GPUDisk → VulkanSunTrace
// ============================================================================
// Converts position-primary GPUDisk to phase-primary VulkanSunTrace for
// rendering via harmonic_phase.comp shader.
//
// This is the bridge between legacy physics (position-based) and
// phase-primary rendering. Eventually physics will be phase-primary too.

__global__ void fillVulkanSunTraceBuffer(
    VulkanSunTrace* output,  // Shared buffer (Vulkan-visible)
    const GPUDisk* disk,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (particle_active(disk, i)) {
        // Store actual positions directly for lossless rendering
        // (Reconstruction from phase state had precision issues causing diagonal artifacts)
        float px = disk->pos_x[i];
        float py = disk->pos_y[i];
        float pz = disk->pos_z[i];

        // Store positions directly in theta/omega/phase12 fields
        // Shader will read these as xyz without reconstruction
        output[i].theta = px;      // x position (was: orbital phase)
        output[i].omega = pz;      // z position (was: angular frequency)
        output[i].phase12 = py;    // y position (was: N12 phase)

        // Pack flags - active particle
        uint32_t flags = SUN_FLAG_ACTIVE;
        if (particle_ejected(disk, i)) flags |= SUN_FLAG_EJECTED;
        output[i].packed_state = ((uint32_t)disk->pump_seam[i] << 8) | (flags << 16);

        // Store rendering properties
        output[i].h1 = 1.0f;
        output[i].h3 = disk->pump_scale[i];
        output[i].coherence = 1.0f - disk->pump_residual[i];
        output[i].w_component = 0.0f;

        // Store radius for temperature calculation
        float r_cyl = sqrtf(px * px + pz * pz);
        output[i].drift = 0.0f;
        output[i].r_target = r_cyl;
    } else {
        // Inactive particle — set flags to show inactive
        output[i].theta = 0.0f;
        output[i].omega = 0.0f;
        output[i].phase12 = 0.0f;
        output[i].packed_state = 0;  // No ACTIVE flag = skip in shader
        output[i].h1 = 0.0f;
        output[i].h3 = 0.0f;
        output[i].drift = 0.0f;
        output[i].coherence = 0.0f;
        output[i].w_component = 0.0f;
        output[i].r_target = 0.0f;
    }
}

// ============================================================================
// Hybrid LOD Particle Fill Kernel
// ============================================================================
// Computes distance-based LOD, fills particle buffer for NEAR/MID particles,
// and accumulates FAR particles into density grid for volumetric rendering.
//
// LOD Zones (with smooth blending):
//   NEAR  (dist < nearThreshold): Full point rendering, alpha = 1.0
//   MID   (near < dist < far):    Blended, point alpha fades, contributes to volume
//   FAR   (dist > farThreshold):  Volume only, point alpha = 0.0
//
// Uses atomicAdd for density grid accumulation (128³ voxels)

// LOD constants and lod_smoothstep now in render_fill.cuh

__global__ void fillVulkanParticleBufferLOD(
    ParticleVertex* output,      // Shared particle buffer (Vulkan-visible)
    float* densityGrid,          // 128³ density grid (4 floats per voxel: scale_sum, temp_sum, count, coherence)
    const GPUDisk* disk,
    int N,
    float camX, float camY, float camZ,  // Camera position
    float nearThreshold,         // Distance below which = full points (default: 150)
    float farThreshold,          // Distance above which = volume only (default: 600)
    float volumeScale,           // World-space extent of density grid (default: 300)
    unsigned int* nearCount      // Atomic counter for near particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip inactive particles entirely
    if (!particle_active(disk, i)) {
        // Zero out this slot
        output[i].position[0] = 0.0f;
        output[i].position[1] = 1e9f;  // Far away, culled by depth
        output[i].position[2] = 0.0f;
        output[i].pump_scale = 0.0f;
        output[i].pump_residual = 1.0f;  // Max residual = invisible
        output[i].temp = 0.0f;
        output[i].velocity[0] = 0.0f;
        output[i].velocity[1] = 0.0f;
        output[i].velocity[2] = 0.0f;
        output[i].elongation = 0.0f;
        return;
    }

    // Get particle position
    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    // Compute distance to camera
    float dx = px - camX;
    float dy = py - camY;
    float dz = pz - camZ;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // Compute LOD weight (1.0 = full point, 0.0 = volume only)
    // Uses smooth blending between thresholds
    float pointWeight = 1.0f - lod_smoothstep(nearThreshold, farThreshold, dist);

    // Get particle data
    float pump_scale = disk->pump_scale[i];
    float pump_residual = disk->pump_residual[i];
    // Compute temp on-demand (saves 4 bytes/particle storage)
    float r_cyl = compute_disk_r(px, pz);
    float temp = compute_temp(r_cyl);
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // Coherence factor: LOCKED particles (pump_state == 1) are coherent
    // This could be used to render coherent particles volumetrically earlier
    float coherence = (disk->pump_state[i] == 1) ? 1.0f : 0.0f;

    // === POINT RENDERING (NEAR/MID zones) ===
    // Only fill particle buffer if pointWeight > 0
    if (pointWeight > 0.01f) {
        output[i].position[0] = px;
        output[i].position[1] = py;
        output[i].position[2] = pz;
        output[i].pump_scale = pump_scale;
        // Encode pointWeight into residual's sign or use elongation
        // For smooth blending, we'll modulate alpha in shader via elongation
        output[i].pump_residual = pump_residual;
        output[i].temp = temp;
        output[i].velocity[0] = vx;
        output[i].velocity[1] = vy;
        output[i].velocity[2] = vz;
        // Encode LOD weight in elongation (shader will use this for alpha blend)
        float vel_mag = sqrtf(vx*vx + vy*vy + vz*vz);
        output[i].elongation = pointWeight * (1.0f + vel_mag * 0.01f);

        // Count near particles for stats
        if (dist < nearThreshold) {
            atomicAdd(nearCount, 1);
        }
    } else {
        // Far particle: hide from point rendering
        output[i].position[0] = 0.0f;
        output[i].position[1] = 1e9f;
        output[i].position[2] = 0.0f;
        output[i].pump_scale = 0.0f;
        output[i].pump_residual = 1.0f;
        output[i].temp = 0.0f;
        output[i].velocity[0] = 0.0f;
        output[i].velocity[1] = 0.0f;
        output[i].velocity[2] = 0.0f;
        output[i].elongation = 0.0f;
    }

    // === DENSITY GRID ACCUMULATION (MID/FAR zones) ===
    // Weight contribution by (1 - pointWeight) so volume fades in as points fade out
    float volumeWeight = 1.0f - pointWeight;
    if (volumeWeight > 0.01f) {
        // Map world position to grid coordinates
        // Grid is centered at origin, spans [-volumeScale/2, volumeScale/2]
        float halfScale = volumeScale * 0.5f;
        float nx = (px + halfScale) / volumeScale;  // Normalize to [0, 1]
        float ny = (py + halfScale) / volumeScale;
        float nz = (pz + halfScale) / volumeScale;

        // Clamp to grid bounds
        if (nx >= 0.0f && nx < 1.0f && ny >= 0.0f && ny < 1.0f && nz >= 0.0f && nz < 1.0f) {
            int gx = (int)(nx * LOD_GRID_SIZE);
            int gy = (int)(ny * LOD_GRID_SIZE);
            int gz = (int)(nz * LOD_GRID_SIZE);

            // Clamp to valid range
            gx = min(max(gx, 0), LOD_GRID_SIZE - 1);
            gy = min(max(gy, 0), LOD_GRID_SIZE - 1);
            gz = min(max(gz, 0), LOD_GRID_SIZE - 1);

            // Linear index into density grid (4 floats per voxel)
            int voxelIdx = (gz * LOD_GRID_SIZE * LOD_GRID_SIZE + gy * LOD_GRID_SIZE + gx) * 4;

            // Atomic accumulate (weighted by volumeWeight for smooth transition)
            atomicAdd(&densityGrid[voxelIdx + 0], pump_scale * volumeWeight);  // scale_sum
            atomicAdd(&densityGrid[voxelIdx + 1], temp * volumeWeight);         // temp_sum
            atomicAdd(&densityGrid[voxelIdx + 2], volumeWeight);                // count (weighted)
            atomicAdd(&densityGrid[voxelIdx + 3], coherence * volumeWeight);    // coherence_sum
        }
    }
}

// Clear density grid kernel (call before fillVulkanParticleBufferLOD)
__global__ void clearDensityGrid(float* densityGrid, int numVoxels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numVoxels * 4) {
        densityGrid[i] = 0.0f;
    }
}

// ============================================================================
// Stream Compaction Kernel for Indirect Draw
// ============================================================================
// This kernel compacts only visible particles (NEAR/MID) into a contiguous
// buffer, allowing Vulkan to skip vertex processing for FAR particles entirely.
// Uses atomic counter for write index - true parallel stream compaction.

// CUDADrawIndirectCommand and HYBRID_R now in render_fill.cuh

// Stream compaction kernel: filters particles by r < HYBRID_R and compacts output
__global__ void compactVisibleParticles(
    ParticleVertex* compactedOutput,       // Compacted output buffer (r < HYBRID_R only)
    CUDADrawIndirectCommand* drawCommand,  // Indirect draw command (unused here)
    float* densityGrid,                  // 128³ density grid (unused with analytic shells)
    const GPUDisk* disk,
    int N,
    float camX, float camY, float camZ,    // Camera pos (unused now, kept for API compat)
    float nearThreshold, float farThreshold, float volumeScale,  // Unused, kept for API
    unsigned int* writeIndex,            // Atomic counter for compaction
    float* maxRadiusOut                  // Debug: track max radius written
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip inactive particles
    if (!particle_active(disk, i)) return;

    // Get particle position
    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    // Radius from origin (black hole center) — the PHYSICAL hybrid boundary
    float r = sqrtf(px*px + py*py + pz*pz);

    // === POINT RENDERING: Only particles inside hybrid boundary ===
    // r < 30 = active pumping region, rendered as particles
    // r > 30 = analytic field region, handled by volume_shells.frag
    if (r < HYBRID_R) {
        // Get particle data
        float pump_scale = disk->pump_scale[i];
        float pump_residual = disk->pump_residual[i];
        // Compute temp on-demand (saves 4 bytes/particle storage)
        float temp = compute_temp(r);  // r is already r_cyl computed above
        float vx = disk->vel_x[i];
        float vy = disk->vel_y[i];
        float vz = disk->vel_z[i];

        // Atomically get write index
        unsigned int writeIdx = atomicAdd(writeIndex, 1);

        // Debug: track max radius being written (atomic max via CAS)
        if (maxRadiusOut) {
            float oldMax = *maxRadiusOut;
            while (r > oldMax) {
                float assumed = oldMax;
                oldMax = __int_as_float(atomicCAS((int*)maxRadiusOut,
                    __float_as_int(assumed), __float_as_int(r)));
                if (oldMax == assumed) break;
            }
        }

        // Write compacted particle
        compactedOutput[writeIdx].position[0] = px;
        compactedOutput[writeIdx].position[1] = py;
        compactedOutput[writeIdx].position[2] = pz;
        compactedOutput[writeIdx].pump_scale = pump_scale;
        compactedOutput[writeIdx].pump_residual = pump_residual;
        compactedOutput[writeIdx].temp = temp;
        compactedOutput[writeIdx].velocity[0] = vx;
        compactedOutput[writeIdx].velocity[1] = vy;
        compactedOutput[writeIdx].velocity[2] = vz;
        // Full opacity for inner region particles
        float vel_mag = sqrtf(vx*vx + vy*vy + vz*vz);
        compactedOutput[writeIdx].elongation = 1.0f + vel_mag * 0.01f;
    }
}

// Kernel to update the indirect draw command with the final count
// Called after compactVisibleParticles completes (minimal overhead - just 4 writes)
__global__ void updateIndirectDrawCommand(
    CUDADrawIndirectCommand* drawCommand,
    unsigned int* writeIndex
) {
    drawCommand->vertexCount = 1;  // 1 vertex per instance (point)
    drawCommand->instanceCount = *writeIndex;  // Number of visible particles
    drawCommand->firstVertex = 0;
    drawCommand->firstInstance = 0;
}
#endif

// ============================================================================
// Octree Kernels - Morton key assignment and tree building
// ============================================================================

// Assign Morton keys to particles for spatial sorting
// Active inner particles (r < HYBRID_R) get real keys, outer particles get max key
__global__ void assignMortonKeys(
    GPUDisk* disk,
    uint64_t* morton_keys,
    uint32_t* xor_corners,
    uint32_t* particle_ids,
    int N,
    float boxSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];
    float r  = sqrtf(px*px + py*py + pz*pz);

    // Outer/inactive particles get max key — sort to end
    if (!particle_active(disk, i) || r >= HYBRID_R) {
        morton_keys[i]  = 0xFFFFFFFFFFFFFFFFULL;
        xor_corners[i]  = 0xFFFFFFFF;
        particle_ids[i] = i;
        return;
    }

    // Map position to integer coordinates
    uint32_t ix = (uint32_t)fminf(fmaxf((px / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iy = (uint32_t)fminf(fmaxf((py / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iz = (uint32_t)fminf(fmaxf((pz / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));

    morton_keys[i]  = (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
    xor_corners[i]  = ix ^ iy ^ iz;
    particle_ids[i] = i;
}

// Build frozen analytic tree (levels 0-5) - run once at init
// Creates nodes for outer region (r > HYBRID_R) with field-derived energy
__global__ void buildAnalyticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    float boxSize,
    float pump_phase,
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all potential nodes at all analytic levels
    for (int level = 0; level <= max_level; level++) {
        int nodes_per_axis = 1 << level;
        int total = nodes_per_axis * nodes_per_axis * nodes_per_axis;
        if (idx >= total) continue;

        // Decode node coordinates from linear index
        int iz =  idx % nodes_per_axis;
        int iy = (idx / nodes_per_axis) % nodes_per_axis;
        int ix =  idx / (nodes_per_axis * nodes_per_axis);

        // Compute node center and half-size
        float half = boxSize / (2.0f * nodes_per_axis);
        float cx = (-boxSize * 0.5f) + (ix + 0.5f) * 2.0f * half;
        float cy = (-boxSize * 0.5f) + (iy + 0.5f) * 2.0f * half;
        float cz = (-boxSize * 0.5f) + (iz + 0.5f) * 2.0f * half;
        float r  = sqrtf(cx*cx + cy*cy + cz*cz);

        // Skip nodes entirely inside stochastic region (r + diagonal < HYBRID_R)
        if (r + half * 1.732f < HYBRID_R) continue;

        // Classify regime based on relationship to HYBRID_R boundary
        uint8_t regime;
        if (r - half * 1.732f > HYBRID_R)
            regime = REGIME_ANALYTIC;      // Entirely outside
        else
            regime = REGIME_BOUNDARY;       // Straddles boundary

        // Atomically allocate node slot
        uint32_t node_idx = atomicAdd(node_count, 1);
        if (node_idx >= OCTREE_MAX_NODES) return;

        // Populate node
        OctreeNode& node = nodes[node_idx];
        node.morton_key     = (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
        node.xor_corner     = ix ^ iy ^ iz;
        node.particle_start = 0;
        node.particle_count = 0;
        node.energy         = fieldEnergy(cx, cy, cz, pump_phase);
        node.center_x       = cx;
        node.center_y       = cy;
        node.center_z       = cz;
        node.half_size      = half;
        node.level          = (uint8_t)level;
        node.regime         = regime;
        node.children_mask  = 0;  // Will be filled in linking pass
        node.padding        = 0;
    }
}

// ============================================================================
// STOCHASTIC TREE BUILD — Levels 6-13 from Morton-sorted particle spans
// Rebuilt every 30 frames alongside Morton sort
// ============================================================================
__global__ void buildStochasticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    const uint64_t* morton_keys,  // Sorted
    uint32_t num_active,
    float boxSize,
    int start_level,              // 6
    int max_level                 // 13
) {
    // Each thread handles one sorted particle position
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active) return;

    uint64_t my_key = morton_keys[i];

    // For each stochastic level, check if this is a boundary
    for (int level = start_level; level <= max_level; level++) {
        // Parent mask: keep bits for levels 0 to level-1
        uint64_t parent_mask = ~((1ULL << (3 * level)) - 1);
        uint64_t my_parent = my_key & parent_mask;

        // Am I the first particle in my level-L cell?
        bool is_boundary = (i == 0) ||
            ((morton_keys[i - 1] & parent_mask) != my_parent);

        if (!is_boundary) continue;

        // Find span end: where parent key changes
        uint32_t span_end = i + 1;
        while (span_end < num_active &&
               (morton_keys[span_end] & parent_mask) == my_parent) {
            span_end++;
        }

        // Allocate node
        uint32_t node_idx = atomicAdd(node_count, 1);
        if (node_idx >= OCTREE_MAX_NODES) return;

        // Decode ix, iy, iz from Morton key at this level
        // Morton key encodes x,y,z interleaved: ...x2y2z2 x1y1z1 x0y0z0
        // Extract level bits by masking and deinterleaving
        uint32_t ix = 0, iy = 0, iz = 0;
        for (int b = 0; b < level; b++) {
            int bit_pos = 3 * b;
            iz |= ((my_key >> bit_pos) & 1) << b;
            iy |= ((my_key >> (bit_pos + 1)) & 1) << b;
            ix |= ((my_key >> (bit_pos + 2)) & 1) << b;
        }

        int nodes_per_axis = 1 << level;
        float half = boxSize / (2.0f * nodes_per_axis);
        float cx = (-boxSize * 0.5f) + (ix + 0.5f) * 2.0f * half;
        float cy = (-boxSize * 0.5f) + (iy + 0.5f) * 2.0f * half;
        float cz = (-boxSize * 0.5f) + (iz + 0.5f) * 2.0f * half;

        OctreeNode& node = nodes[node_idx];
        node.morton_key     = my_parent;  // Parent-level key for this cell
        node.xor_corner     = ix ^ iy ^ iz;
        node.particle_start = i;
        node.particle_count = span_end - i;
        node.energy         = 0.0f;  // Stochastic: energy from particles, not field
        node.center_x       = cx;
        node.center_y       = cy;
        node.center_z       = cz;
        node.half_size      = half;
        node.level          = (uint8_t)level;
        node.regime         = REGIME_STOCHASTIC;
        node.children_mask  = 0;
        node.padding        = 0;
    }
}

// NOTE: octreeRenderTraversal V1 removed — use octreeRenderTraversalV3 instead

// ============================================================================
// EXTRACT LEAF NODE COUNTS — For prefix scan to eliminate atomic contention
// Writes particle_count and node index for each level-13 node into compact arrays
// ============================================================================
__global__ void extractLeafNodeCounts(
    uint32_t* leaf_counts,           // Output: particle counts for level-13 nodes
    uint32_t* leaf_node_indices,     // Output: original node indices for level-13 nodes
    uint32_t* leaf_node_count,       // Output: number of level-13 nodes found
    const OctreeNode* nodes,
    uint32_t total_nodes,
    uint32_t analytic_count,
    int target_level                 // 13
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x + analytic_count;
    if (node_idx >= total_nodes) return;

    const OctreeNode& node = nodes[node_idx];

    // Only count stochastic nodes at target level
    if (node.regime != REGIME_STOCHASTIC) return;
    if (node.level != target_level) return;

    // Allocate slot in leaf array
    uint32_t leaf_idx = atomicAdd(leaf_node_count, 1);
    leaf_counts[leaf_idx] = node.particle_count;
    leaf_node_indices[leaf_idx] = node_idx;
}

// ============================================================================
// BUILD LEAF HASH TABLE — O(1) neighbor lookup (replaces binary search)
// ============================================================================
// Inserts morton_key → leaf_idx mappings into a hash table with linear probing.
// Call after extractLeafNodeCounts populates leaf_node_indices.
// Hash table must be pre-cleared to 0xFF (UINT64_MAX = empty marker).

__global__ void buildLeafHashTable(
    uint64_t* hash_keys,             // Output: hash table keys
    uint32_t* hash_values,           // Output: hash table values (leaf indices)
    uint32_t hash_size,              // Hash table size (power of 2)
    uint32_t hash_mask,              // hash_size - 1 (for fast modulo)
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    uint32_t num_leaves
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    unsigned long long key = (unsigned long long)nodes[node_idx].morton_key;

    // Linear probing insert
    uint32_t slot = (uint32_t)(key & hash_mask);
    for (int probe = 0; probe < 64; probe++) {  // Max 64 probes
        unsigned long long old = atomicCAS((unsigned long long*)&hash_keys[slot],
                                            (unsigned long long)UINT64_MAX, key);
        if (old == (unsigned long long)UINT64_MAX || old == key) {
            // Successfully inserted or key already exists
            hash_values[slot] = leaf_idx;
            return;
        }
        // Collision - try next slot
        slot = (slot + 1) & hash_mask;
    }
    // Should never reach here with 50% load factor
}

// ============================================================================
// LEAF VELOCITY ACCUMULATION — Computes average velocity per leaf node
// ============================================================================
// For vorticity computation, we need velocity at each cell. This kernel
// accumulates particle velocities per leaf using atomics, then divides by count.

__global__ void accumulateLeafVelocities(
    float* leaf_vel_x,               // Output: sum of vx per leaf (then averaged)
    float* leaf_vel_y,
    float* leaf_vel_z,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint32_t* particle_ids,
    const GPUDisk* disk,
    uint32_t num_leaves
) {
    // Each block handles one leaf, threads handle particles within
    int leaf_idx = blockIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    if (node.particle_count == 0) return;

    // Shared memory for warp reduction
    __shared__ float s_vx[256];
    __shared__ float s_vy[256];
    __shared__ float s_vz[256];

    int tid = threadIdx.x;
    float vx_sum = 0.0f, vy_sum = 0.0f, vz_sum = 0.0f;

    // Each thread accumulates multiple particles if needed
    for (int p = tid; p < node.particle_count; p += blockDim.x) {
        uint32_t sorted_pos = node.particle_start + p;
        uint32_t orig_idx = particle_ids[sorted_pos];

        vx_sum += disk->vel_x[orig_idx];
        vy_sum += disk->vel_y[orig_idx];
        vz_sum += disk->vel_z[orig_idx];
    }

    s_vx[tid] = vx_sum;
    s_vy[tid] = vy_sum;
    s_vz[tid] = vz_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_vx[tid] += s_vx[tid + stride];
            s_vy[tid] += s_vy[tid + stride];
            s_vz[tid] += s_vz[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes average
    if (tid == 0) {
        float inv_count = 1.0f / (float)node.particle_count;
        leaf_vel_x[leaf_idx] = s_vx[0] * inv_count;
        leaf_vel_y[leaf_idx] = s_vy[0] * inv_count;
        leaf_vel_z[leaf_idx] = s_vz[0] * inv_count;
    }
}

// FrustumPlanes struct, extractFrustumPlanes, sphereInFrustum now in octree.cuh

// Cull leaf nodes against frustum, zero out particle_count for culled nodes
// Operates on the leaf_counts array (copied from extractLeafNodeCounts output)
__global__ void cullLeafNodesFrustum(
    uint32_t* leaf_counts,              // In/out: particle counts (zeroed if culled)
    const uint32_t* leaf_node_indices,  // Node indices for each leaf
    const OctreeNode* nodes,
    uint32_t num_leaves,
    FrustumPlanes frustum
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    // Bounding sphere: center at node center, radius = half_size * sqrt(3)
    float radius = node.half_size * 1.732051f;

    if (!sphereInFrustum(node.center_x, node.center_y, node.center_z, radius, frustum)) {
        leaf_counts[leaf_idx] = 0;  // Cull this node
    }
    // Note: if visible, leaf_counts already has correct particle_count from extraction
}

#ifdef VULKAN_INTEROP
// ============================================================================
// OCTREE RENDER TRAVERSAL V3 — Warp-cooperative with frustum culling support
// Binary search finds leaf for each output position, handles culled leaves.
// ============================================================================
__global__ void octreeRenderTraversalV3(
    ParticleVertex* compactedOutput,
    const uint32_t* leaf_offsets,        // Exclusive scan of culled counts
    const uint32_t* leaf_node_indices,   // Node index for each leaf
    const OctreeNode* nodes,
    const uint32_t* particle_ids,        // Morton-sorted particle indices
    const GPUDisk* disk,
    uint32_t num_leaves,
    uint32_t total_particles             // Sum of visible particles after culling
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_particles) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_mask = __activemask();

    // Warp-cooperative: lane 0 searches, broadcasts result
    int warp_base_out = out_idx - lane;
    int leaf_idx;

    if (lane == 0) {
        // Binary search: find leaf where offset <= warp_base_out < next_offset
        int lo = 0, hi = num_leaves;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (leaf_offsets[mid] <= warp_base_out) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        leaf_idx = lo - 1;
    }
    leaf_idx = __shfl_sync(warp_mask, leaf_idx, 0);

    // Linear search forward from warp's base leaf
    while (leaf_idx + 1 < (int)num_leaves && leaf_offsets[leaf_idx + 1] <= out_idx) {
        leaf_idx++;
    }

    // Get node and compute particle offset within node
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];
    uint32_t p = out_idx - leaf_offsets[leaf_idx];

    // Lookup original particle index
    uint32_t sorted_pos = node.particle_start + p;
    uint32_t orig_idx = particle_ids[sorted_pos];

    // Read and write particle data
    float px = disk->pos_x[orig_idx];
    float pz = disk->pos_z[orig_idx];
    compactedOutput[out_idx].position[0] = px;
    compactedOutput[out_idx].position[1] = disk->pos_y[orig_idx];
    compactedOutput[out_idx].position[2] = pz;
    compactedOutput[out_idx].pump_scale = disk->pump_scale[orig_idx];
    compactedOutput[out_idx].pump_residual = disk->pump_residual[orig_idx];
    // Compute temp on-demand (saves 4 bytes/particle storage)
    compactedOutput[out_idx].temp = compute_temp(compute_disk_r(px, pz));
    float vx = disk->vel_x[orig_idx];
    float vy = disk->vel_y[orig_idx];
    float vz = disk->vel_z[orig_idx];
    compactedOutput[out_idx].velocity[0] = vx;
    compactedOutput[out_idx].velocity[1] = vy;
    compactedOutput[out_idx].velocity[2] = vz;
    compactedOutput[out_idx].elongation = 1.0f + sqrtf(vx*vx + vy*vy + vz*vz) * 0.01f;
}
#endif  // VULKAN_INTEROP (octree render traversal kernels)

// ============================================================================
// OCTREE PHYSICS KERNEL — XOR neighbor lookup for local stress gradients
// ============================================================================
// For each leaf node, computes stress gradient from 6 face-adjacent neighbors.
// Stress gradient drives particle interactions (pressure, viscosity, etc.)
//
// This kernel operates on nodes, not particles. Each node accumulates:
//   - Neighbor density differences → pressure gradient
//   - Neighbor energy differences → heat flow
//   - Neighbor velocity differences → shear stress
//
// The gradients are stored back in the node for the particle physics kernel
// to use when updating individual particle velocities.

// Compute density gradient for a single leaf node
// Returns gradient vector via output parameters
// Uses O(1) hash lookup instead of O(log N) binary search
__device__ void computeLeafGradient(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx,
    float* grad_x, float* grad_y, float* grad_z
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    *grad_x = 0.0f;
    *grad_y = 0.0f;
    *grad_z = 0.0f;

    if (node.regime != REGIME_STOCHASTIC || node.particle_count == 0) return;

    // Get 6 face-adjacent neighbor keys
    uint64_t neighbor_keys[6];
    getNeighborKeys(node.morton_key, node.level, neighbor_keys);

    float our_density = (float)node.particle_count;
    int neighbor_count = 0;

    // Accumulate gradient: +X, -X, +Y, -Y, +Z, -Z
    float grad[3] = {0.0f, 0.0f, 0.0f};

    for (int n = 0; n < 6; n++) {
        if (neighbor_keys[n] == UINT64_MAX) continue;  // Boundary

        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf_idx = findLeafByHash(
            hash_keys, hash_values, hash_mask, neighbor_keys[n]
        );

        if (neighbor_leaf_idx == UINT32_MAX) continue;  // Empty cell

        uint32_t neighbor_node_idx = leaf_node_indices[neighbor_leaf_idx];
        float their_density = (float)nodes[neighbor_node_idx].particle_count;

        neighbor_count++;

        // n=0: +X, n=1: -X, n=2: +Y, n=3: -Y, n=4: +Z, n=5: -Z
        int axis = n / 2;
        int sign = (n % 2 == 0) ? 1 : -1;
        grad[axis] += sign * (their_density - our_density);
    }

    // Normalize by cell size
    if (neighbor_count > 0) {
        float cell_size = node.half_size * 2.0f;
        float inv_2h = 1.0f / (2.0f * cell_size);
        *grad_x = grad[0] * inv_2h;
        *grad_y = grad[1] * inv_2h;
        *grad_z = grad[2] * inv_2h;
    }
}

// ============================================================================
// VORTICITY COMPUTATION — Curl of velocity field: ω = ∇ × v
// ============================================================================
// Vorticity measures local rotation in the velocity field.
// ω = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
// This enables spiral arm formation, circulation, and turbulence.

__device__ void computeLeafVorticity(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const float* leaf_vel_x,
    const float* leaf_vel_y,
    const float* leaf_vel_z,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx,
    float* omega_x, float* omega_y, float* omega_z
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    *omega_x = 0.0f;
    *omega_y = 0.0f;
    *omega_z = 0.0f;

    if (node.regime != REGIME_STOCHASTIC || node.particle_count == 0) return;

    // Get 6 face-adjacent neighbor keys
    uint64_t neighbor_keys[6];
    getNeighborKeys(node.morton_key, node.level, neighbor_keys);

    // Velocity at this cell
    float vx0 = leaf_vel_x[leaf_idx];
    float vy0 = leaf_vel_y[leaf_idx];
    float vz0 = leaf_vel_z[leaf_idx];

    // Neighbor velocities: [+X, -X, +Y, -Y, +Z, -Z]
    float vx[6], vy[6], vz[6];
    bool have[6] = {false, false, false, false, false, false};

    for (int n = 0; n < 6; n++) {
        if (neighbor_keys[n] == UINT64_MAX) continue;  // Boundary

        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf_idx = findLeafByHash(
            hash_keys, hash_values, hash_mask, neighbor_keys[n]
        );

        if (neighbor_leaf_idx == UINT32_MAX) continue;  // Empty cell

        vx[n] = leaf_vel_x[neighbor_leaf_idx];
        vy[n] = leaf_vel_y[neighbor_leaf_idx];
        vz[n] = leaf_vel_z[neighbor_leaf_idx];
        have[n] = true;
    }

    float cell_size = node.half_size * 2.0f;
    float inv_2h = 1.0f / (2.0f * cell_size);

    // Compute partial derivatives using central differences where possible
    // ∂v/∂x = (v[+X] - v[-X]) / (2h)
    // Only compute off-diagonal terms needed for curl (vorticity)
    float dvy_dx = 0.0f, dvz_dx = 0.0f;
    float dvx_dy = 0.0f, dvz_dy = 0.0f;
    float dvx_dz = 0.0f, dvy_dz = 0.0f;

    // X derivatives (neighbors 0=+X, 1=-X) - need dvy_dx, dvz_dx for curl
    if (have[0] && have[1]) {
        dvy_dx = (vy[0] - vy[1]) * inv_2h;
        dvz_dx = (vz[0] - vz[1]) * inv_2h;
    } else if (have[0]) {
        dvy_dx = (vy[0] - vy0) / cell_size;
        dvz_dx = (vz[0] - vz0) / cell_size;
    } else if (have[1]) {
        dvy_dx = (vy0 - vy[1]) / cell_size;
        dvz_dx = (vz0 - vz[1]) / cell_size;
    }

    // Y derivatives (neighbors 2=+Y, 3=-Y) - need dvx_dy, dvz_dy for curl
    if (have[2] && have[3]) {
        dvx_dy = (vx[2] - vx[3]) * inv_2h;
        dvz_dy = (vz[2] - vz[3]) * inv_2h;
    } else if (have[2]) {
        dvx_dy = (vx[2] - vx0) / cell_size;
        dvz_dy = (vz[2] - vz0) / cell_size;
    } else if (have[3]) {
        dvx_dy = (vx0 - vx[3]) / cell_size;
        dvz_dy = (vz0 - vz[3]) / cell_size;
    }

    // Z derivatives (neighbors 4=+Z, 5=-Z) - need dvx_dz, dvy_dz for curl
    if (have[4] && have[5]) {
        dvx_dz = (vx[4] - vx[5]) * inv_2h;
        dvy_dz = (vy[4] - vy[5]) * inv_2h;
    } else if (have[4]) {
        dvx_dz = (vx[4] - vx0) / cell_size;
        dvy_dz = (vy[4] - vy0) / cell_size;
    } else if (have[5]) {
        dvx_dz = (vx0 - vx[5]) / cell_size;
        dvy_dz = (vy0 - vy[5]) / cell_size;
    }

    // Curl: ω = ∇ × v
    // ωx = ∂vz/∂y - ∂vy/∂z
    // ωy = ∂vx/∂z - ∂vz/∂x
    // ωz = ∂vy/∂x - ∂vx/∂y
    *omega_x = dvz_dy - dvy_dz;
    *omega_y = dvx_dz - dvz_dx;
    *omega_z = dvy_dx - dvx_dy;
}

// Compute phase coherence with neighbors (for pressure modulation)
// Returns average cos(Δphase): 1 = in sync, 0 = random, -1 = anti-phase
__device__ float computePhaseCoherence(
    const float* leaf_phase,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    float my_phase = leaf_phase[leaf_idx];
    uint64_t my_key = node.morton_key;
    int level = node.level;

    uint64_t neighbor_keys[6];
    getNeighborKeys(my_key, level, neighbor_keys);

    float coherence_sum = 0.0f;
    int neighbor_count = 0;

    for (int n = 0; n < 6; n++) {
        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf = findLeafByHash(hash_keys, hash_values,
                                                 hash_mask, neighbor_keys[n]);
        if (neighbor_leaf != UINT32_MAX) {
            float neighbor_phase = leaf_phase[neighbor_leaf];
            coherence_sum += cuda_lut_cos(neighbor_phase - my_phase);
            neighbor_count++;
        }
    }

    if (neighbor_count == 0) return 0.0f;
    return coherence_sum / (float)neighbor_count;
}

// ============================================================================
// STREAMING CELL GRID KERNELS (DNA/RNA Forward-Pass Architecture)
// ============================================================================
// These kernels replace octree rebuild + binary search with forward-only passes.
// No sorting, no binary search, no rebuild. Pure streaming.
//
// Pass 1: scatterParticlesToCells — particles write to cells (atomic accumulation)
// Pass 2: computeCellFields — cells compute gradients/vorticity (fixed stencil)
// Pass 3: gatherCellForcesToParticles — particles read from cells (direct lookup)

// Clear cell grid state before scatter pass (used for cadence mode)
__global__ void clearCellGrid(
    float* density,
    float* momentum_x,
    float* momentum_y,
    float* momentum_z,
    float* phase_sin,
    float* phase_cos,
    float* pressure_x,
    float* pressure_y,
    float* pressure_z,
    float* vorticity_x,
    float* vorticity_y,
    float* vorticity_z
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= d_grid_cells) return;

    density[cell] = 0.0f;
    momentum_x[cell] = 0.0f;
    momentum_y[cell] = 0.0f;
    momentum_z[cell] = 0.0f;
    phase_sin[cell] = 0.0f;
    phase_cos[cell] = 0.0f;
    pressure_x[cell] = 0.0f;
    pressure_y[cell] = 0.0f;
    pressure_z[cell] = 0.0f;
    vorticity_x[cell] = 0.0f;
    vorticity_y[cell] = 0.0f;
    vorticity_z[cell] = 0.0f;
}

// Pass 1: Scatter particles to cells (forward-only atomic accumulation)
// Each particle computes its cell via O(1) hash and accumulates state
__global__ void scatterParticlesToCells(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip inactive particles (but still mark their cell as invalid)
    if (!particle_active(disk, i)) {
        particle_cell[i] = UINT32_MAX;
        return;
    }

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    // O(1) cell assignment — no Morton sort, no binary search
    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;  // Store for Pass 3

    // Atomic accumulation into cell state
    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    // Phase state for Kuramoto coupling (math.md Step 8)
    // Uses the dedicated theta[] field — a continuous rotation, not a pulse.
    // phase_sin/phase_cos accumulate ∑sin(θ_i) and ∑cos(θ_i) per cell, so
    // that the gather kernel can read the mean-field ⟨e^{iθ}⟩ as R_local
    // and apply Kuramoto coupling K·sin(θ_cell − θ_i).
    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Pass 2: Compute cell fields using fixed 6-neighbor stencil
// Central difference for gradients, curl for vorticity — no binary search
__global__ void computeCellFields(
    const float* __restrict__ density,
    const float* __restrict__ momentum_x,
    const float* __restrict__ momentum_y,
    const float* __restrict__ momentum_z,
    float* __restrict__ pressure_x,
    float* __restrict__ pressure_y,
    float* __restrict__ pressure_z,
    float* __restrict__ vorticity_x,
    float* __restrict__ vorticity_y,
    float* __restrict__ vorticity_z,
    float pressure_k,
    float vorticity_k
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= d_grid_cells) return;

    float rho = density[cell];
    if (rho < 0.5f) return;  // Skip nearly empty cells

    // Extract cell coordinates
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);

    // Normalize accumulated momentum to get average velocity
    float inv_rho = 1.0f / rho;
    // ========================================================================
    // PRESSURE GRADIENT: ∇ρ via central difference
    // ========================================================================
    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
    const float inv_2h = 1.0f / (2.0f * d_grid_cell_size);

    // X gradient: (ρ[x+1] - ρ[x-1]) / 2h
    if (cx > 0 && cx < d_grid_dim - 1) {
        float rho_px = density[cell + 1];
        float rho_mx = density[cell - 1];
        grad_x = (rho_px - rho_mx) * inv_2h;
    }

    // Y gradient
    if (cy > 0 && cy < d_grid_dim - 1) {
        float rho_py = density[cell + d_grid_stride_y];
        float rho_my = density[cell - d_grid_stride_y];
        grad_y = (rho_py - rho_my) * inv_2h;
    }

    // Z gradient
    if (cz > 0 && cz < d_grid_dim - 1) {
        float rho_pz = density[cell + d_grid_stride_z];
        float rho_mz = density[cell - d_grid_stride_z];
        grad_z = (rho_pz - rho_mz) * inv_2h;
    }

    // Pressure force = -k * ∇ρ / ρ (pushes toward lower density)
    float pressure_scale = -pressure_k * inv_rho;
    pressure_x[cell] = grad_x * pressure_scale;
    pressure_y[cell] = grad_y * pressure_scale;
    pressure_z[cell] = grad_z * pressure_scale;

    // ========================================================================
    // VORTICITY: ω = ∇ × v (curl of velocity field)
    // ========================================================================
    // ω_x = ∂vz/∂y - ∂vy/∂z
    // ω_y = ∂vx/∂z - ∂vz/∂x
    // ω_z = ∂vy/∂x - ∂vx/∂y

    float dvz_dy = 0.0f, dvy_dz = 0.0f;
    float dvx_dz = 0.0f, dvz_dx = 0.0f;
    float dvy_dx = 0.0f, dvx_dy = 0.0f;

    // Velocity derivatives via central difference on neighbor cells
    // Note: we read accumulated momentum and divide by neighbor density

    if (cy > 0 && cy < d_grid_dim - 1) {
        uint32_t cell_py = cell + d_grid_stride_y;
        uint32_t cell_my = cell - d_grid_stride_y;
        float rho_py = density[cell_py];
        float rho_my = density[cell_my];
        if (rho_py > 0.5f && rho_my > 0.5f) {
            float vz_py = momentum_z[cell_py] / rho_py;
            float vz_my = momentum_z[cell_my] / rho_my;
            dvz_dy = (vz_py - vz_my) * inv_2h;
        }
    }

    if (cz > 0 && cz < d_grid_dim - 1) {
        uint32_t cell_pz = cell + d_grid_stride_z;
        uint32_t cell_mz = cell - d_grid_stride_z;
        float rho_pz = density[cell_pz];
        float rho_mz = density[cell_mz];
        if (rho_pz > 0.5f && rho_mz > 0.5f) {
            float vy_pz = momentum_y[cell_pz] / rho_pz;
            float vy_mz = momentum_y[cell_mz] / rho_mz;
            dvy_dz = (vy_pz - vy_mz) * inv_2h;

            float vx_pz = momentum_x[cell_pz] / rho_pz;
            float vx_mz = momentum_x[cell_mz] / rho_mz;
            dvx_dz = (vx_pz - vx_mz) * inv_2h;
        }
    }

    if (cx > 0 && cx < d_grid_dim - 1) {
        uint32_t cell_px = cell + 1;
        uint32_t cell_mx = cell - 1;
        float rho_px = density[cell_px];
        float rho_mx = density[cell_mx];
        if (rho_px > 0.5f && rho_mx > 0.5f) {
            float vz_px = momentum_z[cell_px] / rho_px;
            float vz_mx = momentum_z[cell_mx] / rho_mx;
            dvz_dx = (vz_px - vz_mx) * inv_2h;

            float vy_px = momentum_y[cell_px] / rho_px;
            float vy_mx = momentum_y[cell_mx] / rho_mx;
            dvy_dx = (vy_px - vy_mx) * inv_2h;

            float vx_px = momentum_x[cell_px] / rho_px;
            float vx_mx = momentum_x[cell_mx] / rho_mx;
            dvx_dy = (vx_px - vx_mx) * inv_2h;  // Reusing for symmetry
        }
    }

    // Compute curl components
    float omega_x = dvz_dy - dvy_dz;
    float omega_y = dvx_dz - dvz_dx;
    float omega_z = dvy_dx - dvx_dy;

    // Scale vorticity force
    vorticity_x[cell] = omega_x * vorticity_k;
    vorticity_y[cell] = omega_y * vorticity_k;
    vorticity_z[cell] = omega_z * vorticity_k;
}

// Pass 3: Gather cell forces to particles (direct O(1) lookup)
// Each particle reads its cell's pressure/vorticity and updates velocity
__global__ void gatherCellForcesToParticles(
    GPUDisk* __restrict__ disk,
    const float* __restrict__ density,
    const float* __restrict__ pressure_x,
    const float* __restrict__ pressure_y,
    const float* __restrict__ pressure_z,
    const float* __restrict__ vorticity_x,
    const float* __restrict__ vorticity_y,
    const float* __restrict__ vorticity_z,
    const float* __restrict__ phase_sin,
    const float* __restrict__ phase_cos,
    const uint32_t* __restrict__ particle_cell,
    const uint8_t* __restrict__ in_active_region,  // Step 3: skip passive particles (may be nullptr if disabled)
    uint32_t N,
    float dt,
    float substrate_k,  // Keplerian substrate coupling (competes with Kuramoto)
    float shear_k,      // Phase-misalignment shear (non-monotonic magnetic friction)
    float rho_ref,      // Reference density for shear normalization (mean × 8)
    float kuramoto_k,   // Kuramoto phase coupling strength (0 = free-running only)
    int   use_n12,      // 1 = apply N12 mixer envelope to coupling, 0 = constant K
    float envelope_scale // Harmonic index multiplier: cos(3s·θ)·cos(4s·θ). s=1 is N12 baseline.
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t cell = particle_cell[i];
    if (cell == UINT32_MAX) return;  // Inactive particle

    // Step 3: passive particles get their physics from advectPassiveParticles.
    // Applying pressure/vorticity/Kuramoto here would corrupt their velocity
    // (passive kernel advances pos azimuthally but doesn't rewrite vel_x/vel_z,
    // so gather increments to stale velocity would cause unbounded divergence).
    if (in_active_region && !in_active_region[i]) return;

    // O(1) direct read — no binary search!
    float press_x = pressure_x[cell];
    float press_y = pressure_y[cell];
    float press_z = pressure_z[cell];
    float ox = vorticity_x[cell];
    float oy = vorticity_y[cell];
    float oz = vorticity_z[cell];

    // Particle position (for substrate torque)
    float pos_x = disk->pos_x[i];
    float pos_z = disk->pos_z[i];

    // Current velocity
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // Apply pressure force
    vx += press_x * dt;
    vy += press_y * dt;
    vz += press_z * dt;

    // Apply vorticity force: F_ω = ω × v (cross product)
    // rsqrtf pattern: get both omega_mag and inv_omega from single SFU call
    float omega_sq = ox*ox + oy*oy + oz*oz;
    if (omega_sq > 1e-8f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float nx = ox * inv_omega;
        float ny = oy * inv_omega;
        float nz = oz * inv_omega;

        // Cross product: omega_hat × v
        float cross_x = ny * vz - nz * vy;
        float cross_y = nz * vx - nx * vz;
        float cross_z = nx * vy - ny * vx;

        vx += cross_x * omega_mag * dt;
        vy += cross_y * omega_mag * dt;
        vz += cross_z * omega_mag * dt;
    }

    // === FRICTIONAL SHEAR — DENSITY × ANGULAR HYBRID ===
    // Gu/Lüders/Bechinger arXiv:2602.11526v1 continuum analog.
    //   effective_k = shear_k × rho_factor × angular_profile
    //
    // rho_factor ∈ [0, 2]: density-weighting (Gemini) — only particles inside
    //   dense shell regions feel friction; inter-shell vacuum stays superfluid.
    //   Scale-invariant: rho_ref is passed from host as mean_density × 8.
    //
    // angular_profile = |sin(2φ)| where φ = atan2(|v_θ − v_Kep|, |v_r|)
    //   (Deepseek's non-monotonic profile) — friction vanishes for pure
    //   circular orbits (φ=0) AND pure radial infall (φ=90°), peaks at 45°
    //   where motion is mixed. This recovers the paper's hysteretic
    //   non-monotonic behavior using a signal that stays nonzero in steady
    //   state: v_r and v_θ − v_Kep are both small but nonzero everywhere.
    //
    // Combined: friction only inside shells AND only where motion is shear-mixed.
    if (shear_k > 0.0f) {
        float rho_cell = density[cell];
        float rho_factor = rho_cell / rho_ref;
        if (rho_factor > 2.0f) rho_factor = 2.0f;

        if (rho_factor > 0.0f) {
            float r2 = pos_x * pos_x + pos_z * pos_z + 1e-6f;
            float inv_r = rsqrtf(r2);
            float r_cyl = r2 * inv_r;
            float rx_hat = pos_x * inv_r;
            float rz_hat = pos_z * inv_r;
            float tx_kep = -rz_hat;  // prograde tangent in XZ
            float tz_kep =  rx_hat;

            float v_theta = vx * tx_kep + vz * tz_kep;
            float v_r = vx * rx_hat + vz * rz_hat;

            // Keplerian orbital speed at this radius: v_K = √(M/r)
            float v_kep = (r_cyl > ISCO_R * 0.5f) ? sqrtf(BH_MASS * inv_r) : 0.0f;
            float dv_tan = fabsf(v_theta - v_kep);
            float abs_vr = fabsf(v_r);

            // Angular profile: sin(2φ) = 2 sin(φ) cos(φ) = 2·dv_tan·|v_r|/(dv_tan² + v_r²)
            // Computed directly from components, no atan2 needed.
            float denom = dv_tan * dv_tan + abs_vr * abs_vr + 1e-8f;
            float angular_profile = 2.0f * dv_tan * abs_vr / denom;  // ∈ [0, 1]

            float drag = shear_k * rho_factor * angular_profile * dt;
            if (drag > 0.5f) drag = 0.5f;  // stability clamp
            float dv = v_theta * drag;
            vx -= dv * tx_kep;
            vz -= dv * tz_kep;
        }
    }

    // === KEPLERIAN SUBSTRATE TORQUE (Competing Interaction) ===
    // Based on magnetic friction paper: friction peaks when competing
    // interactions FRUSTRATE the system. Kuramoto wants phase alignment,
    // substrate wants Keplerian differential rotation. When balanced,
    // neither wins → hysteresis → dissipation → interesting dynamics.
    if (substrate_k > 0.0f) {
        // Use linear version for stability, can switch to sinusoidal later
        apply_keplerian_substrate_linear(pos_x, pos_z, vx, vz, substrate_k);
    }

    // Damping (same as octree path)
    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    disk->vel_x[i] = vx;
    disk->vel_y[i] = vy;
    disk->vel_z[i] = vz;

    // === KURAMOTO PHASE UPDATE (math.md Step 8, Step 10, Step 11) ===
    // Classical Kuramoto mean-field coupling via the grid phase_sin/cos:
    //   dθ/dt = ω_i + K · envelope · R_local · sin(θ_cell − θ_i)
    // The R_local · sin(Δθ) factor comes for free from the cell-averaged
    // phase sums because (mean_sin · cos θ_i − mean_cos · sin θ_i) already
    // has magnitude R_local = |⟨e^{iθ}⟩|. The N12 envelope (math.md Step 11,
    // period LCM(3,4) = 12) modulates coupling strength.
    {
        float theta_i = disk->theta[i];
        float omega_i = disk->omega_nat[i];
        float dtheta = omega_i;

        if (kuramoto_k > 0.0f) {
            float rho_cell = density[cell];
            if (rho_cell > 0.5f) {
                float ps = phase_sin[cell];
                float pc = phase_cos[cell];
                float inv_rho = 1.0f / rho_cell;
                float mean_sin = ps * inv_rho;  // ⟨sin θ⟩ over cell
                float mean_cos = pc * inv_rho;  // ⟨cos θ⟩ over cell
                // coupling = R_local · sin(θ_cell − θ_i)
                float sin_i = cuda_lut_sin(theta_i);
                float cos_i = cuda_lut_cos(theta_i);
                float coupling = mean_sin * cos_i - mean_cos * sin_i;

                // N12 envelope: 0.5 + 0.5·cos(3θ)·cos(4θ), period LCM(3,4)=12
                float envelope = 1.0f;
                if (use_n12) {
                    // envelope_scale controls the harmonic indices; s=1 is N12 baseline.
                    // s=2 → period halves → predicts optimal ω doubles (GPT test).
                    float c3 = cuda_lut_cos(3.0f * envelope_scale * theta_i);
                    float c4 = cuda_lut_cos(4.0f * envelope_scale * theta_i);
                    envelope = 0.5f + 0.5f * c3 * c4;
                }

                dtheta += kuramoto_k * envelope * coupling;
            }
        }

        // Advance and wrap to [0, 2π)
        theta_i += dtheta * dt;
        theta_i = fmodf(theta_i, TWO_PI);
        if (theta_i < 0.0f) theta_i += TWO_PI;
        disk->theta[i] = theta_i;
    }
}

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
__global__ void computeParticleActivityMask(
    const GPUDisk* __restrict__ disk,
    const uint32_t* __restrict__ curr_cell,
    const uint32_t* __restrict__ prev_cell,
    uint8_t* __restrict__ active_mask,
    uint32_t N,
    float velocity_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (!particle_active(disk, i)) {
        active_mask[i] = 0;
        return;
    }

    // Check 1: Did cell change?
    bool cell_changed = (curr_cell[i] != prev_cell[i]);

    // Check 2: Is velocity above threshold?
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];
    float v2 = vx*vx + vy*vy + vz*vz;
    bool moving = (v2 > velocity_threshold * velocity_threshold);

    active_mask[i] = (cell_changed || moving) ? 1 : 0;
}

// Compact active particles using atomic counter (simple but effective)
// For N=10M with ~10% active, atomics are fine (not a bottleneck)
__global__ void compactActiveParticles(
    const uint8_t* __restrict__ active_mask,
    uint32_t* __restrict__ active_list,
    uint32_t* __restrict__ active_count,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (active_mask[i]) {
        uint32_t idx = atomicAdd(active_count, 1);
        active_list[idx] = i;
    }
}

// Scatter ONLY active particles (uses compacted list)
__global__ void scatterActiveParticles(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    const uint32_t* __restrict__ active_list,
    uint32_t active_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int i = active_list[idx];  // Get actual particle index

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;

    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Scatter STATIC particles to bake grid (called once when lock engages)
__global__ void scatterStaticParticles(
    const GPUDisk* __restrict__ disk,
    const uint8_t* __restrict__ active_mask,  // Inverted: scatter where mask=0
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (!particle_active(disk, i) || active_mask[i]) return;  // Skip inactive or active particles

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;

    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Gather forces ONLY to active particles
__global__ void gatherToActiveParticles(
    GPUDisk* __restrict__ disk,
    const float* __restrict__ density,
    const float* __restrict__ pressure_x,
    const float* __restrict__ pressure_y,
    const float* __restrict__ pressure_z,
    const float* __restrict__ vorticity_x,
    const float* __restrict__ vorticity_y,
    const float* __restrict__ vorticity_z,
    const float* __restrict__ phase_sin,
    const float* __restrict__ phase_cos,
    const uint32_t* __restrict__ particle_cell,
    const uint32_t* __restrict__ active_list,
    uint32_t active_count,
    float dt,
    float substrate_k,
    float shear_k,
    float rho_ref,
    float kuramoto_k,
    int   use_n12,
    float envelope_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int i = active_list[idx];
    uint32_t cell = particle_cell[i];
    if (cell == UINT32_MAX) return;

    float press_x = pressure_x[cell];
    float press_y = pressure_y[cell];
    float press_z = pressure_z[cell];
    float ox = vorticity_x[cell];
    float oy = vorticity_y[cell];
    float oz = vorticity_z[cell];

    float pos_x = disk->pos_x[i];
    float pos_z = disk->pos_z[i];
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    vx += press_x * dt;
    vy += press_y * dt;
    vz += press_z * dt;

    float omega_sq = ox*ox + oy*oy + oz*oz;
    if (omega_sq > 1e-8f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float nx = ox * inv_omega;
        float ny = oy * inv_omega;
        float nz = oz * inv_omega;
        float cross_x = ny * vz - nz * vy;
        float cross_y = nz * vx - nx * vz;
        float cross_z = nx * vy - ny * vx;
        vx += cross_x * omega_mag * dt;
        vy += cross_y * omega_mag * dt;
        vz += cross_z * omega_mag * dt;
    }

    // === FRICTIONAL SHEAR — DENSITY × ANGULAR HYBRID (see gatherCellForcesToParticles) ===
    if (shear_k > 0.0f) {
        float rho_cell = density[cell];
        float rho_factor = rho_cell / rho_ref;
        if (rho_factor > 2.0f) rho_factor = 2.0f;

        if (rho_factor > 0.0f) {
            float r2 = pos_x * pos_x + pos_z * pos_z + 1e-6f;
            float inv_r = rsqrtf(r2);
            float r_cyl = r2 * inv_r;
            float rx_hat = pos_x * inv_r;
            float rz_hat = pos_z * inv_r;
            float tx_kep = -rz_hat;
            float tz_kep =  rx_hat;
            float v_theta = vx * tx_kep + vz * tz_kep;
            float v_r = vx * rx_hat + vz * rz_hat;
            float v_kep = (r_cyl > ISCO_R * 0.5f) ? sqrtf(BH_MASS * inv_r) : 0.0f;
            float dv_tan = fabsf(v_theta - v_kep);
            float abs_vr = fabsf(v_r);
            float denom = dv_tan * dv_tan + abs_vr * abs_vr + 1e-8f;
            float angular_profile = 2.0f * dv_tan * abs_vr / denom;
            float drag = shear_k * rho_factor * angular_profile * dt;
            if (drag > 0.5f) drag = 0.5f;
            float dv = v_theta * drag;
            vx -= dv * tx_kep;
            vz -= dv * tz_kep;
        }
    }

    if (substrate_k > 0.0f) {
        apply_keplerian_substrate_linear(pos_x, pos_z, vx, vz, substrate_k);
    }

    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    disk->vel_x[i] = vx;
    disk->vel_y[i] = vy;
    disk->vel_z[i] = vz;

    // === KURAMOTO PHASE UPDATE (see gatherCellForcesToParticles) ===
    {
        float theta_i = disk->theta[i];
        float omega_i = disk->omega_nat[i];
        float dtheta = omega_i;

        if (kuramoto_k > 0.0f) {
            float rho_cell = density[cell];
            if (rho_cell > 0.5f) {
                float ps = phase_sin[cell];
                float pc = phase_cos[cell];
                float inv_rho = 1.0f / rho_cell;
                float mean_sin = ps * inv_rho;
                float mean_cos = pc * inv_rho;
                float sin_i = cuda_lut_sin(theta_i);
                float cos_i = cuda_lut_cos(theta_i);
                float coupling = mean_sin * cos_i - mean_cos * sin_i;
                float envelope = 1.0f;
                if (use_n12) {
                    // envelope_scale controls the harmonic indices; s=1 is N12 baseline.
                    // s=2 → period halves → predicts optimal ω doubles (GPT test).
                    float c3 = cuda_lut_cos(3.0f * envelope_scale * theta_i);
                    float c4 = cuda_lut_cos(4.0f * envelope_scale * theta_i);
                    envelope = 0.5f + 0.5f * c3 * c4;
                }
                dtheta += kuramoto_k * envelope * coupling;
            }
        }

        theta_i += dtheta * dt;
        theta_i = fmodf(theta_i, TWO_PI);
        if (theta_i < 0.0f) theta_i += TWO_PI;
        disk->theta[i] = theta_i;
    }
}

// Copy previous cell indices for next frame's activity detection
__global__ void copyCurrentToPrevCell(
    const uint32_t* __restrict__ curr_cell,
    uint32_t* __restrict__ prev_cell,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    prev_cell[i] = curr_cell[i];
}

// Merge static grid + active scatter into working grid
// dst = static_base (copy first, then active particles atomicAdd on top)
__global__ void copyStaticToWorkingGrid(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[i];
}

// ============================================================================
// FLAGS + COMPACTION KERNELS — O(n) Transcription Pattern
// ============================================================================
// CUB DeviceSelect::Flagged for O(n) compaction (parallel prefix sum, no atomics)
// Sparse clear for O(active_count) flag reset instead of O(GRID_CELLS) memset

// ============================================================================
// HIERARCHICAL TILED FLAGS — "Methylation" Pattern
// ============================================================================
// Two-level flags: tile flags (4096) + cell flags within active tiles
// Scatter marks both tile AND cell, compaction scans tiles then cells

// Scatter particles to cells AND mark tile flags — O(particles)
__global__ void scatterWithTileFlags(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    uint8_t* __restrict__ cell_flags,
    uint8_t* __restrict__ tile_flags,
    uint32_t N, float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (!particle_active(disk, i)) {
        particle_cell[i] = UINT32_MAX;
        return;
    }

    // Compute cell index
    uint32_t cell = cellIndexFromPos(disk->pos_x[i], disk->pos_y[i], disk->pos_z[i]);
    particle_cell[i] = cell;

    // Atomically accumulate to cell
    atomicAdd(&density[cell], alpha);
    atomicAdd(&momentum_x[cell], disk->vel_x[i] * alpha);
    atomicAdd(&momentum_y[cell], disk->vel_y[i] * alpha);
    atomicAdd(&momentum_z[cell], disk->vel_z[i] * alpha);

    // Phase: encode as sin/cos for proper averaging
    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase) * alpha);
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase) * alpha);

    // Mark cell flag (duplicates collapse to same value)
    cell_flags[cell] = FLAG_INITIAL_VALUE;

    // Mark tile flag (coarse level)
    uint32_t tile = cellToTile(cell);
    tile_flags[tile] = FLAG_INITIAL_VALUE;
}

// Compact active tiles — O(NUM_TILES) = O(4096), much smaller than O(2M)
__global__ void compactActiveTiles(
    const uint8_t* __restrict__ tile_flags,
    uint32_t* __restrict__ active_tiles,
    uint32_t* __restrict__ tile_count
) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= NUM_TILES) return;

    if (tile_flags[tile] > 0) {
        uint32_t idx = atomicAdd(tile_count, 1);
        active_tiles[idx] = tile;
    }
}

// Compact cells within active tiles — O(active_tiles × 512)
// This is the key optimization: only scan cells in active tiles
__global__ void compactCellsInTiles(
    const uint8_t* __restrict__ cell_flags,
    const uint32_t* __restrict__ active_tiles,
    uint32_t num_active_tiles,
    uint32_t* __restrict__ active_cells,
    uint32_t* __restrict__ cell_count
) {
    // Each block processes one tile
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_active_tiles) return;

    uint32_t tile = active_tiles[tile_idx];

    // Compute tile origin (use runtime tiles_per_dim, not compile-time TILES_PER_DIM)
    int tiles_per_dim = d_grid_dim / TILE_DIM;
    uint32_t tx = tile % tiles_per_dim;
    uint32_t ty = (tile / tiles_per_dim) % tiles_per_dim;
    uint32_t tz = tile / (tiles_per_dim * tiles_per_dim);

    // Each thread in block handles one cell within the tile
    int local_cell = threadIdx.x;
    if (local_cell >= CELLS_PER_TILE) return;

    // Convert local cell to global cell index
    int lx = local_cell % TILE_DIM;
    int ly = (local_cell / TILE_DIM) % TILE_DIM;
    int lz = local_cell / (TILE_DIM * TILE_DIM);

    uint32_t cx = tx * TILE_DIM + lx;
    uint32_t cy = ty * TILE_DIM + ly;
    uint32_t cz = tz * TILE_DIM + lz;
    uint32_t cell = cx + cy * d_grid_stride_y + cz * d_grid_stride_z;

    if (cell_flags[cell] > 0) {
        uint32_t idx = atomicAdd(cell_count, 1);
        active_cells[idx] = cell;
    }
}

// Clear tile and cell flags for active cells — O(active_cells)
__global__ void sparseClearTileAndCellFlags(
    uint8_t* __restrict__ cell_flags,
    uint8_t* __restrict__ tile_flags,
    const uint32_t* __restrict__ active_cells,
    uint32_t num_active_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_cells) return;

    uint32_t cell = active_cells[i];
    cell_flags[cell] = 0;

    // Also clear tile flag (may write multiple times, that's fine)
    uint32_t tile = cellToTile(cell);
    tile_flags[tile] = 0;
}

// Decay and compute pressure for active cells only — O(active_count), ~4k cells
// No flag propagation: particles scatter to new cells each frame naturally
__global__ void decayAndComputePressure(
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    float* __restrict__ pressure_x,
    float* __restrict__ pressure_y,
    float* __restrict__ pressure_z,
    const uint32_t* __restrict__ active_list,
    uint32_t active_count,
    float decay,
    float pressure_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active_count) return;

    uint32_t cell = active_list[i];

    // Decay accumulated state
    density[cell] *= decay;
    momentum_x[cell] *= decay;
    momentum_y[cell] *= decay;
    momentum_z[cell] *= decay;
    phase_sin[cell] *= decay;
    phase_cos[cell] *= decay;

    // Reset pressure (will be computed below)
    pressure_x[cell] = 0.0f;
    pressure_y[cell] = 0.0f;
    pressure_z[cell] = 0.0f;

    float my_rho = density[cell];
    if (my_rho < 0.1f) return;

    // Extract cell coordinates for neighbor lookup
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);

    // Shell modulation: compute distance from grid center
    // Viviani period-4 interference: constructive at 0,4,8..., destructive at 2,6,10...
    float half_dim = (float)d_grid_dim * 0.5f;
    float rx = ((float)cx - half_dim) * d_grid_cell_size;
    float ry = ((float)cy - half_dim) * d_grid_cell_size;
    float rz = ((float)cz - half_dim) * d_grid_cell_size;
    float r = sqrtf(rx*rx + ry*ry + rz*rz);
    uint32_t shell = cuda_lut_shell_index(r, LAMBDA_OCTREE);
    float shell_mod = cuda_lut_shell_factor(shell);

    // 6-neighbor pressure computation (face neighbors only)
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int n = 0; n < 6; n++) {
        int nx = (int)cx + dx[n];
        int ny = (int)cy + dy[n];
        int nz = (int)cz + dz[n];

        if (nx < 0 || nx >= d_grid_dim || ny < 0 || ny >= d_grid_dim || nz < 0 || nz >= d_grid_dim) continue;

        uint32_t neighbor = nx + ny * d_grid_stride_y + nz * d_grid_stride_z;
        float neighbor_rho = density[neighbor];

        // Compute gradient and pressure force with shell modulation
        // Shell factor: 1.0 at constructive peaks, 0.2 at destructive troughs
        float gradient = (neighbor_rho - my_rho) / d_grid_cell_size;
        float force = -gradient * pressure_k * shell_mod / (my_rho + 0.01f);

        atomicAdd(&pressure_x[cell], force * (float)dx[n]);
        atomicAdd(&pressure_y[cell], force * (float)dy[n]);
        atomicAdd(&pressure_z[cell], force * (float)dz[n]);
    }
}

// ============================================================================
// PRESSURE + VORTICITY FORCE KERNEL
// ============================================================================
// Applies three forces:
//   1. Pressure: F_p = -k_p ∇ρ  (pushes toward lower density)
//   2. Vorticity: F_ω = k_ω (ω × v)  (induces rotation/spiral structure)
//   3. Phase coherence: modulates pressure by neighbor phase alignment
//
// Together these create a self-organizing medium with radial balance,
// rotational structure (spiral arms), and temporal coherence (standing waves).

__global__ void applyPressureVorticityKernel(
    GPUDisk* disk,
    const uint64_t* morton_keys,
    const uint32_t* particle_ids,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const float* leaf_vel_x,
    const float* leaf_vel_y,
    const float* leaf_vel_z,
    const float* leaf_phase,  // S3 phase for direct modulation (zero-cost)
    const uint64_t* hash_keys,    // Hash table for O(1) neighbor lookup
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves,
    uint32_t num_active,
    const uint8_t* __restrict__ in_active_region,  // Step 3: skip passive particles
    float dt,
    float pressure_k,    // Pressure coefficient (~0.03)
    float vorticity_k    // Vorticity coefficient (~0.01)
) {
    int sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= num_active) return;

    uint32_t orig_idx = particle_ids[sorted_idx];
    if (!particle_active(disk, orig_idx)) return;
    // Step 3: passive particles get physics from advectPassiveParticles.
    if (in_active_region && !in_active_region[orig_idx]) return;

    uint64_t my_key = morton_keys[sorted_idx];

    // Find containing leaf node via O(1) hash lookup
    uint64_t level13_mask = ~((1ULL << 39) - 1);
    uint64_t parent_key = my_key & level13_mask;

    uint32_t leaf_idx = findLeafByHash(hash_keys, hash_values, hash_mask, parent_key);
    if (leaf_idx == UINT32_MAX) return;

    // Read current velocity
    float vx = disk->vel_x[orig_idx];
    float vy = disk->vel_y[orig_idx];
    float vz = disk->vel_z[orig_idx];

    // ========================================================================
    // 0. PHASE MODULATION: derive from phase directly (no neighbor lookup)
    // ========================================================================
    // Use sin(phase) for oscillation - creates standing wave patterns
    // Single memory read, already coalesced with leaf access
    // sin(θ) oscillates [-1, 1], so mod ranges [0.5, 1.5]
    float phase_mod = 1.0f;
    if (leaf_phase != nullptr) {
        float phase = leaf_phase[leaf_idx];
        phase_mod = 1.0f + 0.5f * cuda_lut_sin(phase);
    }

    // ========================================================================
    // 1. PRESSURE FORCE: F_p = -k_p ∇ρ × phase_mod
    // ========================================================================
    float grad_x, grad_y, grad_z;
    computeLeafGradient(nodes, leaf_node_indices, hash_keys, hash_values, hash_mask, leaf_idx,
                        &grad_x, &grad_y, &grad_z);

    // rsqrtf pattern: get both grad_mag and inv_grad_mag from single SFU call
    float grad_sq = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z;

    if (grad_sq > 1e-6f) {
        float inv_grad_mag = rsqrtf(grad_sq);
        float grad_mag = grad_sq * inv_grad_mag;
        uint32_t node_idx = leaf_node_indices[leaf_idx];
        float local_density = (float)nodes[node_idx].particle_count + 1.0f;
        // Phase coherence modulates pressure coupling
        float force_scale = pressure_k * phase_mod * grad_mag / local_density;

        // Pressure pushes toward lower density
        vx += -grad_x * inv_grad_mag * force_scale * dt;
        vy += -grad_y * inv_grad_mag * force_scale * dt;
        vz += -grad_z * inv_grad_mag * force_scale * dt;
    }

    // ========================================================================
    // 2. VORTICITY FORCE: F_ω = k_ω (ω × v)
    // ========================================================================
    // Vorticity confinement: amplifies existing rotation
    // The cross product ω × v produces a force perpendicular to both,
    // which induces spiral motion and maintains angular momentum.
    float omega_x, omega_y, omega_z;
    computeLeafVorticity(nodes, leaf_node_indices, leaf_vel_x, leaf_vel_y, leaf_vel_z,
                         hash_keys, hash_values, hash_mask, leaf_idx,
                         &omega_x, &omega_y, &omega_z);

    // rsqrtf pattern: get both omega_mag and inv_omega from single SFU call
    float omega_sq = omega_x*omega_x + omega_y*omega_y + omega_z*omega_z;

    if (omega_sq > 1e-8f && vorticity_k > 0.0f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float ox = omega_x * inv_omega;
        float oy = omega_y * inv_omega;
        float oz = omega_z * inv_omega;

        // Cross product: ω × v
        // This produces a force perpendicular to both, inducing rotation
        float cross_x = oy * vz - oz * vy;
        float cross_y = oz * vx - ox * vz;
        float cross_z = ox * vy - oy * vx;

        // Scale by vorticity magnitude and coefficient
        float vort_force = vorticity_k * omega_mag;
        vx += cross_x * vort_force * dt;
        vy += cross_y * vort_force * dt;
        vz += cross_z * vort_force * dt;
    }

    // ========================================================================
    // 3. VELOCITY DAMPING: v *= (1 - c)
    // ========================================================================
    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    // Write back
    disk->vel_x[orig_idx] = vx;
    disk->vel_y[orig_idx] = vy;
    disk->vel_z[orig_idx] = vz;
}

// ============================================================================
// S3 PHASE STATE — Temporal Coherence Layer
// ============================================================================
// Phase tracking enables:
//   - Resonance detection (nodes with matching phase → standing waves)
//   - Temporal coherence (neighboring phases couple and synchronize)
//   - Wave patterns (phase gradients create propagating structures)
//
// Local frequency ω is derived from density: high density → high frequency
// This creates natural chirp patterns as matter falls inward.

// Initialize phase from spatial position (creates coherent seed pattern)
__global__ void initializeLeafPhase(
    float* leaf_phase,
    float* leaf_frequency,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    uint32_t num_leaves,
    float base_frequency   // Base oscillation frequency
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    // Seed phase from position: creates radial wave fronts
    float cx = node.center_x;
    float cy = node.center_y;
    float cz = node.center_z;
    float r = sqrtf(cx*cx + cy*cy + cz*cz);

    // Phase = radial distance mod 2π (creates concentric rings)
    // Add angular component for spiral pattern
    float theta = cuda_fast_atan2(cz, cx);
    float initial_phase = fmodf(r * 0.1f + theta * 0.5f, 2.0f * M_PI);
    if (initial_phase < 0.0f) initial_phase += 2.0f * M_PI;

    leaf_phase[leaf_idx] = initial_phase;

    // Frequency from density: ω = ω_base * (1 + log(ρ))
    // Denser regions oscillate faster (gravitational time dilation analog)
    float density = (float)node.particle_count + 1.0f;
    float freq = base_frequency * (1.0f + 0.1f * logf(density));
    leaf_frequency[leaf_idx] = freq;
}

// Evolve phase with local frequency and neighbor coupling (Kuramoto model)
__global__ void evolveLeafPhase(
    float* leaf_phase,
    float* leaf_frequency,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves,
    float dt,
    float coupling_k       // Phase coupling strength
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    float my_phase = leaf_phase[leaf_idx];
    float my_freq = leaf_frequency[leaf_idx];

    // === PHASE COUPLING (Kuramoto model) ===
    // dθ/dt = ω + (K/N) * Σ sin(θ_j - θ_i)
    // Neighbors with similar phase reinforce; different phases repel

    uint64_t my_key = node.morton_key;
    int level = node.level;

    // Get 6 face-adjacent neighbors
    uint64_t neighbor_keys[6];
    getNeighborKeys(my_key, level, neighbor_keys);

    float coupling_sum = 0.0f;
    int neighbor_count = 0;

    for (int n = 0; n < 6; n++) {
        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf = findLeafByHash(hash_keys, hash_values,
                                                 hash_mask, neighbor_keys[n]);
        if (neighbor_leaf != UINT32_MAX) {
            float neighbor_phase = leaf_phase[neighbor_leaf];
            // Kuramoto coupling: sin(θ_neighbor - θ_self)
            coupling_sum += cuda_lut_sin(neighbor_phase - my_phase);
            neighbor_count++;
        }
    }

    // Average coupling contribution
    float coupling_term = 0.0f;
    if (neighbor_count > 0) {
        coupling_term = coupling_k * coupling_sum / (float)neighbor_count;
    }

    // Update phase: θ += (ω + coupling) * dt
    float new_phase = my_phase + (my_freq + coupling_term) * dt;

    // Wrap to [0, 2π]
    new_phase = fmodf(new_phase, 2.0f * M_PI);
    if (new_phase < 0.0f) new_phase += 2.0f * M_PI;

    leaf_phase[leaf_idx] = new_phase;

    // Update frequency from current density (tracks changing conditions)
    float density = (float)node.particle_count + 1.0f;
    float base_freq = 0.1f;  // Base frequency
    leaf_frequency[leaf_idx] = base_freq * (1.0f + 0.1f * logf(density));
}

// Pre-compute coherence for all leaves (reduces per-particle neighbor lookups)
__global__ void computeLeafCoherence(
    float* leaf_coherence,
    const float* leaf_phase,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    // Just call the device function and cache the result
    float coherence = computePhaseCoherence(leaf_phase, nodes, leaf_node_indices,
                                             hash_keys, hash_values, hash_mask, leaf_idx);
    leaf_coherence[leaf_idx] = coherence;
}

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


// ============================================================================
// Camera State
// ============================================================================

static struct {
    float dist = 200.0f;   // Larger for bigger disk
    float azimuth = 0.4f;
    float elevation = 0.6f;  // Higher angle to see spiral structure
    double lastX = 0, lastY = 0;
    bool dragging = false;
    bool paused = false;
    uint8_t seam_bits = 0x03;  // Start with full coupling
    float bias = 0.75f;        // Demon efficiency (0.5 = weak, 0.75 = normal, 1.0 = perfect)
    int color_mode = 0;  // 0 = topology, 1 = blackbody, 2 = pump scale, 3 = intrinsic redshift
} g_cam;

// ============================================================================
// Topology Control - GPT's Control Experiment
// ============================================================================
// Toggle between discrete hopfion shells vs smooth gradient (no twist)
// H key toggles: true = hopfion topology, false = smooth gradient
bool g_use_hopfion_topology = true;

// ============================================================================
// Spiral Arm Topology Control - Deepseek's Experiment
// ============================================================================
// Toggle between discrete arm boundaries vs smooth density waves
// A key toggles: true = discrete boundaries, false = smooth waves
bool g_enable_arms = false;         // Enable/disable arm structure (off by default, use --discrete-arms or --smooth-arms)
bool g_spawn_enabled = true;        // Natural growth spawning (--no-spawn disables for clean Kuramoto measurements)
bool g_use_arm_topology = true;     // true = discrete, false = smooth

// Phase-misalignment shear (non-monotonic magnetic friction analog)
// Based on Gu/Lüders/Bechinger arXiv:2602.11526v1 — friction peaks in the
// competing regime where FM and AFM phase orderings frustrate each other.
// Drives collapse in turbulent/frustrated regions, leaves locked shells alone.
float g_shear_k = 0.0f;

// Kuramoto phase coupling — math.md Step 8, Step 10, Step 11
// θ_i advances at rate ω_i + K·sin(θ_cell − θ_i) via mean-field coupling
// through the grid phase_sin/phase_cos fields. Tests synchronization
// threshold, traveling coherence packets, chimera states, breathing clusters.
float g_kuramoto_k = 0.0f;        // Coupling strength K (0 = no coupling)
float g_omega_base = 1.0f;        // Mean natural frequency ω₀
float g_omega_spread = 0.05f;     // Gaussian std-dev σ for ω distribution
bool  g_n12_envelope = true;      // Apply N12 mixer envelope to coupling (math.md Step 11)
float g_envelope_scale = 1.0f;    // Multiplier on envelope harmonic indices: cos(3s·θ)·cos(4s·θ).
                                  // s=1 → period 2π (default N12). s=2 → period π (N6). s=0.5 → period 4π (N24).
                                  // Tests GPT's prediction: optimal ω should scale as 1/envelope_period.

// Tree Architecture Step 4: runtime corner threshold for passive/active classification.
// Particles with |pump_residual| > g_corner_threshold → active (siphon kernel).
// Default 0.15f matches the compile-time constant from Step 3. Tunable via --corner-threshold.
float g_corner_threshold = 0.15f;

// Tree Architecture Step 4: runtime passive residual tau.
// Controls how fast pump_residual decays in passive particles. Units: simulation time.
// Default 5.0f matches PASSIVE_RESIDUAL_TAU from Step 3. Tunable via --passive-tau.
float g_passive_residual_tau = 5.0f;

// Tree Architecture Step 6: shell-aware initialization.
// When true, particles are initialized ON the 8 resonance shells instead of
// uniformly in a box. This skips the settling transient and starts with most
// particles passive. Use --shell-init to enable.
bool g_shell_init = false;

// Per-cell R export: dumps R_cell grid to disk every N frames (0 = disabled)
int g_r_export_interval = 0;

// Dense R(t) logging: prints R_global every N frames (0 = only every 90 like normal stats)
int g_r_log_interval = 0;

// Kuramoto × topology correlation dump — emits one CSV-friendly row per stats
// frame with (frame, R, R_recon, n_peaks, peak_mass_frac, Q, num_shells, active_count).
// Writes to stdout prefixed with [QR-corr] so the stream can be filtered offline.
bool g_qr_corr_log = false;

// Initial rotation direction: prograde (default) or retrograde.
// Used for the chirality / Q-sign test — if Q drift is driven by initial
// rotation direction, flipping this should flip the sign of Q drift.
bool g_retrograde_init = false;

// RNG seed for initial particle positions, phases, and natural frequencies
unsigned int g_rng_seed = 42;

#define NUM_ARMS 3                  // m = 3 (3-armed spiral)
#define ARM_WIDTH_DEG 45.0f         // Width of each arm in degrees
#define ARM_TRAP_STRENGTH 0.15f     // Angular momentum barrier strength

// ============================================================================
// Headless Mode - Performance Testing
// ============================================================================
// Disable all rendering for 10-20x speedup (physics + logging only)
bool g_headless = false;

// ============================================================================
// Hybrid LOD rendering (experimental - requires volume pass implementation)
bool g_hybrid_lod = false;
// Octree-based render traversal (vs flat scan compaction)
bool g_octree_render = false;
// Octree physics - XOR neighbor stress computation
bool g_octree_physics = true;
// Octree phase evolution - Kuramoto coupling via Morton-sorted leaves
// Set to false to use mip-tree for hierarchical coherence instead (A/B test)
bool g_octree_phase = true;
// Octree rebuild - Morton sort + stochastic tree build every 30 frames
// Set to false to skip Morton sorting entirely (mip-tree provides hierarchy)
// DEFAULT: false — mip-tree replaces octree for hierarchical coherence
bool g_octree_rebuild = false;
// Grid physics - DNA/RNA streaming forward-pass model (replaces octree physics)
// DEFAULT: true — enabled with g_grid_flags for sparse 235× speedup
bool g_grid_physics = true;
// Grid flags mode - optimal sparse: presence flags, no lists, no dedup
// DEFAULT: true — 235× speedup on Pass 2 (2M → 3.4k cells)
bool g_grid_flags = true;

// ============================================================================
// Radius-Controlled Termination - GPT's Confounder Test
// ============================================================================
// Terminate based on ring radius instead of frame count to eliminate geometric effects
bool g_terminate_on_radius = false;
float g_target_ring_radius = 250.0f;
int g_target_frames = 50000;  // Configurable target frame count

// ============================================================================
// Test Suite Flags - The Final Trilogy
// ============================================================================
bool g_test_residence_time = false;    // Test A: Track arm vs gap residence time
bool g_matched_amplitude = false;      // Test C: Set discrete boost to 1.25× (match smooth)
float g_arm_boost_override = 0.0f;     // If > 0, override ARM_BOOST in discrete mode

// ============================================================================
// Predictive Locking - Skip Harmonic Recomputation When Shells Are Stable
// ============================================================================
// When the system is in a locked m=0 ground state (8 shells, Q stable, isotropic),
// we can skip expensive per-frame computations:
//   - Mip-tree rebuild (already fast, but skippable)
//   - Full harmonic analysis (m=3 mode detection)
//   - Angular histogram computation
//
// Lock detection: shell count stable + Q variance low + stability low
struct HarmonicLock {
    int prev_shell_count;       // Previous frame shell count
    float prev_Q;               // Previous frame Q estimate
    float Q_variance_acc;       // Running variance accumulator
    int stable_frames;          // Consecutive frames meeting lock criteria
    bool locked;                // Currently in locked state
    int lock_recheck_counter;   // Frames until next full recompute (for verification)

    // Lock thresholds
    // Note: Lock detection runs every 90 frames (diagnostic interval), so thresholds are
    // tuned for that cadence. LOCK_THRESHOLD_FRAMES counts diagnostic intervals, not raw frames.
    static constexpr int LOCK_THRESHOLD_FRAMES = 3;    // Need 3 consecutive diagnostics (270 frames) to lock
    static constexpr int RECHECK_INTERVAL = 256;       // Verify lock every 256 frames
    static constexpr float Q_VARIANCE_THRESHOLD = 50.0f; // Max |ΔQ| between diagnostics (Q swings 0-35 normally)
    static constexpr float STABILITY_THRESHOLD = 0.20f; // Max stability% to stay locked (20%)
};
HarmonicLock g_harmonic_lock = {0, 0.0f, 0.0f, 0, false, 0};
bool g_predictive_locking = true;  // Enable predictive locking by default

// ============================================================================
// Active Particle Compaction - Skip Static Shell Mass
// ============================================================================
// When locked, ~90% of particles are stable shell mass that doesn't need
// scatter/gather every frame. We compact only "active" particles:
//   - Movers: |velocity| > threshold
//   - Cell-changers: current cell != previous cell
//   - Boundary: near active tile edges
//
// This reduces O(N) scatter/gather to O(active_N) where active_N << N.
// Static particles are "baked" into a persistent grid that gets reused.
//
// Memory layout:
//   d_prev_particle_cell[N]     - Previous frame's cell indices
//   d_particle_active[N]        - Activity mask (uint8)
//   d_active_particle_list[N]   - Compacted indices of active particles
//   d_active_particle_count     - Number of active particles
//   d_static_grid_density[G]    - Baked density from static particles
//   d_static_grid_momentum_*[G] - Baked momentum from static particles
//
struct ActiveParticleState {
    uint32_t* d_prev_cell;           // Previous frame cell indices
    uint8_t*  d_active_mask;         // Per-particle activity flag
    uint32_t* d_active_list;         // Compacted active particle indices
    uint32_t* d_active_count;        // Device counter
    uint32_t  h_active_count;        // Host-side count

    // Static grid (baked when lock engages)
    float* d_static_density;
    float* d_static_momentum_x;
    float* d_static_momentum_y;
    float* d_static_momentum_z;
    float* d_static_phase_sin;
    float* d_static_phase_cos;

    bool initialized;
    bool static_baked;               // True after static particles scattered to static grid
    int bake_frame;                  // Frame when bake occurred

    // Thresholds
    // Note: Velocity threshold is less useful than cell-change for Keplerian orbits
    // Particles orbit at v~0.2-0.3 but stay in same cell for many frames
    // The key is whether the particle changes CELL, not whether it's moving
    static constexpr float VELOCITY_THRESHOLD = 10.0f;  // Very high - effectively disabled
    static constexpr int REBAKE_INTERVAL = 256;         // Re-bake static grid periodically
};
ActiveParticleState g_active_particles = {};
#if ENABLE_PASSIVE_ADVECTION
bool g_active_compaction = false;  // Step 3: passive kernel replaces scatter-skip optimization.
                                   // The baked static grid would go stale because passive particles
                                   // move azimuthally. Siphon-skip savings dwarf scatter-skip savings.
#else
bool g_active_compaction = true;   // Pre-passive: active particle compaction for scatter optimization.
#endif

// ============================================================================
// Seam Drift Tracking - Time-Resolved m=3 Phase Logging
// ============================================================================
// Track the m=3 phase angle over time to detect whether seam orientation
// is stationary (locked to arm geometry) or precessing.
// Log format: (frame, M_eff, m3_phase_deg)
#define SEAM_DRIFT_LOG_SIZE 400  // Enough for 60k+ frames at 90-frame intervals
struct SeamDriftEntry {
    int frame;
    float M_eff;
    float phase_deg;    // m=3 phase in degrees [0, 120)
    int sample_count;
};
SeamDriftEntry g_seam_drift_log[SEAM_DRIFT_LOG_SIZE];
int g_seam_drift_count = 0;

// ============================================================================
// Entropy Injection Test - Material Dissolution
// ============================================================================
// Global flag for entropy injection (toggled with E key)
bool g_inject_entropy = false;

// Inject high-entropy star cluster to test thread coherence breakdown
__global__ void injectEntropyCluster(GPUDisk* disk, int N, float sim_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Only inject 10,000 particles starting from a high index
    int inject_start = N - 10000;
    if (idx < inject_start) return;

    // Spawn particles in a spherical cluster at r ~ 200 (outside main disk)
    // Random positions on sphere
    unsigned int seed = idx + (unsigned int)(sim_time * 1000.0f);
    float theta = 2.0f * 3.14159f * (float)(seed % 1000) / 1000.0f;
    float phi = acosf(2.0f * (float)((seed / 1000) % 1000) / 1000.0f - 1.0f);

    float r_inject = 200.0f;
    float x = r_inject * sinf(phi) * cosf(theta);
    float y = r_inject * sinf(phi) * sinf(theta);
    float z = r_inject * cosf(phi);

    // Random inward velocities (thermal + infall)
    float v_thermal = 0.3f;
    float v_infall = -0.5f;  // Aimed at core

    disk->pos_x[idx] = x;
    disk->pos_y[idx] = y;
    disk->pos_z[idx] = z;
    disk->vel_x[idx] = v_infall * x / r_inject + v_thermal * (2.0f * (float)((seed * 7) % 1000) / 1000.0f - 1.0f);
    disk->vel_y[idx] = v_infall * y / r_inject + v_thermal * (2.0f * (float)((seed * 11) % 1000) / 1000.0f - 1.0f);
    disk->vel_z[idx] = v_infall * z / r_inject + v_thermal * (2.0f * (float)((seed * 13) % 1000) / 1000.0f - 1.0f);

    // High entropy state: random pump states, high residual
    disk->pump_state[idx] = seed % 4;  // Random IDLE/UP/DOWN/FULL
    disk->pump_scale[idx] = 0.5f + 1.5f * (float)((seed * 17) % 1000) / 1000.0f;  // 0.5-2.0 (chaotic)
    disk->pump_residual[idx] = 0.8f + 0.4f * (float)((seed * 19) % 1000) / 1000.0f;  // High stress
    disk->pump_work[idx] = 0.0f;
    disk->pump_history[idx] = 0.5f;  // Incoherent
    disk->pump_coherent[idx] = 0;
    disk->pump_seam[idx] = 0x00;  // Closed (will be forced open by stress)

    disk->flags[idx] = PFLAG_ACTIVE;  // active, not ejected
    // NOTE: disk_r, disk_phi, temp, in_disk no longer stored — computed on-demand
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int num_particles = 3500000;  // 3.5M particles for full resolution

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            num_particles = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--no-topology") == 0 || strcmp(argv[i], "--smooth") == 0) {
            extern bool g_use_hopfion_topology;
            g_use_hopfion_topology = false;
        }
        else if (strcmp(argv[i], "--topology") == 0 || strcmp(argv[i], "--hopfion") == 0) {
            extern bool g_use_hopfion_topology;
            g_use_hopfion_topology = true;
        }
        else if (strcmp(argv[i], "--discrete-arms") == 0 || strcmp(argv[i], "--arm-topology") == 0) {
            extern bool g_use_arm_topology;
            extern bool g_enable_arms;
            g_use_arm_topology = true;
            g_enable_arms = true;
        }
        else if (strcmp(argv[i], "--smooth-arms") == 0 || strcmp(argv[i], "--no-arm-topology") == 0) {
            extern bool g_use_arm_topology;
            extern bool g_enable_arms;
            g_use_arm_topology = false;
            g_enable_arms = true;
        }
        else if (strcmp(argv[i], "--no-arms") == 0) {
            extern bool g_enable_arms;
            g_enable_arms = false;
        }
        else if (strcmp(argv[i], "--shear-k") == 0 && i+1 < argc) {
            extern float g_shear_k;
            g_shear_k = (float)atof(argv[++i]);
            printf("[shear] Phase-misalignment shear coefficient: %.4f\n", g_shear_k);
        }
        else if (strcmp(argv[i], "--kuramoto-k") == 0 && i+1 < argc) {
            extern float g_kuramoto_k;
            g_kuramoto_k = (float)atof(argv[++i]);
            printf("[kuramoto] Coupling strength K: %.4f\n", g_kuramoto_k);
        }
        else if (strcmp(argv[i], "--omega-base") == 0 && i+1 < argc) {
            extern float g_omega_base;
            g_omega_base = (float)atof(argv[++i]);
            printf("[kuramoto] Natural frequency ω₀: %.4f\n", g_omega_base);
        }
        else if (strcmp(argv[i], "--omega-spread") == 0 && i+1 < argc) {
            extern float g_omega_spread;
            g_omega_spread = (float)atof(argv[++i]);
            printf("[kuramoto] Frequency spread σ: %.4f\n", g_omega_spread);
        }
        else if (strcmp(argv[i], "--no-n12") == 0) {
            extern bool g_n12_envelope;
            g_n12_envelope = false;
            printf("[kuramoto] N12 envelope DISABLED (constant K)\n");
        }
        else if (strcmp(argv[i], "--envelope-scale") == 0 && i+1 < argc) {
            extern float g_envelope_scale;
            g_envelope_scale = (float)atof(argv[++i]);
            printf("[kuramoto] Envelope harmonic scale: %.3f (period = 2π/%.3f)\n",
                   g_envelope_scale, g_envelope_scale);
        }
        else if (strcmp(argv[i], "--corner-threshold") == 0 && i+1 < argc) {
            extern float g_corner_threshold;
            g_corner_threshold = (float)atof(argv[++i]);
            printf("[passive] Corner threshold: %.4f\n", g_corner_threshold);
        }
        else if (strcmp(argv[i], "--passive-tau") == 0 && i+1 < argc) {
            extern float g_passive_residual_tau;
            g_passive_residual_tau = (float)atof(argv[++i]);
            printf("[passive] Residual decay tau: %.2f\n", g_passive_residual_tau);
        }
        else if (strcmp(argv[i], "--shell-init") == 0) {
            extern bool g_shell_init;
            g_shell_init = true;
            printf("[init] Shell-aware initialization: particles ON resonance shells\n");
        }
        else if (strcmp(argv[i], "--no-spawn") == 0) {
            extern bool g_spawn_enabled;
            g_spawn_enabled = false;
            printf("[spawn] Natural growth DISABLED — particle count locked\n");
        }
        else if (strcmp(argv[i], "--qr-corr") == 0) {
            extern bool g_qr_corr_log;
            g_qr_corr_log = true;
            printf("[qr-corr] Kuramoto × topology correlation dump ENABLED\n");
        }
        else if (strcmp(argv[i], "--retrograde") == 0) {
            extern bool g_retrograde_init;
            g_retrograde_init = true;
            printf("[init] Retrograde initial rotation (counterclockwise → clockwise)\n");
        }
        else if (strcmp(argv[i], "--r-export-interval") == 0 && i+1 < argc) {
            extern int g_r_export_interval;
            g_r_export_interval = atoi(argv[++i]);
            printf("[kuramoto] Per-cell R export every %d frames\n", g_r_export_interval);
        }
        else if (strcmp(argv[i], "--r-log-interval") == 0 && i+1 < argc) {
            extern int g_r_log_interval;
            g_r_log_interval = atoi(argv[++i]);
            printf("[kuramoto] Dense R(t) logging every %d frames\n", g_r_log_interval);
        }
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc) {
            extern unsigned int g_rng_seed;
            g_rng_seed = (unsigned int)atoi(argv[++i]);
            printf("[rng] Initial-condition seed: %u\n", g_rng_seed);
        }
        else if (strcmp(argv[i], "--headless") == 0) {
            extern bool g_headless;
            g_headless = true;
        }
        else if (strcmp(argv[i], "--target-radius") == 0 && i+1 < argc) {
            extern bool g_terminate_on_radius;
            extern float g_target_ring_radius;
            g_terminate_on_radius = true;
            g_target_ring_radius = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc) {
            extern int g_target_frames;
            g_target_frames = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--test-residence") == 0) {
            extern bool g_test_residence_time;
            g_test_residence_time = true;
        }
        else if (strcmp(argv[i], "--matched-amplitude") == 0) {
            extern bool g_matched_amplitude;
            extern float g_arm_boost_override;
            g_matched_amplitude = true;
            g_arm_boost_override = 1.25f;  // Match smooth amplitude
        }
        else if (strcmp(argv[i], "--hybrid") == 0) {
            extern bool g_hybrid_lod;
            g_hybrid_lod = true;
        }
        else if (strcmp(argv[i], "--octree-render") == 0) {
            extern bool g_octree_render;
            g_octree_render = true;
        }
        else if (strcmp(argv[i], "--octree-physics") == 0) {
            extern bool g_octree_physics;
            g_octree_physics = true;
        }
        else if (strcmp(argv[i], "--no-octree-physics") == 0) {
            extern bool g_octree_physics;
            g_octree_physics = false;
        }
        else if (strcmp(argv[i], "--no-octree-phase") == 0) {
            extern bool g_octree_phase;
            g_octree_phase = false;
            printf("[config] Octree phase evolution DISABLED (mip-tree only)\n");
        }
        else if (strcmp(argv[i], "--octree-phase") == 0) {
            extern bool g_octree_phase;
            g_octree_phase = true;
        }
        else if (strcmp(argv[i], "--octree-rebuild") == 0) {
            extern bool g_octree_rebuild;
            g_octree_rebuild = true;
            printf("[config] Octree rebuild ENABLED (Morton sort + stochastic tree)\n");
        }
        else if (strcmp(argv[i], "--no-octree-rebuild") == 0) {
            extern bool g_octree_rebuild;
            g_octree_rebuild = false;
        }
        else if (strcmp(argv[i], "--predictive-lock") == 0) {
            extern bool g_predictive_locking;
            g_predictive_locking = true;
            printf("[config] Predictive locking ENABLED (skip mip-tree when shells locked)\n");
        }
        else if (strcmp(argv[i], "--no-predictive-lock") == 0) {
            extern bool g_predictive_locking;
            g_predictive_locking = false;
            printf("[config] Predictive locking DISABLED (always rebuild mip-tree)\n");
        }
        else if (strcmp(argv[i], "--active-compact") == 0) {
            extern bool g_active_compaction;
            g_active_compaction = true;
            printf("[config] Active particle compaction ENABLED (skip static shell mass)\n");
        }
        else if (strcmp(argv[i], "--no-active-compact") == 0) {
            extern bool g_active_compaction;
            g_active_compaction = false;
            printf("[config] Active particle compaction DISABLED (scatter all particles)\n");
        }
        else if (strcmp(argv[i], "--grid-physics") == 0) {
            extern bool g_grid_physics;
            extern bool g_octree_physics;
            g_grid_physics = true;
            g_octree_physics = false;  // Grid replaces octree physics
        }
        else if (strcmp(argv[i], "--grid-flags") == 0) {
            extern bool g_grid_physics;
            extern bool g_grid_flags;
            extern bool g_octree_physics;
            g_grid_physics = true;
            g_grid_flags = true;
            g_octree_physics = false;  // Grid replaces octree physics
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <num>           Number of particles (default: 3500000)\n");
            printf("  --topology         Start with hopfion topology (default)\n");
            printf("  --hopfion          Alias for --topology\n");
            printf("  --no-topology      Start with smooth gradient (no discrete shells)\n");
            printf("  --smooth           Alias for --no-topology\n");
            printf("  --discrete-arms    Start with discrete arm boundaries (default)\n");
            printf("  --arm-topology     Alias for --discrete-arms\n");
            printf("  --smooth-arms      Start with smooth arm density waves\n");
            printf("  --no-arm-topology  Alias for --smooth-arms\n");
            printf("  --no-arms          Disable spiral arm structure entirely\n");
            printf("  --shear-k <k>      Frictional shear: density × sin(2φ) hybrid (default 0, try 2-10; scale-invariant)\n");
            printf("  --kuramoto-k <K>   Kuramoto phase coupling strength (default 0, sweep 0-2 to find K_c)\n");
            printf("  --omega-base <ω₀>  Mean natural frequency (default 1.0)\n");
            printf("  --omega-spread <σ> Gaussian σ for ω distribution (default 0.05; K_c ≈ 2σ/π)\n");
            printf("  --no-n12           Disable N12 mixer envelope on Kuramoto coupling\n");
            printf("  --r-export-interval <N>  Dump per-cell R grid to r_export/frame_NNNNN.bin every N frames (0=off)\n");
            printf("  --r-log-interval <N>     Print dense R(t) samples every N frames for time-series analysis (0=off)\n");
            printf("  --corner-threshold <f>   Passive/active pump_residual threshold (default 0.15)\n");
            printf("  --passive-tau <f>        Passive residual decay tau in sim-time units (default 5.0)\n");
            printf("  --shell-init             Initialize particles ON resonance shells (skip settling transient)\n");
            printf("  --no-spawn               Disable natural growth — particle count locked for clean measurements\n");
            printf("  --qr-corr                Dump [QR-corr] CSV rows each stats frame: R, Rrec, peaks, mass_frac, Q, shells, ..., active_frac\n");
            printf("  --headless         Disable rendering (physics + logging only, 10-20x speedup)\n");
            printf("  --hybrid           Enable hybrid LOD rendering (experimental)\n");
            printf("  --octree-render    Use octree traversal for render compaction\n");
            printf("  --octree-physics   Enable XOR neighbor stress physics (default: on)\n");
            printf("  --no-octree-physics Disable octree neighbor physics\n");
            printf("  --no-octree-phase  Disable octree phase evolution (mip-tree only)\n");
            printf("  --octree-rebuild   Enable Morton sort + octree rebuild (default: OFF)\n");
            printf("  --predictive-lock  Enable predictive locking (skip mip-tree when locked, default: ON)\n");
            printf("  --no-predictive-lock Disable predictive locking (always rebuild mip-tree)\n");
            printf("  --active-compact   Enable active particle compaction (skip static shell mass, default: ON)\n");
            printf("  --no-active-compact Disable active particle compaction (scatter all particles)\n");
            printf("  --grid-physics     Use streaming cell grid (DNA/RNA 30-frame cadence)\n");
            printf("  --grid-flags       Use sparse flags (optimal: no lists, no sort, no dedup)\n");
            printf("  --target-radius <R> Terminate when photon ring reaches radius R (instead of frame limit)\n");
            printf("  --frames <N>       Terminate after N frames (default: 50000)\n");
            printf("\nTest Suite (The Final Trilogy):\n");
            printf("  --test-residence   Test A: Track residence time (arm vs gap accumulation)\n");
            printf("  --matched-amplitude Test C: Set discrete boost=1.25× (isolate topology from amplitude)\n");
            printf("  --help, -h         Show this help message\n");
            printf("\nControls:\n");
            printf("  H key              Toggle radial topology mode at runtime\n");
            printf("  A key              Toggle arm topology mode at runtime\n");
            printf("  L key              Toggle hybrid LOD culling (requires --hybrid)\n");
            printf("  E key              Inject entropy cluster\n");
            printf("  C key              Cycle color modes\n");
            printf("  Space              Pause/unpause\n");
            printf("  R key              Reset camera\n");
            return 0;
        }
    }

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       BLACKHOLE V20 - Siphon Pump State Machine              ║\n");
    printf("║       12↔16 Dimensional Circulation Visualizer               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // === DYNAMIC VRAM CONFIGURATION ===
    // Query GPU memory and set grid size + particle cap BEFORE any allocations
    // This must happen first so all subsequent code uses correct grid dimensions
    initVRAMConfig();

    // === CUDA LUT INITIALIZATION ===
    // Initialize lookup tables for fast trigonometry in hot loops
    cuda_lut_init();                    // Quarter-sector sine table (2KB)
    cuda_lut_gaussian_init(12.0f);      // Gaussian exp(-r²/2σ²) for density (σ=12)
    cuda_lut_repulsion_init(6.0f);      // Repulsion exp(-r/λ) for soft forces (λ=6)

    // Clamp user-requested particles to VRAM-safe limit
    if (num_particles > g_runtime_particle_cap) {
        printf("[config] Requested %d particles, clamping to VRAM-safe cap %d\n",
               num_particles, g_runtime_particle_cap);
        num_particles = g_runtime_particle_cap;
    }

    printf("[config] Particles: %d\n", num_particles);
    printf("[config] Topology: %s\n", g_use_hopfion_topology ? "Hopfion shells (discrete)" : "Smooth gradient (continuous)");
    printf("[config] Mode: %s\n", g_headless ? "HEADLESS (physics + logging only)" : "INTERACTIVE (with rendering)");

    // === CONDITIONAL RENDERING SETUP ===
#ifdef VULKAN_INTEROP
    // === VULKAN INITIALIZATION ===
    VulkanContext vkCtx;
    // Allocate for full growth potential — spawning can grow 5-10x from seed
    // Using MAX_DISK_PTS ensures buffer never overflows regardless of growth
    int max_render_particles = MAX_DISK_PTS;
    vkCtx.particleCount = max_render_particles;

    // Initialize GLFW for Vulkan (no OpenGL context)
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // No OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Fixed size (helps with tiling WMs)
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);     // Request floating window (Wayland hint)

    vkCtx.window = glfwCreateWindow(WIDTH, HEIGHT, "Siphon Pump Black Hole — V20 (Vulkan)", NULL, NULL);
    if (!vkCtx.window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }

    // Mouse callbacks for camera control
    glfwSetWindowUserPointer(vkCtx.window, &vkCtx);
    glfwSetMouseButtonCallback(vkCtx.window, [](GLFWwindow* w, int button, int action, int) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            ctx->mousePressed = (action == GLFW_PRESS);
            if (ctx->mousePressed) glfwGetCursorPos(w, &ctx->lastMouseX, &ctx->lastMouseY);
        }
    });
    glfwSetCursorPosCallback(vkCtx.window, [](GLFWwindow* w, double x, double y) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        if (ctx->mousePressed) {
            float dx = (float)(x - ctx->lastMouseX);
            float dy = (float)(y - ctx->lastMouseY);
            ctx->cameraYaw += dx * 0.005f;
            ctx->cameraPitch += dy * 0.005f;
            ctx->cameraPitch = fmaxf(-1.5f, fminf(1.5f, ctx->cameraPitch));
            ctx->lastMouseX = x;
            ctx->lastMouseY = y;
        }
    });
    glfwSetScrollCallback(vkCtx.window, [](GLFWwindow* w, double, double yoff) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        ctx->cameraRadius *= (yoff > 0) ? 0.9f : 1.1f;
        ctx->cameraRadius = fmaxf(5.0f, fminf(5000.0f, ctx->cameraRadius));
    });
    glfwSetKeyCallback(vkCtx.window, [](GLFWwindow* w, int key, int, int action, int mods) {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
            else if (key == GLFW_KEY_SPACE) g_cam.paused = !g_cam.paused;
            else if (key == GLFW_KEY_R) {
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                ctx->cameraYaw = 0; ctx->cameraPitch = 0.3f; ctx->cameraRadius = 800.0f;
            }
            else if (key == GLFW_KEY_H) {
                extern bool g_use_hopfion_topology;
                g_use_hopfion_topology = !g_use_hopfion_topology;
                printf("[toggle] Radial topology: %s\n", g_use_hopfion_topology ? "Hopfion shells" : "Smooth gradient");
            }
            else if (key == GLFW_KEY_L) {
                // Toggle hybrid LOD (culling) at runtime for perf testing
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                ctx->useIndirectDraw = !ctx->useIndirectDraw;
                printf("[toggle] Hybrid LOD (particle culling): %s\n", ctx->useIndirectDraw ? "ON" : "OFF");
            }
            else if (key == GLFW_KEY_V) {
                // Cycle shell brightness: 100% → 50% → 25% → OFF → 100%
                // Allows viewing jets without volumetric shell glare
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                if (ctx->shellBrightness > 0.9f) {
                    ctx->shellBrightness = 0.5f;
                    printf("[shells] Brightness: 50%%\n");
                } else if (ctx->shellBrightness > 0.4f) {
                    ctx->shellBrightness = 0.25f;
                    printf("[shells] Brightness: 25%%\n");
                } else if (ctx->shellBrightness > 0.1f) {
                    ctx->shellBrightness = 0.0f;
                    printf("[shells] Brightness: OFF (particles only)\n");
                } else {
                    ctx->shellBrightness = 1.0f;
                    printf("[shells] Brightness: 100%%\n");
                }
            }
            else if (key == GLFW_KEY_P) {
                extern AttractorPipeline g_attractor;
                bool shift_held = (mods & GLFW_MOD_SHIFT) != 0;

                if (shift_held && g_attractor.purePipeline != VK_NULL_HANDLE) {
                    // Shift+P: toggle pure attractor mode (easter egg)
                    if (g_attractor.mode == AttractorMode::PURE_ATTRACTOR) {
                        g_attractor.mode = AttractorMode::POSITION_PRIMARY;
                    } else {
                        g_attractor.mode = AttractorMode::PURE_ATTRACTOR;
                    }
                } else {
                    // P: toggle between position-primary and phase-primary
                    if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                        g_attractor.mode = AttractorMode::POSITION_PRIMARY;
                    } else if (g_attractor.phasePipeline != VK_NULL_HANDLE) {
                        g_attractor.mode = AttractorMode::PHASE_PRIMARY;
                    }
                }

                const char* modeNames[] = {
                    "POSITION-PRIMARY (reads CUDA particle xyz)",
                    "PHASE-PRIMARY (flattened disk, shows shell rings)",
                    "PURE ATTRACTOR (parametric GPU sampling)"
                };
                printf("[V20] Render mode: %s\n", modeNames[static_cast<int>(g_attractor.mode)]);
            }
            else if (key == GLFW_KEY_X) {
                // Export single frame for validation (uses global validation context)
                if (g_validation_ctx.d_disk) {
                    system("mkdir -p frames/");
                    exportFrameBinary(g_validation_ctx.d_disk, g_validation_ctx.N_current,
                                      g_validation_ctx.export_frame_id);
                    exportValidationMetadata(g_validation_ctx.export_frame_id,
                                             g_validation_ctx.N_current,
                                             g_validation_ctx.sim_time,
                                             g_validation_ctx.heartbeat,
                                             g_validation_ctx.avg_scale,
                                             g_validation_ctx.avg_residual,
                                             g_grid_dim, 500.0f);
                    g_validation_ctx.export_frame_id++;
                } else {
                    printf("[validator] Error: Simulation not yet initialized\n");
                }
            }
            else if (key == GLFW_KEY_F) {
                // Start stack capture (F = Frames) - dumps 64 consecutive frames
                if (!isStackCaptureActive()) {
                    startStackCapture();
                } else {
                    printf("[validator] Stack capture already in progress (%d remaining)\n",
                           g_stack_capture_remaining);
                }
            }
            else if (key == GLFW_KEY_T) {
                // Manual topology ring buffer dump (T = Topology)
                printf("[topo] Manual dump triggered...\n");
                topology_recorder_dump("manual");
            }
        }
    });

    // Initialize Vulkan
    printf("[vulkan] Initializing Vulkan renderer...\n");
    vk::createInstance(vkCtx);
    vk::setupDebugMessenger(vkCtx);
    if (glfwCreateWindowSurface(vkCtx.instance, vkCtx.window, nullptr, &vkCtx.surface) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create window surface\n"); return 1;
    }
    vk::pickPhysicalDevice(vkCtx);
    vk::createLogicalDevice(vkCtx);
    vk::createSwapchain(vkCtx);
    vk::createImageViews(vkCtx);
    vk::createRenderPass(vkCtx);
    vk::createDescriptorSetLayout(vkCtx);
    vk::createGraphicsPipeline(vkCtx);
    vk::createDepthResources(vkCtx);
    vk::createFramebuffers(vkCtx);
    vk::createCommandPool(vkCtx);
    vk::createUniformBuffers(vkCtx);
    vk::createDescriptorPool(vkCtx);
    vk::createDescriptorSets(vkCtx);

    // Volume rendering pipeline (analytic far-field shells)
    vk::createVolumeDescriptorSetLayout(vkCtx);
    vk::createVolumePipeline(vkCtx);
    vk::createVolumeUniformBuffers(vkCtx);
    vk::createVolumeDescriptorSets(vkCtx);

    vk::createCommandBuffers(vkCtx);
    vk::createSyncObjects(vkCtx);

    // === CREATE SHARED CUDA-VULKAN BUFFER ===
    // Size for growth capacity (2x initial), not just initial count
    printf("[vulkan] Creating CUDA-Vulkan shared buffer for %d particles (capacity for growth)...\n", max_render_particles);
    SharedBuffer sharedParticleBuffer;
    if (createSharedBuffer(vkCtx.device, vkCtx.physicalDevice, max_render_particles, &sharedParticleBuffer) != 0) {
        fprintf(stderr, "Failed to create shared buffer\n"); return 1;
    }
    if (importBufferToCUDA(&sharedParticleBuffer) != 0) {
        fprintf(stderr, "Failed to import buffer to CUDA\n"); return 1;
    }

    // Store the shared buffer in context for rendering
    vkCtx.particleBuffer = sharedParticleBuffer.vkBuffer;
    vkCtx.particleBufferMemory = sharedParticleBuffer.vkMemory;

    // Get the CUDA pointer for the fill kernel
    ParticleVertex* d_vkParticles = (ParticleVertex*)sharedParticleBuffer.cudaPtr;

    printf("[vulkan] Initialization complete! Shared buffer at CUDA ptr=%p\n", d_vkParticles);

    // Density rendering pipeline (now that particle buffer exists)
    // Press 'M' to toggle density mode
    try {
        vk::createAttractorPipeline(vkCtx, g_attractor);
    } catch (const std::exception& e) {
        fprintf(stderr, "[V20] Warning: Density pipeline not available: %s\n", e.what());
    }

    // === CREATE DENSITY GRID FOR HYBRID LOD ===
    SharedDensityGrid densityGrid = {};
    float* d_densityGrid = nullptr;
    unsigned int* d_nearCount = nullptr;
    // Hybrid LOD is disabled by default until volume rendering is fully implemented
    // The LOD kernel adds overhead without benefit until we can skip far vertices
    extern bool g_hybrid_lod;
    bool hybridLODEnabled = g_hybrid_lod;  // Use --hybrid flag to enable

    // === CREATE INDIRECT DRAW RESOURCES FOR STREAM COMPACTION ===
    // Double-buffered: physics writes to back buffer, renderer reads front buffer
    SharedIndirectDraw indirectDrawBuffers[2] = {};
    int frontBuffer = 0;  // Renderer reads this
    int backBuffer = 1;   // Compaction writes to this

    // Current frame's CUDA pointers (updated each frame based on backBuffer)
    ParticleVertex* d_compactedParticles = nullptr;
    CUDADrawIndirectCommand* d_drawCommand = nullptr;
    unsigned int* d_writeIndex = nullptr;
    bool doubleBufferEnabled = false;

    if (hybridLODEnabled) {
        printf("[hybrid] Creating density grid for volumetric far-field...\n");
        if (createSharedDensityGrid(vkCtx.device, vkCtx.physicalDevice, &densityGrid) != 0) {
            printf("[hybrid] WARNING: Failed to create density grid, falling back to points-only\n");
            hybridLODEnabled = false;
        } else if (importDensityGridToCUDA(&densityGrid) != 0) {
            printf("[hybrid] WARNING: Failed to import density grid to CUDA, falling back to points-only\n");
            destroySharedDensityGrid(vkCtx.device, &densityGrid);
            hybridLODEnabled = false;
        } else {
            d_densityGrid = densityGrid.cudaLinearPtr;
            // Allocate counter for near particles
            cudaMalloc(&d_nearCount, sizeof(unsigned int));
            cudaMemset(d_nearCount, 0, sizeof(unsigned int));
            printf("[hybrid] Density grid created successfully\n");
        }

        // Create DOUBLE-BUFFERED indirect draw buffers for stream compaction
        // This decouples physics from rendering - no sync needed
        if (hybridLODEnabled) {
            printf("[hybrid] Creating double-buffered indirect draw buffers...\n");
            bool buffersOK = true;

            for (int i = 0; i < 2 && buffersOK; i++) {
                if (createSharedIndirectDraw(vkCtx.device, vkCtx.physicalDevice, num_particles, &indirectDrawBuffers[i]) != 0) {
                    printf("[hybrid] WARNING: Failed to create indirect draw buffer %d\n", i);
                    buffersOK = false;
                } else if (importIndirectDrawToCUDA(&indirectDrawBuffers[i]) != 0) {
                    printf("[hybrid] WARNING: Failed to import indirect draw buffer %d to CUDA\n", i);
                    destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
                    buffersOK = false;
                }
            }

            if (!buffersOK) {
                // Cleanup any partially created buffers
                for (int i = 0; i < 2; i++) {
                    if (indirectDrawBuffers[i].compactedBuffer != VK_NULL_HANDLE) {
                        destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
                    }
                }
                printf("[hybrid] Falling back to single-buffered mode\n");
            } else {
                doubleBufferEnabled = true;

                // Initialize CUDA pointers to back buffer (where compaction will write)
                d_compactedParticles = (ParticleVertex*)indirectDrawBuffers[backBuffer].cudaCompactedPtr;
                d_drawCommand = (CUDADrawIndirectCommand*)indirectDrawBuffers[backBuffer].cudaIndirectPtr;
                d_writeIndex = indirectDrawBuffers[backBuffer].cudaWriteIndex;

                // Wire up Vulkan context to front buffer (where renderer will read)
                vkCtx.compactedParticleBuffer = indirectDrawBuffers[frontBuffer].compactedBuffer;
                vkCtx.compactedParticleBufferMemory = indirectDrawBuffers[frontBuffer].compactedMemory;
                vkCtx.indirectDrawBuffer = indirectDrawBuffers[frontBuffer].indirectBuffer;
                vkCtx.indirectDrawBufferMemory = indirectDrawBuffers[frontBuffer].indirectMemory;
                vkCtx.useIndirectDraw = true;

                printf("[hybrid] Double-buffered stream compaction enabled!\n");
                printf("[hybrid]   Buffer A: compacted=%p indirect=%p\n",
                       indirectDrawBuffers[0].cudaCompactedPtr, indirectDrawBuffers[0].cudaIndirectPtr);
                printf("[hybrid]   Buffer B: compacted=%p indirect=%p\n",
                       indirectDrawBuffers[1].cudaCompactedPtr, indirectDrawBuffers[1].cudaIndirectPtr);
                printf("[hybrid] Physics writes to back, renderer reads front - zero sync\n");
            }
        }
    }

    // Dummy window pointer for compatibility with headless checks
    GLFWwindow* window = vkCtx.window;

#else
    // === OPENGL INITIALIZATION ===
    GLFWwindow* window = nullptr;
    if (!g_headless) {
        // GLFW init
        // Force X11 platform for GLX-based GLEW compatibility on Wayland systems
        #if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 4
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        #endif
        if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        window = glfwCreateWindow(WIDTH, HEIGHT,
                                  "Siphon Pump Black Hole — V20", NULL, NULL);
        if (!window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);  // Disable vsync - uncapped FPS

        // GLEW must init after context is current — loads all GL 3.3+ function pointers
        glewExperimental = GL_TRUE;
        GLenum glew_err = glewInit();
        if (glew_err != GLEW_OK) {
            fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(glew_err));
            return 1;
        }
        // glewInit sometimes triggers a benign GL_INVALID_ENUM — clear it
        glGetError();

        glfwSetMouseButtonCallback(window, mouseButtonCB);
        glfwSetCursorPosCallback(window, cursorPosCB);
        glfwSetScrollCallback(window, scrollCB);
        glfwSetKeyCallback(window, keyCB);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_MULTISAMPLE);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Shaders (only in interactive mode)
    GLuint bhProgram = 0, diskProgram = 0;
    if (!g_headless) {
        bhProgram = linkProgram(quadVS, bhFS);
        diskProgram = linkProgram(diskVS, diskFS);
    }
#endif

    // Initialize particles
    int N = num_particles;
    std::vector<float> h_px(N), h_py(N), h_pz(N);
    std::vector<float> h_vx(N), h_vy(N), h_vz(N);  // Velocity for 3D motion
    // NOTE: h_dr, h_dphi, h_temp, h_in_disk no longer needed — computed on-demand
    std::vector<int> h_state(N);
    std::vector<float> h_scale(N);
    std::vector<int> h_coherent(N);
    std::vector<uint8_t> h_seam(N);
    std::vector<float> h_residual(N), h_work(N);
    std::vector<float> h_history(N);  // pump_history for emergence
    std::vector<float> h_theta(N);      // Kuramoto phase, uniform random [0, 2π)
    std::vector<float> h_omega_nat(N);  // Kuramoto natural freq, Gaussian(ω₀, σ)
    std::vector<uint8_t> h_flags(N);    // packed PFLAG_ACTIVE | PFLAG_EJECTED

    std::mt19937 rng(g_rng_seed);
    std::uniform_real_distribution<float> runif(0.0f, 1.0f);
    std::uniform_real_distribution<float> rphase(0, TWO_PI);
    std::normal_distribution<float> rnorm(0.0f, 1.0f);

    // === INITIALIZATION ===
    // Two modes:
    //   Default (uniform box): All particles in a large rectangular volume.
    //     Structure emerges from dynamics over ~5000 frames of settling.
    //   --shell-init: Particles placed ON the 8 resonance shells from
    //     d_shell_radii[] with small radial jitter and full Keplerian velocity.
    //     Skips the settling transient — most particles start passive.

    // Host-side copy of d_shell_radii (device __constant__) for init.
    static const float h_shell_radii[8] = {
        6.0f, 9.7f, 15.7f, 25.4f, 41.1f, 66.5f, 107.5f, 174.0f
    };

    float box_half = DISK_OUTER_R;        // ±120 in X and Z
    float box_height = box_half * 0.3f;   // ±36 in Y (thinner to encourage disk formation)

    for (int i = 0; i < N; i++) {
        float x, y, z;

        if (g_shell_init) {
            // Step 6: shell-aware initialization.
            // Distribute particles across 8 shells weighted by 1/r (inner shells denser).
            // Small radial jitter (σ = 0.5 units) + random azimuthal phase.
            float weight_sum = 0.0f;
            for (int s = 0; s < 8; s++) weight_sum += 1.0f / h_shell_radii[s];
            float pick = runif(rng) * weight_sum;
            float accum = 0.0f;
            int shell = 7;  // fallback to outermost
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
            // Vertical spread proportional to shell radius — flared disk.
            // Inner shells (r=6): σ_y ≈ 1.2, outer shells (r=174): σ_y ≈ 35.
            // Gives a volumetric cloud/nebula appearance, not a paper-thin ring.
            y = rnorm(rng) * r_shell * 0.2f;
        } else {
            // Legacy uniform box initialization.
            x = (runif(rng) * 2.0f - 1.0f) * box_half;
            y = (runif(rng) * 2.0f - 1.0f) * box_height;
            z = (runif(rng) * 2.0f - 1.0f) * box_half;

            // Avoid spawning too close to black hole
            float r = sqrtf(x*x + y*y + z*z);
            if (r < SCHW_R * 3.0f) {
                float scale = (SCHW_R * 3.0f) / r;
                x *= scale; y *= scale; z *= scale;
            }
        }

        h_px[i] = x;
        h_py[i] = y;
        h_pz[i] = z;

        // Initial velocity: Keplerian tangential + thermal noise.
        // --shell-init uses full Keplerian (1.0×); legacy uses 0.3×.
        float r_xz = sqrtf(x*x + z*z);
        if (r_xz > 0.1f) {
            float rot_sign = g_retrograde_init ? -1.0f : 1.0f;
            float v_frac = g_shell_init ? 1.0f : 0.3f;
            float v_rot = rot_sign * v_frac * sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            float thermal = g_shell_init ? 0.01f : 0.05f;
            h_vx[i] = -v_rot * (z / r_xz) + rnorm(rng) * thermal;
            h_vy[i] = rnorm(rng) * (g_shell_init ? 0.005f : 0.03f);
            h_vz[i] = v_rot * (x / r_xz) + rnorm(rng) * thermal;
        } else {
            h_vx[i] = rnorm(rng) * 0.05f;
            h_vy[i] = rnorm(rng) * 0.05f;
            h_vz[i] = rnorm(rng) * 0.05f;
        }

        // NOTE: disk_r, disk_phi, temp, in_disk no longer stored — computed on-demand

        // Pump state: start IDLE, seam closed
        // Particles "turn on" when they enter the disk plane
        h_state[i] = 0;  // PUMP_IDLE
        h_scale[i] = 1.0f;
        h_coherent[i] = 0;
        h_seam[i] = 0x00;  // SEAM_CLOSED - opens based on dynamics
        h_residual[i] = 0.0f;
        h_work[i] = 0.0f;
        h_history[i] = 1.0f;

        // Kuramoto init: uniform random phase, Gaussian natural frequency
        h_theta[i] = rphase(rng);
        h_omega_nat[i] = g_omega_base + g_omega_spread * rnorm(rng);

        h_flags[i] = PFLAG_ACTIVE;  // active, not ejected
    }

    // Debug: verify active count
    int active_count = 0;
    for (int i = 0; i < N; i++) if (h_flags[i] & PFLAG_ACTIVE) active_count++;
    printf("[lattice] %d particles initialized in %.0f×%.0f×%.0f box (active=%d)\n",
           N, box_half*2, box_height*2, box_half*2, active_count);
    printf("[kuramoto-init] theta uniform[0,2π), omega_nat ~ N(%.2f, %.2f)\n",
           g_omega_base, g_omega_spread);

    // GPU allocation
    GPUDisk* d_disk;
    cudaMalloc(&d_disk, sizeof(GPUDisk));

    // V8-STYLE: Zero-initialize ENTIRE struct before uploading seed particles
    // This ensures active[i]=false and ejected[i]=false for all i >= N
    // Without this, uninitialized memory causes garbage reads when N_current grows
    // Reference: aizawa_slab.cuh line 209 — V8 zero-inits pools at allocation
    cudaMemset(d_disk, 0, sizeof(GPUDisk));

    // Upload using offsetof to avoid dereferencing device pointer on host
    #define UPLOAD(field, hvec) \
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, field), hvec.data(), N*sizeof(hvec[0]), cudaMemcpyHostToDevice)

    UPLOAD(pos_x, h_px);
    UPLOAD(pos_y, h_py);
    UPLOAD(pos_z, h_pz);
    UPLOAD(vel_x, h_vx);
    UPLOAD(vel_y, h_vy);
    UPLOAD(vel_z, h_vz);
    // NOTE: disk_r, disk_phi, temp, in_disk no longer uploaded — computed on-demand
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
    #undef UPLOAD

    StressCounters* d_stress;
    cudaMalloc(&d_stress, sizeof(StressCounters));

    // === KURAMOTO ORDER PARAMETER BUFFERS ===
    // Pre-allocate for the largest expected particle count so we don't
    // reallocate if N_current grows via spawning.
    const int KR_THREADS = 256;
    int kr_max_blocks = (RUNTIME_PARTICLE_CAP + KR_THREADS - 1) / KR_THREADS;
    float* d_kr_sin_sum;
    float* d_kr_cos_sum;
    int* d_kr_count;
    cudaMalloc(&d_kr_sin_sum, kr_max_blocks * sizeof(float));
    cudaMalloc(&d_kr_cos_sum, kr_max_blocks * sizeof(float));
    cudaMalloc(&d_kr_count, kr_max_blocks * sizeof(int));
    // Host buffers for the block partial sums (small — one float per block)
    std::vector<float> h_kr_sin_sum(kr_max_blocks);
    std::vector<float> h_kr_cos_sum(kr_max_blocks);
    std::vector<int> h_kr_count(kr_max_blocks);
    float R_global_cached = 0.0f;  // Last computed order parameter

    // Phase histogram (for multi-domain clustering confirmation) and
    // per-bin ω statistics (for velocity filter / sieve verification)
    int* d_phase_hist;
    float* d_phase_omega_sum;
    float* d_phase_omega_sq;
    cudaMalloc(&d_phase_hist, PHASE_HIST_BINS * sizeof(int));
    cudaMalloc(&d_phase_omega_sum, PHASE_HIST_BINS * sizeof(float));
    cudaMalloc(&d_phase_omega_sq, PHASE_HIST_BINS * sizeof(float));
    std::vector<int> h_phase_hist(PHASE_HIST_BINS);
    std::vector<float> h_phase_omega_sum(PHASE_HIST_BINS);
    std::vector<float> h_phase_omega_sq(PHASE_HIST_BINS);

    // === NATURAL GROWTH: Spawn counter for atomic allocation ===
    // V8-style: separate counter for attempted slots vs successful spawns
    unsigned int* d_spawn_idx;      // Atomic slot allocation (may exceed capacity)
    unsigned int* d_spawn_success;  // Only counts SUCCESSFUL spawns (V8 pattern)
    cudaMalloc(&d_spawn_idx, sizeof(unsigned int));
    cudaMalloc(&d_spawn_success, sizeof(unsigned int));
    cudaMemset(d_spawn_idx, 0, sizeof(unsigned int));
    cudaMemset(d_spawn_success, 0, sizeof(unsigned int));
    int N_current = N;  // Track current particle count (grows over time)

    // === STRATIFIED SAMPLING SETUP ===
    // Generate 128 sample indices distributed across radial/angular bins
    std::vector<int> h_sample_indices(SAMPLE_COUNT);
    generateStratifiedSamples(h_sample_indices.data(), h_px.data(), h_pz.data(), N, ISCO_R, DISK_OUTER_R);

    // Upload sample indices to GPU
    int* d_sample_indices;
    cudaMalloc(&d_sample_indices, SAMPLE_COUNT * sizeof(int));
    cudaMemcpy(d_sample_indices, h_sample_indices.data(), SAMPLE_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    // === ASYNC DOUBLE-BUFFERING SETUP ===
    // Two buffers for async transfer - read from one while writing to the other
    SampleMetrics* d_sample_metrics[2];
    cudaMalloc(&d_sample_metrics[0], sizeof(SampleMetrics));
    cudaMalloc(&d_sample_metrics[1], sizeof(SampleMetrics));

    // Host-side pinned memory for async transfer (faster than pageable)
    SampleMetrics* h_sample_metrics;
    cudaMallocHost(&h_sample_metrics, sizeof(SampleMetrics));
    h_sample_metrics->avg_scale = 1.0f;
    h_sample_metrics->avg_residual = 0.0f;
    h_sample_metrics->total_work = 0.0f;

    // CUDA stream for async operations
    cudaStream_t sample_stream;
    cudaStreamCreate(&sample_stream);

    // Async stats stream - reduction runs without blocking main loop
    cudaStream_t stats_stream;
    cudaStreamCreate(&stats_stream);
    cudaEvent_t stats_ready;
    cudaEventCreate(&stats_ready);
    StressCounters* d_stress_async = nullptr;  // Async buffer (separate from d_stress)
    cudaMalloc(&d_stress_async, sizeof(StressCounters));
    StressCounters h_stats_cache = {};  // Cached results from last completed reduction
    bool stats_pending = false;  // True if async reduction is in flight

    // Async spawn count stream - read previous frame's spawns without blocking
    cudaStream_t spawn_stream;
    cudaStreamCreate(&spawn_stream);
    cudaEvent_t spawn_ready;
    cudaEventCreate(&spawn_ready);
    unsigned int* h_spawn_pinned = nullptr;  // Pinned memory for async copy
    cudaMallocHost(&h_spawn_pinned, sizeof(unsigned int));
    *h_spawn_pinned = 0;
    bool spawn_pending = false;  // True if spawn count copy is in flight

    int current_buffer = 0;  // Double-buffer index

    // === TOPOLOGY RING BUFFER INITIALIZATION ===
    // Records downsampled m-field (64³) every frame for crystal detection
    if (!topology_recorder_init()) {
        fprintf(stderr, "[topo] WARNING: Failed to initialize topology recorder\n");
    }

    // === OCTREE ALLOCATIONS ===
    // Morton-sorted spatial tree for unified physics/rendering
    uint64_t* d_morton_keys = nullptr;
    uint32_t* d_xor_corners = nullptr;
    uint32_t* d_particle_ids = nullptr;
    OctreeNode* d_octree_nodes = nullptr;
    uint32_t* d_node_count = nullptr;
    uint32_t* d_leaf_counts = nullptr;       // Particle counts for level-13 nodes (pristine)
    uint32_t* d_leaf_counts_culled = nullptr; // Working copy for frustum culling
    uint32_t* d_leaf_offsets = nullptr;      // Exclusive scan of counts (output positions)
    uint32_t* d_leaf_node_indices = nullptr; // Original node indices for level-13 nodes
    uint32_t* d_leaf_node_count = nullptr;   // Number of level-13 nodes
    extern bool g_octree_rebuild;
    extern bool g_octree_render;
    // Octree subsystem is lazy-allocated: only built if either the rebuild
    // path or the render traversal is opted into via CLI flag. With neither
    // flag set (the default), the entire octree block at line ~4570 is
    // skipped and ~320 MB of morton/xor/ids buffers plus ~88 MB of
    // node/leaf/hash buffers stay unallocated. Downstream guards at
    // g_octree_rebuild && octreeEnabled and h_leaf_node_count > 0 ensure
    // no kernel tries to read null pointers.
    const bool octreeEnabled = (g_octree_rebuild || g_octree_render);
    bool useOctreeTraversal = g_octree_render;  // Toggle to use octree-based compaction
    uint32_t h_analytic_node_count = 0;
    uint32_t h_total_node_count = 0;  // Updated after each stochastic rebuild
    uint32_t h_leaf_node_count = 0;   // Number of level-13 nodes
    uint32_t h_cached_total_particles = 0;  // Sum of particle_counts in all leaves (no culling)
    uint32_t h_culled_total_particles = 0;  // Sum after frustum culling
    uint32_t h_num_active = 0;              // Active particle count (for physics kernel)
    extern bool g_octree_physics;
    bool useOctreePhysics = g_octree_physics;  // Enable octree-based neighbor physics
    float pressure_k = 0.03f;                  // Pressure coefficient for F = -k∇ρ
    float vorticity_k = 0.01f;                 // Vorticity coefficient for F = k(ω × v)
    float substrate_k = 0.05f;                 // Keplerian substrate coupling (competes with Kuramoto)
    float phase_coupling_k = 0.05f;            // Phase coupling coefficient (Kuramoto K)
    // Leaf velocity buffers for vorticity computation
    float* d_leaf_vel_x = nullptr;
    float* d_leaf_vel_y = nullptr;
    float* d_leaf_vel_z = nullptr;
    // S3: Phase state buffers for temporal coherence
    float* d_leaf_phase = nullptr;             // θ ∈ [0, 2π] - oscillation phase
    float* d_leaf_frequency = nullptr;         // ω - local oscillation frequency
    float* d_leaf_coherence = nullptr;         // Pre-computed phase coherence
    // Leaf hash table for O(1) neighbor lookup (replaces binary search)
    uint64_t* d_leaf_hash_keys = nullptr;      // Morton keys in hash table
    uint32_t* d_leaf_hash_values = nullptr;    // Leaf indices in hash table
    uint32_t  h_leaf_hash_size = 0;            // Hash table size (power of 2)

    // Octree buffers sized to g_octree_particle_cap (VRAM-aware, not N*2)
    // The octree is the crystallization: 24 = LAMBDA_OCTREE (finished stone layer)
    // Morton buffers scale with particle cap, fixed buffers use OCTREE_MAX_NODES
    int morton_capacity = g_octree_particle_cap;
    if (morton_capacity > MAX_DISK_PTS) morton_capacity = MAX_DISK_PTS;

    if (octreeEnabled) {
        // Per-particle buffers: sized to octree cap (VRAM-aware)
        // NOTE: Morton sort removed — keys used only for active counting
        cudaMalloc(&d_morton_keys, morton_capacity * sizeof(uint64_t));
        cudaMalloc(&d_xor_corners, morton_capacity * sizeof(uint32_t));
        cudaMalloc(&d_particle_ids, morton_capacity * sizeof(uint32_t));
        // Fixed buffers: always OCTREE_MAX_NODES
        cudaMalloc(&d_octree_nodes, OCTREE_MAX_NODES * sizeof(OctreeNode));  // 48 MB
        cudaMalloc(&d_node_count, sizeof(uint32_t));
        cudaMemset(d_node_count, 0, sizeof(uint32_t));
        // Prefix scan buffers for atomic-free traversal
        cudaMalloc(&d_leaf_counts, OCTREE_MAX_NODES * sizeof(uint32_t));        // 4 MB
        cudaMalloc(&d_leaf_counts_culled, OCTREE_MAX_NODES * sizeof(uint32_t)); // 4 MB (working copy)
        cudaMalloc(&d_leaf_offsets, OCTREE_MAX_NODES * sizeof(uint32_t));       // 4 MB
        cudaMalloc(&d_leaf_node_indices, OCTREE_MAX_NODES * sizeof(uint32_t));  // 4 MB
        cudaMalloc(&d_leaf_node_count, sizeof(uint32_t));
        // Leaf velocity buffers for vorticity computation (~192 KB each for 16k leaves)
        cudaMalloc(&d_leaf_vel_x, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&d_leaf_vel_y, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&d_leaf_vel_z, OCTREE_MAX_NODES * sizeof(float));
        // S3: Phase state buffers (~64 KB each for 16k leaves)
        cudaMalloc(&d_leaf_phase, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&d_leaf_frequency, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&d_leaf_coherence, OCTREE_MAX_NODES * sizeof(float));
        // Initialize phase to zero (will be seeded from position on first use)
        cudaMemset(d_leaf_phase, 0, OCTREE_MAX_NODES * sizeof(float));
        cudaMemset(d_leaf_frequency, 0, OCTREE_MAX_NODES * sizeof(float));
        cudaMemset(d_leaf_coherence, 0, OCTREE_MAX_NODES * sizeof(float));
        // Leaf hash table: Fixed L2-resident size
        // 256K entries = 3MB (uses full RTX 2060 L2 cache)
        // Supports up to 128K leaves at 50% load factor
        h_leaf_hash_size = 262144;
        cudaMalloc(&d_leaf_hash_keys, h_leaf_hash_size * sizeof(uint64_t));
        cudaMalloc(&d_leaf_hash_values, h_leaf_hash_size * sizeof(uint32_t));
        // Initialize keys to UINT64_MAX (empty marker)
        cudaMemset(d_leaf_hash_keys, 0xFF, h_leaf_hash_size * sizeof(uint64_t));

        printf("[octree] Allocated: morton cap=%d (%.1f MB), nodes=%zuMB\n",
               morton_capacity, morton_capacity * 16 / 1e6,
               OCTREE_MAX_NODES * sizeof(OctreeNode) / (1024 * 1024));
        printf("[octree] Hash: %u entries (%.1f MB) — L2 resident, max %u leaves\n",
               h_leaf_hash_size, h_leaf_hash_size * 12 / 1e6, h_leaf_hash_size / 2);

        // Build frozen analytic tree once at init
        float boxSize = 500.0f;  // Covers DISK_OUTER_R * 2
        int maxNodesAtLevel5 = 1 << (ANALYTIC_MAX_LEVEL * 3);  // 32768
        int analyticBlocks = (maxNodesAtLevel5 + 255) / 256;
        buildAnalyticTree<<<analyticBlocks, 256>>>(
            d_octree_nodes, d_node_count, boxSize, 0.0f, ANALYTIC_MAX_LEVEL
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_analytic_node_count, d_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("[octree] Frozen analytic tree: %u nodes (levels 0-%d)\n",
               h_analytic_node_count, ANALYTIC_MAX_LEVEL);
    }

    // === CELL GRID ALLOCATION (DNA/RNA Streaming Architecture) ===
    // Fixed-topology grid for forward-only physics passes
    extern bool g_grid_physics;
    float* d_grid_density = nullptr;
    float* d_grid_momentum_x = nullptr;
    float* d_grid_momentum_y = nullptr;
    float* d_grid_momentum_z = nullptr;
    float* d_grid_phase_sin = nullptr;
    float* d_grid_phase_cos = nullptr;
    float* d_grid_pressure_x = nullptr;
    float* d_grid_pressure_y = nullptr;
    float* d_grid_pressure_z = nullptr;
    float* d_grid_vorticity_x = nullptr;
    float* d_grid_vorticity_y = nullptr;
    float* d_grid_vorticity_z = nullptr;
    float* d_grid_R_cell = nullptr;      // per-cell Kuramoto order parameter
    uint32_t* d_particle_cell = nullptr;

    // Tree Architecture Step 2: passive/active region membership flag.
    // 1 = particle is inside an active region (handled by siphonDiskKernel).
    // 0 = particle is passive (handled by advectPassiveParticles).
    // Initialized to all-1s below so Step 2 is zero-behavior-change: the
    // passive kernel early-returns on every particle. Sized to
    // g_runtime_particle_cap (not N_current) because spawning can grow N.
    uint8_t* d_in_active_region = nullptr;

    // Radial profile of R_cell — 16 bins from center to box edge
    const int RC_RADIAL_BINS = 16;
    float* d_rc_bin_R = nullptr;
    float* d_rc_bin_W = nullptr;
    float* d_rc_bin_N = nullptr;  // cell count per bin
    std::vector<float> h_rc_bin_R(RC_RADIAL_BINS);
    std::vector<float> h_rc_bin_W(RC_RADIAL_BINS);
    std::vector<float> h_rc_bin_N(RC_RADIAL_BINS);

    if (g_grid_physics) {
        size_t cell_array_size = GRID_CELLS * sizeof(float);
        // CRITICAL: Must allocate for MAX particles, not initial N
        // Otherwise spawning beyond N causes OOB access
        size_t particle_cell_size = (size_t)g_runtime_particle_cap * sizeof(uint32_t);

        // Accumulated fields (Pass 1)
        cudaMalloc(&d_grid_density, cell_array_size);
        cudaMalloc(&d_grid_momentum_x, cell_array_size);
        cudaMalloc(&d_grid_momentum_y, cell_array_size);
        cudaMalloc(&d_grid_momentum_z, cell_array_size);
        cudaMalloc(&d_grid_phase_sin, cell_array_size);
        cudaMalloc(&d_grid_phase_cos, cell_array_size);

        // Derived fields (Pass 2)
        cudaMalloc(&d_grid_pressure_x, cell_array_size);
        cudaMalloc(&d_grid_pressure_y, cell_array_size);
        cudaMalloc(&d_grid_pressure_z, cell_array_size);
        cudaMalloc(&d_grid_vorticity_x, cell_array_size);
        cudaMalloc(&d_grid_vorticity_y, cell_array_size);
        cudaMalloc(&d_grid_vorticity_z, cell_array_size);

        // Per-cell Kuramoto order parameter and radial profile bins
        cudaMalloc(&d_grid_R_cell, cell_array_size);
        cudaMalloc(&d_rc_bin_R, RC_RADIAL_BINS * sizeof(float));
        cudaMalloc(&d_rc_bin_W, RC_RADIAL_BINS * sizeof(float));
        cudaMalloc(&d_rc_bin_N, RC_RADIAL_BINS * sizeof(float));

        // Per-particle cell assignment
        cudaMalloc(&d_particle_cell, particle_cell_size);

        size_t total_grid_mem = 12 * cell_array_size + particle_cell_size;
        printf("[grid] DNA/RNA streaming grid allocated: %zuMB\n",
               total_grid_mem / (1024 * 1024));
        printf("[grid] Grid: %dx%dx%d cells (%.2f units/cell)\n",
               GRID_DIM, GRID_DIM, GRID_DIM, GRID_CELL_SIZE);

        // Initialize mip-tree for hierarchical coherence
        // This replaces Morton-sorted octree for scale coupling
        // Pass actual grid dimension (g_grid_dim can be 128, 96, or 64)
        if (!mip_tree_init(g_grid_dim, 0.1f)) {
            fprintf(stderr, "[mip] WARNING: Failed to initialize mip-tree (hierarchy disabled)\n");
        }
    }

    // === SPARSE FLAGS BUFFER ALLOCATION ===
    // Hierarchical tiled flags: O(tiles) + O(active_tiles × cells_per_tile)
    // Instead of scanning 2M cells, scan 4096 tiles then ~150 × 512 cells
    extern bool g_grid_flags;
    uint8_t* d_active_flags = nullptr;       // Cell-level flags [2M]
    uint8_t* d_tile_flags = nullptr;         // Tile-level flags [4096]
    uint32_t* d_compact_active_list = nullptr;
    uint32_t* d_compact_active_count = nullptr;
    uint32_t* d_active_tiles = nullptr;      // Compacted tile list
    uint32_t* d_active_tile_count = nullptr;
    uint32_t h_compact_active_count = 0;
    uint32_t h_active_tile_count = 0;

    if (g_grid_flags && g_grid_physics) {
        size_t cell_flags_size = GRID_CELLS * sizeof(uint8_t);  // 2MB
        size_t tile_flags_size = NUM_TILES * sizeof(uint8_t);   // 4KB
        size_t active_list_size = GRID_CELLS * sizeof(uint32_t);  // 8MB worst case
        size_t tile_list_size = NUM_TILES * sizeof(uint32_t);   // 16KB

        cudaMalloc(&d_active_flags, cell_flags_size);
        cudaMalloc(&d_tile_flags, tile_flags_size);
        cudaMalloc(&d_compact_active_list, active_list_size);
        cudaMalloc(&d_compact_active_count, sizeof(uint32_t));
        cudaMalloc(&d_active_tiles, tile_list_size);
        cudaMalloc(&d_active_tile_count, sizeof(uint32_t));

        // Initialize all flags to zero
        cudaMemset(d_active_flags, 0, cell_flags_size);
        cudaMemset(d_tile_flags, 0, tile_flags_size);
        cudaMemset(d_compact_active_count, 0, sizeof(uint32_t));
        cudaMemset(d_active_tile_count, 0, sizeof(uint32_t));

        size_t total_mem = cell_flags_size + tile_flags_size + active_list_size + tile_list_size;
        printf("[flags+tiles] Hierarchical tiled compaction: %.1fMB\n", total_mem / (1024.0 * 1024.0));
        printf("[flags+tiles] Cell flags: %.1fMB | Tile flags: %.1fKB | Tiles: %d\n",
               cell_flags_size / (1024.0 * 1024.0), tile_flags_size / 1024.0, NUM_TILES);
    }

    // === ACTIVE PARTICLE COMPACTION BUFFERS ===
    // Skip static shell mass by only scatter/gather "active" (moving/cell-changed) particles
    extern bool g_active_compaction;
    extern ActiveParticleState g_active_particles;
    if (g_active_compaction && g_grid_physics) {
        size_t particle_cap = (size_t)g_runtime_particle_cap;
        size_t cell_array_size = GRID_CELLS * sizeof(float);

        // Per-particle tracking
        cudaMalloc(&g_active_particles.d_prev_cell, particle_cap * sizeof(uint32_t));
        cudaMalloc(&g_active_particles.d_active_mask, particle_cap * sizeof(uint8_t));
        cudaMalloc(&g_active_particles.d_active_list, particle_cap * sizeof(uint32_t));
        cudaMalloc(&g_active_particles.d_active_count, sizeof(uint32_t));

        // Static grid (baked contribution from non-moving particles)
        cudaMalloc(&g_active_particles.d_static_density, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_x, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_y, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_z, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_phase_sin, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_phase_cos, cell_array_size);

        // Initialize prev_cell to invalid (forces all particles active on first frame)
        cudaMemset(g_active_particles.d_prev_cell, 0xFF, particle_cap * sizeof(uint32_t));
        cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));

        g_active_particles.initialized = true;
        g_active_particles.static_baked = false;
        g_active_particles.h_active_count = 0;

        size_t total_active_mem = particle_cap * (sizeof(uint32_t) * 2 + sizeof(uint8_t)) + 6 * cell_array_size;
        printf("[active-compact] Active particle compaction: %.1fMB\n", total_active_mem / (1024.0 * 1024.0));
        printf("[active-compact] Particle tracking: %.1fMB | Static grid: %.1fMB\n",
               particle_cap * (sizeof(uint32_t) * 2 + sizeof(uint8_t)) / (1024.0 * 1024.0),
               6 * cell_array_size / (1024.0 * 1024.0));
    }

    // === TREE ARCHITECTURE STEP 2: in_active_region buffer ===
    // Unconditional allocation — the passive kernel is orthogonal to grid
    // physics and active-compaction. Initialized to all-1s so every particle
    // is "in the all-encompassing bootstrap region"; the passive kernel
    // early-returns on every particle in Step 2 (zero behavior change).
    {
        size_t in_active_region_size = (size_t)g_runtime_particle_cap * sizeof(uint8_t);
        cudaMalloc(&d_in_active_region, in_active_region_size);
        cudaMemset(d_in_active_region, 0xFF, in_active_region_size);  // all-1s = all in region
        printf("[passive] d_in_active_region allocated: %zu bytes, init=all-in-region\n",
               in_active_region_size);
    }

    // === TREE ARCHITECTURE STEP 2: bootstrap ActiveRegion ===
    // Allocate a small fixed-size array of ActiveRegion slots and seed
    // exactly one all-encompassing region that covers the entire simulation
    // volume. Every alive particle will test as "inside" this region, so
    // computeInActiveRegionMask will write 1 to in_active_region[i] for
    // every alive particle, the passive kernel's third early-return will
    // trigger on every particle, and behavior is byte-identical to
    // pre-Step-2. Step 3 will replace this with dynamic region lifecycle.
    ActiveRegion* d_active_regions = nullptr;
    int h_num_active_regions = 0;
    {
        cudaMalloc(&d_active_regions, MAX_ACTIVE_REGIONS * sizeof(ActiveRegion));
        cudaMemset(d_active_regions, 0, MAX_ACTIVE_REGIONS * sizeof(ActiveRegion));

        ActiveRegion h_bootstrap = {};
        h_bootstrap.gate_positions[0] = make_float3(-500.0f, -500.0f, -500.0f);
        h_bootstrap.gate_positions[1] = make_float3( 500.0f,  500.0f,  500.0f);
        h_bootstrap.gate_positions[2] = make_float3(0.0f, 0.0f, 0.0f);
        h_bootstrap.parent_shell = -1;
        h_bootstrap.birth_frame = 0;
        h_bootstrap.stability_integral = 0.0f;
        h_bootstrap.state = REGION_STATE_ACTIVE;
        cudaMemcpy(d_active_regions, &h_bootstrap, sizeof(ActiveRegion),
                   cudaMemcpyHostToDevice);
        h_num_active_regions = 1;
        printf("[passive] ActiveRegion bootstrap: 1 all-encompassing region seeded (state=ACTIVE, bounds=±500)\n");
    }

    // Timing
    float sim_time = 0.0f;
    int frame = 0;
    int threads = 256;
    // blocks computed dynamically each frame based on N_current
    auto t0 = std::chrono::steady_clock::now();
    double fps_acc = 0; int fps_frames = 0;

    // === BRIDGE METRICS: Smoothed pump state for raymarcher ===
    // These are exponentially smoothed to prevent visual jitter
    PumpMetrics pump_bridge = {1.0f, 0.0f, 0.0f, 0.0f};

    printf("[run] Controls: drag=orbit, scroll=zoom, R=reset, Space=pause, C=color, V=shell brightness\n");
    printf("[run] Seam: 1=closed 2=up 3=down 4=full | Bias: [/] or T=turbo\n");
    printf("[run] PURE PHYSICS MODE - no template forcing\n");

    // Render loop
    // Main loop (headless runs until frame limit, interactive until window closes)
    bool running = true;
    while (running) {
        // Compute blocks based on current particle count (grows via spawning)
        int blocks = (N_current + threads - 1) / threads;

        if (!g_headless && glfwWindowShouldClose(window)) {
            running = false;
            break;
        }
        auto t1 = std::chrono::steady_clock::now();
        float dt_wall = std::chrono::duration<float>(t1 - t0).count();
        t0 = t1;

        // === FIXED TIMESTEP PHYSICS ===
        // Decouple simulation time from wall-clock time to prevent
        // "Aizawa fling" when frame export causes slowdown spikes.
        // Physics always advances in fixed increments regardless of render lag.
        //
        // Simple approach: dt_sim is ALWAYS FIXED_DT, regardless of wall time.
        // During stalls (frame export), simulation effectively pauses rather
        // than trying to "catch up" which would cause instability.
        constexpr float FIXED_DT = 1.0f / 60.0f;  // 60 Hz physics tick

        // dt_sim is ALWAYS fixed - never variable, never scaled by wall time
        float dt_sim = FIXED_DT;

        // Advance sim_time only when not paused (one tick per frame)
        // During export stalls, frames still advance but wall-clock is ignored
        bool should_simulate = g_headless || !g_cam.paused;
        if (should_simulate) {
            sim_time += FIXED_DT;
        }

        // === KERNEL TIMING (every 900 frames) ===
        static cudaEvent_t t_start, t_siphon, t_octree, t_physics, t_render;
        static bool timing_init = false;
        static float ms_siphon = 0, ms_physics = 0, ms_render = 0;
        bool do_timing = (frame > 0 && frame % 900 == 0);
        if (!timing_init) {
            cudaEventCreate(&t_start); cudaEventCreate(&t_siphon);
            cudaEventCreate(&t_octree); cudaEventCreate(&t_physics);
            cudaEventCreate(&t_render);
            timing_init = true;
        }

        // Simulate (fixed timestep - one physics step per frame, dt always constant)
        if (should_simulate) {
            if (do_timing) cudaEventRecord(t_start);

            // Update arm topology device constants ONLY when changed
            // (cudaMemcpyToSymbol is expensive - ~1ms per call, 5 calls = 5ms/frame wasted)
            static int cached_num_arms = -1;
            static bool cached_use_topology = false;
            static float cached_boost_override = -999.0f;
            static bool arm_constants_dirty = true;  // First frame always updates

            int h_num_arms = g_enable_arms ? NUM_ARMS : 0;
            extern float g_arm_boost_override;

            if (arm_constants_dirty ||
                h_num_arms != cached_num_arms ||
                g_use_arm_topology != cached_use_topology ||
                g_arm_boost_override != cached_boost_override) {

                float h_arm_width = ARM_WIDTH_DEG;
                float h_arm_trap = ARM_TRAP_STRENGTH;
                cudaMemcpyToSymbol(d_NUM_ARMS, &h_num_arms, sizeof(int));
                cudaMemcpyToSymbol(d_ARM_WIDTH_DEG, &h_arm_width, sizeof(float));
                cudaMemcpyToSymbol(d_ARM_TRAP_STRENGTH, &h_arm_trap, sizeof(float));
                cudaMemcpyToSymbol(d_USE_ARM_TOPOLOGY, &g_use_arm_topology, sizeof(bool));
                cudaMemcpyToSymbol(d_ARM_BOOST_OVERRIDE, &g_arm_boost_override, sizeof(float));

                cached_num_arms = h_num_arms;
                cached_use_topology = g_use_arm_topology;
                cached_boost_override = g_arm_boost_override;
                arm_constants_dirty = false;
            }

            // Use dynamic particle count (N_current grows via spawning)
            int spawn_blocks = (N_current + threads - 1) / threads;

#if ENABLE_PASSIVE_ADVECTION
            // Tree Architecture Step 3 dispatch ordering:
            // 1. computeInActiveRegionMask reads PREVIOUS frame's pump_residual
            //    to classify particles as active (siphon) or passive (advect).
            //    The one-frame lag is acceptable — pump_residual changes slowly.
            // 2. siphonDiskKernel reads the mask and skips passive particles.
            // 3. advectPassiveParticles reads the mask and processes passive ones.
            // INVARIANT: for every alive particle, exactly one of siphon or
            // passive does position/theta writes per frame. Both kernels'
            // early-return conditions are mutually exclusive on in_active_region.
            computeInActiveRegionMask<<<spawn_blocks, threads>>>(
                d_disk, d_active_regions, h_num_active_regions,
                d_in_active_region, N_current, g_corner_threshold);
#endif

            siphonDiskKernel<<<spawn_blocks, threads>>>(d_disk, d_in_active_region, N_current, sim_time, dt_sim * 2.0f, g_cam.seam_bits, g_cam.bias);

#if ENABLE_PASSIVE_ADVECTION
            advectPassiveParticles<<<spawn_blocks, threads>>>(
                d_disk, d_in_active_region, N_current,
                dt_sim * 2.0f, g_passive_residual_tau);
#endif

            // === NATURAL GROWTH: Spawn new particles in coherent regions ===
            if (SPAWN_ENABLE && N_current < MAX_DISK_PTS) {
                // === ASYNC SPAWN: Apply previous frame's spawns first ===
                // This avoids blocking - we read spawns one frame late, which is fine
                if (spawn_pending) {
                    cudaError_t status = cudaEventQuery(spawn_ready);
                    if (status == cudaSuccess) {
                        unsigned int h_spawned = *h_spawn_pinned;
                        spawn_pending = false;

                        if (h_spawned > 0) {
                            // OOM protection: cap at RUNTIME_PARTICLE_CAP
                            int new_total = N_current + (int)h_spawned;
                            if (new_total > RUNTIME_PARTICLE_CAP) {
                                static bool oom_warned = false;
                                if (!oom_warned) {
                                    printf("[OOM PROTECTION] Particle cap reached: %d (limit %d) — spawning disabled\n",
                                           N_current, RUNTIME_PARTICLE_CAP);
                                    oom_warned = true;
                                }
                                h_spawned = 0;
                            }
                            N_current += h_spawned;
                            spawn_blocks = (N_current + threads - 1) / threads;

                            // Log growth events (every 10k frames to reduce spam)
                            static int last_log_frame = 0;
                            if (frame - last_log_frame >= 10000) {
                                printf("[growth] Frame %d: %d particles (%.1f%% capacity)\n",
                                       frame, N_current,
                                       100.0f * N_current / (float)MAX_DISK_PTS);
                                last_log_frame = frame;
                            }
                        }
                    }
                    // If not ready, just wait until next frame (no stall)
                }

                // Launch spawn kernel for THIS frame (unless disabled for clean measurements)
                if (g_spawn_enabled) {
                    cudaMemsetAsync(d_spawn_idx, 0, sizeof(unsigned int), spawn_stream);
                    cudaMemsetAsync(d_spawn_success, 0, sizeof(unsigned int), spawn_stream);

                    unsigned int spawn_seed = (unsigned int)(frame * 12345 + (int)(sim_time * 1000));
                    spawnParticlesKernel<<<spawn_blocks, threads, 0, spawn_stream>>>(
                        d_disk, N_current, MAX_DISK_PTS, d_spawn_idx, d_spawn_success, sim_time, spawn_seed,
                        d_in_active_region
                    );

                    // Async copy spawn count (will be read NEXT frame)
                    cudaMemcpyAsync(h_spawn_pinned, d_spawn_success, sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, spawn_stream);
                    cudaEventRecord(spawn_ready, spawn_stream);
                    spawn_pending = true;
                }
            }
            if (do_timing) cudaEventRecord(t_siphon);

            // === OCTREE UPDATE (every N frames) ===
            // Rebuild Morton-sorted tree for active particles
            // The octree is the crystallization — 24 (LAMBDA_OCTREE) is the finished stone layer.
            // DISABLED BY DEFAULT: Mip-tree provides hierarchical coherence without Morton sort
            // Use --octree-rebuild to re-enable if needed for comparison
            extern bool g_octree_rebuild;
            static int octreeRebuildCounter = 0;
            const int OCTREE_REBUILD_INTERVAL = 30;  // Rebuild every 30 frames (~0.2s at 150 FPS)
            static int octreeSkipCount = 0;          // Track consecutive skips

            if (g_octree_rebuild && octreeEnabled && ++octreeRebuildCounter >= OCTREE_REBUILD_INTERVAL) {
                // VRAM-aware check: test if rebuild will fit before attempting
                // This replaces the hard threshold (was 200k) with dynamic VRAM test
                if (!canOctreeFit(N_current)) {
                    // Skip this rebuild — not enough VRAM for thrust temp buffers
                    octreeSkipCount++;
                    if (octreeSkipCount == 1 || (octreeSkipCount % 100) == 0) {
                        size_t free_now = 0, total = 0;
                        cudaError_t memErr = cudaMemGetInfo(&free_now, &total);
                        cudaError_t lastErr = cudaGetLastError();  // Check if CUDA is in error state
                        printf("[octree] Skip #%d: %d particles, %.1f MB free (need ~%.1f MB), CUDA err=%d/%d\n",
                               octreeSkipCount, N_current, free_now / 1e6,
                               (float)N_current * 28 / 0.75 / 1e6, (int)memErr, (int)lastErr);
                    }
                    // Retry in 10 frames instead of full interval
                    octreeRebuildCounter = OCTREE_REBUILD_INTERVAL - 10;
                } else {
                    octreeSkipCount = 0;  // Reset skip counter on successful rebuild
                    octreeRebuildCounter = 0;

                // 1. Assign Morton keys to all particles (active inner get real keys)
                float boxSize = 500.0f;
                assignMortonKeys<<<blocks, threads>>>(
                    d_disk, d_morton_keys, d_xor_corners, d_particle_ids, N_current, boxSize
                );

                // 2. Count active particles directly (no sort needed)
                // Keys < 0xFFFF... are active (inner particles)
                h_num_active = gpuCountLessThan(d_morton_keys, N_current, 0xFFFFFFFFFFFFFFFFULL);

                // Octree enabled for all particle counts (morton sort restored)

                // 4. Reset node count to analytic tree size (discard old stochastic)
                cudaMemcpy(d_node_count, &h_analytic_node_count,
                           sizeof(uint32_t), cudaMemcpyHostToDevice);

                // 5. Build stochastic tree (levels 6-13) from sorted active particles
                if (h_num_active > 0) {
                    int stochBlocks = (h_num_active + 255) / 256;
                    buildStochasticTree<<<stochBlocks, 256>>>(
                        d_octree_nodes,
                        d_node_count,
                        d_morton_keys,
                        h_num_active,
                        boxSize,    // 500.0f
                        6,          // start_level (first stochastic)
                        13          // max_level
                    );
                }

                // 6. Get total node count for render traversal
                cudaMemcpy(&h_total_node_count, d_node_count,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost);

                // 7. Pre-compute leaf node count for traversal and physics
                // Needed for both render traversal AND physics neighbor lookup
                if ((useOctreeTraversal || useOctreePhysics) && h_total_node_count > h_analytic_node_count) {
                    cudaMemsetAsync(d_leaf_node_count, 0, sizeof(uint32_t));
                    uint32_t stochastic_count = h_total_node_count - h_analytic_node_count;
                    int extractBlocks = (stochastic_count + 255) / 256;
                    extractLeafNodeCounts<<<extractBlocks, 256>>>(
                        d_leaf_counts,
                        d_leaf_node_indices,
                        d_leaf_node_count,
                        d_octree_nodes,
                        h_total_node_count,
                        h_analytic_node_count,
                        13
                    );
                    cudaMemcpy(&h_leaf_node_count, d_leaf_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

                    // Safety check: warn if leaf count exceeds hash capacity
                    if (h_leaf_node_count > h_leaf_hash_size / 2) {
                        printf("[octree] WARNING: leaf count %u exceeds 50%% hash capacity (%u) — increase h_leaf_hash_size\n",
                               h_leaf_node_count, h_leaf_hash_size / 2);
                    }

                    // Build hash table for O(1) neighbor lookup (replaces binary search)
                    if (h_leaf_node_count > 0) {
                        // Clear hash table (set all keys to UINT64_MAX = empty)
                        cudaMemsetAsync(d_leaf_hash_keys, 0xFF, h_leaf_hash_size * sizeof(uint64_t));
                        // Build hash table from leaf indices
                        int hashBlocks = (h_leaf_node_count + 255) / 256;
                        buildLeafHashTable<<<hashBlocks, 256>>>(
                            d_leaf_hash_keys,
                            d_leaf_hash_values,
                            h_leaf_hash_size,
                            h_leaf_hash_size - 1,  // hash_mask
                            d_octree_nodes,
                            d_leaf_node_indices,
                            h_leaf_node_count
                        );
                    }

                    // Cache total particle count for V3 flat dispatch (avoids per-frame sync)
                    if (h_leaf_node_count > 0) {
                        h_cached_total_particles = gpuReduceSum(d_leaf_counts, h_leaf_node_count);
                    } else {
                        h_cached_total_particles = 0;
                    }
                }

                // 8. Print stats occasionally (every 30 rebuilds = 900 frames)
                static int stochastic_rebuild_count = 0;
                stochastic_rebuild_count++;
                if (stochastic_rebuild_count == 1 || stochastic_rebuild_count % 30 == 0) {
                    printf("[octree] Tree: %u analytic + %u stochastic = %u total, %u active, %u leaves\n",
                           h_analytic_node_count, h_total_node_count - h_analytic_node_count,
                           h_total_node_count, h_num_active, h_leaf_node_count);
                }

                // 9. Initialize phase state on first rebuild (once we have leaves)
                static bool phase_initialized = false;
                if (!phase_initialized && h_leaf_node_count > 0) {
                    int leafBlocks = (h_leaf_node_count + 255) / 256;
                    float base_frequency = 0.1f;  // Base oscillation frequency
                    initializeLeafPhase<<<leafBlocks, 256>>>(
                        d_leaf_phase,
                        d_leaf_frequency,
                        d_octree_nodes,
                        d_leaf_node_indices,
                        h_leaf_node_count,
                        base_frequency
                    );
                    phase_initialized = true;
                    printf("[phase] S3 phase state initialized: %u leaves, ω_base=%.2f, coupling=%.3f\n",
                           h_leaf_node_count, base_frequency, phase_coupling_k);
                }
                }  // end else (canOctreeFit succeeded)
            }

            // === OCTREE PHYSICS — Pressure + Vorticity Forces ===
            // 1. Pressure: F_p = -k_p ∇ρ (radial balance, shell formation)
            // 2. Vorticity: F_ω = k_ω (ω × v) (spiral arms, rotation)
            if (useOctreePhysics && octreeEnabled && h_leaf_node_count > 0 && h_num_active > 0) {
                // Step 1: Accumulate average velocity per leaf node (for vorticity)
                accumulateLeafVelocities<<<h_leaf_node_count, 256>>>(
                    d_leaf_vel_x,
                    d_leaf_vel_y,
                    d_leaf_vel_z,
                    d_octree_nodes,
                    d_leaf_node_indices,
                    d_particle_ids,
                    d_disk,
                    h_leaf_node_count
                );

                // Step 2: Apply pressure + vorticity forces to all active particles
                // Phase modulates pressure via sin(θ) - zero-cost oscillation
                int particleBlocks = (h_num_active + 255) / 256;
                applyPressureVorticityKernel<<<particleBlocks, 256>>>(
                    d_disk,
                    d_morton_keys,
                    d_particle_ids,
                    d_octree_nodes,
                    d_leaf_node_indices,
                    d_leaf_vel_x,
                    d_leaf_vel_y,
                    d_leaf_vel_z,
                    d_leaf_phase,  // Direct phase read (no coherence lookup)
                    d_leaf_hash_keys,     // Hash table for O(1) neighbor lookup
                    d_leaf_hash_values,
                    h_leaf_hash_size - 1, // Hash mask
                    h_leaf_node_count,
                    h_num_active,
                    d_in_active_region,   // Step 3: skip passive particles
                    dt_sim,
                    pressure_k,
                    vorticity_k
                );

                // Step 3: Evolve S3 phase state (Kuramoto model)
                // Phase coupling creates temporal coherence and enables resonance
                // Update every 10 frames - neighbor lookups only here, not in pressure
                // Can be disabled with --no-octree-phase to use mip-tree for coherence instead
                extern bool g_octree_phase;
                if (g_octree_phase && frame % 10 == 0) {
                    int leafBlocks = (h_leaf_node_count + 255) / 256;

                    // Evolve phase with Kuramoto coupling (neighbor lookups here only)
                    evolveLeafPhase<<<leafBlocks, 256>>>(
                        d_leaf_phase,
                        d_leaf_frequency,
                        d_octree_nodes,
                        d_leaf_node_indices,
                        d_leaf_hash_keys,      // Hash table for O(1) neighbor lookup
                        d_leaf_hash_values,
                        h_leaf_hash_size - 1,  // Hash mask
                        h_leaf_node_count,
                        dt_sim * 10.0f,  // Compensate for reduced update rate
                        phase_coupling_k
                    );
                }
            }

            // === GRID PHYSICS — DNA/RNA Streaming Forward-Pass Model ===
            // Four modes:
            //   Cadence mode (--grid-physics): scatter/stencil every 30 frames, gather every frame
            //   Flags mode (--grid-flags): presence flags, no lists, no sort, no dedup (optimal)
            extern bool g_grid_physics;
            extern bool g_grid_flags;
            static int gridRebuildCounter = 0;
            const int GRID_REBUILD_INTERVAL = 30;

            // Flags mode state
            static bool flagsInitialized = false;

            if (g_grid_physics && d_grid_density != nullptr) {
                int clearBlocks = (GRID_CELLS + 255) / 256;

                // Per-pass timing events (static to avoid allocation overhead)
                static cudaEvent_t e_scatter_start, e_scatter_end;
                static cudaEvent_t e_compact_end, e_pressure_end, e_gather_end;
                static bool timing_events_init = false;
                static float ms_scatter = 0, ms_compact = 0, ms_pressure = 0, ms_gather = 0;
                if (!timing_events_init) {
                    cudaEventCreate(&e_scatter_start);
                    cudaEventCreate(&e_scatter_end);
                    cudaEventCreate(&e_compact_end);
                    cudaEventCreate(&e_pressure_end);
                    cudaEventCreate(&e_gather_end);
                    timing_events_init = true;
                }

                if (g_grid_flags && d_active_flags != nullptr && d_compact_active_list != nullptr) {
                    // === SPARSE FLAGS + O(n) COMPACTION — Transcription Pattern ===
                    // 1. Scatter marks flags (duplicates collapse)
                    // 2. Compact flags → active_list (O(n), no sort)
                    // 3. Process only active_list (~4k cells, not 2M)

                    // Single flags buffer - no double-buffering needed without propagation
                    // Pipeline: scatter→flags, compact, sparse_clear, repeat

                    // Initialize on first frame
                    if (!flagsInitialized) {
                        // Clear grid
                        clearCellGrid<<<clearBlocks, 256>>>(
                            d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z
                        );

                        flagsInitialized = true;
                        printf("[flags+tiles] Initialized with hierarchical tiled compaction\n");
                    }

                    // Full cadence: only scatter+compact+decay every N frames
                    // Between rebuilds, only gather (like cadence mode)
                    static const int FLAGS_CADENCE = 30;
                    static int flagsCadenceCounter = 0;
                    bool doRebuild = (flagsCadenceCounter == 0);

                    if (doRebuild) {
                        // === ACTIVE PARTICLE COMPACTION (when locked) ===
                        // Instead of scatter(N), do scatter(active_N) + baked static grid
                        extern bool g_active_compaction;
                        extern ActiveParticleState g_active_particles;
                        extern HarmonicLock g_harmonic_lock;
                        bool use_active_compact = g_active_compaction &&
                                                  g_active_particles.initialized &&
                                                  g_harmonic_lock.locked;

                        // Force full scatter if static grid needs rebaking
                        if (use_active_compact && !g_active_particles.static_baked) {
                            use_active_compact = false;  // Will bake on this frame
                        }

                        // Periodic rebake even when locked (every 256 frames)
                        if (use_active_compact &&
                            (frame - g_active_particles.bake_frame) >= ActiveParticleState::REBAKE_INTERVAL) {
                            use_active_compact = false;  // Force rebake
                        }

                        cudaEventRecord(e_scatter_start);

                        if (use_active_compact) {
                            // === ACTIVE COMPACTION PATH ===
                            // 1. Compute activity mask
                            computeParticleActivityMask<<<blocks, threads>>>(
                                d_disk, d_particle_cell, g_active_particles.d_prev_cell,
                                g_active_particles.d_active_mask, N_current,
                                ActiveParticleState::VELOCITY_THRESHOLD
                            );

                            // 2. Compact active particles
                            cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));
                            compactActiveParticles<<<blocks, threads>>>(
                                g_active_particles.d_active_mask,
                                g_active_particles.d_active_list,
                                g_active_particles.d_active_count,
                                N_current
                            );
                            cudaMemcpy(&g_active_particles.h_active_count,
                                       g_active_particles.d_active_count,
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost);

                            // 3. Copy static grid to working grid (base layer)
                            int gridCopyBlocks = (GRID_CELLS + 255) / 256;
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_density, d_grid_density, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_x, d_grid_momentum_x, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_y, d_grid_momentum_y, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_z, d_grid_momentum_z, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_phase_sin, d_grid_phase_sin, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_phase_cos, d_grid_phase_cos, GRID_CELLS);

                            // 4. Scatter ONLY active particles on top
                            if (g_active_particles.h_active_count > 0) {
                                int activeBlks = (g_active_particles.h_active_count + 255) / 256;
                                scatterActiveParticles<<<activeBlks, 256>>>(
                                    d_disk, d_grid_density,
                                    d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                    d_grid_phase_sin, d_grid_phase_cos, d_particle_cell,
                                    g_active_particles.d_active_list, g_active_particles.h_active_count
                                );
                            }

                            // 5. Update prev_cell for next frame
                            copyCurrentToPrevCell<<<blocks, threads>>>(
                                d_particle_cell, g_active_particles.d_prev_cell, N_current
                            );

                            // Mark tile flags for sparse pressure (estimate from active particles)
                            // For now, use full tile marking - could optimize later
                            cudaMemset(d_tile_flags, 0, NUM_TILES * sizeof(uint8_t));
                            cudaMemset(d_active_flags, 0, GRID_CELLS * sizeof(uint8_t));

                            // Log active compaction stats occasionally
                            static int active_log_counter = 0;
                            if (++active_log_counter >= 30) {
                                active_log_counter = 0;
                                float active_pct = 100.0f * g_active_particles.h_active_count / N_current;
                                printf("[active-compact] frame=%d | active=%u (%.1f%%) | saved %.1f%% scatter ops\n",
                                       frame, g_active_particles.h_active_count, active_pct, 100.0f - active_pct);
                            }
                        } else {
                            // === FULL SCATTER PATH (or baking) ===
                            // Pass 1: Scatter particles → marks BOTH cell and tile flags
                            scatterWithTileFlags<<<blocks, threads>>>(
                                d_disk, d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                d_grid_phase_sin, d_grid_phase_cos, d_particle_cell,
                                d_active_flags, d_tile_flags, N_current, 1.0f
                            );

                            // === BAKE STATIC GRID when lock just engaged ===
                            if (g_active_compaction && g_active_particles.initialized &&
                                g_harmonic_lock.locked && !g_active_particles.static_baked) {
                                // First time locked: bake static grid

                                // Compute activity mask for baking
                                computeParticleActivityMask<<<blocks, threads>>>(
                                    d_disk, d_particle_cell, g_active_particles.d_prev_cell,
                                    g_active_particles.d_active_mask, N_current,
                                    ActiveParticleState::VELOCITY_THRESHOLD
                                );

                                // Clear static grid
                                cudaMemset(g_active_particles.d_static_density, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_x, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_y, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_z, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_phase_sin, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_phase_cos, 0, GRID_CELLS * sizeof(float));

                                // Scatter static (non-active) particles to static grid
                                scatterStaticParticles<<<blocks, threads>>>(
                                    d_disk, g_active_particles.d_active_mask,
                                    g_active_particles.d_static_density,
                                    g_active_particles.d_static_momentum_x,
                                    g_active_particles.d_static_momentum_y,
                                    g_active_particles.d_static_momentum_z,
                                    g_active_particles.d_static_phase_sin,
                                    g_active_particles.d_static_phase_cos,
                                    d_particle_cell, N_current
                                );

                                // Count static vs active
                                cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));
                                compactActiveParticles<<<blocks, threads>>>(
                                    g_active_particles.d_active_mask,
                                    g_active_particles.d_active_list,
                                    g_active_particles.d_active_count,
                                    N_current
                                );
                                cudaMemcpy(&g_active_particles.h_active_count,
                                           g_active_particles.d_active_count,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost);

                                g_active_particles.static_baked = true;
                                g_active_particles.bake_frame = frame;

                                uint32_t static_count = N_current - g_active_particles.h_active_count;
                                printf("[active-compact] BAKED: %u static (%.1f%%) + %u active (%.1f%%)\n",
                                       static_count, 100.0f * static_count / N_current,
                                       g_active_particles.h_active_count,
                                       100.0f * g_active_particles.h_active_count / N_current);
                            }

                            // Update prev_cell for activity tracking
                            if (g_active_compaction && g_active_particles.initialized) {
                                copyCurrentToPrevCell<<<blocks, threads>>>(
                                    d_particle_cell, g_active_particles.d_prev_cell, N_current
                                );
                            }

                            // Invalidate bake when lock breaks
                            if (!g_harmonic_lock.locked) {
                                g_active_particles.static_baked = false;
                            }
                        }
                        cudaEventRecord(e_scatter_end);

                        // Pass 2a: Compact tiles — O(4096) instead of O(2M)!
                        cudaMemset(d_active_tile_count, 0, sizeof(uint32_t));
                        int tileBlocks = (NUM_TILES + 255) / 256;
                        compactActiveTiles<<<tileBlocks, 256>>>(
                            d_tile_flags, d_active_tiles, d_active_tile_count
                        );
                        cudaMemcpy(&h_active_tile_count, d_active_tile_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

                        // Pass 2b: Compact cells within active tiles — O(active_tiles × 512)
                        // Each block handles one tile (512 threads max per tile)
                        cudaMemset(d_compact_active_count, 0, sizeof(uint32_t));
                        if (h_active_tile_count > 0) {
                            compactCellsInTiles<<<h_active_tile_count, CELLS_PER_TILE>>>(
                                d_active_flags, d_active_tiles, h_active_tile_count,
                                d_compact_active_list, d_compact_active_count
                            );
                        }
                        cudaMemcpy(&h_compact_active_count, d_compact_active_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaEventRecord(e_compact_end);

                        // Sparse clear: only clear flags for active cells (also clears tile flags)
                        int activeBlocks = (h_compact_active_count + 255) / 256;
                        if (activeBlocks == 0) activeBlocks = 1;
                        sparseClearTileAndCellFlags<<<activeBlocks, 256>>>(
                            d_active_flags, d_tile_flags, d_compact_active_list, h_compact_active_count
                        );

                        // Pass 3: Compute Pressure for active cells
                        decayAndComputePressure<<<activeBlocks, 256>>>(
                            d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_compact_active_list, h_compact_active_count,
                            1.0f, pressure_k
                        );
                        cudaEventRecord(e_pressure_end);

                        // Pass 3b: Build mip-tree hierarchy for scale coupling
                        // This replaces Morton-sorted octree for coherence
                        // PREDICTIVE LOCKING: Skip when shells are locked (m=0 ground state)
                        extern bool g_predictive_locking;
                        extern HarmonicLock g_harmonic_lock;
                        bool skip_mip = g_predictive_locking && g_harmonic_lock.locked;

                        // Periodic recheck: rebuild every RECHECK_INTERVAL frames even when locked
                        if (skip_mip && g_harmonic_lock.lock_recheck_counter >= HarmonicLock::RECHECK_INTERVAL) {
                            skip_mip = false;  // Force rebuild for verification
                            g_harmonic_lock.lock_recheck_counter = 0;
                        }

                        if (g_mip_tree.initialized && !skip_mip) {
                            mip_tree_from_grid(d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                               d_grid_phase_sin, d_grid_phase_cos);
                            mip_tree_build_up();
                            mip_tree_propagate_down(0.05f);  // Mild coupling
                        }
                    }

                    flagsCadenceCounter = (flagsCadenceCounter + 1) % FLAGS_CADENCE;

                    // Pass 4: Gather cell forces to particles
                    // Half-rate gather: when locked, only gather every other frame
                    // (Keplerian orbits are smooth enough to extrapolate between frames)
                    extern bool g_active_compaction;
                    extern ActiveParticleState g_active_particles;
                    extern HarmonicLock g_harmonic_lock;

                    static int gather_phase = 0;
                    bool use_active_gather = g_active_compaction &&
                                             g_active_particles.initialized &&
                                             g_harmonic_lock.locked &&
                                             g_active_particles.static_baked &&
                                             g_active_particles.h_active_count > 0;

                    // Half-rate: skip gather on odd frames when locked
                    bool skip_gather = use_active_gather && (gather_phase & 1);
                    gather_phase++;

                    // Scale-invariant reference density for shear weighting:
                    // mean density × 8 (shells are ~8× denser than grid average)
                    float shear_rho_ref = (float)N_current / (float)GRID_CELLS * 8.0f;

                    if (skip_gather) {
                        // Skip gather this frame - particles extrapolate from last frame's forces
                        // (Static counter to log occasionally)
                        static int skip_count = 0;
                        if (++skip_count % 500 == 1) {
                            printf("[half-rate] Skipped %d gathers (locked, extrapolating)\n", skip_count);
                        }
                    } else if (use_active_gather) {
                        // Gather only to active particles (static particles don't need updates)
                        int activeBlks = (g_active_particles.h_active_count + 255) / 256;
                        gatherToActiveParticles<<<activeBlks, 256>>>(
                            d_disk, d_grid_density,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_particle_cell, g_active_particles.d_active_list,
                            g_active_particles.h_active_count, dt_sim, substrate_k, g_shear_k, shear_rho_ref,
                            g_kuramoto_k, g_n12_envelope ? 1 : 0, g_envelope_scale
                        );
                    } else {
                        // Full gather to all particles (Step 3: passive particles skipped inside kernel)
                        gatherCellForcesToParticles<<<blocks, threads>>>(
                            d_disk, d_grid_density, d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_particle_cell, d_in_active_region,
                            N_current, dt_sim, substrate_k, g_shear_k, shear_rho_ref,
                            g_kuramoto_k, g_n12_envelope ? 1 : 0, g_envelope_scale
                        );
                    }
                    cudaEventRecord(e_gather_end);

                    // Collect timing every rebuild cycle
                    if (doRebuild) {
                        cudaEventSynchronize(e_gather_end);
                        cudaEventElapsedTime(&ms_scatter, e_scatter_start, e_scatter_end);
                        cudaEventElapsedTime(&ms_compact, e_scatter_end, e_compact_end);
                        cudaEventElapsedTime(&ms_pressure, e_compact_end, e_pressure_end);
                        cudaEventElapsedTime(&ms_gather, e_pressure_end, e_gather_end);
                    }

                    // Debug stats on rebuild frames, every 30 rebuilds (~900 frames)
                    static int rebuild_count = 0;
                    if (doRebuild) {
                        rebuild_count++;
                        if (rebuild_count % 30 == 0 || rebuild_count == 1) {
                            printf("[flags+tiles] frame=%u rebuild=%d | Active: %u cells in %u tiles (%.2f%% of grid)\n",
                                   frame, rebuild_count, h_compact_active_count, h_active_tile_count,
                                   100.0f * h_compact_active_count / GRID_CELLS);
                            printf("[grid timing] scatter=%.2fms compact=%.2fms pressure=%.2fms gather=%.2fms total=%.2fms\n",
                                   ms_scatter, ms_compact, ms_pressure, ms_gather,
                                   ms_scatter + ms_compact + ms_pressure + ms_gather);
                        }
                    }
                } else {
                    // === CADENCE MODE — Rebuild every 30 frames ===

                    // Pass 0+1+2: Rebuild cell state every 30 frames (amortized cost)
                    if (++gridRebuildCounter >= GRID_REBUILD_INTERVAL) {
                        gridRebuildCounter = 0;

                        // Pass 0: Clear cell state
                        clearCellGrid<<<clearBlocks, 256>>>(
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_phase_sin,
                            d_grid_phase_cos,
                            d_grid_pressure_x,
                            d_grid_pressure_y,
                            d_grid_pressure_z,
                            d_grid_vorticity_x,
                            d_grid_vorticity_y,
                            d_grid_vorticity_z
                        );

                        // Pass 1: Scatter particles to cells (atomic accumulation)
                        scatterParticlesToCells<<<blocks, threads>>>(
                            d_disk,
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_phase_sin,
                            d_grid_phase_cos,
                            d_particle_cell,
                            N
                        );

                        // Pass 2: Compute cell fields (fixed 6-neighbor stencil)
                        computeCellFields<<<clearBlocks, 256>>>(
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_pressure_x,
                            d_grid_pressure_y,
                            d_grid_pressure_z,
                            d_grid_vorticity_x,
                            d_grid_vorticity_y,
                            d_grid_vorticity_z,
                            pressure_k,
                            vorticity_k
                        );
                    }

                    // Pass 3: Gather cell forces to particles (every frame, O(1) lookup)
                    // Step 3: passive particles skipped inside kernel via in_active_region check.
                    float shear_rho_ref_cadence = (float)N / (float)GRID_CELLS * 8.0f;
                    gatherCellForcesToParticles<<<blocks, threads>>>(
                        d_disk,
                        d_grid_density,
                        d_grid_pressure_x,
                        d_grid_pressure_y,
                        d_grid_pressure_z,
                        d_grid_vorticity_x,
                        d_grid_vorticity_y,
                        d_grid_vorticity_z,
                        d_grid_phase_sin,
                        d_grid_phase_cos,
                        d_particle_cell,
                        d_in_active_region,
                        N,
                        dt_sim,
                        substrate_k,
                        g_shear_k,
                        shear_rho_ref_cadence,
                        g_kuramoto_k,
                        g_n12_envelope ? 1 : 0,
                        g_envelope_scale
                    );

                    // Debug stats every 900 frames
                    if (frame % 900 == 0) {
                        printf("[grid] Cadence mode: gather every frame, scatter/stencil every %d frames\n",
                               GRID_REBUILD_INTERVAL);
                    }
                }
            }

            if (do_timing) cudaEventRecord(t_physics);

            // === ENTROPY INJECTION TEST ===
            // Inject high-entropy cluster when E key is pressed
            if (g_inject_entropy) {
                injectEntropyCluster<<<blocks, threads>>>(d_disk, N_current, sim_time);
                g_inject_entropy = false;  // One-shot injection
                printf("[ENTROPY] Cluster injected at r=200, falling toward core...\n");
            }
        }

        // === RENDERING: Fill instance buffers ===
        // Skip all rendering work in headless mode
        if (!g_headless) {
        // Fill Vulkan shared buffer with LOD-aware kernel
        // Camera position for LOD distance calculation
        float camX_fill = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * sinf(vkCtx.cameraYaw);
        float camY_fill = vkCtx.cameraRadius * sinf(vkCtx.cameraPitch);
        float camZ_fill = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * cosf(vkCtx.cameraYaw);

        // LOD thresholds (adjust with camera distance for infinite zoom effect)
        // Near threshold scales with camera radius: closer camera = tighter near zone
        float baseFactor = fminf(vkCtx.cameraRadius / 400.0f, 2.0f);  // Scale factor 0.5-2x
        float nearThreshold = vkCtx.lodConfig.nearThreshold * baseFactor;
        float farThreshold = vkCtx.lodConfig.farThreshold * baseFactor;
        float volumeScale = vkCtx.lodConfig.volumeScale;

        // Check if we should use stream compaction (runtime toggleable via L key)
        bool useCompaction = hybridLODEnabled && d_densityGrid != nullptr &&
                             vkCtx.useIndirectDraw && d_compactedParticles != nullptr;

        if (useCompaction) {
            // === HYBRID LOD WITH STREAM COMPACTION ===
            // True vertex culling: only visible particles go through the pipeline

            // 1. Clear density grid
            int gridVoxels = LOD_GRID_SIZE * LOD_GRID_SIZE * LOD_GRID_SIZE;
            clearDensityGrid<<<(gridVoxels * 4 + 255) / 256, 256>>>(d_densityGrid, gridVoxels);

            // 2. Reset write index for compaction
            cudaMemsetAsync(d_writeIndex, 0, sizeof(unsigned int));

            // 3. Compact visible particles (two paths: flat scan or octree traversal)
            // Adaptive: only use octree if previous frame had >10% culling
            // Probe every 30 frames to check if culling rate changed
            static float last_cull_ratio = 0.0f;
            static int probe_counter = 0;
            bool shouldProbe = (probe_counter++ % 30 == 0);
            bool useOctreePath = useOctreeTraversal && octreeEnabled && h_leaf_node_count > 0
                                 && (last_cull_ratio > 0.10f || shouldProbe);

            if (useOctreePath) {
                // === OCTREE TRAVERSAL PATH WITH FRUSTUM CULLING ===

                // Build view-projection matrix for frustum extraction
                Vec3 eye = {camX_fill, camY_fill, camZ_fill};
                Vec3 center = {0, 0, 0};
                Mat4 view = Mat4::lookAt(eye, center, {0, 1, 0});

                int fb_w, fb_h;
                glfwGetFramebufferSize(vkCtx.window, &fb_w, &fb_h);
                float aspect = (float)fb_w / (float)fb_h;
                Mat4 proj = Mat4::perspective(PI / 4.0f, aspect, 0.1f, 2000.0f);
                proj.m[5] *= -1.0f;  // Vulkan Y inversion
                Mat4 vp = Mat4::mul(proj, view);

                // Extract frustum planes
                FrustumPlanes frustum;
                extractFrustumPlanes(vp.m, frustum);

                // Copy pristine leaf counts to working buffer
                cudaMemcpyAsync(d_leaf_counts_culled, d_leaf_counts,
                               h_leaf_node_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

                // Cull leaves against frustum (zeros out particle_count for culled nodes)
                int cullBlocks = (h_leaf_node_count + 255) / 256;
                cullLeafNodesFrustum<<<cullBlocks, 256>>>(
                    d_leaf_counts_culled,
                    d_leaf_node_indices,
                    d_octree_nodes,
                    h_leaf_node_count,
                    frustum
                );

                // Exclusive scan on culled counts to get output offsets (CUB)
                gpuExclusiveScan(d_leaf_counts_culled, d_leaf_offsets, h_leaf_node_count);

                // Compute total from scan tail: total = offset[last] + count[last]
                // Read both values to compute total
                uint32_t tail_values[2];
                cudaMemcpy(&tail_values[0], d_leaf_offsets + h_leaf_node_count - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(&tail_values[1], d_leaf_counts_culled + h_leaf_node_count - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost);
                h_culled_total_particles = tail_values[0] + tail_values[1];

                // Update culling ratio for adaptive fallback
                last_cull_ratio = 1.0f - (float)h_culled_total_particles / (float)h_cached_total_particles;

                // Print culling stats occasionally
                static int cull_stat_frame = 0;
                if (cull_stat_frame++ % 300 == 0) {
                    printf("[frustum] %u → %u particles (%.1f%% culled), %d blocks\n",
                           h_cached_total_particles, h_culled_total_particles, last_cull_ratio * 100.0f,
                           (h_culled_total_particles + 255) / 256);
                }

                // Dispatch V3 traversal with culled particle count
                if (h_culled_total_particles > 0) {
                    int v3Blocks = (h_culled_total_particles + 255) / 256;
                    octreeRenderTraversalV3<<<v3Blocks, 256>>>(
                        d_compactedParticles,
                        d_leaf_offsets,
                        d_leaf_node_indices,
                        d_octree_nodes,
                        d_particle_ids,
                        d_disk,
                        h_leaf_node_count,
                        h_culled_total_particles
                    );
                }

                // Set total (culled count)
                cudaMemcpyAsync(d_writeIndex, &h_culled_total_particles,
                               sizeof(unsigned int), cudaMemcpyHostToDevice);
            } else {
                // === FLAT SCAN PATH (original) ===
                compactVisibleParticles<<<blocks, threads>>>(
                    d_compactedParticles,    // Output: compacted visible particles
                    d_drawCommand,           // Unused in this kernel
                    d_densityGrid,           // Output: density grid for volume
                    d_disk,
                    N,
                    camX_fill, camY_fill, camZ_fill,
                    nearThreshold,
                    farThreshold,
                    volumeScale,
                    d_writeIndex,            // Atomic counter
                    nullptr                  // maxRadius disabled for performance
                );
            }

            // 4. Update indirect draw command with final count
            updateIndirectDrawCommand<<<1, 1>>>(d_drawCommand, d_writeIndex);

            // 5. DOUBLE BUFFER SWAP: back buffer is now ready, swap with front
            // After swap: renderer reads just-written data, compaction writes to old front
            if (doubleBufferEnabled) {
                // Swap indices
                int temp = frontBuffer;
                frontBuffer = backBuffer;
                backBuffer = temp;

                // Update Vulkan context to point to new front buffer (for renderer)
                vkCtx.compactedParticleBuffer = indirectDrawBuffers[frontBuffer].compactedBuffer;
                vkCtx.compactedParticleBufferMemory = indirectDrawBuffers[frontBuffer].compactedMemory;
                vkCtx.indirectDrawBuffer = indirectDrawBuffers[frontBuffer].indirectBuffer;
                vkCtx.indirectDrawBufferMemory = indirectDrawBuffers[frontBuffer].indirectMemory;

                // Update CUDA pointers to new back buffer (for next frame's compaction)
                d_compactedParticles = (ParticleVertex*)indirectDrawBuffers[backBuffer].cudaCompactedPtr;
                d_drawCommand = (CUDADrawIndirectCommand*)indirectDrawBuffers[backBuffer].cudaIndirectPtr;
                d_writeIndex = indirectDrawBuffers[backBuffer].cudaWriteIndex;
            }

            // Visible count readback removed — caused stutter from GPU sync
        } else if (hybridLODEnabled && d_densityGrid != nullptr && !vkCtx.useIndirectDraw) {
            // === HYBRID LOD WITHOUT COMPACTION (toggle OFF for perf comparison) ===
            // Fill ALL particles to main buffer - no culling, for baseline comparison
            if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                fillVulkanSunTraceBuffer<<<blocks, threads>>>(
                    (VulkanSunTrace*)d_vkParticles, d_disk, N_current);
            } else {
                fillVulkanParticleBuffer<<<blocks, threads>>>(d_vkParticles, d_disk, N_current);
            }
            vkCtx.nearParticleCount = 0;  // Clear count to indicate no culling
        } else if (hybridLODEnabled && d_densityGrid != nullptr) {
            // === HYBRID LOD WITHOUT COMPACTION (fallback) ===
            // LOD kernel sets alpha=0 for far particles but still processes all vertices
            int gridVoxels = LOD_GRID_SIZE * LOD_GRID_SIZE * LOD_GRID_SIZE;
            clearDensityGrid<<<(gridVoxels * 4 + 255) / 256, 256>>>(d_densityGrid, gridVoxels);

            cudaMemsetAsync(d_nearCount, 0, sizeof(unsigned int));

            fillVulkanParticleBufferLOD<<<blocks, threads>>>(
                d_vkParticles,
                d_densityGrid,
                d_disk,
                N_current,
                camX_fill, camY_fill, camZ_fill,
                nearThreshold,
                farThreshold,
                volumeScale,
                d_nearCount
            );

            // Near count readback removed — caused stutter from GPU sync
        } else {
            // Simple fill (no LOD)
            // When phase-primary mode is enabled, fill with SunTrace data instead
            // Both structs are 40 bytes so they share the same buffer
            if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                fillVulkanSunTraceBuffer<<<blocks, threads>>>(
                    (VulkanSunTrace*)d_vkParticles, d_disk, N_current);
            } else {
                fillVulkanParticleBuffer<<<blocks, threads>>>(d_vkParticles, d_disk, N_current);
            }
        }
        // Sync CUDA before Vulkan reads the particle buffer
        cudaDeviceSynchronize();
        } // End of !g_headless check for Vulkan rendering

        // === ASYNC DOUBLE-BUFFERED BRIDGE METRICS ===
        // Zero-sync sampling: launch reduction on stream, read previous frame's result
        // This eliminates ALL pipeline stalls from GPU↔CPU synchronization

        // 1. Determine which buffer to write to (other one has previous frame's data)
        int write_buffer = 1 - current_buffer;

        // 2. Launch async reduction on the sample stream (128 particles only, O(1))
        sampleReductionKernel<<<1, 128, 0, sample_stream>>>(
            d_disk, d_sample_indices, SAMPLE_COUNT, d_sample_metrics[write_buffer], g_use_hopfion_topology);

        // 3. Async memcpy to pinned host memory
        static SampleMetrics* h_sample_metrics_back = nullptr;  // Back buffer
        static cudaEvent_t sample_copy_ready;
        static bool sample_event_init = false;
        static bool sample_copy_pending = false;
        if (!sample_event_init) {
            cudaMallocHost(&h_sample_metrics_back, sizeof(SampleMetrics));
            *h_sample_metrics_back = *h_sample_metrics;  // Initialize
            cudaEventCreate(&sample_copy_ready);
            sample_event_init = true;
        }

        // Check if previous copy is done before launching new one
        if (sample_copy_pending) {
            if (cudaEventQuery(sample_copy_ready) == cudaSuccess) {
                // Swap: back becomes front, front becomes back
                SampleMetrics* tmp = h_sample_metrics;
                h_sample_metrics = h_sample_metrics_back;
                h_sample_metrics_back = tmp;
                sample_copy_pending = false;
            }
            // If not done, skip this frame's copy (use stale data)
        }

        if (!sample_copy_pending) {
            cudaMemcpyAsync(h_sample_metrics_back, d_sample_metrics[write_buffer],
                            sizeof(SampleMetrics), cudaMemcpyDeviceToHost, sample_stream);
            cudaEventRecord(sample_copy_ready, sample_stream);
            sample_copy_pending = true;
        }

        // 4. Swap buffers for next frame
        current_buffer = write_buffer;

        // 5. Smooth the metrics using FRONT buffer (always safe to read, no sync)
        const float BRIDGE_SMOOTH = 0.25f;
        pump_bridge.avg_scale = pump_bridge.avg_scale * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->avg_scale * BRIDGE_SMOOTH;
        pump_bridge.avg_residual = pump_bridge.avg_residual * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->avg_residual * BRIDGE_SMOOTH;
        pump_bridge.total_work = pump_bridge.total_work * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->total_work * BRIDGE_SMOOTH;

        // === HEARTBEAT: Always updates (cheap CPU-side calculation) ===
        float scale_phase = pump_bridge.avg_scale * 0.3f + sim_time * 2.0f;
        pump_bridge.heartbeat = sinf(scale_phase);

        // === UPDATE VALIDATION CONTEXT (for key handler access) ===
        g_validation_ctx.d_disk = d_disk;
        g_validation_ctx.N_current = N_current;
        g_validation_ctx.sim_time = sim_time;
        g_validation_ctx.heartbeat = pump_bridge.heartbeat;
        g_validation_ctx.avg_scale = pump_bridge.avg_scale;
        g_validation_ctx.avg_residual = pump_bridge.avg_residual;

        // === VALIDATION FRAME EXPORT ===
        // Stack capture mode: exports 64 consecutive frames with sync
        if (isStackCaptureActive()) {
            cudaDeviceSynchronize();  // Ensure frame is complete before export
            maybeExportStackFrame(d_disk, N_current, sim_time,
                                  pump_bridge.heartbeat, pump_bridge.avg_scale,
                                  pump_bridge.avg_residual, g_grid_dim, 500.0f);
        }
        // Legacy continuous mode (every N frames)
        maybeExportFrame(d_disk, N_current, frame, sim_time,
                         pump_bridge.heartbeat, pump_bridge.avg_scale,
                         pump_bridge.avg_residual, g_grid_dim, 500.0f);

        // === TOPOLOGY RING BUFFER UPDATE ===
        // Records downsampled m-field to detect crystallization events
        // Uses h_stats_cache (may be 1-2 frames stale, acceptable for detection)
        {
            // Compute stability from cached shell data
            float stability = 0.0f;
            float mean_n = 1.0f;
            if (h_sample_metrics->num_shells > 0) {
                float n_min = h_sample_metrics->shell_n[0];
                float n_max = h_sample_metrics->shell_n[0];
                float n_sum = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float n = h_sample_metrics->shell_n[i];
                    if (n < n_min) n_min = n;
                    if (n > n_max) n_max = n;
                    n_sum += n;
                }
                mean_n = n_sum / h_sample_metrics->num_shells;
                float delta_n = n_max - n_min;
                stability = (mean_n > 1.001f) ? delta_n / (mean_n - 1.0f) : 1.0f;
            }

            // Get particle position/velocity device pointers via offsetof
            const float* d_pos_x = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_x));
            const float* d_pos_y = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_y));
            const float* d_pos_z = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_z));
            const float* d_vel_x = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_x));
            const float* d_vel_y = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_y));
            const float* d_vel_z = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_z));

            bool crystal_detected = topology_recorder_update(
                d_pos_x, d_pos_y, d_pos_z,
                d_vel_x, d_vel_y, d_vel_z,
                N_current,
                frame,
                sim_time,
                h_stats_cache.total_kinetic_energy,
                stability,
                pump_bridge.avg_scale,
                mean_n,
                250.0f,  // grid_half_size
                g_harmonic_lock.locked  // lock-aware topology gating
            );

            // Auto-dump on crystal detection and pause for user verification
            if (crystal_detected) {
                printf("\n");
                printf("╔══════════════════════════════════════════════════════════════╗\n");
                printf("║           *** CRYSTAL DETECTED ***                           ║\n");
                printf("║                                                              ║\n");
                printf("║   Topology ring buffer auto-dumped.                          ║\n");
                printf("║   Simulation PAUSED for visual verification.                 ║\n");
                printf("║                                                              ║\n");
                printf("║   Press SPACE to continue (disables further detection)       ║\n");
                printf("╚══════════════════════════════════════════════════════════════╝\n");
                printf("\n");
                topology_recorder_dump("auto_crystal");
                g_cam.paused = true;  // Pause simulation
            }
        }

        // Check if user has unpaused after crystal detection
        if (!g_cam.paused && topology_recorder_awaiting_continue()) {
            topology_recorder_acknowledge_crystal();
        }

        // === ASYNC STATS COLLECTION (fully non-blocking) ===
        // Check if previous async reduction AND copy are complete
        StressCounters sc = h_stats_cache;  // Use cached results by default
        static bool stats_copy_pending = false;
        static cudaEvent_t stats_copy_ready;
        static bool stats_copy_event_init = false;
        if (!stats_copy_event_init) {
            cudaEventCreate(&stats_copy_ready);
            stats_copy_event_init = true;
        }

        // Check if copy is done (non-blocking)
        if (stats_copy_pending) {
            if (cudaEventQuery(stats_copy_ready) == cudaSuccess) {
                sc = h_stats_cache;  // Now safe to use
                stats_copy_pending = false;
            }
        }

        // Check if reduction is done, then start async copy
        if (stats_pending && !stats_copy_pending) {
            if (cudaEventQuery(stats_ready) == cudaSuccess) {
                // Reduction done - launch async copy (NO sync!)
                cudaMemcpyAsync(&h_stats_cache, d_stress_async, sizeof(StressCounters),
                               cudaMemcpyDeviceToHost, stats_stream);
                cudaEventRecord(stats_copy_ready, stats_stream);
                stats_copy_pending = true;
                stats_pending = false;
            }
        }

        // === Dense R(t) logging (optional) ===
        // Compact per-frame Kuramoto order parameter print for time-series
        // analysis. Minimal overhead (one block reduce + small DtoH).
        if (g_r_log_interval > 0 && frame % g_r_log_interval == 0 && frame > 0) {
            int kr_blocks = (N_current + KR_THREADS - 1) / KR_THREADS;
            if (kr_blocks > kr_max_blocks) kr_blocks = kr_max_blocks;
            reduceKuramotoR<<<kr_blocks, KR_THREADS>>>(
                d_disk, N_current, d_kr_sin_sum, d_kr_cos_sum, d_kr_count);
            cudaMemcpy(h_kr_sin_sum.data(), d_kr_sin_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_cos_sum.data(), d_kr_cos_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_count.data(), d_kr_count, kr_blocks * sizeof(int), cudaMemcpyDeviceToHost);
            double sum_sin = 0.0, sum_cos = 0.0;
            long long total_count = 0;
            for (int b = 0; b < kr_blocks; b++) {
                sum_sin += h_kr_sin_sum[b];
                sum_cos += h_kr_cos_sum[b];
                total_count += h_kr_count[b];
            }
            if (total_count > 0) {
                double inv_n = 1.0 / (double)total_count;
                double mean_sin = sum_sin * inv_n;
                double mean_cos = sum_cos * inv_n;
                float R_t = (float)sqrt(mean_sin * mean_sin + mean_cos * mean_cos);
                printf("[rt] frame=%d R=%.6f\n", frame, R_t);
            }
        }

        // Print stats every 90 frames using sample metrics (no stall)
        if (frame % 90 == 0) {
            // === Compute Kuramoto order parameter R = |⟨e^{iθ}⟩| ===
            int kr_blocks = (N_current + KR_THREADS - 1) / KR_THREADS;
            if (kr_blocks > kr_max_blocks) kr_blocks = kr_max_blocks;
            reduceKuramotoR<<<kr_blocks, KR_THREADS>>>(
                d_disk, N_current, d_kr_sin_sum, d_kr_cos_sum, d_kr_count);
            cudaMemcpy(h_kr_sin_sum.data(), d_kr_sin_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_cos_sum.data(), d_kr_cos_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_count.data(), d_kr_count, kr_blocks * sizeof(int), cudaMemcpyDeviceToHost);
            double sum_sin = 0.0, sum_cos = 0.0;
            long long total_count = 0;
            for (int b = 0; b < kr_blocks; b++) {
                sum_sin += h_kr_sin_sum[b];
                sum_cos += h_kr_cos_sum[b];
                total_count += h_kr_count[b];
            }
            if (total_count > 0) {
                double inv_n = 1.0 / (double)total_count;
                double mean_sin = sum_sin * inv_n;
                double mean_cos = sum_cos * inv_n;
                R_global_cached = (float)sqrt(mean_sin * mean_sin + mean_cos * mean_cos);
            }

            // === Compute per-cell R and radial profile ===
            float r_inner = 0.0f, r_mid = 0.0f, r_outer = 0.0f;
            float R_recon = 0.0f;  // grid-reconstructed global R for consistency check
            if (g_grid_physics && d_grid_R_cell != nullptr) {
                int cell_blocks = (GRID_CELLS + 255) / 256;
                computeRcell<<<cell_blocks, 256>>>(
                    d_grid_density, d_grid_phase_sin, d_grid_phase_cos,
                    d_grid_R_cell, GRID_CELLS);

                // === CONSISTENCY CHECK: R_recon from grid vector sums ===
                // Summing phase_sin[cell] across cells gives total Σ sin(θ_i).
                // Dividing by total density gives ⟨sin θ⟩. R_recon must equal
                // R_global (particle-level reduction) if the grid and particle
                // paths agree. Any discrepancy flags inconsistency (inactive
                // particles, double-counting, sampling bias, etc.).
                {
                    std::vector<float> h_grid_ps(GRID_CELLS);
                    std::vector<float> h_grid_pc(GRID_CELLS);
                    std::vector<float> h_grid_rho(GRID_CELLS);
                    cudaMemcpy(h_grid_ps.data(), d_grid_phase_sin, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_grid_pc.data(), d_grid_phase_cos, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_grid_rho.data(), d_grid_density, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    double tot_s = 0.0, tot_c = 0.0, tot_rho = 0.0;
                    for (int c = 0; c < GRID_CELLS; c++) {
                        tot_s += h_grid_ps[c];
                        tot_c += h_grid_pc[c];
                        tot_rho += h_grid_rho[c];
                    }
                    if (tot_rho > 0.0) {
                        double inv_rho = 1.0 / tot_rho;
                        double ms = tot_s * inv_rho;
                        double mc = tot_c * inv_rho;
                        R_recon = (float)sqrt(ms * ms + mc * mc);
                    }
                }

                // Radial profile: 16 bins from center to edge, noise-floor corrected
                cudaMemsetAsync(d_rc_bin_R, 0, RC_RADIAL_BINS * sizeof(float));
                cudaMemsetAsync(d_rc_bin_W, 0, RC_RADIAL_BINS * sizeof(float));
                cudaMemsetAsync(d_rc_bin_N, 0, RC_RADIAL_BINS * sizeof(float));
                reduceRcellRadialProfile<<<cell_blocks, 256>>>(
                    d_grid_R_cell, d_grid_density,
                    g_grid_dim, RC_RADIAL_BINS, g_grid_cell_size,
                    d_rc_bin_R, d_rc_bin_W, d_rc_bin_N);
                cudaMemcpy(h_rc_bin_R.data(), d_rc_bin_R, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rc_bin_W.data(), d_rc_bin_W, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rc_bin_N.data(), d_rc_bin_N, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);

                // Collapse 16 bins into 3 zones (inner / mid / outer) for printing
                float sum_R[3] = {0.0f, 0.0f, 0.0f};
                float sum_W[3] = {0.0f, 0.0f, 0.0f};
                for (int b = 0; b < RC_RADIAL_BINS; b++) {
                    int zone = (b < 6) ? 0 : (b < 12) ? 1 : 2;
                    sum_R[zone] += h_rc_bin_R[b];
                    sum_W[zone] += h_rc_bin_W[b];
                }
                r_inner = (sum_W[0] > 0.0f) ? sum_R[0] / sum_W[0] : 0.0f;
                r_mid   = (sum_W[1] > 0.0f) ? sum_R[1] / sum_W[1] : 0.0f;
                r_outer = (sum_W[2] > 0.0f) ? sum_R[2] / sum_W[2] : 0.0f;

                // Optional: dump full R_cell grid to disk for offline analysis
                if (g_r_export_interval > 0 && (frame % g_r_export_interval == 0)) {
                    system("mkdir -p r_export");
                    char fname[256];
                    snprintf(fname, sizeof(fname), "r_export/frame_%05d.bin", frame);
                    std::vector<float> h_R(GRID_CELLS);
                    cudaMemcpy(h_R.data(), d_grid_R_cell, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    FILE* fp = fopen(fname, "wb");
                    if (fp) {
                        // Header: grid_dim (int), grid_cell_size (float), frame (int), 4 bytes pad
                        int hdr_dim = g_grid_dim;
                        float hdr_cell = g_grid_cell_size;
                        int hdr_frame = frame;
                        int hdr_pad = 0;
                        fwrite(&hdr_dim, sizeof(int), 1, fp);
                        fwrite(&hdr_cell, sizeof(float), 1, fp);
                        fwrite(&hdr_frame, sizeof(int), 1, fp);
                        fwrite(&hdr_pad, sizeof(int), 1, fp);
                        fwrite(h_R.data(), sizeof(float), GRID_CELLS, fp);
                        fclose(fp);
                        printf("[r-export] Wrote %s (%.1f MB)\n", fname, GRID_CELLS * sizeof(float) / 1.0e6);
                    }
                }
            }

            // === Phase histogram: multi-domain clustering check ===
            cudaMemsetAsync(d_phase_hist, 0, PHASE_HIST_BINS * sizeof(int));
            cudaMemsetAsync(d_phase_omega_sum, 0, PHASE_HIST_BINS * sizeof(float));
            cudaMemsetAsync(d_phase_omega_sq, 0, PHASE_HIST_BINS * sizeof(float));
            int hist_blocks = (N_current + 255) / 256;
            reducePhaseHistogram<<<hist_blocks, 256>>>(d_disk, N_current, d_phase_hist, d_phase_omega_sum, d_phase_omega_sq);
            cudaMemcpy(h_phase_hist.data(), d_phase_hist, PHASE_HIST_BINS * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_phase_omega_sum.data(), d_phase_omega_sum, PHASE_HIST_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_phase_omega_sq.data(), d_phase_omega_sq, PHASE_HIST_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            // Compute max bin for normalization
            int max_bin_count = 0;
            long long hist_total = 0;
            for (int b = 0; b < PHASE_HIST_BINS; b++) {
                if (h_phase_hist[b] > max_bin_count) max_bin_count = h_phase_hist[b];
                hist_total += h_phase_hist[b];
            }

            printf("[frame %5d] fps=%.0f | particles=%d "
            "avg_scale=%.2f (sample=%.2f) | bridge: s=%.2f r=%.3f hb=%.2f | R=%.4f/Rrec=%.4f [%.3f/%.3f/%.3f]",
                   frame, fps_acc > 0 ? fps_frames / fps_acc : 0.0,
                   N_current,
                   h_sample_metrics->avg_scale, h_sample_metrics->avg_scale,
                   pump_bridge.avg_scale, pump_bridge.avg_residual, pump_bridge.heartbeat,
                   R_global_cached, R_recon, r_inner, r_mid, r_outer);

#ifdef VULKAN_INTEROP
            // Show hybrid LOD stats (visible particle count and culling percentage)
            if (vkCtx.useIndirectDraw && vkCtx.nearParticleCount > 0) {
                float cull_pct = 100.0f * (1.0f - (float)vkCtx.nearParticleCount / (float)N);
                printf(" | LOD: %u visible (%.0f%% culled)", vkCtx.nearParticleCount, cull_pct);
            }
#endif

            // Entropy dissolution diagnostic
            if (sc.high_stress_count > 0) {
                float dissolution_pct = 100.0f * (float)sc.high_stress_count / (float)sc.active_count;
                printf(" | DISSOLVING: %u (%.1f%%) at >0.95 stress", sc.high_stress_count, dissolution_pct);
            }
            printf("\n");

            // Phase histogram (32 bins, ASCII heat map + numeric peak analysis).
            // Multi-peak → multi-domain clustering. Flat → uniform. Single
            // peak → global lock. Bin 0 = θ ∈ [0, 2π/32).
            //
            // n_peaks and peak_frac are hoisted to outer scope so the
            // Kuramoto × topology correlation dump below can read them.
            int n_peaks = 0;
            float peak_frac = 0.0f;
            if (hist_total > 0 && max_bin_count > 0) {
                printf("[phase-hist] ");
                const char* ramps = " .,-:;=+*#%@";
                int n_ramps = 12;
                float expected = (float)hist_total / (float)PHASE_HIST_BINS;  // flat baseline
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    float ratio = (float)h_phase_hist[b] / expected;
                    if (ratio > 3.0f) ratio = 3.0f;
                    int r = (int)(ratio * (n_ramps - 1) / 3.0f);
                    if (r < 0) r = 0;
                    if (r >= n_ramps) r = n_ramps - 1;
                    putchar(ramps[r]);
                }
                printf(" max/avg=%.2f\n", (float)max_bin_count / expected);

                // Peak detection: count local maxima that are ≥ 1.5× expected
                // and compute how much of total mass is in peaks vs background.
                long long peak_mass = 0;
                float peak_threshold = 1.5f * expected;
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    int prev = h_phase_hist[(b + PHASE_HIST_BINS - 1) % PHASE_HIST_BINS];
                    int curr = h_phase_hist[b];
                    int next = h_phase_hist[(b + 1) % PHASE_HIST_BINS];
                    if (curr > prev && curr > next && (float)curr > peak_threshold) {
                        n_peaks++;
                        peak_mass += curr;
                    }
                }
                peak_frac = (float)peak_mass / (float)hist_total;
                printf("[phase-hist] peaks=%d  peak_mass_frac=%.6f  total_hist=%lld\n", n_peaks, peak_frac, hist_total);

                // === Velocity-filter check (Gemini's "Sieve" hypothesis) ===
                // Do clustered particles have a different ω distribution than
                // background particles? If the clusters are selecting particles
                // whose ω matches the envelope beat frequency, the clustered
                // subset should have a tighter ω variance and a mean ω closer
                // to the envelope-determined attractor frequency.
                double cluster_w_sum = 0.0, cluster_w_sq = 0.0;
                long long cluster_n = 0;
                double bg_w_sum = 0.0, bg_w_sq = 0.0;
                long long bg_n = 0;
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    int prev = h_phase_hist[(b + PHASE_HIST_BINS - 1) % PHASE_HIST_BINS];
                    int curr = h_phase_hist[b];
                    int next = h_phase_hist[(b + 1) % PHASE_HIST_BINS];
                    bool is_peak = (curr > prev && curr > next && (float)curr > peak_threshold);
                    if (is_peak) {
                        cluster_w_sum += h_phase_omega_sum[b];
                        cluster_w_sq += h_phase_omega_sq[b];
                        cluster_n += curr;
                    } else {
                        bg_w_sum += h_phase_omega_sum[b];
                        bg_w_sq += h_phase_omega_sq[b];
                        bg_n += curr;
                    }
                }
                if (cluster_n > 0 && bg_n > 0) {
                    double cluster_mean = cluster_w_sum / cluster_n;
                    double cluster_var = cluster_w_sq / cluster_n - cluster_mean * cluster_mean;
                    double cluster_std = (cluster_var > 0) ? sqrt(cluster_var) : 0.0;
                    double bg_mean = bg_w_sum / bg_n;
                    double bg_var = bg_w_sq / bg_n - bg_mean * bg_mean;
                    double bg_std = (bg_var > 0) ? sqrt(bg_var) : 0.0;
                    printf("[omega-filter] cluster: μ=%.4f σ=%.4f n=%lld  bg: μ=%.4f σ=%.4f n=%lld\n",
                           cluster_mean, cluster_std, cluster_n,
                           bg_mean, bg_std, bg_n);
                }

                // Numeric dump: ratio per bin for quantitative inspection
                printf("[phase-hist-num] ");
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    float ratio = (float)h_phase_hist[b] / expected;
                    printf("%.2f ", ratio);
                }
                printf("\n");
            }
            float latest_Q = topology_recorder_get_latest_Q();
            printf("[%s] num=%d Q=%.2f | ",
                   g_use_hopfion_topology ? "hopfion shells" : "smooth gradient",
                   h_sample_metrics->num_shells, latest_Q);
            for (int i = 0; i < h_sample_metrics->num_shells && i < 4; i++) {
                printf("r=%.1f n=%.3f | ", h_sample_metrics->shell_radii[i], h_sample_metrics->shell_n[i]);
            }
            printf("\n");

            // === Kuramoto × topology correlation dump ===
            // One CSV-friendly row per stats frame with all scalars needed to
            // correlate phase-cluster structure with Hopfion invariant Q.
            // Columns: frame, R_global, R_recon, n_peaks, peak_mass_frac,
            //          Q, num_shells, N, R_inner, R_mid, active_frac
            // Step 4: active_frac = fraction of alive particles classified as active (siphon).
            if (g_qr_corr_log) {
                // Step 4: count active-region particles via host readback.
                // 1MB every 90 frames = negligible bandwidth.
                float active_frac = 1.0f;
#if ENABLE_PASSIVE_ADVECTION
                {
                    uint32_t h_active_region_count = 0;
                    std::vector<uint8_t> h_region_mask(N_current);
                    cudaMemcpy(h_region_mask.data(), d_in_active_region,
                               N_current * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                    for (int j = 0; j < N_current; j++)
                        if (h_region_mask[j]) h_active_region_count++;
                    active_frac = (float)h_active_region_count / (float)N_current;
                }
#endif
                printf("[QR-corr] %d %.6f %.6f %d %.6f %.4f %d %d %.4f %.4f %.4f\n",
                       frame,
                       R_global_cached,
                       R_recon,
                       n_peaks,
                       peak_frac,
                       latest_Q,
                       h_sample_metrics->num_shells,
                       N_current,
                       r_inner,
                       r_mid,
                       active_frac);
            }

            // === RING STABILITY DIAGNOSTIC ===
            // Compute observational stability: Δn / (n_avg - 1)
            if (h_sample_metrics->num_shells > 0) {
                // Find min and max refractive index across shells
                float n_min = h_sample_metrics->shell_n[0];
                float n_max = h_sample_metrics->shell_n[0];
                float n_sum = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float n = h_sample_metrics->shell_n[i];
                    if (n < n_min) n_min = n;
                    if (n > n_max) n_max = n;
                    n_sum += n;
                }
                float n_avg = n_sum / h_sample_metrics->num_shells;
                float delta_n = n_max - n_min;

                // Instability ratio (target: ~10% for EHT match)
                float stability_pct = 100.0f * delta_n / (n_avg - 1.0f);

                // === M_EFF CALCULATION (Option D: Active Region Only) ===
                // Only include shells within the active pumping region where mass
                // is being fed. Shells beyond M_EFF_ACTIVE_RADIUS are "fossil" structures
                // that propagate the refractive index wave but no longer accumulate mass.
                float M_eff = 0.0f;
                int active_shells = 0;
                float innermost_active_r = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float r = h_sample_metrics->shell_radii[i];
                    if (r < M_EFF_ACTIVE_RADIUS) {
                        M_eff += (h_sample_metrics->shell_n[i] - 1.0f) * r;
                        active_shells++;
                        if (innermost_active_r == 0.0f || r < innermost_active_r) {
                            innermost_active_r = r;
                        }
                    }
                }

                // Log shell migration for wave propagation analysis
                // (useful even when shells leave active region)
                float innermost_r = h_sample_metrics->shell_radii[0];
                static float prev_innermost_r = 0.0f;
                float migration_rate = 0.0f;
                if (prev_innermost_r > 0.0f && frame > 100) {
                    migration_rate = (innermost_r - prev_innermost_r);  // units per diagnostic interval
                }
                prev_innermost_r = innermost_r;

                printf("[lensing] stability=%.1f%% (target ~10%%) | M_eff=%.3f (%d active shells, r<%.0f) | n_avg=%.4f Δn=%.4f",
                       stability_pct, M_eff, active_shells, M_EFF_ACTIVE_RADIUS, n_avg, delta_n);

                // Show migration warning if shells are leaving active region
                if (active_shells < h_sample_metrics->num_shells && innermost_r > M_EFF_ACTIVE_RADIUS * 0.5f) {
                    printf(" | wave: r=%.0f (migration=%.2f/frame)", innermost_r, migration_rate);
                }
                printf("\n");

                // === PREDICTIVE LOCKING UPDATE ===
                // Check if we should enter/exit locked state based on stability metrics
                extern bool g_predictive_locking;
                if (g_predictive_locking) {
                    extern HarmonicLock g_harmonic_lock;
                    HarmonicLock& lock = g_harmonic_lock;

                    // Get current Q
                    float current_Q = latest_Q;
                    float Q_delta = fabsf(current_Q - lock.prev_Q);
                    int shell_count = h_sample_metrics->num_shells;

                    // Check lock criteria:
                    // 1. Shell count stable (8 shells)
                    // 2. Q variance low (not jumping around)
                    // 3. Stability low (shells well-formed)
                    bool shell_stable = (shell_count == lock.prev_shell_count && shell_count >= 6);
                    bool q_stable = (Q_delta < HarmonicLock::Q_VARIANCE_THRESHOLD);
                    bool stability_ok = (stability_pct < HarmonicLock::STABILITY_THRESHOLD * 100.0f);

                    if (shell_stable && q_stable && stability_ok) {
                        lock.stable_frames++;
                        if (!lock.locked && lock.stable_frames >= HarmonicLock::LOCK_THRESHOLD_FRAMES) {
                            lock.locked = true;
                            lock.lock_recheck_counter = 0;
                            printf("[lock] LOCKED: %d shells, Q=%.2f, stability=%.1f%% — skipping mip-tree rebuild\n",
                                   shell_count, current_Q, stability_pct);
                        }
                    } else {
                        // Lost stability
                        if (lock.locked) {
                            printf("[lock] UNLOCKED: shells=%d→%d, ΔQ=%.2f, stability=%.1f%%\n",
                                   lock.prev_shell_count, shell_count, Q_delta, stability_pct);
                        }
                        lock.stable_frames = 0;
                        lock.locked = false;
                    }

                    // Update previous values
                    lock.prev_shell_count = shell_count;
                    lock.prev_Q = current_Q;

                    // Periodic recheck when locked
                    if (lock.locked) {
                        lock.lock_recheck_counter++;
                    }
                }

                // === GEOMETRIC COHERENCE PUMP TRACKING ===
                // Power injected: P = (m·n)² — self-limiting
                // Equilibrium: E = (m·n)²/γ — bounded by |m|²/γ
                // With coherence filter: only aligned modes survive
                printf("[coherence] E_kin=%.2e E/N=%.3f | λ=%.3f γ=%.3f (bounded)\n",
                       sc.total_kinetic_energy, sc.energy_per_particle,
                       COHERENCE_LAMBDA, COHERENCE_GAMMA);

                // === SEAM DRIFT TRACKING (m=3 phase over time) ===
                // Log the m=3 phase angle to detect whether seam orientation drifts as M_eff grows
                if (sc.stress_sample_count > 10 && g_seam_drift_count < SEAM_DRIFT_LOG_SIZE) {
                    // Compute m=3 phase from circular mean: atan2(sin_sum, cos_sum)
                    float m3_phase = atan2f(sc.stress_sin_sum, sc.stress_cos_sum);
                    // Convert to degrees [0, 120) - the m=3 fundamental domain
                    float phase_deg = m3_phase * 180.0f / 3.14159265f;  // [-180, 180]
                    if (phase_deg < 0) phase_deg += 360.0f;             // [0, 360)
                    phase_deg = fmodf(phase_deg, 120.0f);               // [0, 120) - m=3 symmetry

                    g_seam_drift_log[g_seam_drift_count].frame = frame;
                    g_seam_drift_log[g_seam_drift_count].M_eff = M_eff;
                    g_seam_drift_log[g_seam_drift_count].phase_deg = phase_deg;
                    g_seam_drift_log[g_seam_drift_count].sample_count = sc.stress_sample_count;
                    g_seam_drift_count++;
                }

                // === PHOTON RING RADIUS TRACKING (EHT Observable) ===
                // Track the Einstein ring radius over time to verify geometric stability
                // EHT measurements show < 2% variation in M87* and Sgr A*
                static float ring_history[100] = {0};
                static int ring_idx = 0;
                static int ring_count = 0;

                float current_ring = h_sample_metrics->photon_ring_radius;
                ring_history[ring_idx] = current_ring;
                ring_idx = (ring_idx + 1) % 100;
                if (ring_count < 100) ring_count++;

                // Compute ring radius variation over last 100 frames
                if (ring_count > 10) {
                    float ring_min = ring_history[0];
                    float ring_max = ring_history[0];
                    float ring_sum = 0.0f;
                    for (int i = 0; i < ring_count; i++) {
                        if (ring_history[i] < ring_min) ring_min = ring_history[i];
                        if (ring_history[i] > ring_max) ring_max = ring_history[i];
                        ring_sum += ring_history[i];
                    }
                    float ring_avg = ring_sum / ring_count;
                    float ring_variation = 100.0f * (ring_max - ring_min) / ring_avg;

                    printf("[photon ring] R=%.2f | ΔR/R=%.2f%% (EHT target <2%%) | [%.2f ... %.2f]",
                           ring_avg, ring_variation, ring_min, ring_max);

                    // === SHELL-RING CONVERGENCE WARNING ===
                    // Gemini observed: innermost shell (r~90) expanding, ring (R~105) contracting
                    // When they meet, we test GPT's necessity question:
                    //   - Does topology cause graceful re-weaving? (proves necessity)
                    //   - Or does geometry collapse? (disproves necessity)
                    float r_inner = h_sample_metrics->shell_radii[h_sample_metrics->num_shells - 1];
                    float convergence_ratio = r_inner / ring_avg;

                    if (convergence_ratio > 0.95f) {
                        // CRITICAL: Shell has reached/passed the photon ring
                        printf(" | 🔥 TOPOLOGY TEST: Shell swallowing photon ring!");
                    } else if (convergence_ratio > 0.8f) {
                        // WARNING: Approaching convergence
                        printf(" | ⚠ CONVERGENCE: r_inner=%.1f (%.0f%% of R)", r_inner, convergence_ratio * 100.0f);

                        // Calculate shell spacing to predict re-weaving
                        if (h_sample_metrics->num_shells > 1) {
                            float r_next = h_sample_metrics->shell_radii[h_sample_metrics->num_shells - 2];
                            float shell_gap = r_next - r_inner;
                            printf(" | gap_to_next=%.1f", shell_gap);
                        }
                    }
                    printf("\n");

                    // === CONTROLLED EXPERIMENT TERMINATION ===
                    // Only auto-terminate in headless mode - interactive runs until user closes window
                    const int TARGET_FRAMES = g_target_frames;
                    const float TARGET_RADIUS = g_target_ring_radius;

                    bool termination_condition = false;
                    if (g_headless) {
                        if (g_terminate_on_radius) {
                            // Radius-based: terminate when ring reaches target (need at least 10 samples)
                            termination_condition = (ring_avg >= TARGET_RADIUS && ring_count >= 10);
                        } else {
                            // Frame-based: terminate at fixed frame count
                            termination_condition = (frame >= TARGET_FRAMES);
                        }
                    }

                    if (termination_condition) {
                        printf("\n");
                        printf("╔════════════════════════════════════════════════════════════════╗\n");
                        if (g_terminate_on_radius) {
                            printf("║  RADIUS-CONTROLLED EXPERIMENT COMPLETE                         ║\n");
                        } else {
                            printf("║  CONTROLLED EXPERIMENT COMPLETE (Equal Time)                   ║\n");
                        }
                        printf("╚════════════════════════════════════════════════════════════════╝\n");

                        if (g_terminate_on_radius) {
                            printf("[TERMINATION] Target ring radius reached:\n");
                            printf("  ✓ Ring R = %.2f (target: %.2f)\n", ring_avg, TARGET_RADIUS);
                            printf("  ✓ Frames = %d\n", frame);
                        } else {
                            printf("[TERMINATION] Target frames reached:\n");
                            printf("  ✓ Frames = %d (target: %d)\n", frame, TARGET_FRAMES);
                        }
                        printf("  ✓ M_eff = %.3f (compare mass retention)\n", M_eff);
                        printf("\nFinal Results:\n");
                        printf("  Radial Topology: %s\n", g_use_hopfion_topology ? "Hopfion shells" : "Smooth gradient");
                        printf("  Arm Mode: %s\n", !g_enable_arms ? "DISABLED" : g_use_arm_topology ? "Discrete boundaries" : "Smooth waves");
                        printf("  Photon Ring R: %.2f\n", ring_avg);
                        printf("  Ring Stability ΔR/R: %.2f%% (EHT target <2%%)\n", ring_variation);
                        printf("  Lensing Stability: %.1f%% (target ~10%%)\n", stability_pct);
                        printf("  Active Particles: %u\n", sc.active_count);
                        printf("  Ejected Particles: %u\n", sc.ejected_count);

                        // === AZIMUTHAL EJECTION DISTRIBUTION ===
                        // Tracks WHERE particles are ejected around the disk
                        // If pump creates m=3 pattern, we expect 3-fold symmetry in ejections
                        printf("\n");
                        printf("╔════════════════════════════════════════════════════════════╗\n");
                        printf("║  AZIMUTHAL EJECTION DISTRIBUTION (m=3 pump hypothesis)    ║\n");
                        printf("╚════════════════════════════════════════════════════════════╝\n");
                        printf("\n");

                        // Find max for scaling
                        unsigned int ejection_max = 0;
                        for (int b = 0; b < 16; b++) {
                            if (sc.ejection_bins[b] > ejection_max) ejection_max = sc.ejection_bins[b];
                        }

                        // Print histogram
                        printf("  φ (deg)   Count     Distribution\n");
                        printf("  ════════════════════════════════════════════════════\n");
                        unsigned int ejection_total = 0;
                        for (int b = 0; b < 16; b++) {
                            ejection_total += sc.ejection_bins[b];
                        }

                        for (int b = 0; b < 16; b++) {
                            float angle = b * 22.5f;
                            float pct = ejection_total > 0 ? 100.0f * sc.ejection_bins[b] / ejection_total : 0.0f;
                            int bar_len = ejection_max > 0 ? (sc.ejection_bins[b] * 40 / ejection_max) : 0;

                            printf("  %5.1f°  %7u   ", angle, sc.ejection_bins[b]);
                            for (int j = 0; j < bar_len; j++) printf("█");
                            printf(" %.1f%%\n", pct);
                        }

                        // Calculate m=3 asymmetry metric
                        // If pump creates 3 peaks, bins 0,5,10 (or shifted) should dominate
                        // Compute FFT-like m=3 component
                        float cos_sum = 0.0f, sin_sum = 0.0f;
                        for (int b = 0; b < 16; b++) {
                            float phi = b * 22.5f * 3.14159f / 180.0f;  // Convert to radians
                            float weight = ejection_total > 0 ? (float)sc.ejection_bins[b] / ejection_total : 0.0f;
                            cos_sum += weight * cosf(3.0f * phi);  // m=3 mode
                            sin_sum += weight * sinf(3.0f * phi);
                        }
                        float m3_amplitude = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum);
                        float m3_phase_deg = atan2f(sin_sum, cos_sum) * 180.0f / 3.14159f;

                        printf("\n  m=3 Mode Analysis:\n");
                        printf("    Amplitude: %.3f (0=uniform, 1=perfect 3-fold)\n", m3_amplitude);
                        printf("    Phase: %.1f° (orientation of pattern)\n", m3_phase_deg);

                        if (m3_amplitude > 0.3f) {
                            printf("    → STRONG m=3 asymmetry detected! Pump-driven ejection.\n");
                        } else if (m3_amplitude > 0.15f) {
                            printf("    → Moderate m=3 signal. Partial pump influence.\n");
                        } else {
                            printf("    → Weak m=3. Ejections appear isotropic.\n");
                        }

                        // === BEAT FREQUENCY CROSSOVER TEST ===
                        // Tests whether m=3 clustering is due to beat frequency (ω_orb - ω_pump)
                        // Crossover radius r≈185: inner zone has ω_orb > ω_pump, outer has ω_orb < ω_pump
                        // If beat frequency model is correct: clustering differs between zones
                        if (sc.inner_ejection_total > 0 || sc.outer_ejection_total > 0) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  BEAT FREQUENCY CROSSOVER TEST (Claude's ω_orb vs ω_pump)     ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Crossover radius: r = %.0f (where ω_orb = ω_pump ≈ 0.125/step)\n", BEAT_CROSSOVER_RADIUS);
                            printf("Inner zone (r < %.0f): %u ejections\n", BEAT_CROSSOVER_RADIUS, sc.inner_ejection_total);
                            printf("Outer zone (r ≥ %.0f): %u ejections\n", BEAT_CROSSOVER_RADIUS, sc.outer_ejection_total);
                            printf("\n");

                            // Print side-by-side comparison
                            printf("  Sector    INNER ZONE (ω_orb > ω_pump)    OUTER ZONE (ω_orb < ω_pump)\n");
                            printf("  ═══════════════════════════════════════════════════════════════════\n");

                            // Find max for scaling
                            unsigned int inner_max = 0, outer_max = 0;
                            for (int b = 0; b < 16; b++) {
                                if (sc.inner_ejection_bins[b] > inner_max) inner_max = sc.inner_ejection_bins[b];
                                if (sc.outer_ejection_bins[b] > outer_max) outer_max = sc.outer_ejection_bins[b];
                            }

                            for (int b = 0; b < 16; b++) {
                                float angle = b * 22.5f;
                                float inner_pct = sc.inner_ejection_total > 0 ?
                                    100.0f * sc.inner_ejection_bins[b] / sc.inner_ejection_total : 0.0f;
                                float outer_pct = sc.outer_ejection_total > 0 ?
                                    100.0f * sc.outer_ejection_bins[b] / sc.outer_ejection_total : 0.0f;

                                // Bar charts (10 chars max each)
                                int inner_bar = inner_max > 0 ? (sc.inner_ejection_bins[b] * 10 / inner_max) : 0;
                                int outer_bar = outer_max > 0 ? (sc.outer_ejection_bins[b] * 10 / outer_max) : 0;

                                printf("  %3.0f°    %3u (%5.1f%%) ", angle, sc.inner_ejection_bins[b], inner_pct);
                                for (int i = 0; i < inner_bar; i++) printf("█");
                                for (int i = inner_bar; i < 10; i++) printf(" ");
                                printf("    %3u (%5.1f%%) ", sc.outer_ejection_bins[b], outer_pct);
                                for (int i = 0; i < outer_bar; i++) printf("█");
                                printf("\n");
                            }

                            // Analyze clustering in each zone
                            printf("\n");
                            printf("Analysis:\n");

                            // Find dominant sectors in each zone (m=3 means ~120° spacing)
                            int inner_peaks[3] = {-1, -1, -1};
                            int outer_peaks[3] = {-1, -1, -1};
                            unsigned int inner_peak_vals[3] = {0, 0, 0};
                            unsigned int outer_peak_vals[3] = {0, 0, 0};

                            for (int b = 0; b < 16; b++) {
                                // Insert into sorted top-3 for inner
                                if (sc.inner_ejection_bins[b] > inner_peak_vals[2]) {
                                    inner_peak_vals[2] = sc.inner_ejection_bins[b];
                                    inner_peaks[2] = b;
                                    // Bubble sort
                                    for (int i = 2; i > 0 && inner_peak_vals[i] > inner_peak_vals[i-1]; i--) {
                                        unsigned int tv = inner_peak_vals[i]; inner_peak_vals[i] = inner_peak_vals[i-1]; inner_peak_vals[i-1] = tv;
                                        int tp = inner_peaks[i]; inner_peaks[i] = inner_peaks[i-1]; inner_peaks[i-1] = tp;
                                    }
                                }
                                // Insert into sorted top-3 for outer
                                if (sc.outer_ejection_bins[b] > outer_peak_vals[2]) {
                                    outer_peak_vals[2] = sc.outer_ejection_bins[b];
                                    outer_peaks[2] = b;
                                    for (int i = 2; i > 0 && outer_peak_vals[i] > outer_peak_vals[i-1]; i--) {
                                        unsigned int tv = outer_peak_vals[i]; outer_peak_vals[i] = outer_peak_vals[i-1]; outer_peak_vals[i-1] = tv;
                                        int tp = outer_peaks[i]; outer_peaks[i] = outer_peaks[i-1]; outer_peaks[i-1] = tp;
                                    }
                                }
                            }

                            if (sc.inner_ejection_total >= 3 && inner_peaks[0] >= 0) {
                                printf("  Inner zone peaks: %d° (%u), %d° (%u), %d° (%u)\n",
                                       inner_peaks[0] * 22, inner_peak_vals[0],
                                       inner_peaks[1] * 22, inner_peak_vals[1],
                                       inner_peaks[2] * 22, inner_peak_vals[2]);
                                // Check for ~120° spacing
                                int d1 = abs(inner_peaks[1] - inner_peaks[0]);
                                int d2 = abs(inner_peaks[2] - inner_peaks[1]);
                                if (d1 >= 4 && d1 <= 6 && d2 >= 4 && d2 <= 6) {
                                    printf("    → m=3 clustering (~120° spacing) DETECTED\n");
                                }
                            }

                            if (sc.outer_ejection_total >= 3 && outer_peaks[0] >= 0) {
                                printf("  Outer zone peaks: %d° (%u), %d° (%u), %d° (%u)\n",
                                       outer_peaks[0] * 22, outer_peak_vals[0],
                                       outer_peaks[1] * 22, outer_peak_vals[1],
                                       outer_peaks[2] * 22, outer_peak_vals[2]);
                                int d1 = abs(outer_peaks[1] - outer_peaks[0]);
                                int d2 = abs(outer_peaks[2] - outer_peaks[1]);
                                if (d1 >= 4 && d1 <= 6 && d2 >= 4 && d2 <= 6) {
                                    printf("    → m=3 clustering (~120° spacing) DETECTED\n");
                                }
                            }

                            printf("\n");
                            printf("Interpretation:\n");
                            printf("  If INNER shows m=3 clustering but OUTER doesn't:\n");
                            printf("    → Beat frequency model CONFIRMED (clustering = ω_orb - ω_pump)\n");
                            printf("  If BOTH zones show same clustering pattern:\n");
                            printf("    → Clustering is arm-geometry driven, not beat frequency\n");
                            printf("\n");
                        }

                        // === HIGH-STRESS FIELD SPATIAL DISTRIBUTION ===
                        // Track pump_residual > 0.7 across the full disk by (r, θ)
                        // This reveals the spatial structure of pump instability across all radii
                        unsigned int total_high_stress = 0;
                        for (int r = 0; r < STRESS_RADIAL_BINS; r++) {
                            total_high_stress += sc.stress_radial_totals[r];
                        }

                        if (total_high_stress > 10) {  // Need sufficient data
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  HIGH-STRESS FIELD SPATIAL DISTRIBUTION (pump_residual > 0.7) ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");

                            // Radial bin boundaries (at ISCO = 6)
                            const char* radial_labels[STRESS_RADIAL_BINS] = {
                                "ISCO-2×ISCO  (6-12)",
                                "2×-4×ISCO   (12-24)",
                                "4×-8×ISCO   (24-48)",
                                "8×ISCO+     (48+)  "
                            };

                            // Print header with radial context
                            printf("Radial structure of high-stress particles:\n");
                            printf("  (ω_pump ≈ 0.125/step, crossover at r≈185 where ω_orb = ω_pump)\n\n");

                            // Print azimuthal distribution for each radial bin
                            for (int r = 0; r < STRESS_RADIAL_BINS; r++) {
                                if (sc.stress_radial_totals[r] < 3) continue;  // Skip sparse bins

                                printf("═══ %s: %u high-stress particles ═══\n",
                                       radial_labels[r], sc.stress_radial_totals[r]);

                                // Find max in this radial bin for scaling
                                unsigned int rmax = 0;
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    if (sc.stress_field[r][a] > rmax) rmax = sc.stress_field[r][a];
                                }

                                // Print angular distribution with bar chart
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    float angle = a * 22.5f;
                                    float pct = 100.0f * sc.stress_field[r][a] / sc.stress_radial_totals[r];
                                    int bar = rmax > 0 ? (sc.stress_field[r][a] * 15 / rmax) : 0;

                                    printf("  %5.1f° %4u (%5.1f%%) ", angle, sc.stress_field[r][a], pct);
                                    for (int i = 0; i < bar; i++) printf("█");
                                    printf("\n");
                                }

                                // Detect m-mode for this radial bin
                                // Find top 3 peaks
                                int peaks[3] = {-1, -1, -1};
                                unsigned int peak_vals[3] = {0, 0, 0};
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    if (sc.stress_field[r][a] > peak_vals[2]) {
                                        peak_vals[2] = sc.stress_field[r][a];
                                        peaks[2] = a;
                                        for (int i = 2; i > 0 && peak_vals[i] > peak_vals[i-1]; i--) {
                                            unsigned int tv = peak_vals[i]; peak_vals[i] = peak_vals[i-1]; peak_vals[i-1] = tv;
                                            int tp = peaks[i]; peaks[i] = peaks[i-1]; peaks[i-1] = tp;
                                        }
                                    }
                                }

                                if (peaks[0] >= 0 && peaks[1] >= 0 && peaks[2] >= 0) {
                                    // Sort peaks by angle for spacing calculation
                                    int sorted[3] = {peaks[0], peaks[1], peaks[2]};
                                    for (int i = 0; i < 2; i++) {
                                        for (int j = i+1; j < 3; j++) {
                                            if (sorted[j] < sorted[i]) {
                                                int t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                                            }
                                        }
                                    }

                                    // Calculate angular spacing (wrap-around aware)
                                    int d01 = sorted[1] - sorted[0];
                                    int d12 = sorted[2] - sorted[1];
                                    int d20 = (16 - sorted[2]) + sorted[0];  // wrap-around

                                    printf("  Peaks: %.0f°, %.0f°, %.0f° (spacing: %d, %d, %d sectors)\n",
                                           sorted[0] * 22.5f, sorted[1] * 22.5f, sorted[2] * 22.5f,
                                           d01, d12, d20);

                                    // Check for m=3 (~5-6 sectors = 112-135°)
                                    bool is_m3 = (d01 >= 4 && d01 <= 7) && (d12 >= 4 && d12 <= 7) && (d20 >= 4 && d20 <= 7);
                                    // Check for m=2 (~8 sectors = 180°)
                                    bool is_m2 = (d01 >= 7 && d01 <= 9) || (d12 >= 7 && d12 <= 9) || (d20 >= 7 && d20 <= 9);

                                    if (is_m3) {
                                        printf("  → m=3 mode detected (~120° spacing)\n");
                                    } else if (is_m2) {
                                        printf("  → m=2 mode detected (~180° spacing)\n");
                                    } else {
                                        printf("  → No clear m-mode (irregular spacing)\n");
                                    }
                                }
                                printf("\n");
                            }

                            // Summary: does m-mode change with radius?
                            printf("═══ M-MODE GRADIENT SUMMARY ═══\n");
                            printf("Beat frequency model predicts:\n");
                            printf("  Inner disk (ω_orb >> ω_pump): higher m-modes (m=3 or higher)\n");
                            printf("  Outer disk (ω_orb → ω_pump): lower m-modes (m=2 or m=1)\n");
                            printf("  Past crossover (ω_orb < ω_pump): azimuthally uniform\n");
                            printf("\n");
                        }

                        // === SEAM DRIFT TIME SERIES ===
                        // Output the logged m=3 phase over time to detect precession
                        if (g_seam_drift_count > 5) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  SEAM DRIFT TIME SERIES (m=3 phase vs M_eff)                   ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Tracking: Does the ~113° seam gap orientation drift as M_eff grows?\n");
                            printf("  If stationary: seam locked to arm geometry (fixed in lab frame)\n");
                            printf("  If drifting:   seam precessing on long timescale\n");
                            printf("\n");
                            printf("  Frame     M_eff    m=3 Phase   Samples\n");
                            printf("  ═════════════════════════════════════════\n");

                            // Print every Nth entry to keep output manageable
                            int stride = (g_seam_drift_count > 20) ? g_seam_drift_count / 20 : 1;
                            for (int i = 0; i < g_seam_drift_count; i += stride) {
                                printf("  %5d   %7.2f   %6.1f°     %d\n",
                                       g_seam_drift_log[i].frame,
                                       g_seam_drift_log[i].M_eff,
                                       g_seam_drift_log[i].phase_deg,
                                       g_seam_drift_log[i].sample_count);
                            }

                            // Compute phase drift statistics
                            float phase_sum = 0.0f, phase_sq_sum = 0.0f;
                            float phase_first = g_seam_drift_log[0].phase_deg;
                            float phase_last = g_seam_drift_log[g_seam_drift_count - 1].phase_deg;
                            for (int i = 0; i < g_seam_drift_count; i++) {
                                phase_sum += g_seam_drift_log[i].phase_deg;
                                phase_sq_sum += g_seam_drift_log[i].phase_deg * g_seam_drift_log[i].phase_deg;
                            }
                            float phase_mean = phase_sum / g_seam_drift_count;
                            float phase_var = phase_sq_sum / g_seam_drift_count - phase_mean * phase_mean;
                            float phase_std = sqrtf(fmaxf(phase_var, 0.0f));
                            float phase_drift = phase_last - phase_first;

                            // === LIMIT CYCLE VS FIXED POINT ANALYSIS ===
                            // Split data into thirds and compare oscillation amplitude
                            // If amplitude decreases: approaching fixed point (damped)
                            // If amplitude constant: limit cycle (persistent oscillation)
                            int third = g_seam_drift_count / 3;
                            float early_sum = 0.0f, early_sq = 0.0f, early_mean;
                            float mid_sum = 0.0f, mid_sq = 0.0f, mid_mean;
                            float late_sum = 0.0f, late_sq = 0.0f, late_mean;

                            for (int i = 0; i < third; i++) {
                                early_sum += g_seam_drift_log[i].phase_deg;
                            }
                            early_mean = early_sum / third;
                            for (int i = 0; i < third; i++) {
                                float d = g_seam_drift_log[i].phase_deg - early_mean;
                                early_sq += d * d;
                            }
                            float early_std = sqrtf(early_sq / third);

                            for (int i = third; i < 2*third; i++) {
                                mid_sum += g_seam_drift_log[i].phase_deg;
                            }
                            mid_mean = mid_sum / third;
                            for (int i = third; i < 2*third; i++) {
                                float d = g_seam_drift_log[i].phase_deg - mid_mean;
                                mid_sq += d * d;
                            }
                            float mid_std = sqrtf(mid_sq / third);

                            for (int i = 2*third; i < g_seam_drift_count; i++) {
                                late_sum += g_seam_drift_log[i].phase_deg;
                            }
                            int late_count = g_seam_drift_count - 2*third;
                            late_mean = late_sum / late_count;
                            for (int i = 2*third; i < g_seam_drift_count; i++) {
                                float d = g_seam_drift_log[i].phase_deg - late_mean;
                                late_sq += d * d;
                            }
                            float late_std = sqrtf(late_sq / late_count);

                            printf("\n");
                            printf("Summary:\n");
                            printf("  Phase mean: %.1f° ± %.1f° (std dev)\n", phase_mean, phase_std);
                            printf("  Net drift:  %.1f° (first→last)\n", phase_drift);
                            printf("\n");
                            printf("Oscillation Amplitude by Phase (limit cycle vs fixed point test):\n");
                            printf("  Early (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[third/2].M_eff, early_mean, early_std);
                            printf("  Mid   (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[third + third/2].M_eff, mid_mean, mid_std);
                            printf("  Late  (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[2*third + late_count/2].M_eff, late_mean, late_std);
                            printf("\n");

                            // Determine behavior
                            float amp_ratio = (early_std > 0.1f) ? late_std / early_std : 1.0f;
                            if (phase_std < 10.0f && fabsf(phase_drift) < 15.0f) {
                                printf("  → STATIONARY: Seam orientation is locked to arm geometry\n");
                            } else if (fabsf(phase_drift) > 30.0f) {
                                printf("  → PRECESSING: Seam drifting %.1f° over run\n", phase_drift);
                                if (amp_ratio < 0.5f) {
                                    printf("  → DAMPED: Oscillation amplitude decreasing (%.1f× early→late)\n", amp_ratio);
                                    printf("     Approaching FIXED POINT - seam will phase-lock\n");
                                } else if (amp_ratio > 0.8f && amp_ratio < 1.2f) {
                                    printf("  → LIMIT CYCLE: Oscillation amplitude stable (%.1f× early→late)\n", amp_ratio);
                                    printf("     Persistent oscillation - pump still driving seam\n");
                                } else if (amp_ratio > 1.5f) {
                                    printf("  → UNSTABLE: Oscillation amplitude growing (%.1f× early→late)\n", amp_ratio);
                                    printf("     System may be diverging\n");
                                }
                            } else {
                                printf("  → UNCERTAIN: Moderate scatter, needs longer run\n");
                            }
                            printf("\n");
                        }

                        // === RESIDENCE TIME DIAGNOSTIC (Test A) ===
                        extern bool g_test_residence_time;
                        if (g_test_residence_time && g_enable_arms && sc.active_count > 0) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  RESIDENCE TIME DIAGNOSTIC (Test A: Weak Barrier Test)        ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Accumulated residence time over %d frames:\n", frame);
                            printf("\n");

                            float total_time = sc.arm_residence_time + sc.gap_residence_time;
                            float arm_fraction = (total_time > 0) ? sc.arm_residence_time / total_time : 0.0f;
                            float gap_fraction = (total_time > 0) ? sc.gap_residence_time / total_time : 0.0f;

                            printf("  Arm Regions:  %.0f particle-frames (%.1f%% of total)\n",
                                   sc.arm_residence_time, arm_fraction * 100.0f);
                            printf("  Gap Regions:  %.0f particle-frames (%.1f%% of total)\n",
                                   sc.gap_residence_time, gap_fraction * 100.0f);
                            printf("\n");
                            printf("  Current distribution:\n");
                            printf("    Particles in arms: %u (%.1f%%)\n",
                                   sc.arm_particle_count,
                                   100.0f * sc.arm_particle_count / (float)sc.active_count);
                            printf("    Particles in gaps: %u (%.1f%%)\n",
                                   sc.gap_particle_count,
                                   100.0f * sc.gap_particle_count / (float)sc.active_count);
                            printf("\n");

                            float arm_width = ARM_WIDTH_DEG / 360.0f;  // Fraction of circle
                            float expected_arm_fraction = arm_width * NUM_ARMS;
                            float residence_enhancement = arm_fraction / expected_arm_fraction;

                            printf("  Expected arm fraction (geometric): %.1f%%\n", expected_arm_fraction * 100.0f);
                            printf("  Observed arm fraction (residence): %.1f%%\n", arm_fraction * 100.0f);
                            printf("  Residence enhancement factor: %.2fx\n", residence_enhancement);
                            printf("\n");

                            if (g_use_arm_topology) {
                                printf("Interpretation (DISCRETE ARMS):\n");
                                printf("  If residence enhancement > 1.5× → Weak barriers trap particles\n");
                                printf("  If residence enhancement ≈ 1.0× → No trapping effect\n");
                                printf("\n");
                                if (residence_enhancement > 1.5f) {
                                    printf("  ✓ BARRIER CONFIRMED: Particles accumulate in arms\n");
                                } else if (residence_enhancement > 1.2f) {
                                    printf("  ~ WEAK BARRIER: Modest accumulation in arms\n");
                                } else {
                                    printf("  ✗ NO BARRIER: Particles drift through arms freely\n");
                                }
                            } else {
                                printf("Interpretation (SMOOTH ARMS):\n");
                                printf("  Expected: residence ≈ geometric (no preferential trapping)\n");
                                printf("  If enhancement > 1.2× → Density gradient causes accumulation\n");
                            }
                            printf("\n");
                        }
                        printf("\n");

                        // Clean exit
                        if (!g_headless) {
                            glfwSetWindowShouldClose(window, 1);
                        }
                        running = false;
                    }
                }
            }

            fps_acc = 0; fps_frames = 0;
        }

        // === RENDERING ===
#ifdef VULKAN_INTEROP
        // Vulkan rendering
        {
            // NOTE: No cudaDeviceSynchronize() - CUDA and Vulkan work on same GPU
            // The shared buffer is written by CUDA kernel, read by Vulkan vertex shader
            // GPU naturally serializes operations on same memory

            // Update camera from Vulkan context
            float camX = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * sinf(vkCtx.cameraYaw);
            float camY = vkCtx.cameraRadius * sinf(vkCtx.cameraPitch);
            float camZ = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * cosf(vkCtx.cameraYaw);

            // Build view-projection matrix
            Vec3 eye = {camX, camY, camZ};
            Vec3 center = {0, 0, 0};
            Mat4 view = Mat4::lookAt(eye, center, {0, 1, 0});

            int fb_w, fb_h;
            glfwGetFramebufferSize(vkCtx.window, &fb_w, &fb_h);
            float aspect = (float)fb_w / (float)fb_h;
            Mat4 proj = Mat4::perspective(PI / 4.0f, aspect, 0.1f, 2000.0f);

            // Vulkan uses inverted Y compared to OpenGL
            proj.m[5] *= -1.0f;

            Mat4 vp = Mat4::mul(proj, view);

            // Update uniform buffer
            GlobalUBO ubo;
            memcpy(ubo.viewProj, vp.m, sizeof(float) * 16);
            ubo.cameraPos[0] = eye.x;
            ubo.cameraPos[1] = eye.y;
            ubo.cameraPos[2] = eye.z;
            ubo.time = sim_time;
            ubo.avgScale = pump_bridge.avg_scale;
            ubo.avgResidual = pump_bridge.avg_residual;
            ubo.heartbeat = pump_bridge.heartbeat;
            ubo.pump_phase = sim_time * 0.125f * 2.0f * 3.14159f;  // ω_pump = 0.125

            // Copy to mapped uniform buffer
            memcpy(vkCtx.uniformBuffersMapped[vkCtx.currentFrame], &ubo, sizeof(ubo));

            // Update attractor pipeline (density rendering mode)
            if (g_attractor.enabled) {
                // Update particle count for compute dispatch
                g_attractor.particleCount = N_current;
                // Update time for phase evolution (in phase-primary mode)
                g_attractor.time = sim_time;

                // Update camera (viewProj already computed)
                float aspect = (float)vkCtx.swapchainExtent.width / (float)vkCtx.swapchainExtent.height;
                vk::updateAttractorCamera(g_attractor, ubo.viewProj, 1.0f, aspect,
                                          vkCtx.swapchainExtent.width, vkCtx.swapchainExtent.height);

                // Update attractor state for pure mode (use bridge state)
                // w oscillates based on heartbeat, phase rotates with time
                float w = fmaxf(0.0f, fminf(0.5f, pump_bridge.heartbeat * 0.25f + 0.25f));
                float phase = fmodf(sim_time * 0.5f, 6.2831853f);  // Slow rotation
                vk::updateAttractorState(g_attractor, w, phase, pump_bridge.avg_residual);
            }

            // Draw frame (skip in headless mode for pure physics benchmark)
            if (!g_headless) {
                vk::drawFrame(vkCtx);
            }

            // === KERNEL TIMING OUTPUT (async - no stall) ===
            // Record end event, but don't sync - we'll print NEXT time if ready
            static bool timing_pending = false;
            if (do_timing) {
                cudaEventRecord(t_render);
                timing_pending = true;
            }
            // Check if previous timing is ready (non-blocking)
            if (timing_pending) {
                cudaError_t status = cudaEventQuery(t_render);
                if (status == cudaSuccess) {
                    cudaEventElapsedTime(&ms_siphon, t_start, t_siphon);
                    cudaEventElapsedTime(&ms_physics, t_siphon, t_physics);
                    cudaEventElapsedTime(&ms_render, t_physics, t_render);
                    printf("[profile] siphon=%.2fms physics=%.2fms render=%.2fms total=%.2fms\n",
                           ms_siphon, ms_physics, ms_render, ms_siphon + ms_physics + ms_render);
                    timing_pending = false;
                }
            }

            glfwPollEvents();
        }
#else
        // === OPENGL RENDERING (only in interactive mode) ===
        if (!g_headless) {
            // Framebuffer
            int fb_w, fb_h;
            glfwGetFramebufferSize(window, &fb_w, &fb_h);
            glViewport(0, 0, fb_w, fb_h);
            float fb_aspect = (float)fb_w / (float)fb_h;

            // Camera
            float camX = g_cam.dist * cosf(g_cam.elevation) * sinf(g_cam.azimuth);
            float camY = g_cam.dist * sinf(g_cam.elevation);
            float camZ = g_cam.dist * cosf(g_cam.elevation) * cosf(g_cam.azimuth);
            Vec3 eye = {camX, camY, camZ};
            Vec3 center = {0,0,0};
            Vec3 fwd = (center - eye).norm();
            Vec3 right = fwd.cross({0,1,0}).norm();
            Vec3 camup = right.cross(fwd);

            Mat4 view = Mat4::lookAt(eye, center, {0,1,0});
            Mat4 proj = Mat4::perspective(PI/4.0f, fb_aspect, 0.1f, 1000.0f);  // Larger far plane
            Mat4 vp = Mat4::mul(proj, view);

            float fovTan = tanf(PI/8.0f);

            // Render BH background
            glDepthMask(GL_FALSE);
            glUseProgram(bhProgram);
            glUniform3f(u_camPos_bh, eye.x, eye.y, eye.z);
            glUniform3f(u_camFwd_bh, fwd.x, fwd.y, fwd.z);
            glUniform3f(u_camRight_bh, right.x, right.y, right.z);
            glUniform3f(u_camUp_bh, camup.x, camup.y, camup.z);
            glUniform1f(u_time_bh, sim_time);
            glUniform1f(u_aspect_bh, fb_aspect);
            glUniform1f(u_fov_bh, fovTan);

            // === BRIDGE UNIFORMS: Feed pump state to raymarcher ===
            glUniform1f(u_avgScale_bh, pump_bridge.avg_scale);
            glUniform1f(u_avgResidual_bh, pump_bridge.avg_residual);
            glUniform1f(u_pumpWork_bh, pump_bridge.total_work);
            glUniform1f(u_heartbeat_bh, pump_bridge.heartbeat);

            // === HOPFION SHELLS: Upload EM confinement layer structure ===
            glUniform1fv(u_shellRadii_bh, h_sample_metrics->num_shells, h_sample_metrics->shell_radii);
            glUniform1fv(u_shellN_bh, h_sample_metrics->num_shells, h_sample_metrics->shell_n);
            glUniform1i(u_numShells_bh, h_sample_metrics->num_shells);

            glBindVertexArray(fsVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glDepthMask(GL_TRUE);

            // Render disk
            glClear(GL_DEPTH_BUFFER_BIT);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glDepthMask(GL_FALSE);
            glUseProgram(diskProgram);
            glUniformMatrix4fv(u_viewProj, 1, GL_FALSE, vp.m);
            glUniform3f(u_camPos_d, eye.x, eye.y, eye.z);

            // === HOPFION SHELLS: Upload same shell structure to particle shader ===
            glUniform1fv(u_shellRadii_d, h_sample_metrics->num_shells, h_sample_metrics->shell_radii);
            glUniform1fv(u_shellN_d, h_sample_metrics->num_shells, h_sample_metrics->shell_n);
            glUniform1i(u_numShells_d, h_sample_metrics->num_shells);

            glBindVertexArray(quadVAO);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, N_current);
            glDepthMask(GL_TRUE);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }  // End of !g_headless rendering
#endif

        frame++;
        fps_acc += dt_wall;
        fps_frames++;
    }

    // === CLEANUP: Graceful shutdown for 3.5M particles ===
    printf("\n[shutdown] Beginning cleanup...\n");

    // 1. Synchronize all CUDA operations before cleanup
    cudaDeviceSynchronize();
    printf("[shutdown] CUDA synchronized\n");

#ifdef VULKAN_INTEROP
    // 2. Wait for Vulkan to finish all GPU work BEFORE destroying shared buffers
    // This prevents "buffer in use by command buffer" errors
    vkDeviceWaitIdle(vkCtx.device);
    printf("[shutdown] Vulkan device idle\n");

    // Vulkan cleanup - now safe to destroy shared resources
    destroySharedBuffer(vkCtx.device, &sharedParticleBuffer);
    // Clear handles to prevent double-free in vk::cleanup
    vkCtx.particleBuffer = VK_NULL_HANDLE;
    vkCtx.particleBufferMemory = VK_NULL_HANDLE;
    printf("[shutdown] Shared buffer destroyed\n");

    // Clean up hybrid LOD resources
    if (hybridLODEnabled) {
        // Clean up double-buffered indirect draw resources
        if (doubleBufferEnabled) {
            for (int i = 0; i < 2; i++) {
                destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
            }
            // Clear handles to prevent double-free
            vkCtx.compactedParticleBuffer = VK_NULL_HANDLE;
            vkCtx.compactedParticleBufferMemory = VK_NULL_HANDLE;
            vkCtx.indirectDrawBuffer = VK_NULL_HANDLE;
            vkCtx.indirectDrawBufferMemory = VK_NULL_HANDLE;
            printf("[shutdown] Double-buffered indirect draw resources destroyed\n");
        }

        destroySharedDensityGrid(vkCtx.device, &densityGrid);
        printf("[shutdown] Density grid destroyed\n");
        if (d_nearCount) {
            cudaFree(d_nearCount);
            printf("[shutdown] Near count buffer freed\n");
        }
    }

    // Cleanup attractor pipeline
    if (g_attractor.enabled) {
        vk::destroyAttractorPipeline(vkCtx, g_attractor);
    }

    vk::cleanup(vkCtx);
    printf("[shutdown] Vulkan cleaned up\n");

    glfwDestroyWindow(vkCtx.window);
    glfwTerminate();
    printf("[shutdown] Window closed, GLFW terminated\n");
#else
    // 2. Unregister graphics resources (only if rendering was enabled)
    if (!g_headless) {
        cudaGraphicsUnregisterResource(posRes);
        cudaGraphicsUnregisterResource(colorRes);
        cudaGraphicsUnregisterResource(sizeRes);
        cudaGraphicsUnregisterResource(velocityRes);
        cudaGraphicsUnregisterResource(elongationRes);
        printf("[shutdown] Graphics resources unregistered\n");
    }

    // Cleanup Vulkan shared memory (file-based IPC mode)
    cleanupVulkanSharedMemory();

    // Delete OpenGL resources (only if rendering was enabled)
    if (!g_headless) {
        glDeleteProgram(bhProgram);
        glDeleteProgram(diskProgram);
        printf("[shutdown] Shaders deleted\n");

        glfwDestroyWindow(window);
        glfwTerminate();
        printf("[shutdown] Window closed, GLFW terminated\n");
    }
#endif

    // 3. Destroy streams before freeing resources allocated in them
    cudaStreamDestroy(sample_stream);
    cudaStreamDestroy(stats_stream);
    cudaEventDestroy(stats_ready);
    cudaFree(d_stress_async);
    printf("[shutdown] CUDA streams destroyed\n");

    // 4. Free large allocations (GPUDisk is ~280MB)
    cudaFree(d_disk);
    printf("[shutdown] Main particle data freed\n");

    // 4b. Free topology ring buffer
    topology_recorder_cleanup();
    printf("[shutdown] Topology ring buffer freed\n");

    // 4c. Free mip-tree hierarchy
    mip_tree_cleanup();
    printf("[shutdown] Mip-tree hierarchy freed\n");

    // 5. Free smaller allocations
    cudaFree(d_stress);
    cudaFree(d_sample_indices);
    cudaFree(d_sample_metrics[0]);
    cudaFree(d_sample_metrics[1]);
    cudaFreeHost(h_sample_metrics);
    printf("[shutdown] Auxiliary data freed\n");

    // 6. Free octree allocations
    if (octreeEnabled) {
        cudaFree(d_morton_keys);
        cudaFree(d_xor_corners);
        cudaFree(d_particle_ids);
        cudaFree(d_octree_nodes);
        cudaFree(d_node_count);
        cudaFree(d_leaf_counts);
        cudaFree(d_leaf_counts_culled);
        cudaFree(d_leaf_offsets);
        cudaFree(d_leaf_node_indices);
        cudaFree(d_leaf_node_count);
        printf("[shutdown] Octree data freed\n");
    }

    printf("[shutdown] Cleanup complete!\n");

    return 0;
}
