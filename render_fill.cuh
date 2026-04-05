// render_fill.cuh — Vulkan Particle Buffer Fill Kernels
// ======================================================
//
// Kernels for filling Vulkan-visible particle buffers via CUDA interop.
// Includes LOD (Level-of-Detail) support and stream compaction.
//
// Main kernels:
//   - fillVulkanParticleBuffer: Simple full-buffer fill
//   - fillVulkanSunTraceBuffer: Phase-primary export for GPU reconstruction
//   - fillVulkanParticleBufferLOD: Distance-based LOD with density grid
//   - compactVisibleParticles: Stream compaction for hybrid rendering
//
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Forward declarations
struct GPUDisk;
struct ParticleVertex;

// ============================================================================
// Phase-Primary Export Struct (matches harmonic_phase.comp EXACTLY)
// ============================================================================
// 40 bytes, packed for Vulkan SSBO. Uses uint32 for the packed fields
// instead of separate uint8/uint16 to match GLSL std430 layout.

struct VulkanSunTrace {
    // Phase state (16 bytes)
    float theta;           // [0, 2π) orbital phase
    float omega;           // angular frequency
    float phase12;         // N12 envelope phase
    uint32_t packed_state; // shell_idx (8) + seam_bits (8) + flags (16)

    // Harmonic coefficients (8 bytes)
    float h1;              // fundamental amplitude
    float h3;              // third harmonic amplitude

    // Emergent state (12 bytes)
    float drift;           // radial drift from equilibrium
    float coherence;       // phase lock strength [0,1]
    float w_component;     // 4D accumulator

    // Anchor (4 bytes)
    float r_target;        // equilibrium radius
};

// ============================================================================
// LOD (Level of Detail) Constants
// ============================================================================

// Density grid dimensions for volumetric rendering
#define LOD_GRID_SIZE 128
#define LOD_GRID_HALF 64.0f

// Hybrid boundary: particles inside this radius render as points,
// particles outside are handled by the analytic shell shader.
// This is a PHYSICAL boundary (radius from BH), not view-dependent.
#define HYBRID_R 30.0f

// ============================================================================
// Indirect Draw Command (matches Vulkan VkDrawIndirectCommand)
// ============================================================================
// Named differently to avoid conflict with vulkan_core.h definition

struct CUDADrawIndirectCommand {
    unsigned int vertexCount;
    unsigned int instanceCount;
    unsigned int firstVertex;
    unsigned int firstInstance;
};

// ============================================================================
// Kernel Declarations (implementations in blackhole_v20.cu)
// ============================================================================

#ifdef VULKAN_INTEROP

// Simple particle buffer fill (legacy version)
__global__ void fillVulkanParticleBuffer(
    ParticleVertex* output,  // Shared buffer (Vulkan-visible)
    const GPUDisk* disk,
    int N
);

// Phase-primary export for GPU position reconstruction
__global__ void fillVulkanSunTraceBuffer(
    VulkanSunTrace* output,  // Shared buffer (Vulkan-visible)
    const GPUDisk* disk,
    int N
);

// LOD-aware particle buffer fill with density grid accumulation
__global__ void fillVulkanParticleBufferLOD(
    ParticleVertex* output,      // Shared particle buffer (Vulkan-visible)
    float* densityGrid,          // 128³ density grid (4 floats per voxel)
    const GPUDisk* disk,
    int N,
    float camX, float camY, float camZ,  // Camera position
    float nearThreshold,         // Distance below which = full points
    float farThreshold,          // Distance above which = volume only
    float volumeScale,           // World-space extent of density grid
    unsigned int* nearCount      // Atomic counter for near particles
);

// Clear density grid before LOD fill
__global__ void clearDensityGrid(float* densityGrid, int numVoxels);

// Stream compaction for hybrid rendering (r < HYBRID_R only)
__global__ void compactVisibleParticles(
    ParticleVertex* compactedOutput,       // Compacted output buffer
    CUDADrawIndirectCommand* drawCommand,  // Indirect draw command
    float* densityGrid,                    // Density grid (for API compat)
    const GPUDisk* disk,
    int N,
    float camX, float camY, float camZ,    // Camera position
    float nearThreshold, float farThreshold, float volumeScale,
    unsigned int* writeIndex,              // Atomic counter for compaction
    float* maxRadiusOut                    // Debug: track max radius written
);

// Update indirect draw command after compaction
__global__ void updateIndirectDrawCommand(
    CUDADrawIndirectCommand* drawCommand,
    unsigned int* writeIndex
);

#endif  // VULKAN_INTEROP

// ============================================================================
// LOD Helper Function
// ============================================================================

// Smoothstep helper for LOD blending
__device__ inline float lod_smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// End of render_fill.cuh
