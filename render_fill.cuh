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
// LOD Helper Function
// ============================================================================

// Smoothstep helper for LOD blending
__device__ inline float lod_smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ============================================================================
// Kernel Implementations
// ============================================================================
// These kernels depend on disk.cuh (compute_disk_r, compute_temp,
// particle_active, particle_ejected) and sun_trace.cuh (d_shell_radii).
// Both must be #included before this header.

#ifdef VULKAN_INTEROP

// ============================================================================
// Simple Particle Buffer Fill (Legacy)
// ============================================================================
// Packs GPUDisk arrays into ParticleVertex format for Vulkan rendering.
// Writes directly to a Vulkan-visible buffer via CUDA interop.

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

// ============================================================================
// Phase-Primary Fill Kernel — GPUDisk → VulkanSunTrace
// ============================================================================
// Converts position-primary GPUDisk to phase-primary VulkanSunTrace for
// rendering via harmonic_phase.comp shader.

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

__global__ void fillVulkanParticleBufferLOD(
    ParticleVertex* output,      // Shared particle buffer (Vulkan-visible)
    float* densityGrid,          // 128³ density grid (4 floats per voxel: scale_sum, temp_sum, count, coherence)
    const GPUDisk* disk,
    int N,
    float camX, float camY, float camZ,  // Camera position
    float nearThreshold,         // Distance below which = full points (default: 150)
    float farThreshold,          // Distance above which = volume only (default: 600)
    float volumeScale,           // World-space extent of density grid (default: 300)
    unsigned int* nearCount,     // Atomic counter for near particles
    bool shellLensing            // Step 6: enable shell lensing distortion
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

    // === SHELL LENSING (gravitational-lens-like distortion) ===
    // Toggle via --shell-lensing flag. Each shell between the particle
    // and camera adds a small fixed radial displacement to the particle's
    // apparent position — like looking through concentric glass shells.
    // The displacement is proportional to shell radius (outer shells bend
    // more because light crosses them at a shallower angle).
    if (shellLensing) {
        float r_cam_cyl = sqrtf(camX * camX + camZ * camZ);
        float r_part = r_cyl;

        // Count shells between particle and camera, accumulate displacement.
        // Each shell crossing adds a small outward (or inward) nudge scaled
        // by the shell's radius relative to the outermost shell.
        float deflection = 0.0f;
        const float lens_strength = 0.3f;  // total max displacement in world units

        for (int s = 0; s < 8; s++) {
            float r_s = d_shell_radii[s];
            bool cam_outside = (r_cam_cyl > r_s);
            bool part_inside = (r_part < r_s);
            if (cam_outside && part_inside) {
                deflection += lens_strength * (r_s / 174.0f);  // stronger for outer shells
            } else if (!cam_outside && !part_inside) {
                deflection -= lens_strength * (r_s / 174.0f);
            }
        }

        // Apply radial displacement in the XZ plane.
        if (fabsf(deflection) > 0.001f) {
            float r_xz = sqrtf(px * px + pz * pz);
            if (r_xz > 0.1f) {
                px += deflection * (px / r_xz);
                pz += deflection * (pz / r_xz);
            }
        }
    }

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
// Compacts only visible particles (r < HYBRID_R) into a contiguous buffer,
// allowing Vulkan to skip vertex processing for FAR particles entirely.

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

// Update the indirect draw command with the final count
__global__ void updateIndirectDrawCommand(
    CUDADrawIndirectCommand* drawCommand,
    unsigned int* writeIndex
) {
    drawCommand->vertexCount = 1;  // 1 vertex per instance (point)
    drawCommand->instanceCount = *writeIndex;  // Number of visible particles
    drawCommand->firstVertex = 0;
    drawCommand->firstInstance = 0;
}

#endif  // VULKAN_INTEROP

// End of render_fill.cuh
