// active_region.cuh — ActiveRegion struct and mask kernel
// =========================================================
//
// Part of Tree Architecture Step 2. See docs/active_flag_audit.md
// and /home/zaiken/sanity/math.md Part V for full context.
//
// WHAT IS AN ActiveRegion?
// ------------------------
// A localized volume of the simulation in which particles need the
// full siphonDiskKernel physics (pump state machine, ejection,
// Kuramoto coupling, grid scatter/gather). Particles OUTSIDE every
// active region are "passive": they only need the cheap Keplerian
// advection in passive_advection.cuh.
//
// In Step 2 there is exactly one ActiveRegion: an all-encompassing
// bootstrap that covers the entire simulation volume. Every alive
// particle tests as "inside" it, so `in_active_region[i]` is 1 for
// every alive particle, and the passive kernel early-returns on
// every particle. Step 3 will replace this with dynamic region
// lifecycle management (spawn, stability integral, N12-gated
// promotion to persistent sub-anchors).
//
// NAMING NOTE
// -----------
// NOT to be confused with the existing `d_xor_corners` buffer in
// blackhole_v20.cu:4598, which is a Morton-XOR neighbor lookup key
// (`ix ^ iy ^ iz` in assignMortonKeys at 1134-1166) used for the
// octree rebuild. That XOR is integer bit manipulation of cell
// indices. The tree-architecture "XOR corners" discussed in the
// design conversations become these `ActiveRegion` objects in code
// to avoid the collision.

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"

// Upper bound on concurrently active regions. Step 2 uses only 1.
// Step 3 will tune this; 64 is generous for the envisioned workload
// (~8 shells × a handful of regions per shell).
#ifndef MAX_ACTIVE_REGIONS
#define MAX_ACTIVE_REGIONS 64
#endif

// Per-particle pump_residual threshold for active/passive classification.
// Particles with |pump_residual| > CORNER_THRESHOLD need full siphon
// physics. Particles below threshold are "settled" and use the cheap
// Keplerian advection kernel. Step 4 runtime candidate — tune empirically
// from headless sweep data before wiring to a CLI flag or keybinding.
#ifndef CORNER_THRESHOLD
#define CORNER_THRESHOLD 0.15f
#endif

// State machine for an active region's lifecycle. Step 2 only uses
// UNUSED and ACTIVE; the intermediate states are reserved for Step 3.
#define REGION_STATE_UNUSED      0
#define REGION_STATE_SEEDED      1   // Step 3: candidate, accumulating stability
#define REGION_STATE_ACTIVE      2   // Full physics inside this region
#define REGION_STATE_RETIRING    3   // Step 3: dissolving back to passive

// An active region is defined by three "gate positions" — reserved
// slots for the Step 3 triple-XOR promotion geometry. In Step 2 the
// bootstrap region uses gate_positions[0] and gate_positions[1] as
// an axis-aligned bounding box (min/max corners) for a simple
// containment test. Step 3 will use all three for a proper
// triple-gate predicate.
struct ActiveRegion {
    float3 gate_positions[3];    // Step 2: gate[0]=box_min, gate[1]=box_max, gate[2]=unused
    int    parent_shell;         // -1 = root (all shells); 0..7 = specific shell for Step 3
    int    birth_frame;
    float  stability_integral;   // Reserved for Step 3 promotion gate
    int    state;                // See REGION_STATE_* above
};

// ============================================================================
// computeInActiveRegionMask
// ============================================================================
// For each alive particle, test containment against every ACTIVE region.
// Writes 1 to in_active_region[i] if the particle is inside any region,
// 0 otherwise. Dead particles get 0 (they're owned by neither kernel).
//
// Step 2 performance note: this kernel runs every frame. With 1 active
// region and the simple bounding-box test, it's a few ALU ops per
// particle plus a few memory loads. At 1M particles this is negligible
// compared to siphonDiskKernel. Step 3 may need an acceleration
// structure if MAX_ACTIVE_REGIONS × N becomes expensive.
__global__ void computeInActiveRegionMask(
    GPUDisk* disk,                                     // non-const: velocity fix-up writes vel_x/vel_z
    const ActiveRegion* __restrict__ regions,           // unused in Step 3a threshold path; kept for future region lifecycle
    int num_regions,                                    // unused in Step 3a threshold path
    uint8_t* __restrict__ in_active_region,
    int N,
    float corner_threshold)                             // Step 4: runtime threshold (replaces CORNER_THRESHOLD macro)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Dead particles don't belong to any region.
    if (!particle_active(disk, i)) {
        in_active_region[i] = 0;
        return;
    }

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    // ================================================================
    // Step 3b: per-particle threshold classification.
    // Uses previous frame's pump_residual (one-frame lag, acceptable).
    // ================================================================
    float residual = disk->pump_residual[i];
    float r_cyl_sq = px * px + pz * pz;
    float r_cyl = sqrtf(r_cyl_sq);

    // Step 5: inject residual proportional to deviation from nearest
    // resonance shell. Particles off-shell accumulate instability;
    // on-shell particles stay passive. This breaks the "all passive
    // forever" feedback loop and makes CORNER_THRESHOLD meaningful.
    // The injection is computed HERE (in the mask kernel, same frame
    // as the threshold comparison) to avoid the one-frame-lag issue
    // that would occur if it were in the passive kernel.
    // Shell-deviation injection — REMOVED. Shells should emerge from
    // the field, not be used as hardcoded classification boundaries.
    // Active/passive split based purely on pump residual now.
    float effective_residual = fabsf(residual);

    // Force active (siphon owns) if ANY of these hold:
    //   - effective_residual > corner_threshold: off-shell or dynamically active
    //   - r_cyl near ISCO or at boundary: violent physics / recycle territory
    //   - ejected: in Aizawa jet, siphon owns entirely
    //   - pump_history < 0.7: newborn (cf. SPAWN_COHERENCE_THRESH in disk.cuh:114)
    bool force_active = (effective_residual > corner_threshold)
                     || (r_cyl < PASSIVE_R_MIN)
                     || (r_cyl > PASSIVE_R_MAX)
                     || particle_ejected(disk, i)
                     || (disk->pump_history[i] < 0.7f)
                     || (disk->pump_state[i] <= 1);  // IDLE or PRIMED — pump hasn't cycled yet

    uint8_t new_val = force_active ? 1 : 0;
    uint8_t old_val = in_active_region[i];  // previous frame's classification

    // Hysteresis: once active, stay active until effective residual drops
    // well below threshold. Prevents frame-to-frame oscillation.
    if (old_val != 0 && !force_active) {
        if (effective_residual > corner_threshold * 0.5f)
            new_val = 1;
    }

    // Velocity fix-up on passive→active promotion.
    // Snap to Keplerian tangential using 3D local orbital frame so siphon
    // gets a consistent state regardless of orbital plane orientation.
    if (new_val == 1 && old_val == 0) {
        float vx_cur = disk->vel_x[i];
        float vy_cur = disk->vel_y[i];
        float vz_cur = disk->vel_z[i];

        // Compute local frame from current L
        float Lx = py * vz_cur - pz * vy_cur;
        float Ly = pz * vx_cur - px * vz_cur;
        float Lz = px * vy_cur - py * vx_cur;
        float inv_L = rsqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
        float lx = Lx * inv_L, ly = Ly * inv_L, lz = Lz * inv_L;

        float r3d_sq = px*px + py*py + pz*pz;
        float inv_r3d = rsqrtf(r3d_sq + 1e-8f);
        float rx = px * inv_r3d, ry = py * inv_r3d, rz = pz * inv_r3d;

        // t_hat = L_hat × r_hat
        float tx = ly * rz - lz * ry;
        float ty = lz * rx - lx * rz;
        float tz = lx * ry - ly * rx;
        float inv_t = rsqrtf(tx*tx + ty*ty + tz*tz + 1e-8f);
        tx *= inv_t; ty *= inv_t; tz *= inv_t;

        float v_kep = sqrtf(BH_MASS * inv_r3d);
        disk->vel_x[i] = tx * v_kep;
        disk->vel_y[i] = ty * v_kep;
        disk->vel_z[i] = tz * v_kep;
    }

    in_active_region[i] = new_val;
}
