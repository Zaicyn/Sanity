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
    const GPUDisk* disk,
    const ActiveRegion* __restrict__ regions,
    int num_regions,
    uint8_t* __restrict__ in_active_region,
    int N)
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

    uint8_t inside = 0;
    for (int r = 0; r < num_regions; r++) {
        if (regions[r].state != REGION_STATE_ACTIVE) continue;

        // Step 2 bounding-box test: gate[0]=min corner, gate[1]=max corner.
        float3 box_min = regions[r].gate_positions[0];
        float3 box_max = regions[r].gate_positions[1];
        bool inside_this =
            (px >= box_min.x) && (px <= box_max.x) &&
            (py >= box_min.y) && (py <= box_max.y) &&
            (pz >= box_min.z) && (pz <= box_max.z);

        if (inside_this) {
            inside = 1;
            break;
        }
    }

    in_active_region[i] = inside;
}
