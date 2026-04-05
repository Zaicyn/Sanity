// topology.cuh — Spiral Arm Structure and Azimuthal Topology
// ===========================================================
//
// This header handles the spiral arm structure:
//   - MODE 1: Discrete arm boundaries (topological traps)
//   - MODE 2: Smooth density waves (sinusoidal modulation)
//
// Math.md context:
//   - Arms create azimuthal variation in the coupling K
//   - Inside arm: boosted pump_scale → stronger coupling
//   - Outside arm: baseline pump_scale → weaker coupling
//   - The trap force pushes particles back toward arms
//
// This is a GEOMETRIC constraint (exists regardless of emergence),
// but it shapes WHERE emergent behavior happens.

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"

// Forward declarations for LUT functions
__device__ float cuda_lut_sin(float x);

// Note: Arm constants (d_NUM_ARMS, d_ARM_WIDTH_DEG, etc.) are defined in blackhole_v20.cu
// and accessed directly from constant memory. No extern needed in CUDA.

// ============================================================================
// Arm Sector Identification
// ============================================================================
// Divide 2π into num_arms sectors, each with defined width.
//
// Returns:
//   arm_id: which arm sector (0 to num_arms-1)
//   arm_center: center angle of that sector
//   dist_from_center: angular distance from arm center

__device__ __forceinline__ void identify_arm_sector(
    float orb_phi,
    int num_arms,
    int& arm_id,
    float& arm_center,
    float& dist_from_center)
{
    float sector_size = TWO_PI / (float)num_arms;
    arm_id = (int)floorf(orb_phi / sector_size);
    arm_center = (arm_id + 0.5f) * sector_size;
    dist_from_center = fabsf(orb_phi - arm_center);
}

// ============================================================================
// Mode 1: Discrete Arm Boundaries (Topological Traps)
// ============================================================================
// Particles inside arms get boosted pump_scale.
// Particles outside arms feel a restoring force toward nearest arm.
//
// Arguments:
//   orb_phi: orbital angle
//   r_cyl: cylindrical radius
//   scale: pump scale (modified by arm_boost)
//   tidal_stress: local tidal stress
//   vx, vz: velocity components (modified by trap force)
//   dt: timestep
//
// Returns: arm_boost factor (1.0 if outside, 1.5 if inside)

__device__ __forceinline__ float apply_discrete_arms(
    float orb_phi, float r_cyl,
    float scale, float tidal_stress,
    float px, float pz,
    float& vx, float& vz,
    float dt)
{
    float arm_width_rad = d_ARM_WIDTH_DEG * PI / 180.0f;

    int arm_id;
    float arm_center, dist_from_center;
    identify_arm_sector(orb_phi, d_NUM_ARMS, arm_id, arm_center, dist_from_center);

    // Normalize distance to [0, 1] where 0=center, 1=boundary
    float normalized_dist = dist_from_center / (arm_width_rad * 0.5f);

    if (normalized_dist < 1.0f) {
        // Inside arm: boosted pump_scale
        // Test C: Use override if set (matched amplitude test)
        return (d_ARM_BOOST_OVERRIDE > 0.0f) ? d_ARM_BOOST_OVERRIDE : 1.5f;
    } else {
        // Outside arm: apply restoring force toward nearest arm
        float trap_force = d_ARM_TRAP_STRENGTH * scale * tidal_stress;
        float direction = (dist_from_center > arm_center) ? -1.0f : 1.0f;

        // Apply azimuthal force (push particle back toward arm)
        float dv_tangential = direction * trap_force * dt;

        // Convert to Cartesian velocity change
        float sin_phi = pz / r_cyl;
        float cos_phi = px / r_cyl;
        vx += dv_tangential * sin_phi;   // Tangential direction
        vz -= dv_tangential * cos_phi;

        return 1.0f;  // Baseline outside arm
    }
}

// ============================================================================
// Mode 2: Smooth Density Waves
// ============================================================================
// Sinusoidal modulation, no discrete boundaries.
// arm_boost = 1.0 + 0.25 * sin(NUM_ARMS * phi)

__device__ __forceinline__ float apply_smooth_arms(float orb_phi)
{
    float wave_phase = orb_phi * (float)d_NUM_ARMS;
    return 1.0f + 0.25f * cuda_lut_sin(wave_phase);
}

// ============================================================================
// Combined Arm Topology Application
// ============================================================================
// Selects between discrete and smooth modes based on USE_ARM_TOPOLOGY.
//
// Arguments:
//   orb_phi: orbital angle
//   r_cyl: cylindrical radius
//   scale: pump scale (will be multiplied by arm_boost)
//   tidal_stress: local tidal stress
//   px, pz: position components
//   vx, vz: velocity components (may be modified)
//   dt: timestep
//
// Returns: arm_boost factor to apply to scale

__device__ __forceinline__ float apply_arm_topology(
    float orb_phi, float r_cyl,
    float scale, float tidal_stress,
    float px, float pz,
    float& vx, float& vz,
    float dt)
{
    if (d_NUM_ARMS <= 0) {
        return 1.0f;  // Arms disabled (--no-arms flag)
    }

    if (d_USE_ARM_TOPOLOGY) {
        return apply_discrete_arms(orb_phi, r_cyl, scale, tidal_stress, px, pz, vx, vz, dt);
    } else {
        return apply_smooth_arms(orb_phi);
    }
}
