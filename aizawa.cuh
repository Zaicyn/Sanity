// aizawa.cuh — Phase-Breathing Aizawa Attractor Dynamics
// =======================================================
//
// The Aizawa attractor governs ejected particle behavior (jets).
// VIV §4.7: Phase-modulated strange attractor evolution.
//
// Math.md mapping:
//   - Aizawa is similar to r⃗(θ) = r⃗_ω + ½r⃗_3ω (fundamental + 3rd harmonic)
//   - The attractor shape breathes with the particle's own phase history
//   - Bifurcation at high stress (>0.98) → period-doubling
//
// The Aizawa attractor has the harmonic structure we want:
//   - Primary oscillation (ω)
//   - Cubic nonlinearity creates 3ω component
//   - Phase modulation couples to particle's jet_phase
//
// This is where ejected particles "orbit" before re-entraining.

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"

// Forward declarations for LUT functions
__device__ float cuda_lut_sin(float x);
__device__ float cuda_lut_cos(float x);
__device__ float cuda_lut_sin3(float x);
__device__ float cuda_fast_atan2(float y, float x);

// ============================================================================
// Aizawa Parameters (Phase-Modulated)
// ============================================================================
// The attractor parameters vary with the particle's jet_phase:
//   a: primary oscillation amplitude (0.95 + 0.05*sin(phase))
//   b: damping coefficient (0.70 + 0.02*cos(2*phase))
//   c: z-axis drive (3.5 + bifurcation)
//   d: rotation coupling (0.7)
//   e: nonlinear z coupling (0.25)
//   f: cubic x³ coupling (0.10 + 0.02*sin3(phase))

__device__ __forceinline__ void compute_aizawa_params(
    float jet_phase, float phase_stress,
    float& a, float& b, float& c, float& d, float& e, float& f)
{
    // Phase-modulated parameters
    a = 0.95f + 0.05f * cuda_lut_sin(jet_phase);
    b = 0.70f + 0.02f * cuda_lut_cos(2.0f * jet_phase);
    f = 0.10f + 0.02f * cuda_lut_sin3(jet_phase);
    d = 0.7f;
    e = 0.25f;

    // Bifurcation: at high stress (>0.98), attractor splits
    // Period-doubling via cos(2θ) modulation of c parameter
    float bifurcation = 0.0f;
    if (phase_stress > 0.98f) {
        bifurcation = (phase_stress - 0.98f) * 50.0f;
        bifurcation *= cuda_lut_cos(2.0f * jet_phase);
    }
    c = 3.5f + bifurcation * 0.5f;
}

// ============================================================================
// Aizawa ODE
// ============================================================================
// dx/dt = (z - b)x - dy
// dy/dt = dx + (z - b)y
// dz/dt = c + az - z³/3 - (x² + y²)(1 + ez) + fx³
//
// Note: Aizawa coordinates are different from simulation coordinates:
//   Aizawa xy = simulation xz (disk plane)
//   Aizawa z  = simulation y (jet axis)

__device__ __forceinline__ void aizawa_derivatives(
    float x, float y, float z,
    float a, float b, float c, float d, float e, float f,
    float& dx, float& dy, float& dz)
{
    dx = (z - b) * x - d * y;
    dy = d * x + (z - b) * y;
    dz = c + a * z - (z * z * z) / 3.0f
       - (x * x + y * y) * (1.0f + e * z)
       + f * z * x * x * x;
}

// ============================================================================
// Apply Aizawa Dynamics
// ============================================================================
// Called for ejected particles in the jet region.
// Blends Aizawa derivatives with existing velocity (30% Aizawa, 70% ballistic).
//
// Arguments:
//   px, py, pz: simulation coordinates
//   vx, vy, vz: velocity (modified in place)
//   jet_phase: particle's phase memory
//   phase_stress: current interface polarization
//   jet_phase_delta: accumulated phase change (output)

__device__ __forceinline__ void apply_aizawa_dynamics(
    float px, float py, float pz,
    float& vx, float& vy, float& vz,
    float jet_phase, float phase_stress,
    float& jet_phase_delta)
{
    // Compute phase-modulated parameters
    float a, b, c, d, e, f;
    compute_aizawa_params(jet_phase, phase_stress, a, b, c, d, e, f);

    // Normalize position to Aizawa scale
    float aiz_x = px * 0.1f;
    float aiz_y = pz * 0.1f;  // Aizawa xy = simulation xz
    float aiz_z = py * 0.1f;  // Aizawa z = simulation y

    // Compute derivatives
    float dax, day, daz;
    aizawa_derivatives(aiz_x, aiz_y, aiz_z, a, b, c, d, e, f, dax, day, daz);

    // Accumulate phase shift from Aizawa orbit
    jet_phase_delta += cuda_fast_atan2(day, dax) * 0.01f;

    // Apply as velocity perturbation (30% Aizawa, 70% ballistic)
    const float aizawa_blend = 0.3f;
    vx += dax * aizawa_blend;
    vz += day * aizawa_blend;
    vy += daz * aizawa_blend;
}

// ============================================================================
// Jet Synchronization: Kuramoto Coupling
// ============================================================================
// Pull toward mean phase of nearby particles (creates helical jets).
// Uses orbital phi as proxy for local phase field.
//
// Math.md Step 8: θ̇_i = ω_i + K sin(θ_j - θ_i)
// Here K = 0.05 (gentle coupling)

__device__ __forceinline__ void apply_jet_synchronization(
    float orb_phi, float jet_phase,
    float& vx, float& vz,
    float& jet_phase_delta)
{
    // Use orbital phi as proxy for local field
    float local_sin = cuda_lut_sin(orb_phi);
    float local_cos = cuda_lut_cos(orb_phi);
    float local_phase = cuda_fast_atan2(local_sin, local_cos);
    float phase_diff = local_phase - jet_phase;

    // Wrap to [-π, π]
    if (phase_diff > 3.14159f) phase_diff -= 6.28318f;
    if (phase_diff < -3.14159f) phase_diff += 6.28318f;

    // Gentle coupling (K = 0.05)
    float coupling = 0.05f * cuda_lut_sin(phase_diff);
    jet_phase_delta += coupling;

    // Apply coupling to velocity (creates helical structure)
    vx += coupling * cuda_lut_sin(jet_phase) * 0.1f;
    vz += coupling * cuda_lut_cos(jet_phase) * 0.1f;
}

// ============================================================================
// Full Jet Evolution
// ============================================================================
// Combines Aizawa dynamics + jet synchronization for ejected particles.
//
// Returns true if particle should be re-entrained (exit jet state)

__device__ __forceinline__ bool evolve_jet_particle(
    float& px, float& py, float& pz,
    float& vx, float& vy, float& vz,
    float jet_phase, float phase_stress,
    float orb_phi,
    int& state, int& coherent,
    float& residual, float& scale,
    float& jet_phase_delta)
{
    float jet_height = fabsf(py);

    // Deep in jet - escape and recycle
    if (jet_height > 20.0f) {
        scale = 1.0f;
        coherent = 0;
        state = PUMP_IDLE;
        return true;  // Exit ejected state
    }

    // VIV STRESS_RETURN = 0.82: re-entrain if phase depolarized
    if (phase_stress < 0.82f && jet_height < 10.0f) {
        state = PUMP_PRIMED;
        coherent = 0;

        // Inject jet phase back into pump cycle (phase memory)
        float phase_injection = cuda_lut_sin(jet_phase) * 0.2f;
        residual = phase_injection;

        // Damp velocity for smooth re-entry
        vx *= 0.5f;
        vy *= 0.3f;
        vz *= 0.5f;
        return true;  // Exit ejected state
    }

    // Still in jet: evolve under phase-breathing Aizawa
    apply_aizawa_dynamics(px, py, pz, vx, vy, vz, jet_phase, phase_stress, jet_phase_delta);
    apply_jet_synchronization(orb_phi, jet_phase, vx, vz, jet_phase_delta);

    return false;  // Stay in jet
}
