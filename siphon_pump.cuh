// siphon_pump.cuh — 12↔16 Dimensional Siphon State Machine
// =========================================================
//
// The siphon pump is the thermodynamic engine of the simulation:
//   - 8-state machine cycling through 12→16→12 dimensional transitions
//   - Residual accumulation drives ejection (phase stress > 0.95)
//   - Scale regulation with soft decay (prevents hard-snap ripples)
//
// Math.md mapping:
//   - pump_scale tracks coherence level
//   - pump_residual measures overflow (short to ground)
//   - Ejection = phase catastrophe (loss of rhythm)
//
// The pump creates the CONDITIONS for emergence:
//   - High coupling (K) near ISCO → tight synchronization
//   - Low coupling far from ISCO → weak synchronization
//   - This K gradient is what makes stars form near the siphon

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"
#include "harmonic.cuh"

// ============================================================================
// Coupling Strength Calculation
// ============================================================================
// Math.md: K factor determines synchronization threshold
//
// Coupling depends on:
//   1. Proximity to ISCO (high coupling near event horizon)
//   2. Disk alignment (high coupling for orbiting particles)
//   3. Height (low coupling far from disk plane)

__device__ __forceinline__ float compute_coupling_strength(
    float r3d, float L_disk_align, float py)
{
    float proximity_factor = fmaxf(0.0f, 1.0f - r3d / (ISCO_R * 3.0f));
    float align_factor = fmaxf(0.0f, L_disk_align);
    float height_penalty = expf(-fabsf(py) / (DISK_THICKNESS * 10.0f));

    return proximity_factor * 0.5f + align_factor * 0.3f + height_penalty * 0.2f;
}

// ============================================================================
// Seam Bits Selection
// ============================================================================
// Seam bits control the pump's dimensional coupling:
//   0x00 (SEAM_CLOSED): minimal pump activity
//   0x01 (SEAM_UP_ONLY): upstroke bias (accumulation)
//   0x02 (SEAM_DOWN_ONLY): downstroke bias (release)
//   0x03 (SEAM_FULL): full pump cycle

__device__ __forceinline__ uint8_t select_seam_bits(float coupling, float r3d)
{
    if (coupling > 0.7f || r3d < ISCO_R * 1.5f) {
        return SEAM_FULL;      // Strong coupling: full pump cycle
    } else if (coupling > 0.3f) {
        return SEAM_UP_ONLY;   // Medium coupling: upstroke bias
    } else {
        return SEAM_CLOSED;    // Weak coupling: minimal activity
    }
}

// ============================================================================
// Pressure Calculation
// ============================================================================
// Pressure = φ_excess × √scale
// This drives the pump's work output

__device__ __forceinline__ float compute_pressure(float scale) {
    return PHI_EXCESS * sqrtf(scale);
}

// ============================================================================
// Siphon State Machine Step
// ============================================================================
// 8 states:
//   0: PUMP_IDLE        — waiting for coupling
//   1: PUMP_PRIMED      — ready to pump
//   2: PUMP_UPSTROKE_COHERENT — projection-subtraction (energy conserving)
//   3: PUMP_UPSTROKE_HOP — stronger filter (φ-jump)
//   4: PUMP_EXPAND      — 12→16 expansion
//   5: PUMP_DOWNSTROKE  — 16→12 compression
//   6: PUMP_VALVE_OPEN  — residual accumulation
//   7: PUMP_RECIRCULATE — return to primed
//
// Note: States 2 and 3 apply the coherence filter from harmonic.cuh
// This function handles the OTHER states and transitions.

__device__ __forceinline__ void siphon_state_step(
    int& state,
    float& scale,
    int& coherent,
    float& residual,
    float& work,
    uint8_t seam_bits,
    float pressure,
    float global_bias,
    float tidal_stress,
    float history)
{
    switch (state) {
        case PUMP_IDLE:
            if (seam_bits != 0) state = PUMP_PRIMED;
            residual = 0.0f;
            break;

        case PUMP_PRIMED:
            residual = 0.0f;
            if (seam_bits == SEAM_FULL) {
                state = (coherent < PUMP_COHERENT_MAX) ? PUMP_UPSTROKE_COHERENT : PUMP_UPSTROKE_HOP;
            }
            break;

        // States 2 and 3 (coherent upstroke and hop) are handled in physics.cu
        // because they need velocity access for the coherence filter

        case PUMP_EXPAND:
            state = PUMP_DOWNSTROKE;
            break;

        case PUMP_DOWNSTROKE:
            state = PUMP_VALVE_OPEN;
            break;

        case PUMP_VALVE_OPEN:
        {
            // Pure pump physics: bias = k = 0.75 (the Demon's operating point)
            float effective_bias = global_bias;

            if (scale > 10.0f) {
                // Logarithmic throttle: pump efficiency drops at high scales
                effective_bias *= 1.0f / (1.0f + 0.1f * log2f(scale / 10.0f));
            }
            effective_bias = fminf(effective_bias, 1.0f);

            // Base residual from pump
            residual = pressure * (1.0f - effective_bias);

            // Tidal stress injection: high-history particles feel tidal forces
            float tidal_coupling = tidal_stress * history * 0.1f;
            residual += tidal_coupling;

            work += pressure * residual;
            state = PUMP_RECIRCULATE;
        }
        break;

        case PUMP_RECIRCULATE:
            state = PUMP_PRIMED;
            break;
    }
}

// ============================================================================
// Scale Regulation
// ============================================================================
// Soft decay prevents hard-snap ripples.
// The "Demon" becomes less efficient at high scales.

__device__ __forceinline__ void regulate_scale(
    float& scale,
    int& coherent)
{
    const float SCALE_SOFT_CAP = 30.0f;
    const float SCALE_HARD_CAP = 100.0f;
    const float DECAY_RATE = 0.05f;

    if (scale > SCALE_SOFT_CAP) {
        // Apply back-pressure: decay proportional to overshoot
        float overshoot = (scale - SCALE_SOFT_CAP) / (SCALE_HARD_CAP - SCALE_SOFT_CAP);
        overshoot = fminf(overshoot, 1.0f);

        // Exponential decay with cubic falloff
        scale *= (1.0f - DECAY_RATE * overshoot * overshoot * overshoot);

        // Throttle coherent count
        if (overshoot > 0.3f && coherent > 0) {
            coherent--;
        }
    }

    // Hard cap with proportional landing
    if (scale > SCALE_HARD_CAP) {
        scale = SCALE_SOFT_CAP * 0.8f;
        coherent = 0;
    }
}

// ============================================================================
// Ejection Check
// ============================================================================
// Math.md: Phase stress > 0.95 + residual threshold = "short to ground"
//
// VIV §4.7: Phase stress measures interface polarization
//   - Balanced: stress = 0.5
//   - Maximally polarized: stress → 1.0
//
// When polarized AND close enough AND high residual → EJECT

__device__ __forceinline__ bool check_ejection(
    float phase_stress,
    float r3d, float r_cyl,
    float residual)
{
    bool close_enough = (r3d < ISCO_R * 2.5f) || (r_cyl < ISCO_R * 2.0f);
    bool phase_polarized = (phase_stress > 0.95f);  // VIV STRESS_EJECT
    return phase_polarized && close_enough && (fabsf(residual) > PUMP_EJECT_THRESHOLD * 0.5f);
}

// ============================================================================
// Initial Ejection Kick
// ============================================================================
// Particles near the rotation axis get collimated into coherent jets.
// Particles further out get more lateral kick.

__device__ __forceinline__ void apply_ejection_kick(
    float px, float py, float pz,
    float& vx, float& vy, float& vz,
    float r_cyl,
    float residual, float history,
    float Ly, float orb_phi,
    float dt)
{
    // Determine jet direction
    float jet_sign = (py >= 0.0f) ? 1.0f : -1.0f;
    if (fabsf(py) < 0.5f) {
        // Near disk plane - use angular momentum
        jet_sign = (Ly >= 0.0f) ? 1.0f : -1.0f;
    }

    // Collimation factor: tighter near center
    float collimation = fmaxf(0.1f, 1.0f - (r_cyl - SCHW_R) / (ISCO_R * 2.0f));

    // Muzzle velocity from pump history
    float muzzle_velocity = residual * (1.0f + history * 0.5f);

    // Vertical kick
    float vertical_kick = muzzle_velocity * collimation * jet_sign;
    vy += vertical_kick * 15.0f;

    // Lateral drift for non-collimated particles
    float lateral_drift = muzzle_velocity * (1.0f - collimation) * 0.3f;

    // Forward declarations for LUT - defined elsewhere
    extern __device__ float cuda_lut_sin(float x);
    extern __device__ float cuda_lut_cos(float x);

    vx += lateral_drift * cuda_lut_cos(orb_phi);
    vz += lateral_drift * cuda_lut_sin(orb_phi);
}

// ============================================================================
// Re-entrainment Check
// ============================================================================
// VIV STRESS_RETURN = 0.82: re-entrain if phase depolarized

__device__ __forceinline__ bool check_reentrain(
    float phase_stress, float jet_height)
{
    return (phase_stress < 0.82f && jet_height < 10.0f);
}

// ============================================================================
// Pump History Update
// ============================================================================
// Exponential smoothing of scale — tracks pump activity over time

__device__ __forceinline__ float update_pump_history(
    float old_history, float scale)
{
    const float HISTORY_SMOOTH = 0.02f;
    return old_history * (1.0f - HISTORY_SMOOTH) + scale * HISTORY_SMOOTH;
}
