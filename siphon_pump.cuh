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
// Initial Ejection Kick — Hopf Fiber Geometry
// ============================================================================
// Jets follow the Hopf fiber direction at the particle's position rather than
// firing straight up/down along Y. This produces curved, spiraling jets that
// express the hopfion field topology instead of vertical column artifacts.
//
// The fiber direction is a blend of two orthogonal components:
//
//   Toroidal (t): tangent to the orbital circle in the XZ plane
//     t = (-pz/r_cyl, 0, px/r_cyl)
//     → pure orbital rotation, no vertical component
//
//   Poloidal (p): up the meridional circle in the r-Y plane
//     p = (-px/r3d, py/r3d, -pz/r3d)  [inward-and-up along field line]
//     → escape direction along the field
//
//   fiber = cos(pitch)*t + sin(pitch)*p * jet_sign
//
// Pitch angle = PI/2 * (1 - collimation):
//   collimation=1 (near center):  pitch→0    → pure toroidal → tight spiral
//   collimation=0.5 (mid):        pitch=PI/4  → 45° helix
//   collimation=0 (outer edge):   pitch=PI/2  → pure poloidal → field escape
//
// This inverts the old logic: near the center jets spiral tightly (bound by
// the hopfion field), far out they open and escape along field lines.
//
// JET_SPEED replaces the old 15x vertical hack. Tune this constant to match
// the desired jet velocity scale.

#ifndef JET_SPEED
#define JET_SPEED 4.0f
#endif

__device__ __forceinline__ void apply_ejection_kick(
    float px, float py, float pz,
    float& vx, float& vy, float& vz,
    float r_cyl,
    float residual, float history,
    float Ly, float orb_phi,
    float dt)
{
    // Forward declarations for LUT
    extern __device__ float cuda_lut_sin(float x);
    extern __device__ float cuda_lut_cos(float x);

    // Jet sign: which hemisphere the particle escapes toward
    float jet_sign = (py >= 0.0f) ? 1.0f : -1.0f;
    if (fabsf(py) < 0.5f) {
        // Near disk plane — use angular momentum direction
        jet_sign = (Ly >= 0.0f) ? 1.0f : -1.0f;
    }

    // 3D radius (safe floor to avoid div by zero)
    float r3d = sqrtf(px*px + py*py + pz*pz);
    float r3d_safe  = fmaxf(r3d,  0.01f);
    float r_cyl_safe = fmaxf(r_cyl, 0.01f);

    // Toroidal unit vector: tangent to orbit in XZ plane
    // t = (-pz, 0, px) / r_cyl
    float inv_rcyl = 1.0f / r_cyl_safe;
    float tx = -pz * inv_rcyl;
    float ty =  0.0f;
    float tz =  px * inv_rcyl;

    // Poloidal unit vector: inward-and-up along the meridional field line
    // p = (-px, py, -pz) / r3d  (points toward axis and up)
    float inv_r3d = 1.0f / r3d_safe;
    float polx = -px * inv_r3d;
    float poly =  py * inv_r3d;
    float polz = -pz * inv_r3d;

    // Collimation: 1.0 near center, 0.0 at outer edge
    float collimation = fmaxf(0.0f, fminf(1.0f,
        1.0f - (r_cyl - SCHW_R) / (ISCO_R * 2.0f)));

    // Pitch angle: near center→tight spiral, far out→field escape
    float pitch = (PI * 0.5f) * (1.0f - collimation);
    float cos_pitch = cuda_lut_cos(pitch);
    float sin_pitch = cuda_lut_sin(pitch);

    // Hopf fiber direction = blend of toroidal and poloidal
    float kx = cos_pitch * tx + sin_pitch * polx * jet_sign;
    float ky = cos_pitch * ty + sin_pitch * poly * jet_sign;
    float kz = cos_pitch * tz + sin_pitch * polz * jet_sign;

    // Normalize (rsqrtf pattern)
    float inv_k = rsqrtf(kx*kx + ky*ky + kz*kz + 1e-8f);
    kx *= inv_k;
    ky *= inv_k;
    kz *= inv_k;

    // Muzzle velocity: residual × history boost × jet speed constant
    float muzzle = residual * (1.0f + history * 0.5f) * JET_SPEED;

    vx += muzzle * kx;
    vy += muzzle * ky;
    vz += muzzle * kz;
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
