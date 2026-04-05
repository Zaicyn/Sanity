// sun_trace.cuh — Phase-Primary Particle State
// =============================================
//
// Core idea: Store PHASE STATE, not position. Position is derived.
//
// Math.md mapping:
//   - theta: base phase θ(t) = ωt (orbital position on shell)
//   - omega: angular frequency (includes bias from pump state)
//   - phase12: N12 envelope phase (12↔16 dimensional breathing)
//   - shell_idx: discrete eigenspectrum index [0-7]
//   - h1, h3: harmonic ladder coefficients (1:3 ratio)
//   - drift: radial migration accumulator
//   - coherence: phase lock strength (Kuramoto order parameter)
//   - w_component: 4D phase accumulator (w=1 → black hole lock)
//
// AI consensus (GPT/Gemini/DeepSeek):
//   - Don't store derived values — compute per-step
//   - Phase is primary — theta, omega, phase offsets are the real state
//   - Shells are discrete — 8 shells, indexed, not continuous
//   - Particles are tracers — they visualize the field, they don't define it
//
// Memory: 40 bytes/particle vs 68 bytes for position-primary GPUDisk
// Savings: 41% VRAM reduction at 10M particles (280 MB)

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifndef PI
#define PI 3.14159265358979f
#endif
#ifndef TWO_PI
#define TWO_PI 6.28318530717959f
#endif

// LUT forward declarations
__device__ float cuda_lut_sin(float x);
__device__ float cuda_lut_cos(float x);
__device__ float cuda_lut_cos3(float x);
__device__ float cuda_fast_atan2(float y, float x);

// ============================================================================
// Shell Radii — The 8-Shell Eigenspectrum
// ============================================================================
// These are the equilibrium radii for each shell, derived from the
// golden ratio cascade: r_n = r_0 * φ^(n/2) for shells 0-7
//
// Shell 0: Core (ISCO vicinity)
// Shell 1-2: Inner disk
// Shell 3-4: Mid disk
// Shell 5-7: Outer disk / habitable margin

__constant__ float d_shell_radii[8] = {
    6.0f,      // Shell 0: ISCO
    9.7f,      // Shell 1: 6 * φ^0.5
    15.7f,     // Shell 2: 6 * φ^1.0
    25.4f,     // Shell 3: 6 * φ^1.5
    41.1f,     // Shell 4: 6 * φ^2.0
    66.5f,     // Shell 5: 6 * φ^2.5
    107.5f,    // Shell 6: 6 * φ^3.0
    174.0f     // Shell 7: 6 * φ^3.5
};

// ============================================================================
// Flag Bits
// ============================================================================
#define SUN_FLAG_ACTIVE     0x0001  // Particle is simulated
#define SUN_FLAG_EJECTED    0x0002  // In jet/ejection phase
#define SUN_FLAG_LOCKED     0x0004  // Phase-locked (coherent)
#define SUN_FLAG_SPAWNING   0x0008  // About to spawn child
#define SUN_FLAG_CHILD      0x0010  // Recently spawned

// ============================================================================
// SunTrace — Phase-Primary Particle State (40 bytes)
// ============================================================================
struct SunTrace {
    // === PHASE STATE (Primary) — 16 bytes ===
    float theta;           // Base phase [0, 2π) — orbital position
    float omega;           // Angular frequency (rad/frame)
    float phase12;         // N12 envelope phase (12↔16 breathing)
    uint8_t shell_idx;     // Discrete shell index [0-7]
    uint8_t seam_bits;     // 2-bit seam phase (bits 0-1) + pump state (bits 2-4)
    uint16_t flags;        // SUN_FLAG_* bits

    // === HARMONIC COEFFICIENTS — 8 bytes ===
    float h1;              // Fundamental amplitude (ω term)
    float h3;              // Third harmonic amplitude (3ω term, cubic bias)

    // === EMERGENT STATE — 12 bytes ===
    float drift;           // Radial drift from shell equilibrium
    float coherence;       // Phase lock strength [0,1] (Kuramoto R)
    float w_component;     // 4D phase accumulator (0→1 = singularity)

    // === ANCHOR — 4 bytes ===
    float r_target;        // Current equilibrium radius (interpolated from shell)
};

// ============================================================================
// SunTraceBuffer — SoA Layout for GPU (coalesced access)
// ============================================================================
#ifndef MAX_SUN_TRACES
#define MAX_SUN_TRACES 20000000  // 20M capacity
#endif

struct SunTraceBuffer {
    // Phase state (16 bytes/particle)
    float* theta;
    float* omega;
    float* phase12;
    uint8_t* shell_idx;
    uint8_t* seam_bits;
    uint16_t* flags;

    // Harmonic coefficients (8 bytes/particle)
    float* h1;
    float* h3;

    // Emergent state (12 bytes/particle)
    float* drift;
    float* coherence;
    float* w_component;

    // Anchor (4 bytes/particle)
    float* r_target;
};

// ============================================================================
// Position Reconstruction — Derive XYZ from Phase
// ============================================================================
// The key insight: position is NOT stored, it's computed from phase state.
// This saves memory bandwidth and ensures geometric consistency.

__device__ __forceinline__ float3 sun_reconstruct_position(
    float theta,
    float phase12,
    float r_target,
    float drift
) {
    // Effective radius = shell equilibrium + accumulated drift
    float r = r_target + drift;

    // Orbital position from base phase
    float cos_theta = cuda_lut_cos(theta);
    float sin_theta = cuda_lut_sin(theta);

    // Vertical oscillation from N12 envelope (breathing)
    // Amplitude proportional to shell (inner shells breathe less)
    float breath_amp = 0.8f * (1.0f - r_target / 200.0f);
    float z_osc = breath_amp * cuda_lut_sin(phase12);

    return make_float3(
        r * cos_theta,   // x
        z_osc,           // y (vertical breathing)
        r * sin_theta    // z
    );
}

// Overload for SunTrace struct
__device__ __forceinline__ float3 sun_reconstruct_position(const SunTrace& s) {
    return sun_reconstruct_position(s.theta, s.phase12, s.r_target, s.drift);
}

// ============================================================================
// Velocity Reconstruction — Derive from Phase Derivatives
// ============================================================================
// Velocity = dr/dt, which depends on omega and phase12 rate

__device__ __forceinline__ float3 sun_reconstruct_velocity(
    float theta,
    float omega,
    float phase12,
    float r_target,
    float drift,
    float omega12  // N12 breathing frequency (typically omega/12)
) {
    float r = r_target + drift;
    float cos_theta = cuda_lut_cos(theta);
    float sin_theta = cuda_lut_sin(theta);

    // Tangential velocity from orbital motion: v_t = r * omega
    float vt = r * omega;

    // Breathing velocity
    float breath_amp = 0.8f * (1.0f - r_target / 200.0f);
    float vy = breath_amp * omega12 * cuda_lut_cos(phase12);

    return make_float3(
        -vt * sin_theta,  // vx = -r*omega*sin(theta)
        vy,               // vy = breathing
        vt * cos_theta    // vz = r*omega*cos(theta)
    );
}

// ============================================================================
// Mix Function — The Siphon Coupling
// ============================================================================
// mix = cos(θ)cos(3θ) — the 1:3 harmonic product that drives the pump

__device__ __forceinline__ float sun_compute_mix(float theta) {
    return cuda_lut_cos(theta) * cuda_lut_cos3(theta);
}

// Full mixer with harmonic amplitudes
__device__ __forceinline__ float sun_compute_mix_weighted(float theta, float h1, float h3) {
    return h1 * cuda_lut_cos(theta) * h3 * cuda_lut_cos3(theta);
}

// ============================================================================
// Temperature Reconstruction — Blackbody from Shell Position
// ============================================================================
// Inner shells are hotter: T ~ (ISCO/r)^0.75

__device__ __forceinline__ float sun_compute_temp(float r) {
    const float ISCO = 6.0f;
    return (r > ISCO) ? powf(ISCO / r, 0.75f) : 1.0f;
}

// ============================================================================
// Phase Evolution — The Core Update
// ============================================================================
// This replaces position integration with phase integration

__device__ __forceinline__ void sun_evolve_phase(
    SunTrace& s,
    float dt,
    float coupling_k,     // Kuramoto coupling strength
    float neighbor_phase  // Mean phase of neighbors (for synchronization)
) {
    // 1. Base phase evolution: θ += ω*dt
    s.theta += s.omega * dt;

    // 2. Kuramoto coupling: θ += K*sin(θ_mean - θ)*dt
    float phase_diff = neighbor_phase - s.theta;
    s.theta += coupling_k * cuda_lut_sin(phase_diff) * dt;

    // 3. Wrap to [0, 2π)
    while (s.theta >= TWO_PI) s.theta -= TWO_PI;
    while (s.theta < 0.0f) s.theta += TWO_PI;

    // 4. N12 envelope evolution (slower, 1/12 of base frequency)
    s.phase12 += (s.omega / 12.0f) * dt;
    while (s.phase12 >= TWO_PI) s.phase12 -= TWO_PI;

    // 5. Drift evolution (radial migration based on mix)
    float mix = sun_compute_mix(s.theta);
    s.drift += mix * 0.001f * dt;  // Slow radial drift

    // 6. Coherence update (exponential smoothing toward lock)
    float target_coherence = fabsf(cuda_lut_cos(phase_diff));
    s.coherence = 0.99f * s.coherence + 0.01f * target_coherence;

    // 7. w-component accumulation (toward 4D singularity)
    if (s.coherence > 0.9f) {
        s.w_component += 0.0001f * dt;
        if (s.w_component > 1.0f) s.w_component = 1.0f;
    }
}

// ============================================================================
// Shell Transition — Discrete Jumps Between Eigenspectrum Levels
// ============================================================================

__device__ __forceinline__ void sun_transition_shell(SunTrace& s, int delta) {
    int new_shell = (int)s.shell_idx + delta;
    if (new_shell < 0) new_shell = 0;
    if (new_shell > 7) new_shell = 7;

    s.shell_idx = (uint8_t)new_shell;
    s.r_target = d_shell_radii[new_shell];
    s.drift = 0.0f;  // Reset drift on shell transition
}

// ============================================================================
// Conversion: GPUDisk ↔ SunTrace
// ============================================================================
// For gradual migration from position-primary to phase-primary

__device__ __forceinline__ SunTrace sun_from_gpudisk(
    float px, float py, float pz,
    float vx, float vy, float vz,
    float pump_scale,
    float pump_residual,
    uint8_t pump_seam
) {
    SunTrace s;

    // Compute cylindrical coordinates from position
    float r = sqrtf(px * px + pz * pz);
    float phi = cuda_fast_atan2(pz, px);
    if (phi < 0) phi += TWO_PI;

    // Assign to nearest shell
    s.shell_idx = 0;
    for (int i = 7; i >= 0; i--) {
        if (r >= d_shell_radii[i] * 0.8f) {
            s.shell_idx = i;
            break;
        }
    }
    s.r_target = d_shell_radii[s.shell_idx];
    s.drift = r - s.r_target;

    // Phase from angular position
    s.theta = phi;

    // Omega from tangential velocity
    float v_tangent = sqrtf(vx*vx + vz*vz);
    s.omega = (r > 0.1f) ? v_tangent / r : 0.1f;

    // Store actual Y position for accurate reconstruction
    // (Previously tried asin encoding but it lost precision for |py| > 0.8)
    s.phase12 = py;

    // Harmonic amplitudes (default to balanced)
    s.h1 = 1.0f;
    s.h3 = pump_scale;  // Use pump_scale as h3 amplitude

    // Emergent state
    s.coherence = 1.0f - pump_residual;
    s.w_component = 0.0f;

    // Flags
    s.seam_bits = pump_seam;
    s.flags = SUN_FLAG_ACTIVE;

    return s;
}

// Convert back to position for rendering
__device__ __forceinline__ void sun_to_position(
    const SunTrace& s,
    float& px, float& py, float& pz
) {
    float3 pos = sun_reconstruct_position(s);
    px = pos.x;
    py = pos.y;
    pz = pos.z;
}
