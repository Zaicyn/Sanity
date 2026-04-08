// forces.cuh — Force Models and Energy Terms
// ============================================
//
// This header contains:
//   - Viviani field force (topology-derived anisotropic force)
//   - Angular momentum preservation
//   - Disk plane damping
//   - Angular momentum sink
//   - Ion kick (Langevin noise term)
//   - Core anchor (1/(1+r²) attractor)
//
// Math.md mapping:
//   - Ion kick: External entropy source (σR(t) term)
//   - Core anchor: The siphon's "K factor" — increases coupling near center
//   - These are the INTERACTION terms that enable emergent behavior

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"

// Forward declarations for LUT functions
__device__ float cuda_lut_sin(float x);
__device__ float cuda_lut_cos(float x);
__device__ float cuda_lut_cos3(float x);
__device__ float cuda_lut_repulsion_var(float r, float lambda);
__device__ float cuda_fast_atan2(float y, float x);

// ============================================================================
// Viviani Field Force Model
// ============================================================================
// Force derived directly from the Viviani curve geometry:
//   x(θ) = sin(θ) - ½·sin(3θ)
//   y(θ) = -cos(θ) + ½·cos(3θ)
//   z(θ) = cos(θ)·cos(3θ)
//   w(θ) = ⅓·sin(5θ)
//
// The field direction at each particle is the Viviani tangent vector
// evaluated at the particle's orbital phase θ = atan2(pz, px).
// The force arises from the difference between the particle's velocity
// and the field direction — particles are accelerated along the field.
//
// This IS gravity in this framework: G_μν = 8πG(T^ideal + ΔE/c² g_μν)
// The curve encodes the metric's topology. No -GM/r² anywhere.

__device__ __forceinline__ void apply_viviani_field_force(
    float px, float py, float pz,
    float vx, float vy, float vz,
    float r3d, float inv_r3d,
    float& ax, float& ay, float& az)
{
    float r_safe = fmaxf(r3d, SCHW_R * 0.5f);
    float inv_r = (r3d >= SCHW_R * 0.5f) ? inv_r3d : (1.0f / r_safe);

    // Orbital phase from position in the disk plane
    float theta = cuda_fast_atan2(pz, px);

    // Viviani curve evaluated at θ — the field geometry
    //   x(θ) = sin(θ) - ½·sin(3θ)
    //   y(θ) = -cos(θ) + ½·cos(3θ)
    //   z(θ) = cos(θ)·cos(3θ)
    float s1 = cuda_lut_sin(theta);
    float c1 = cuda_lut_cos(theta);
    float s3 = cuda_lut_sin3(theta);
    float c3 = cuda_lut_cos3(theta);

    // Tangent vector: d/dθ of the curve = field direction
    //   dx/dθ = cos(θ) - 3/2·cos(3θ)
    //   dy/dθ = sin(θ) - 3/2·sin(3θ)
    //   dz/dθ = -sin(θ)·cos(3θ) - 3·cos(θ)·sin(3θ)
    float fx = c1 - 1.5f * c3;
    float fy = s1 - 1.5f * s3;
    float fz = -s1 * c3 - 3.0f * c1 * s3;

    // Normalize field direction
    float inv_f = rsqrtf(fx*fx + fy*fy + fz*fz + 1e-8f);
    fx *= inv_f;
    fy *= inv_f;
    fz *= inv_f;

    // Force weight: field strength / (1 + r²/falloff)
    // At large r, weight ≈ STRENGTH × FALLOFF / r² ≈ BH_MASS / r²
    float weight = FIELD_FORCE_STRENGTH / (1.0f + r_safe * r_safe / FIELD_FORCE_FALLOFF);

    // The force has two components:
    //   1. Tangential: accelerate along the field direction (drives rotation)
    //   2. Radial: pull toward center proportional to weight (provides centripetal)

    // Radial unit vector
    float rx = px * inv_r;
    float ry = py * inv_r;
    float rz = pz * inv_r;

    // Tangential component: field direction projected perpendicular to radial
    float f_dot_r = fx * rx + fy * ry + fz * rz;
    float tx = fx - f_dot_r * rx;
    float ty = fy - f_dot_r * ry;
    float tz = fz - f_dot_r * rz;

    // Centripetal (radial inward) — the "gravity" component
    ax += -rx * weight;
    ay += -ry * weight;
    az += -rz * weight;

    // Tangential — the rotation-driving component (scaled by TANGENT_SCALE)
    ax += tx * weight * TANGENT_SCALE;
    ay += ty * weight * TANGENT_SCALE;
    az += tz * weight * TANGENT_SCALE;
}

// ============================================================================
// Angular Momentum Calculation
// ============================================================================
// L = r × v (specific angular momentum)
// Returns Lx, Ly, Lz and inv_L_mag (using rsqrtf pattern)

__device__ __forceinline__ void compute_angular_momentum(
    float px, float py, float pz,
    float vx, float vy, float vz,
    float& Lx, float& Ly, float& Lz, float& inv_L_mag)
{
    Lx = py * vz - pz * vy;
    Ly = pz * vx - px * vz;
    Lz = px * vy - py * vx;
    // rsqrtf pattern: get 1/|L| directly, avoids sqrtf + division
    inv_L_mag = rsqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
}

// Disk alignment factor: Lz / L_mag = Lz * inv_L_mag
// +1 = prograde (orbiting with disk)
// -1 = retrograde
__device__ __forceinline__ float compute_disk_alignment(float Lz, float inv_L_mag) {
    return Lz * inv_L_mag;
}

// ============================================================================
// Local Orbital Frame — 3D generalization of the XZ-plane tangent/radial
// ============================================================================
// Computes an orthonormal frame from each particle's actual angular momentum:
//   r_hat = r / |r|             — 3D radial (away from center)
//   L_hat = L / |L|             — orbital plane normal
//   t_hat = L_hat × r_hat       — in-plane tangential (prograde direction)
//
// This replaces hardcoded (-pz, 0, px)/r_cyl tangent vectors. Particles can
// orbit in ANY plane — the frame adapts to each particle's actual trajectory.
// Cost: ~12 FMA + 1 rsqrtf on top of the already-computed L and r3d.

__device__ __forceinline__ void compute_local_frame(
    float px, float py, float pz,
    float Lx, float Ly, float Lz, float inv_L_mag,
    float inv_r3d,
    float& rx, float& ry, float& rz,    // radial unit vector (outward)
    float& tx, float& ty, float& tz,    // tangential unit vector (prograde)
    float& lx, float& ly, float& lz)    // orbital normal unit vector
{
    // Radial: r_hat = r / |r|
    rx = px * inv_r3d;
    ry = py * inv_r3d;
    rz = pz * inv_r3d;

    // Orbital normal: L_hat = L / |L|
    lx = Lx * inv_L_mag;
    ly = Ly * inv_L_mag;
    lz = Lz * inv_L_mag;

    // Tangential: t_hat = L_hat × r_hat (right-hand rule: prograde direction)
    tx = ly * rz - lz * ry;
    ty = lz * rx - lx * rz;
    tz = lx * ry - ly * rx;

    // Normalize t_hat (should already be ~unit length since L_hat ⊥ r_hat,
    // but numerical drift in L can make them non-orthogonal)
    float inv_t = rsqrtf(tx*tx + ty*ty + tz*tz + 1e-8f);
    tx *= inv_t;
    ty *= inv_t;
    tz *= inv_t;
}

// ============================================================================
// Orbital-Plane Damping (was: Disk Plane Damping)
// ============================================================================
// Damps velocity perpendicular to the particle's orbital plane (defined by L).
// This is the 3D generalization of the old vy-only damping that forced
// particles toward Y=0. Now particles settle into their OWN orbital plane,
// not a hardcoded XZ plane.

__device__ __forceinline__ void apply_orbital_damping(
    float vx_in, float vy_in, float vz_in,
    float lx, float ly, float lz,  // orbital normal (L_hat from local frame)
    float r3d,
    float& vx, float& vy, float& vz)
{
    if (r3d > SCHW_R * 2.0f && r3d < ION_KICK_OUTER_R) {
        // Only damp if L is well-defined (particle is actually orbiting).
        // When |L| is small, L_hat is garbage and damping kills all velocity.
        float L_sq = lx*lx + ly*ly + lz*lz;  // L_hat is already normalized, but check input L
        float v_sq = vx*vx + vy*vy + vz*vz;
        // Skip if velocity is too low to define an orbital plane
        if (v_sq > 0.001f) {
            float v_normal = vx * lx + vy * ly + vz * lz;
            // Radius-scaled damping matched to vertical epicyclic frequency:
            //   ω_z(r) = √(BH_MASS / r³)
            //   γ(r) = overdamp × ω_z(r)
            // With overdamp ≥ 1.0, vertical oscillations are critically damped
            // at all radii simultaneously → cone flattens into disk.
            // overdamp = 2.0 gives slightly supercritical damping (no ringing).
            float r3 = r3d * r3d * r3d;
            float omega_z = sqrtf(BH_MASS / fmaxf(r3, 1.0f));
            float damping = 2.0f * omega_z;  // critically damped
            damping = fminf(damping, 0.50f);  // stability cap
            vx -= damping * v_normal * lx;
            vy -= damping * v_normal * ly;
            vz -= damping * v_normal * lz;
        }
    }
}

// Legacy wrapper for callers that haven't been updated yet
__device__ __forceinline__ void apply_disk_damping(
    float py, float r_cyl,
    float& vy)
{
    if (fabsf(py) < DISK_THICKNESS * 3.0f &&
        r_cyl > SCHW_R * 2.0f &&
        r_cyl < ION_KICK_OUTER_R) {  // All 8 shells (max 174.0) within 200.0
        float disk_damping = 0.10f * cuda_lut_repulsion_var(fabsf(py), DISK_THICKNESS);
        vy *= (1.0f - disk_damping);
    }
}

// Angular momentum sink REMOVED — the Viviani field self-regulates
// through its topology. The sink drained 1-5% of tangential velocity
// per frame (scaling as r²) with no energy source to replenish it,
// causing all orbits to decay to zero velocity within ~60 frames.

// ============================================================================
// Ion Kick: Langevin Noise Term (σR(t))
// ============================================================================
// Math.md: This is the energy SOURCE that balances damping.
// Modulated by period-4 heartbeat: "inject, inject, inject, relax" pattern.
//
// Equilibrium: ⟨v²⟩ = σ²/(2γ) — automatic balance, no tuning needed.

__device__ __forceinline__ void apply_ion_kick(
    float px, float pz, float r_xz,
    float global_heartbeat,
    float& vx, float& vz,
    // 3D mode: pass full position and vy for 3D energy injection
    float py = 0.0f, float* vy_ptr = nullptr, float r3d_override = 0.0f)
{
    // Period-4 modulation: on on on off (1110 pattern)
    float hb_mod = (global_heartbeat > -0.5f) ? 1.0f : 0.0f;

    // Use 3D radius if provided, otherwise fall back to XZ
    float r = (r3d_override > 0.0f) ? r3d_override : r_xz;

    if (r > ION_KICK_INNER_R && r < ION_KICK_OUTER_R) {
        float ion_coupling = (r - ION_KICK_INNER_R) / (ION_KICK_OUTER_R - ION_KICK_INNER_R);
        float sigma_eff = LANGEVIN_SIGMA * hb_mod;

        // 3D radial inward direction
        float inv_r = 1.0f / (r + 1e-8f);
        vx += sigma_eff * ion_coupling * (-px * inv_r);
        vz += sigma_eff * ion_coupling * (-pz * inv_r);
        if (vy_ptr) {
            *vy_ptr += sigma_eff * ion_coupling * (-py * inv_r);
        }
    }
}

// ============================================================================
// Boundary Recycling
// ============================================================================
// Extreme escapees get respawned at ION_KICK_RESPAWN_R.
// Preserves jet_phase (coherence memory from jet excursion).

__device__ __forceinline__ bool apply_boundary_recycle(
    float& px, float& py, float& pz,
    float& vx, float& vy, float& vz,
    float r_xz, float global_heartbeat,
    int& state, int& coherent)
{
    // Use 3D radius for boundary check
    float r3d = sqrtf(px*px + py*py + pz*pz);
    if (r3d > ION_KICK_OUTER_R) {
        // Respawn along current 3D radial direction (preserves orbital plane)
        float inv_r = 1.0f / (r3d + 1e-8f);
        float scale = ION_KICK_RESPAWN_R * inv_r;
        px *= scale;
        py *= scale;
        pz *= scale;

        // Gentle inward velocity along radial direction
        float hb_mod = (global_heartbeat > -0.5f) ? 1.0f : 0.0f;
        float sigma = LANGEVIN_SIGMA * hb_mod;
        float v_inward = -0.1f * (0.5f + sigma);
        float rx = px * inv_r * scale;  // post-scale radial
        float ry = py * inv_r * scale;
        float rz = pz * inv_r * scale;
        float inv_rr = rsqrtf(rx*rx + ry*ry + rz*rz + 1e-8f);
        vx = v_inward * rx * inv_rr;
        vy = v_inward * ry * inv_rr;
        vz = v_inward * rz * inv_rr;

        state = 0;  // IDLE
        coherent = 0;
        return true;
    }
    return false;
}

// ============================================================================
// Core Anchor: 1/(1+r²) Attractor
// ============================================================================
// Math.md: The siphon's "K factor" — increases coupling near center.
// Prevents shell detachment — keeps shells periodically touching core.
// Creates 3-zone field: core pull + mid drift + outer return.

__device__ __forceinline__ void apply_core_anchor(
    float px, float pz, float r_xz,
    float& vx, float& vz,
    // 3D mode: pass full position and vy
    float py = 0.0f, float* vy_ptr = nullptr, float r3d_override = 0.0f)
{
    float r = (r3d_override > 0.0f) ? r3d_override : r_xz;
    if (r > CORE_ANCHOR_INNER_R) {
        float r_scaled = r / CORE_PULL_SCALE;
        float core_pull = CORE_PULL_STRENGTH / (1.0f + r_scaled * r_scaled);
        float onset = fminf((r - CORE_ANCHOR_INNER_R) / 20.0f, 1.0f);
        // 3D radial inward
        float inv_r = 1.0f / (r + 1e-8f);
        vx += core_pull * onset * (-px * inv_r);
        vz += core_pull * onset * (-pz * inv_r);
        if (vy_ptr) {
            *vy_ptr += core_pull * onset * (-py * inv_r);
        }
    }
}

// ============================================================================
// Tidal Forces
// ============================================================================
// 1. Keplerian shear: |dω/dr| = 1.5 × √(M/r⁵) — disk-plane phenomenon
// 2. Radial tidal: ∝ 1/r³ — stretches along radial direction
// Both contribute to destabilization when multiplied by pump history.

__device__ __forceinline__ void compute_tidal_forces(
    float r_cyl, float r3d,
    float scale, float L_disk_align,
    float& shear_out, float& tidal_radial_out)
{
    float shear = 0.0f;
    float tidal_radial = 0.0f;

    // Keplerian shear — use 3D radius, no disk-alignment penalty.
    // Shear exists for any orbiting particle regardless of orbital plane.
    if (r3d > SCHW_R * 1.5f) {
        shear = 1.5f * sqrtf(BH_MASS / (r3d * r3d * r3d * r3d * r3d));
        shear *= sqrtf(scale);
        shear *= fmaxf(0.0f, fabsf(L_disk_align));  // Orbital coherence, not direction
    }

    // Radial tidal (always present)
    if (r3d > SCHW_R * 1.2f) {
        tidal_radial = BH_MASS / (r3d * r3d * r3d);
        tidal_radial *= sqrtf(scale);
    }

    shear_out = shear;
    tidal_radial_out = tidal_radial;
}

// ============================================================================
// Keplerian Substrate Torque (Competing Interaction Model)
// ============================================================================
// Based on: "Nonmonotonic Magnetic Friction from Collective Rotor Dynamics"
// (Gu, Lüders, Bechinger 2024)
//
// KEY INSIGHT: Maximum dissipation occurs when competing interactions FRUSTRATE
// the system, preventing it from settling into either:
//   - FM (ferromagnetic) = all phases aligned = rigid vortex
//   - AFM (antiferromagnetic) = alternating phases = locked shear
//
// The "competing regime" (CP) where neither wins produces:
//   - Hysteresis cycles
//   - Energy dissipation
//   - Turbulent dynamics
//   - Spiral structure
//
// Physics analogy:
//   - Kuramoto coupling (neighbor sync) = intralayer FM interaction
//   - Keplerian substrate = rotating external field (like magnetic substrate)
//   - When K_kuramoto ≈ K_substrate → frustration → interesting dynamics
//
// The substrate "wants" particles at radius r to have angular velocity:
//   Ω_kepler(r) = √(GM/r³)
//
// But Kuramoto coupling "wants" neighbors to sync phases.
// These are incompatible → system never settles → dissipation.
//
// Arguments:
//   px, pz: position (disk plane)
//   vx, vz: velocity (modified in place)
//   orb_phase: current orbital phase θ
//   substrate_k: substrate coupling strength (compete with kuramoto_k)

__device__ __forceinline__ void apply_keplerian_substrate_torque(
    float px, float pz,
    float& vx, float& vz,
    float orb_phase,
    float substrate_k)
{
    float r_cyl = sqrtf(px * px + pz * pz);

    // Skip particles too close to center (numerical stability)
    if (r_cyl < ISCO_R * 0.5f) return;

    float inv_r = 1.0f / r_cyl;

    // Keplerian angular velocity: Ω_k = √(GM/r³)
    float r_eff = fmaxf(r_cyl, ISCO_R);
    float omega_kepler = sqrtf(BH_MASS / (r_eff * r_eff * r_eff));

    // Current angular velocity: Ω = v_θ / r
    float v_theta = (-pz * vx + px * vz) * inv_r;
    float omega_current = v_theta * inv_r;

    // Phase mismatch between current rotation and Keplerian "substrate"
    // This is the key: we use sin() to create oscillating torque, not linear relaxation
    // The substrate field rotates at Ω_kepler, particle is at orb_phase
    // Mismatch creates torque that can overshoot, creating hysteresis
    float omega_mismatch = omega_kepler - omega_current;

    // Sinusoidal torque: T = S × sin(Δω × τ)
    // Using sin() creates the hysteresis loops seen in the paper
    // τ is effective coupling timescale (absorbed into substrate_k)
    // For small mismatches, sin(x) ≈ x, so this reduces to linear
    // For large mismatches, sin() saturates, preventing runaway
    float torque = substrate_k * cuda_lut_sin(omega_mismatch * 10.0f);

    // Tangential unit vector: θ_hat = (-z/r, x/r)
    float theta_x = -pz * inv_r;
    float theta_z = px * inv_r;

    // Apply torque as velocity change in tangential direction
    // This competes with Kuramoto coupling for phase control
    vx += torque * theta_x;
    vz += torque * theta_z;
}

// ============================================================================
// Simplified Linear Substrate (for comparison/tuning)
// ============================================================================
// Linear version without sin() - useful for understanding the dynamics
// before adding the nonlinear hysteresis effects.

__device__ __forceinline__ void apply_keplerian_substrate_linear(
    float px, float pz,
    float& vx, float& vz,
    float substrate_k)
{
    float r_cyl = sqrtf(px * px + pz * pz);
    if (r_cyl < ISCO_R * 0.5f) return;

    float inv_r = 1.0f / r_cyl;

    // Target Keplerian velocity
    float r_eff = fmaxf(r_cyl, ISCO_R);
    float v_kepler = sqrtf(BH_MASS / r_eff);

    // Current tangential velocity
    float v_theta = (-pz * vx + px * vz) * inv_r;

    // Linear competing force (not relaxation - can overshoot)
    // Clamped to prevent instability while maintaining competition
    float delta_v = v_kepler - v_theta;
    float torque = substrate_k * fminf(fmaxf(delta_v, -1.0f), 1.0f);

    float theta_x = -pz * inv_r;
    float theta_z = px * inv_r;

    vx += torque * theta_x;
    vz += torque * theta_z;
}
