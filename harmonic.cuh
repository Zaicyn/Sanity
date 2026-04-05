// harmonic.cuh — Quartic Harmonic Ladder Implementation
// ======================================================
//
// Math.md Part II mapping:
//
// GEOMETRIC (Boundary Conditions):
//   - heartbeat = cos(θ) * cos(3θ)     → The Z-axis mixer (Step 7)
//   - Generates ω, 2ω, 3ω, 4ω ladder   → Quartic harmonic structure
//   - Period-4 zeroing at θ = nπ/2     → "1110" pattern
//   - LCM(1,3,4) = 12                  → N12 closure (harmonic route)
//   - Mode cycling (3 modes × period-4) → N12 closure (geometric route)
//
// TWO INDEPENDENT ROUTES TO N12:
//   Route 1: LCM(1,3,4) = 12 from harmonic content (cos θ, cos 3θ, cos 4θ)
//   Route 2: LCM(3,4) = 12 from mode cycling (3 coherence directions × period-4 heartbeat)
//   Both are GEOMETRIC — no coupling required. See compute_coherence_direction().
//
// EMERGENT (from coupling):
//   - Synchronization via Kuramoto     → K > Δω threshold
//   - Coherence filter (-λm_⊥)         → Entropy removal
//   - Anisotropic damping (-γm_∥)      → Energy removal
//
// The heartbeat is the GEOMETRIC core — it exists even for a single node.
// Synchronization is EMERGENT — requires many interacting nodes.

#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"

// Forward declarations for LUT functions
__device__ float cuda_lut_sin(float x);
__device__ float cuda_lut_cos(float x);
__device__ float cuda_lut_cos3(float x);
__device__ void cuda_lut_sincos(float x, float* s, float* c);

// ============================================================================
// The Z-Axis Mixer: cos(θ) * cos(3θ)
// ============================================================================
// Math.md Step 7:
//   z(θ) = cos(θ) * cos(3θ)
//        = ½[cos(2θ) + cos(4θ)]
//
// This self-generates the complete quartic harmonic ladder:
//   ω   - Fundamental rotation
//   2ω  - Mixer difference (cos(θ-3θ) = cos(-2θ))
//   3ω  - Explicit distortion
//   4ω  - Mixer sum (cos(θ+3θ) = cos(4θ))
//
// Period-4 zeroing: at θ = π/2, 3π/2 the mixer hits zero
// This creates the "1110" pattern: 3 expansion phases, 1 compression

__device__ __forceinline__ float compute_heartbeat(float theta) {
    return cuda_lut_cos(theta) * cuda_lut_cos3(theta);
}

// Rate modulation based on heartbeat
// k = 0.1 gives gentle bias without forcing behavior
__device__ __forceinline__ float compute_rate_mod(float heartbeat) {
    return 1.0f + 0.1f * heartbeat;
}

// ============================================================================
// Phase Stress: Interface Polarization Metric
// ============================================================================
// VIV §4.7: stress(φ) = max(cos²φ, sin²φ)
//   - Balanced: stress = 0.5 (both quadratures equal)
//   - Polarized: stress → 1.0 (one quadrature dominates)
//
// Math.md: This measures how close the system is to "shorting"
// When stress > 0.95, the particle is maximally polarized → ejection

__device__ __forceinline__ float compute_phase_stress(float phi) {
    float sin_phi, cos_phi;
    cuda_lut_sincos(phi, &sin_phi, &cos_phi);
    return fmaxf(cos_phi * cos_phi, sin_phi * sin_phi);
}

// ============================================================================
// Coherence Filter: -λm_⊥ (Entropy Removal)
// ============================================================================
// Math.md Step 8: GPT's corrected coherence pump
//   ṁ = λ[(m·n)n - m] = -λm_⊥
//
// This REMOVES orthogonal components instead of ADDING aligned ones.
// Energy conserving: rotates phase, doesn't inject energy.
//
// Implementation:
//   m_⊥ = m - (m·n)n           (orthogonal component)
//   m_new = m - λ*m_⊥          (remove orthogonal)
//         = (1-λ)m + λ(m·n)n   (interpolate toward aligned)
//
// Arguments:
//   vx, vy, vz: velocity components (modified in place)
//   nx, ny, nz: coherence direction (unit vector)
//   lambda: filter strength (COHERENCE_LAMBDA * rate_mod)

__device__ __forceinline__ void apply_coherence_filter(
    float& vx, float& vy, float& vz,
    float nx, float ny, float nz,
    float lambda)
{
    // Projection: m·n
    float dot_mn = vx*nx + vy*ny + vz*nz;

    // Orthogonal component: m_⊥ = m - (m·n)n
    float ox = vx - dot_mn * nx;
    float oy = vy - dot_mn * ny;
    float oz = vz - dot_mn * nz;

    // Remove orthogonal: m_new = m - λ*m_⊥
    vx = vx - lambda * ox;
    vy = vy - lambda * oy;
    vz = vz - lambda * oz;
}

// ============================================================================
// Anisotropic Damping: -γ(m·n)n (Energy Removal)
// ============================================================================
// Math.md Step 8: Only damp the PARALLEL component
//   v_new = v - γ*(v·n)n
//
// The coherence pump removes orthogonal (entropy).
// The anisotropic damping removes parallel (energy).
// Together they create the breathing equilibrium.
//
// Breathing amplitude: A = P_kick / √((2γ)² + ω²)
//   - Small γ → large breathing (active galaxy, AGN-like)
//   - Large γ → tight phase lock (condensate, elliptical)

__device__ __forceinline__ void apply_anisotropic_damping(
    float& vx, float& vy, float& vz,
    float nx, float ny, float nz,
    float gamma)
{
    // Parallel component: (v·n)n
    float dot_vn = vx*nx + vy*ny + vz*nz;
    float px = dot_vn * nx;
    float py = dot_vn * ny;
    float pz = dot_vn * nz;

    // Remove parallel: v_new = v - γ*(v·n)n
    vx = vx - gamma * px;
    vy = vy - gamma * py;
    vz = vz - gamma * pz;
}

// ============================================================================
// Coherence Direction Selection
// ============================================================================
// The coherence direction n varies with heartbeat phase:
//   heartbeat > 0.5:  radial direction (in disk plane)
//   heartbeat > -0.5: tangential direction
//   heartbeat < -0.5: vertical direction (DMRG freeze)
//
// ACCIDENTAL N12 EMERGENCE (Geometric, Pre-Coupling):
//   This function implements a period-3 mode pattern (3 distinct states).
//   The heartbeat cos(θ)*cos(3θ) is period-4 (zeros at θ = nπ/2).
//
//   Duty cycle analysis:
//     heartbeat > +0.5  → radial    (~25% of cycle)
//     heartbeat ∈ [-0.5, +0.5] → tangential (~50% of cycle)
//     heartbeat < -0.5  → vertical  (~25% of cycle)
//
//   LCM(3,4) = 12 → full mode pattern repeats every 12 cycles.
//
//   This means N12 emerges from two INDEPENDENT geometric constraints
//   with no particle coupling required. This is distinct from (and precedes)
//   the Kuramoto N12 envelope (Math.md Step 11), which is emergent from
//   interaction. The mode-cycling N12 is already executing in the simulation.

__device__ __forceinline__ void compute_coherence_direction(
    float px, float py, float pz,
    float heartbeat,
    float inv_r_cyl,  // Pre-computed: 1/(r_cyl + epsilon), avoids redundant sqrtf
    float& nx, float& ny, float& nz)
{
    if (heartbeat > 0.5f) {
        // Radial direction (in disk plane)
        nx = px * inv_r_cyl;
        ny = 0.0f;
        nz = pz * inv_r_cyl;
    } else if (heartbeat > -0.5f) {
        // Tangential direction
        nx = -pz * inv_r_cyl;
        ny = 0.0f;
        nz = px * inv_r_cyl;
    } else {
        // Orthogonal phase: align toward vertical (DMRG freeze)
        nx = 0.0f;
        ny = 1.0f;
        nz = 0.0f;
    }
}

// ============================================================================
// Coherence Scale Calculation
// ============================================================================
// Scale tracks coherence level: |aligned| / |total|
// Higher scale = more phase-locked = closer to "star" state

__device__ __forceinline__ float compute_coherence_scale(
    float vx, float vy, float vz,
    float nx, float ny, float nz,
    float phi_factor)  // 1.0 for coherent, d_PHI for hop
{
    float dot_mn = vx*nx + vy*ny + vz*nz;
    float vmag = sqrtf(vx*vx + vy*vy + vz*vz) + 0.001f;
    return 1.0f + fabsf(dot_mn) / vmag * phi_factor;
}

// ============================================================================
// N12 Envelope (TODO: Not Yet Implemented)
// ============================================================================
// Math.md Step 11: The envelope coupling field
//   F(t) = A(θ) * cos(ωt)
// where A(θ) = period-12 envelope
//
// Currently: heartbeat modulates rate_mod but no 12-cycle accumulation
// Gap: Need to track phase over 12 cycles and accumulate bias
//
// Proposed implementation:
//   - Track pump_phase per particle (already exists as jet_phase)
//   - Accumulate N12 bias: only commit coupling when phase % 12 == 0
//   - This would make synchronization truly emergent from N12 closure

// Placeholder for future N12 envelope implementation
__device__ __forceinline__ float compute_n12_envelope(float phase) {
    // TODO: Implement proper 12-cycle envelope
    // For now, return constant (no N12 modulation)
    return 1.0f;
}

// ============================================================================
// Kuramoto Coupling (TODO: Not Yet Implemented)
// ============================================================================
// Math.md Step 8: θ̇_i = ω_i + K Σ_j sin(θ_j - θ_i) * A_12(θ_i)
//
// Currently: coupling is position-based (grid forces), not phase-based
// Gap: Need phase-coherent coupling for true Kuramoto synchronization
//
// Proposed implementation:
//   - Store pump_phase per particle
//   - In grid force calculation, add sin(θ_j - θ_i) term
//   - Weight by N12 envelope

// Placeholder for future Kuramoto coupling
__device__ __forceinline__ float kuramoto_coupling(
    float phase_i, float phase_j, float K)
{
    // TODO: Implement proper Kuramoto coupling
    // For now, return 0 (no phase coupling)
    return 0.0f;
}
