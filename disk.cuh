// disk.cuh — Core particle data structure and constants
// ======================================================
//
// This header defines:
//   - GPUDisk struct (SoA layout for coalesced access)
//   - Physical constants (ISCO, Schwarzschild radius, etc.)
//   - Device-side __constant__ variables
//   - Inline compute functions for derived properties
//
// Math.md mapping:
//   - GPUDisk stores position/velocity/pump state per particle
//   - pump_phase drives θ(t) = ωt evolution
//   - pump_scale tracks coherence level (proxy for envelope amplitude)
//
// V8 philosophy: "Everything correct stays untouched forever."

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Forward declaration for LUT functions (defined in cuda_lut.cuh)
__device__ float cuda_fast_atan2(float y, float x);

// ============================================================================
// Configuration Constants
// ============================================================================

#ifndef PI
#define PI            3.14159265358979f
#endif
#ifndef TWO_PI
#define TWO_PI        6.28318530717959f
#endif

#ifndef MAX_DISK_PTS
// Upper bound for particle data structure allocation
// GPUDisk struct: ~20 float arrays + 3 bool arrays per particle
// At 10M particles: ~800MB just for GPUDisk
// At 20M particles: ~1.6GB just for GPUDisk
#define MAX_DISK_PTS  20000000  // 20M max capacity (requires 6GB+ VRAM)
#endif

// ============================================================================
// Black Hole Geometry
// ============================================================================

#ifndef BH_MASS
#define BH_MASS       1.0f
#endif
#ifndef ISCO_R
#define ISCO_R        6.0f
#endif
#ifndef DISK_OUTER_R
#define DISK_OUTER_R  120.0f     // Much larger disk for galactic scale
#endif
#ifndef DISK_THICKNESS
#define DISK_THICKNESS 0.8f      // Thin disk for galactic scale
#endif
#ifndef SCHW_R
#define SCHW_R        2.0f
#endif

// M_eff calculation: include shells within the circulating region
#ifndef M_EFF_ACTIVE_RADIUS
#define M_EFF_ACTIVE_RADIUS  100.0f
#endif

// ============================================================================
// Siphon Pump Parameters
// ============================================================================

#ifndef PUMP_EJECT_THRESHOLD
#define PUMP_EJECT_THRESHOLD  0.88f   // Residual threshold for ejection
#endif
#ifndef PUMP_COHERENT_MAX
#define PUMP_COHERENT_MAX     3       // Coherent strokes before φ hop
#endif

// ============================================================================
// Background Plasma Coupling (Ion Kick + Core Anchor)
// ============================================================================
// Creates 3-zone circulation: core pull + mid drift + outer return

#ifndef ION_KICK_EPSILON
#define ION_KICK_EPSILON      0.0008f // ~8e-4, outer return strength
#endif
#ifndef ION_KICK_INNER_R
#define ION_KICK_INNER_R      50.0f   // Start outer ion kick at this radius
#endif
#ifndef ION_KICK_OUTER_R
#define ION_KICK_OUTER_R      200.0f  // Recycle particles beyond this radius
#endif
#ifndef ION_KICK_RESPAWN_R
#define ION_KICK_RESPAWN_R    150.0f  // Respawn recycled particles here
#endif

// Core anchor: 1/(1+r²) attractor — only active outside shells to return escapees.
// Inside the shell region, Keplerian orbits provide structure; the anchor's
// persistent inward pull has no counterpart and collapses shells into pillars.
#ifndef CORE_PULL_STRENGTH
#define CORE_PULL_STRENGTH    0.002f
#endif
#ifndef CORE_PULL_SCALE
#define CORE_PULL_SCALE       30.0f
#endif
#ifndef CORE_ANCHOR_INNER_R
#define CORE_ANCHOR_INNER_R   DISK_OUTER_R  // Only pull back particles that escape the disk
#endif

// ============================================================================
// Natural Growth (Star Formation)
// ============================================================================

#ifndef SPAWN_ENABLE
#define SPAWN_ENABLE          true
#endif
#ifndef SPAWN_COHERENCE_THRESH
#define SPAWN_COHERENCE_THRESH 0.7f
#endif
#ifndef SPAWN_PROB_BASE
#define SPAWN_PROB_BASE       0.00001f
#endif
#ifndef SPAWN_SCALE_BOOST
#define SPAWN_SCALE_BOOST     2.0f
#endif
#ifndef SPAWN_ENERGY_TAX
#define SPAWN_ENERGY_TAX      0.1f
#endif
#ifndef SPAWN_MIN_PARENT_KE
#define SPAWN_MIN_PARENT_KE   0.01f
#endif

// ============================================================================
// Viviani Field Force Model
// ============================================================================
// Force derived from field line direction, not central mass

#ifndef FIELD_FORCE_STRENGTH
#define FIELD_FORCE_STRENGTH    0.02f
#endif
#ifndef FIELD_FORCE_FALLOFF
#define FIELD_FORCE_FALLOFF     10.0f
#endif
#ifndef AXIAL_THRESHOLD
#define AXIAL_THRESHOLD         0.5f
#endif
#ifndef TANGENT_SCALE
#define TANGENT_SCALE           0.5f
#endif

// ============================================================================
// Geometric Coherence Pump
// ============================================================================
// Math.md Step 8: ṁ = -λm_⊥ - γ(m·n)n + kick + gravity
//
// ANISOTROPIC DISSIPATION:
//   - Coherence filter (-λm_⊥): removes orthogonal components (entropy)
//   - Energy damping (-γm_∥): removes parallel components (energy)
//   - Ion kick: injects energy (external entropy source)
//
// Breathing amplitude: A = P_kick / √((2γ)² + ω²)

#ifndef COHERENCE_LAMBDA
#define COHERENCE_LAMBDA  0.1f    // Orthogonal decay (entropy removal)
#endif
#ifndef COHERENCE_GAMMA
#define COHERENCE_GAMMA   0.02f   // Parallel decay (energy removal)
#endif
#ifndef KEPLER_RESTORE_RATE
#define KEPLER_RESTORE_RATE 6.0f  // Tangential velocity restore toward v_kep (per sim-time unit)
#endif                            // At dt=1/60: ~10% nudge per frame. Balances L_sink at outer shells.
#ifndef LANGEVIN_SIGMA
#define LANGEVIN_SIGMA    0.02f   // Ion kick strength
#endif

// ============================================================================
// Spiral Arm Topology
// ============================================================================

#ifndef NUM_ARMS
#define NUM_ARMS 3
#endif
#ifndef ARM_WIDTH_DEG
#define ARM_WIDTH_DEG 45.0f
#endif
#ifndef ARM_TRAP_STRENGTH
#define ARM_TRAP_STRENGTH 0.15f
#endif
#ifndef USE_ARM_TOPOLOGY
#define USE_ARM_TOPOLOGY true
#endif
#ifndef ARM_BOOST_OVERRIDE
#define ARM_BOOST_OVERRIDE 0.0f
#endif

// ============================================================================
// Golden Ratio Constants
// ============================================================================

#ifndef PHI
#define PHI 1.6180339887498948f
#endif
#ifndef PHI_EXCESS
#define PHI_EXCESS 0.09017f      // φ - 1.5 ≈ 0.118 (pump asymmetry)
#endif
#ifndef SCALE_RATIO
#define SCALE_RATIO 1.6875f      // 27/16 (coherent scale step)
#endif
#ifndef BIAS
#define BIAS 0.75f               // Demon's operating point
#endif

// ============================================================================
// Hybrid Rendering
// ============================================================================

#ifndef HYBRID_R
#define HYBRID_R 30.0f
#endif

// ============================================================================
// Device Constants — Defined in blackhole_v20.cu
// ============================================================================
// The following device constants are defined in blackhole_v20.cu:
//   d_PI, TWO_PI, ISCO_R, d_BH_MASS, d_SCHW_R, DISK_THICKNESS
//   d_PHI, d_SCALE_RATIO, d_BIAS, d_PHI_EXCESS
//   d_NUM_ARMS, d_ARM_WIDTH_DEG, d_ARM_TRAP_STRENGTH, d_USE_ARM_TOPOLOGY, d_ARM_BOOST_OVERRIDE
//   d_current_particle_count, d_spawn_count
//   d_grid_dim, d_grid_cells, d_grid_cell_size, d_grid_stride_y, d_grid_stride_z
//
// The modular headers use preprocessor #defines (e.g., ISCO_R, BH_MASS) instead,
// which avoids the need for extern declarations and works with both modular and
// monolithic compilation.

// ============================================================================
// GPU Disk Structure — SoA Layout for Coalesced Access
// ============================================================================
// Math.md mapping:
//   - pos_x/y/z, vel_x/y/z: particle state
//   - pump_state: 8-state siphon machine
//   - pump_scale: tracks coherence level
//   - pump_history: exponentially smoothed activity (emergence metric)
//   - jet_phase: phase memory through ejection cycle

// Packed flag bits for GPUDisk.flags — replaces separate bool arrays.
// Bit layout was chosen to leave room for future state bits without
// widening the byte. Reserved bits must be zero.
#define PFLAG_ACTIVE    0x01
#define PFLAG_EJECTED   0x02
// bits 2-7 reserved

struct GPUDisk {
    // Position (3 floats × N)
    float pos_x[MAX_DISK_PTS];
    float pos_y[MAX_DISK_PTS];
    float pos_z[MAX_DISK_PTS];

    // Velocity (3 floats × N)
    float vel_x[MAX_DISK_PTS];
    float vel_y[MAX_DISK_PTS];
    float vel_z[MAX_DISK_PTS];

    // NOTE: disk_r, disk_phi, temp, in_disk removed — computed on-demand
    // Saves 13 bytes/particle × 10M = 130 MB VRAM and ~37 GB/s bandwidth

    // Siphon pump state per particle — float fields grouped together
    // to avoid alignment padding around 1-byte fields.
    int   pump_state[MAX_DISK_PTS];      // siphon_state_t (0-7)
    float pump_scale[MAX_DISK_PTS];      // scale_factor (coherence level)
    int   pump_coherent[MAX_DISK_PTS];   // coherent_count
    float pump_residual[MAX_DISK_PTS];   // current residual
    float pump_work[MAX_DISK_PTS];       // accumulated work
    float pump_history[MAX_DISK_PTS];    // smoothed pump activity (emergence)
    float jet_phase[MAX_DISK_PTS];       // phase memory through ejection cycle

    // Kuramoto phase coupling (math.md Step 8, Step 10, Step 11)
    // theta advances at rate omega_nat per frame, plus mean-field coupling
    // from grid phase_sin/phase_cos via the gather kernel. Separate from the
    // pump state machine so Kuramoto dynamics are not entangled with ejections.
    float theta[MAX_DISK_PTS];       // phase in [0, 2π), continuous rotation
    float omega_nat[MAX_DISK_PTS];   // natural frequency, Gaussian spread

    // Byte-sized fields packed at the end to avoid alignment padding that
    // would otherwise cost 3 bytes per particle (previous layout had
    // pump_seam as a uint8_t between two floats, forcing 3 bytes of pad).
    uint8_t pump_seam[MAX_DISK_PTS]; // seam phase bits
    uint8_t flags[MAX_DISK_PTS];     // bit0=active, bit1=ejected (see PFLAG_*)
};

// Packed-flag accessors for the ejected/active bits previously stored as
// separate bool arrays. Keeps call sites readable and makes the bit
// manipulation explicit in one place.
__device__ __forceinline__ bool particle_active(const GPUDisk* disk, int i) {
    return (disk->flags[i] & PFLAG_ACTIVE) != 0;
}

__device__ __forceinline__ bool particle_ejected(const GPUDisk* disk, int i) {
    return (disk->flags[i] & PFLAG_EJECTED) != 0;
}

__device__ __forceinline__ void set_particle_active(GPUDisk* disk, int i, bool v) {
    if (v) disk->flags[i] |= PFLAG_ACTIVE;
    else   disk->flags[i] &= (uint8_t)~PFLAG_ACTIVE;
}

__device__ __forceinline__ void set_particle_ejected(GPUDisk* disk, int i, bool v) {
    if (v) disk->flags[i] |= PFLAG_EJECTED;
    else   disk->flags[i] &= (uint8_t)~PFLAG_EJECTED;
}

// ============================================================================
// Derived Particle Properties — Compute On-Demand
// ============================================================================
// These were previously stored but are cheap to compute.
// Math.md: These are GEOMETRIC (single-node), not emergent.

__device__ __forceinline__ float compute_disk_r(float px, float pz) {
    return sqrtf(px * px + pz * pz);
}

__device__ __forceinline__ float compute_disk_phi(float px, float pz) {
    float phi = cuda_fast_atan2(pz, px);
    return (phi < 0) ? phi + TWO_PI : phi;
}

__device__ __forceinline__ float compute_temp(float r_cyl) {
    return (r_cyl > ISCO_R) ? powf(ISCO_R / r_cyl, 0.75f) : 1.0f;
}

__device__ __forceinline__ bool compute_in_disk(float py, float px, float pz,
                                                 float vx, float vy, float vz) {
    // Angular momentum (rsqrtf pattern avoids sqrtf + division)
    float Lx = py * vz - pz * vy;
    float Ly = pz * vx - px * vz;
    float Lz = px * vy - py * vx;
    float inv_L_mag = rsqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
    float L_disk_align = Lz * inv_L_mag;
    return (fabsf(py) < DISK_THICKNESS * 2.0f && L_disk_align > 0.3f);
}

// ============================================================================
// Siphon Pump States — Use siphon_pump.h definitions
// ============================================================================
// The canonical enum is defined in siphon_pump.h as siphon_state_t.
// PUMP_IDLE, PUMP_PRIMED, etc. are available from that header.

// ============================================================================
// Seam Bits — Compatible with siphon_pump.h definitions
// ============================================================================
// These match SEAM_* from siphon_pump.h but with guards to avoid redefinition

#ifndef SEAM_CLOSED
#define SEAM_CLOSED    0x00
#endif
#ifndef SEAM_UP_ONLY
#define SEAM_UP_ONLY   0x01
#endif
#ifndef SEAM_DOWN_ONLY
#define SEAM_DOWN_ONLY 0x02
#endif
#ifndef SEAM_FULL
#define SEAM_FULL      0x03
#endif
