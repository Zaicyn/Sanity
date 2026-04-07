// sim_globals.h — Simulation Global State and Configuration
// ==========================================================
// All global variables for camera, physics toggles, test flags,
// and runtime configuration. Included once into blackhole_v20.cu.
#pragma once

#include <cstdint>
#include "disk.cuh"  // PFLAG_ACTIVE

// Camera State
// ============================================================================

static struct {
    float dist = 200.0f;   // Larger for bigger disk
    float azimuth = 0.4f;
    float elevation = 0.6f;  // Higher angle to see spiral structure
    double lastX = 0, lastY = 0;
    bool dragging = false;
    bool paused = false;
    uint8_t seam_bits = 0x03;  // Start with full coupling
    float bias = 0.75f;        // Demon efficiency (0.5 = weak, 0.75 = normal, 1.0 = perfect)
    int color_mode = 0;  // 0 = topology, 1 = blackbody, 2 = pump scale, 3 = intrinsic redshift
} g_cam;

// ============================================================================
// Topology Control - GPT's Control Experiment
// ============================================================================
// Toggle between discrete hopfion shells vs smooth gradient (no twist)
// H key toggles: true = hopfion topology, false = smooth gradient
bool g_use_hopfion_topology = true;

// ============================================================================
// Spiral Arm Topology Control - Deepseek's Experiment
// ============================================================================
// Toggle between discrete arm boundaries vs smooth density waves
// A key toggles: true = discrete boundaries, false = smooth waves
bool g_enable_arms = false;         // Enable/disable arm structure (off by default, use --discrete-arms or --smooth-arms)
bool g_spawn_enabled = true;        // Natural growth spawning (--no-spawn disables for clean Kuramoto measurements)
bool g_use_arm_topology = true;     // true = discrete, false = smooth

// Phase-misalignment shear (non-monotonic magnetic friction analog)
// Based on Gu/Lüders/Bechinger arXiv:2602.11526v1 — friction peaks in the
// competing regime where FM and AFM phase orderings frustrate each other.
// Drives collapse in turbulent/frustrated regions, leaves locked shells alone.
float g_shear_k = 0.0f;

// Kuramoto phase coupling — math.md Step 8, Step 10, Step 11
// θ_i advances at rate ω_i + K·sin(θ_cell − θ_i) via mean-field coupling
// through the grid phase_sin/phase_cos fields. Tests synchronization
// threshold, traveling coherence packets, chimera states, breathing clusters.
float g_kuramoto_k = 0.0f;        // Coupling strength K (0 = no coupling)
float g_omega_base = 1.0f;        // Mean natural frequency ω₀
float g_omega_spread = 0.05f;     // Gaussian std-dev σ for ω distribution
bool  g_n12_envelope = true;      // Apply N12 mixer envelope to coupling (math.md Step 11)
float g_envelope_scale = 1.0f;    // Multiplier on envelope harmonic indices: cos(3s·θ)·cos(4s·θ).
                                  // s=1 → period 2π (default N12). s=2 → period π (N6). s=0.5 → period 4π (N24).
                                  // Tests GPT's prediction: optimal ω should scale as 1/envelope_period.

// Tree Architecture Step 4: runtime corner threshold for passive/active classification.
// Particles with |pump_residual| > g_corner_threshold → active (siphon kernel).
// Default 0.15f matches the compile-time constant from Step 3. Tunable via --corner-threshold.
float g_corner_threshold = 0.15f;

// Tree Architecture Step 4: runtime passive residual tau.
// Controls how fast pump_residual decays in passive particles. Units: simulation time.
// Default 5.0f matches PASSIVE_RESIDUAL_TAU from Step 3. Tunable via --passive-tau.
float g_passive_residual_tau = 5.0f;

// Tree Architecture Step 6: shell-aware initialization.
// When true, particles are initialized ON the 8 resonance shells instead of
// uniformly in a box. This skips the settling transient and starts with most
// particles passive. Use --shell-init to enable.
bool g_shell_init = false;

// Shell lensing: displace particle apparent positions based on shell
// refractive indices, creating a gravitational-lensing-like distortion.
// Toggle via --shell-lensing or L key at runtime.
bool g_shell_lensing = false;

// Per-cell R export: dumps R_cell grid to disk every N frames (0 = disabled)
int g_r_export_interval = 0;

// Dense R(t) logging: prints R_global every N frames (0 = only every 90 like normal stats)
int g_r_log_interval = 0;

// Kuramoto × topology correlation dump — emits one CSV-friendly row per stats
// frame with (frame, R, R_recon, n_peaks, peak_mass_frac, Q, num_shells, active_count).
// Writes to stdout prefixed with [QR-corr] so the stream can be filtered offline.
bool g_qr_corr_log = false;

// Initial rotation direction: prograde (default) or retrograde.
// Used for the chirality / Q-sign test — if Q drift is driven by initial
// rotation direction, flipping this should flip the sign of Q drift.
bool g_retrograde_init = false;

// RNG seed for initial particle positions, phases, and natural frequencies
unsigned int g_rng_seed = 42;

#define NUM_ARMS 3                  // m = 3 (3-armed spiral)
#define ARM_WIDTH_DEG 45.0f         // Width of each arm in degrees
#define ARM_TRAP_STRENGTH 0.15f     // Angular momentum barrier strength

// ============================================================================
// Headless Mode - Performance Testing
// ============================================================================
// Disable all rendering for 10-20x speedup (physics + logging only)
bool g_headless = false;

// ============================================================================
// Hybrid LOD rendering (experimental - requires volume pass implementation)
bool g_hybrid_lod = false;
// Octree-based render traversal (vs flat scan compaction)
bool g_octree_render = false;
// Octree physics - XOR neighbor stress computation
bool g_octree_physics = true;
// Octree phase evolution - Kuramoto coupling via Morton-sorted leaves
// Set to false to use mip-tree for hierarchical coherence instead (A/B test)
bool g_octree_phase = true;
// Octree rebuild - Morton sort + stochastic tree build every 30 frames
// Set to false to skip Morton sorting entirely (mip-tree provides hierarchy)
// DEFAULT: false — mip-tree replaces octree for hierarchical coherence
bool g_octree_rebuild = false;
// Grid physics - DNA/RNA streaming forward-pass model (replaces octree physics)
// DEFAULT: true — enabled with g_grid_flags for sparse 235× speedup
bool g_grid_physics = true;
// Grid flags mode - optimal sparse: presence flags, no lists, no dedup
// DEFAULT: true — 235× speedup on Pass 2 (2M → 3.4k cells)
bool g_grid_flags = true;

// ============================================================================
// Radius-Controlled Termination - GPT's Confounder Test
// ============================================================================
// Terminate based on ring radius instead of frame count to eliminate geometric effects
bool g_terminate_on_radius = false;
float g_target_ring_radius = 250.0f;
int g_target_frames = 50000;  // Configurable target frame count

// ============================================================================
// Test Suite Flags - The Final Trilogy
// ============================================================================
bool g_test_residence_time = false;    // Test A: Track arm vs gap residence time
bool g_matched_amplitude = false;      // Test C: Set discrete boost to 1.25× (match smooth)
float g_arm_boost_override = 0.0f;     // If > 0, override ARM_BOOST in discrete mode

// ============================================================================
// Predictive Locking - Skip Harmonic Recomputation When Shells Are Stable
// ============================================================================
// When the system is in a locked m=0 ground state (8 shells, Q stable, isotropic),
// we can skip expensive per-frame computations:
//   - Mip-tree rebuild (already fast, but skippable)
//   - Full harmonic analysis (m=3 mode detection)
//   - Angular histogram computation
//
// Lock detection: shell count stable + Q variance low + stability low
struct HarmonicLock {
    int prev_shell_count;       // Previous frame shell count
    float prev_Q;               // Previous frame Q estimate
    float Q_variance_acc;       // Running variance accumulator
    int stable_frames;          // Consecutive frames meeting lock criteria
    bool locked;                // Currently in locked state
    int lock_recheck_counter;   // Frames until next full recompute (for verification)

    // Lock thresholds
    // Note: Lock detection runs every 90 frames (diagnostic interval), so thresholds are
    // tuned for that cadence. LOCK_THRESHOLD_FRAMES counts diagnostic intervals, not raw frames.
    static constexpr int LOCK_THRESHOLD_FRAMES = 3;    // Need 3 consecutive diagnostics (270 frames) to lock
    static constexpr int RECHECK_INTERVAL = 256;       // Verify lock every 256 frames
    static constexpr float Q_VARIANCE_THRESHOLD = 50.0f; // Max |ΔQ| between diagnostics (Q swings 0-35 normally)
    static constexpr float STABILITY_THRESHOLD = 0.20f; // Max stability% to stay locked (20%)
};
HarmonicLock g_harmonic_lock = {0, 0.0f, 0.0f, 0, false, 0};
bool g_predictive_locking = true;  // Enable predictive locking by default

// ============================================================================
// Active Particle Compaction - Skip Static Shell Mass
// ============================================================================
// When locked, ~90% of particles are stable shell mass that doesn't need
// scatter/gather every frame. We compact only "active" particles:
//   - Movers: |velocity| > threshold
//   - Cell-changers: current cell != previous cell
//   - Boundary: near active tile edges
//
// This reduces O(N) scatter/gather to O(active_N) where active_N << N.
// Static particles are "baked" into a persistent grid that gets reused.
//
// Memory layout:
//   d_prev_particle_cell[N]     - Previous frame's cell indices
//   d_particle_active[N]        - Activity mask (uint8)
//   d_active_particle_list[N]   - Compacted indices of active particles
//   d_active_particle_count     - Number of active particles
//   d_static_grid_density[G]    - Baked density from static particles
//   d_static_grid_momentum_*[G] - Baked momentum from static particles
//
struct ActiveParticleState {
    uint32_t* d_prev_cell;           // Previous frame cell indices
    uint8_t*  d_active_mask;         // Per-particle activity flag
    uint32_t* d_active_list;         // Compacted active particle indices
    uint32_t* d_active_count;        // Device counter
    uint32_t  h_active_count;        // Host-side count

    // Static grid (baked when lock engages)
    float* d_static_density;
    float* d_static_momentum_x;
    float* d_static_momentum_y;
    float* d_static_momentum_z;
    float* d_static_phase_sin;
    float* d_static_phase_cos;

    bool initialized;
    bool static_baked;               // True after static particles scattered to static grid
    int bake_frame;                  // Frame when bake occurred

    // Thresholds
    // Note: Velocity threshold is less useful than cell-change for Keplerian orbits
    // Particles orbit at v~0.2-0.3 but stay in same cell for many frames
    // The key is whether the particle changes CELL, not whether it's moving
    static constexpr float VELOCITY_THRESHOLD = 10.0f;  // Very high - effectively disabled
    static constexpr int REBAKE_INTERVAL = 256;         // Re-bake static grid periodically
};
ActiveParticleState g_active_particles = {};
#if ENABLE_PASSIVE_ADVECTION
bool g_active_compaction = false;  // Step 3: passive kernel replaces scatter-skip optimization.
                                   // The baked static grid would go stale because passive particles
                                   // move azimuthally. Siphon-skip savings dwarf scatter-skip savings.
#else
bool g_active_compaction = true;   // Pre-passive: active particle compaction for scatter optimization.
#endif

// ============================================================================
// Seam Drift Tracking - Time-Resolved m=3 Phase Logging
// ============================================================================
// Track the m=3 phase angle over time to detect whether seam orientation
// is stationary (locked to arm geometry) or precessing.
// Log format: (frame, M_eff, m3_phase_deg)
#define SEAM_DRIFT_LOG_SIZE 400  // Enough for 60k+ frames at 90-frame intervals
struct SeamDriftEntry {
    int frame;
    float M_eff;
    float phase_deg;    // m=3 phase in degrees [0, 120)
    int sample_count;
};
SeamDriftEntry g_seam_drift_log[SEAM_DRIFT_LOG_SIZE];
int g_seam_drift_count = 0;

// ============================================================================
// Entropy Injection Test - Material Dissolution
// ============================================================================
// Global flag for entropy injection (toggled with E key)
bool g_inject_entropy = false;

// injectEntropyCluster now in spawn.cuh


// End of sim_globals.h
