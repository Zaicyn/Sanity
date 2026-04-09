// physics.cu — Main Physics Kernel Assembly (REFERENCE IMPLEMENTATION)
// =====================================================================
//
// STATUS: This file is a REFERENCE IMPLEMENTATION showing how the modular
// headers (disk.cuh, harmonic.cuh, forces.cuh, etc.) combine to form the
// complete siphonDiskKernel. The actual kernel still lives in blackhole_v20.cu.
//
// USAGE: This file is NOT compiled directly. It serves as:
//   1. A clean specification of each physics component
//   2. A reference for auditing against math.md
//   3. A future modular replacement when fully validated
//
// To use the modular physics instead of the monolithic kernel:
//   1. Remove siphonDiskKernel from blackhole_v20.cu
//   2. Include this file (or copy its kernel) into blackhole_v20.cu
//   3. Ensure only one set of device constant definitions exist
//
// Structure:
//   1. Load particle state (Schism-style local accumulation)
//   2. Apply forces (Viviani field, angular momentum)
//   3. Compute harmonic terms (heartbeat, phase stress)
//   4. Run siphon pump state machine
//   5. Handle ejection / jet dynamics (Aizawa)
//   6. Apply background coupling (ion kick, core anchor)
//   7. Write back (single coalesced write)
//
// Math.md mapping:
//   - GEOMETRIC: heartbeat, coherence filter, damping (harmonic.cuh)
//   - EMERGENT: synchronization arises from coupling (forces.cuh)
//   - The pump creates CONDITIONS for emergence, not emergence itself

#include "disk.cuh"
#include "harmonic.cuh"
#include "forces.cuh"
#include "siphon_pump.cuh"
#include "aizawa.cuh"
// topology.cuh included after device constants are defined (needs d_NUM_ARMS etc.)

// LUT include (for implementations)
#include "cuda_lut.cuh"

// ============================================================================
// Device Constant Definitions
// ============================================================================
// These are declared extern in disk.cuh, defined here.
// When included from blackhole_v20.cu (USE_MODULAR_PHYSICS), the constants
// are already defined in blackhole_v20.cu, so we skip them here.

#ifndef PHYSICS_CONSTANTS_DEFINED
#define PHYSICS_CONSTANTS_DEFINED

__device__ __constant__ float d_PI = 3.14159265358979f;
__device__ __constant__ float d_TWO_PI = 6.28318530717959f;
__device__ __constant__ float d_ISCO = 6.0f;
__device__ __constant__ float d_BH_MASS = BH_MASS;  // Must match BH_MASS (100.0f) from disk.cuh
__device__ __constant__ float d_SCHW_R = 2.0f;
__device__ __constant__ float d_DISK_THICKNESS = 0.8f;
__device__ __constant__ float d_PHI = 1.6180339887498948f;
__device__ __constant__ float d_SCALE_RATIO = 1.6875f;        // 27/16
__device__ __constant__ float d_BIAS = 0.75f;
__device__ __constant__ float d_PHI_EXCESS = 0.09017f;        // φ - 1.5 ≈ 0.118

// Spiral arm topology
__device__ __constant__ int d_NUM_ARMS = 3;
__device__ __constant__ float d_ARM_WIDTH_DEG = 45.0f;
__device__ __constant__ float d_ARM_TRAP_STRENGTH = 0.15f;
__device__ __constant__ bool d_USE_ARM_TOPOLOGY = true;
__device__ __constant__ float d_ARM_BOOST_OVERRIDE = 0.0f;

// Dynamic particle count
__device__ unsigned int d_current_particle_count = 0;
__device__ unsigned int d_spawn_count = 0;

// Include topology.cuh here AFTER arm constants are defined
#include "topology.cuh"

#endif // PHYSICS_CONSTANTS_DEFINED

// ============================================================================
// Siphon Disk Kernel — Main Physics Integration
// ============================================================================
//
// This kernel processes all particles through:
//   - Force accumulation (Viviani field model)
//   - Harmonic dynamics (heartbeat, coherence filter)
//   - Siphon pump state machine
//   - Jet dynamics (Aizawa attractor for ejected particles)
//   - Background coupling (ion kick, core anchor)
//
// Uses LOCAL ACCUMULATION PATTERN (Schism-style):
//   - Load once at start
//   - Accumulate all modifications in registers
//   - Write once at end
// This ensures coalesced memory access and avoids scattered writes.

__global__ void siphonDiskKernel(
    GPUDisk* disk,
    const uint8_t* __restrict__ in_active_region,
    int N,
    float time,
    float dt,
    uint8_t global_seam,
    float global_bias)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Step 3: skip passive particles (in_active_region[i] == 0).
    // The passive kernel owns them. See active_region.cuh for the
    // threshold-based classification that decides which is which.
    if (i >= N || !particle_active(disk, i) || !in_active_region[i]) return;

    // ========================================================================
    // STEP 1: LOAD PARTICLE STATE (Single Coalesced Read)
    // ========================================================================

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // Pump state
    int state = disk->pump_state[i];
    float scale = disk->pump_scale[i];
    int coherent = disk->pump_coherent[i];
    float residual = disk->pump_residual[i];
    float work = disk->pump_work[i];
    float history = disk->pump_history[i];
    bool was_ejected = particle_ejected(disk, i);

    // ========================================================================
    // STEP 2: COMPUTE DISTANCES AND ANGULAR MOMENTUM
    // ========================================================================
    // rsqrtf pattern: compute r² once, get r and 1/r from single SFU call

    float r3d_sq = px*px + py*py + pz*pz;
    float inv_r3d = rsqrtf(r3d_sq + 1e-8f);
    float r3d = r3d_sq * inv_r3d;

    float r_cyl_sq = px*px + pz*pz;
    float inv_r_cyl = rsqrtf(r_cyl_sq + 1e-8f);
    float r_cyl = r_cyl_sq * inv_r_cyl;

    float r_safe = fmaxf(r3d, d_SCHW_R * 0.5f);

    // Angular momentum (L = r × v) - rsqrtf pattern for inv_L_mag
    float Lx, Ly, Lz, inv_L_mag;
    compute_angular_momentum(px, py, pz, vx, vy, vz, Lx, Ly, Lz, inv_L_mag);
    float L_disk_align = compute_disk_alignment(Lz, inv_L_mag);

    // Local orbital frame: 3D radial, tangential, and orbital normal
    float frx, fry, frz;  // radial (outward from center)
    float ftx, fty, ftz;  // tangential (prograde in orbital plane)
    float flx, fly, flz;  // orbital normal (L_hat)
    compute_local_frame(px, py, pz, Lx, Ly, Lz, inv_L_mag, inv_r3d,
                        frx, fry, frz, ftx, fty, ftz, flx, fly, flz);

    // ========================================================================
    // STEP 3: APPLY VIVIANI FIELD FORCE
    // ========================================================================
    // Math.md: This is part of the INTERACTION that enables emergence

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    apply_viviani_field_force(px, py, pz, vx, vy, vz, r3d, inv_r3d, ax, ay, az);

    // ========================================================================
    // STEP 4: ORBITAL-PLANE DAMPING (was: disk plane damping)
    // ========================================================================
    // Damps velocity perpendicular to the particle's own orbital plane,
    // not the hardcoded Y axis. Particles settle into their own orbits.

    apply_orbital_damping(vx, vy, vz, flx, fly, flz, r3d, vx, vy, vz);

    // ========================================================================
    // STEP 5: UPDATE VELOCITY AND POSITION (First Integration)
    // ========================================================================

    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;

    // Angular momentum sink REMOVED — the Viviani field self-regulates
    // through its topology. The sink was draining tangential velocity
    // with no physical basis in the waveform gravity framework.

    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    // Recalculate distances after update (rsqrtf pattern)
    r3d_sq = px*px + py*py + pz*pz;
    inv_r3d = rsqrtf(r3d_sq + 1e-8f);
    r3d = r3d_sq * inv_r3d;

    r_cyl_sq = px*px + pz*pz;
    inv_r_cyl = rsqrtf(r_cyl_sq + 1e-8f);
    r_cyl = r_cyl_sq * inv_r_cyl;

    // ========================================================================
    // STEP 6: COMPUTE HARMONIC TERMS (Math.md Step 7)
    // ========================================================================
    // This is GEOMETRIC — exists even for single node

    float orb_phi = compute_disk_phi(px, pz);

    // The Z-axis mixer: cos(θ) * cos(3θ) = ½[cos(2θ) + cos(4θ)]
    // Generates the quartic harmonic ladder: ω, 2ω, 3ω, 4ω
    float heartbeat = compute_heartbeat(orb_phi);
    float rate_mod = compute_rate_mod(heartbeat);

    // Phase stress: interface polarization metric
    // stress > 0.95 → maximally polarized → ejection candidate
    float phase_stress = compute_phase_stress(orb_phi);

    // ========================================================================
    // STEP 7: COMPUTE COUPLING AND SEAM BITS
    // ========================================================================
    // Math.md: Coupling K determines synchronization threshold

    // Orbital circularity: |L| / (r × |v|) = 1 for circular, 0 for radial.
    // Works for orbits in ANY plane, unlike L_disk_align which only measures Lz.
    float L_actual = 1.0f / (inv_L_mag + 1e-8f);  // |L| from pre-computed inv_L_mag
    float v_mag = sqrtf(vx*vx + vy*vy + vz*vz + 1e-8f);
    float circularity = L_actual / (r3d * v_mag + 1e-8f);
    circularity = fminf(circularity, 1.0f);
    float coupling = compute_coupling_strength(r3d, circularity, py);
    uint8_t seam_bits = select_seam_bits(coupling, r3d);
    disk->pump_seam[i] = seam_bits;


    // Pressure from pump
    float pressure = compute_pressure(scale);

    // ========================================================================
    // STEP 8: COMPUTE TIDAL FORCES
    // ========================================================================

    float shear, tidal_radial;
    compute_tidal_forces(r_cyl, r3d, scale, L_disk_align, shear, tidal_radial);
    float tidal_stress = fmaxf(shear, tidal_radial);

    // STEP 9: SPIRAL ARM TOPOLOGY — REMOVED
    // Azimuthal confinement with no field derivation. Arms should emerge
    // from the Viviani field topology, not be forced.
    float arm_boost = 1.0f;  // neutral (no boost)
    scale *= arm_boost;

    // ========================================================================
    // STEP 10: SIPHON PUMP STATE MACHINE
    // ========================================================================
    // States 2 and 3 need special handling (coherence filter)

    if (state == PUMP_UPSTROKE_COHERENT) {
        // Math.md Step 8: Coherence filter -λm_⊥
        // Removes orthogonal components (entropy)
        float nx, ny, nz;
        compute_coherence_direction(px, py, pz, heartbeat, inv_r_cyl, nx, ny, nz,
                                    frx, fry, frz, ftx, fty, ftz, flx, fly, flz);

        float lambda = COHERENCE_LAMBDA * rate_mod;
        apply_coherence_filter(vx, vy, vz, nx, ny, nz, lambda);

        scale = compute_coherence_scale(vx, vy, vz, nx, ny, nz, 1.0f);
        coherent++;
        state = PUMP_EXPAND;
    }
    else if (state == PUMP_UPSTROKE_HOP) {
        // Stronger filter for φ-hop — use 3D radial from local frame
        float nx = frx;
        float ny = fry;
        float nz = frz;

        float lambda = COHERENCE_LAMBDA * d_PHI * rate_mod;
        apply_coherence_filter(vx, vy, vz, nx, ny, nz, lambda);

        scale = compute_coherence_scale(vx, vy, vz, nx, ny, nz, d_PHI);
        coherent = 0;
        state = PUMP_EXPAND;
    }
    else {
        // Other states handled by state machine
        siphon_state_step(state, scale, coherent, residual, work,
                          seam_bits, pressure, global_bias, tidal_stress, history);
    }

    // Scale regulation (prevents runaway)
    regulate_scale(scale, coherent);

    // ========================================================================
    // STEP 11: EJECTION CHECK
    // ========================================================================

    bool eject = check_ejection(phase_stress, r3d, r_cyl, residual);

    if (eject && !was_ejected) {
        // Initial ejection kick
        disk->jet_phase[i] = orb_phi;  // Capture phase for jet memory
        apply_ejection_kick(px, py, pz, vx, vy, vz, r_cyl, residual, history, Ly, orb_phi, dt,
                            ftx, fty, ftz, flx, fly, flz);
    }

    // ========================================================================
    // STEP 12: JET DYNAMICS (Aizawa Attractor)
    // ========================================================================
    // Math.md: Aizawa has harmonic structure similar to r⃗(θ) = r⃗_ω + ½r⃗_3ω

    // STEP 12: AIZAWA JET DYNAMICS — REMOVED
    // Strange attractor blended into jet trajectories had no derivation
    // from the Viviani field. Ejected particles follow ballistic + field.
    float jet_phase_delta = 0.0f;

    // STEPS 13, 13b: BOLTED-ON FORCES — ALL REMOVED
    // Ion kick, core anchor, boundary recycling, Keplerian orbit maintenance
    // were all independent hacks fighting the Viviani field. Orbits should
    // emerge from the field's radial force balance, not be forced.

    // ========================================================================
    // STEP 14: ANISOTROPIC DISSIPATION (Energy Sink)
    // ========================================================================
    // Math.md Step 8: -γ(m·n)n removes parallel component
    // NOTE: reuse inv_r_cyl (no redundant sqrt)

    {
        // Use 3D radial from local frame (not XZ projection)
        apply_anisotropic_damping(vx, vy, vz, frx, fry, frz, COHERENCE_GAMMA);
    }

    // ========================================================================
    // STEP 15: FINAL COALESCED WRITE-BACK
    // ========================================================================

    disk->pos_x[i] = px;
    disk->pos_y[i] = py;
    disk->pos_z[i] = pz;
    disk->vel_x[i] = vx;
    disk->vel_y[i] = vy;
    disk->vel_z[i] = vz;

    disk->pump_state[i] = state;
    disk->pump_scale[i] = scale;
    disk->pump_coherent[i] = coherent;
    disk->pump_residual[i] = residual;
    disk->pump_work[i] = work;
    set_particle_ejected(disk, i, eject);

    // Write back jet_phase delta
    if (was_ejected && jet_phase_delta != 0.0f) {
        disk->jet_phase[i] += jet_phase_delta;
    }

    // Update pump history (exponential smoothing)
    disk->pump_history[i] = update_pump_history(history, scale);

    // ========================================================================
    // STEP 16: w-COMPONENT EVOLUTION (4D Phase / Transport Axis)
    // ========================================================================
    // math.md: w(θ) = ⅓ sin 5θ. The w-component accumulates from the pump
    // bias residual — each pump cycle leaves (1 - BIAS) = 25% unrecovered.
    // This energy rotates into the 4D phase, thinning the 3D projection:
    //   s(θ) = sqrt(1 - w²)  →  visible size scales down as w grows.
    // Jets reset w toward zero (re-entry into 3D from the transport channel).
    {
        float w = disk->w_component[i];

        // Accumulation: pump residual leaks into w at rate proportional to
        // the bias gap. Higher residual = faster w accumulation.
        float w_rate = (1.0f - d_BIAS) * fabsf(residual) * 1.0f;
        w += w_rate * dt;

        // Jet reset: ejection dumps w back toward zero (3D re-entry).
        // The jets are the mechanism for resetting the 4D accumulation.
        if (eject) {
            w *= 0.8f;  // 20% reset per frame while ejected
        }

        // Clamp to [0, 1] — w=1 is fully in the transport channel
        w = fminf(fmaxf(w, 0.0f), 1.0f);

        disk->w_component[i] = w;
    }
}

// ============================================================================
// Natural Growth Spawning Kernel
// ============================================================================
// Models gravitational collapse in coherent gas. When pump_history is high,
// the region is gravitationally bound and can form new stars.
//
// ENERGY CONSERVATION (per GPT's audit):
//   E_parent_before = T_parent + S_parent
//   E_child = fraction of parent energy
//   E_parent_after = E_parent_before - E_child * (1 + tax)

__global__ void spawnParticlesKernel(
    GPUDisk* disk,
    int N_current,
    int N_max,
    unsigned int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_current || !particle_active(disk, i)) return;

    // Check spawn eligibility
    float history = disk->pump_history[i];
    if (history < SPAWN_COHERENCE_THRESH) return;

    float scale = disk->pump_scale[i];

    // Spawn probability based on coherence and scale
    float spawn_prob = SPAWN_PROB_BASE * (1.0f + (scale - 1.0f) * SPAWN_SCALE_BOOST);
    spawn_prob *= history;  // Higher history = more likely

    // Simple hash-based RNG
    unsigned int hash = seed ^ (i * 2654435761u);
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    float rng = (float)(hash & 0xFFFFFF) / (float)0xFFFFFF;

    if (rng > spawn_prob) return;

    // Try to claim a slot
    unsigned int new_idx = atomicAdd(&d_current_particle_count, 1);
    if (new_idx >= (unsigned int)N_max) {
        atomicSub(&d_current_particle_count, 1);
        return;
    }

    // Get parent state
    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // Calculate parent energy
    float v_sq = vx*vx + vy*vy + vz*vz;
    float E_kinetic = 0.5f * v_sq;
    float E_pump = scale * 0.1f + history * 0.05f;
    float E_parent = E_kinetic + E_pump;

    if (E_parent < 0.1f) {
        atomicSub(&d_current_particle_count, 1);
        return;
    }

    // Energy split: child gets fraction, parent keeps rest minus tax
    float child_fraction = 0.3f;
    float E_child = E_parent * child_fraction;
    float E_tax = E_child * SPAWN_ENERGY_TAX;
    float E_parent_new = E_parent - E_child - E_tax;

    // Ensure parent keeps minimum KE
    if (E_parent_new < SPAWN_MIN_PARENT_KE * E_parent) {
        atomicSub(&d_current_particle_count, 1);
        return;
    }

    // Update parent velocity (reduced by energy loss)
    float velocity_factor = sqrtf(fmaxf(0.0f, E_parent_new / E_parent));
    disk->vel_x[i] = vx * velocity_factor;
    disk->vel_y[i] = vy * velocity_factor;
    disk->vel_z[i] = vz * velocity_factor;
    disk->pump_scale[i] = scale * (1.0f - child_fraction);
    disk->pump_history[i] = history * 0.8f;

    // Small random offset for child position
    float offset_scale = 0.5f;
    hash *= 0x27d4eb2d;
    float ox = ((float)((hash >> 0) & 0xFF) / 127.5f - 1.0f) * offset_scale;
    float oy = ((float)((hash >> 8) & 0xFF) / 127.5f - 1.0f) * offset_scale;
    float oz = ((float)((hash >> 16) & 0xFF) / 127.5f - 1.0f) * offset_scale;

    // Initialize child
    disk->pos_x[new_idx] = px + ox;
    disk->pos_y[new_idx] = py + oy;
    disk->pos_z[new_idx] = pz + oz;

    // Child inherits parent velocity with slight perturbation
    float child_v_scale = sqrtf(E_child / E_kinetic) * 0.8f;
    disk->vel_x[new_idx] = vx * child_v_scale + ox * 0.1f;
    disk->vel_y[new_idx] = vy * child_v_scale + oy * 0.1f;
    disk->vel_z[new_idx] = vz * child_v_scale + oz * 0.1f;

    // Initialize pump state
    disk->pump_state[new_idx] = PUMP_IDLE;
    disk->pump_scale[new_idx] = 1.0f;
    disk->pump_coherent[new_idx] = 0;
    disk->pump_seam[new_idx] = SEAM_CLOSED;
    disk->pump_residual[new_idx] = 0.0f;
    disk->pump_work[new_idx] = 0.0f;
    disk->pump_history[new_idx] = history * 0.5f;
    disk->jet_phase[new_idx] = disk->jet_phase[i];
    disk->flags[new_idx] = PFLAG_ACTIVE;  // active, not ejected

    // Hopfion: child inherits parent topo_state with one axis flipped
    uint8_t parent_topo = disk->topo_state[i];
    uint8_t child_topo = parent_topo;
    if (topo_dim(parent_topo) > 0) {
        unsigned int rng2 = seed * (i + 1);
        int a = (int)(rng2 & 0x03);
        for (int t = 0; t < 4; t++) {
            int ax = (a + t) & 0x03;
            if (topo_get_axis(parent_topo, ax) != 0) {
                child_topo = hopfion_phason_flip(parent_topo, ax);
                break;
            }
        }
    }
    disk->topo_state[new_idx] = child_topo;

    atomicAdd(&d_spawn_count, 1);
}
