// spawn.cuh — Natural Growth Spawning + Entropy Injection Kernels
// ==============================================================
// - spawnParticlesKernel: Energy-conserving star formation
// - injectEntropyCluster: High-entropy test cluster injection
//
// Dependencies: disk.cuh, forces.cuh, cuda_lut.cuh
#pragma once
#include <cuda_runtime.h>

// ============================================================================
// Natural Growth Spawning Kernel - ENERGY-CONSERVING Star Formation
// ============================================================================
// Models gravitational collapse in coherent gas. When pump_history is high
// (indicating sustained pumping activity), the region is gravitationally bound
// and can form new stars.
//
// ENERGY CONSERVATION (per GPT's audit):
// Total energy E = T + S where:
//   T = 0.5 * m * v² (kinetic)
//   S = α * pump_scale + β * pump_history (pump/phase energy)
//
// When spawning:
//   E_parent_before = T_parent + S_parent
//   E_child = fraction of parent energy (split ratio)
//   E_parent_after = E_parent_before - E_child * (1 + tax)
//
// Parent velocity reduced: v_new = v_old * sqrt(1 - ε)
// Parent pump_scale reduced: scale_new = scale_old * (1 - ε)
// where ε = E_child / E_parent (including tax)

__global__ void spawnParticlesKernel(
    GPUDisk* disk,
    int N_current,           // Current active particle count
    int N_max,               // Maximum allowed particles
    unsigned int* spawn_idx, // Atomic counter for spawn slot allocation
    unsigned int* spawn_success, // V8-style: counts SUCCESSFUL spawns only
    float time,
    unsigned int seed,       // Per-frame random seed
    const uint8_t* __restrict__ in_active_region  // Step 3: skip passive parents
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_current || !particle_active(disk, i)) return;
    // Step 3d: passive parents don't spawn. Their pump_history is frozen
    // (siphon doesn't update them), so a stale high history could trigger
    // spawning from a settled particle. Also avoids a vel write race:
    // spawn reduces parent vel on spawn_stream while passive kernel writes
    // vel_y on default stream.
    if (in_active_region && !in_active_region[i]) return;

    // Only coherent particles can spawn (sustained pumping = gravitationally bound)
    float history = disk->pump_history[i];
    if (history < SPAWN_COHERENCE_THRESH) return;

    // Spawn probability scales with pump_scale (high D = more accretion = more spawn)
    float scale = disk->pump_scale[i];
    float spawn_prob = SPAWN_PROB_BASE * (1.0f + scale * SPAWN_SCALE_BOOST);

    // Simple LCG random for spawn decision (per-particle, per-frame)
    unsigned int rng = seed ^ (i * 1664525u + 1013904223u);
    rng = rng * 1664525u + 1013904223u;
    float rand_val = (float)(rng & 0xFFFFFF) / 16777216.0f;

    if (rand_val > spawn_prob) return;

    // === ENERGY BUDGET CHECK ===
    // Compute parent's current energy
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];
    float v2_parent = vx*vx + vy*vy + vz*vz;
    float T_parent = 0.5f * v2_parent;  // Kinetic (m=1)

    // Pump energy: α * scale + β * history (using α=0.1, β=0.05)
    float S_parent = 0.1f * scale + 0.05f * history;
    float E_parent = T_parent + S_parent;

    // Child gets 20% of parent's energy (after tax)
    float split_ratio = 0.2f;
    float E_child_raw = E_parent * split_ratio;
    float E_removed = E_child_raw * (1.0f + SPAWN_ENERGY_TAX);  // Include 10% tax

    // Check if parent has enough energy to spawn
    // Parent must retain at least SPAWN_MIN_PARENT_KE of original KE
    float E_parent_after = E_parent - E_removed;
    if (E_parent_after < E_parent * SPAWN_MIN_PARENT_KE) return;

    // === V8-STYLE SLOT ALLOCATION ===
    // Pre-check capacity before atomic increment (reduces contention)
    int available = N_max - N_current;
    if (available <= 0) return;

    // Allocate spawn slot atomically
    unsigned int slot = atomicAdd(spawn_idx, 1);

    // V8 pattern: reject if slot exceeds available capacity
    // This prevents phantom particles from being counted
    if (slot >= (unsigned int)available) return;

    int new_idx = N_current + slot;

    // === ENERGY-CONSERVING VELOCITY SPLIT ===
    // Parent loses kinetic energy: v_new = v_old * sqrt(1 - ε_kinetic)
    // where ε_kinetic = fraction of KE going to child
    float ke_fraction = (E_removed * 0.7f) / fmaxf(T_parent, 0.001f);  // 70% from KE
    ke_fraction = fminf(ke_fraction, 0.5f);  // Cap at 50% KE loss
    float vel_factor = sqrtf(1.0f - ke_fraction);

    // Reduce parent velocity (ENERGY COST)
    disk->vel_x[i] *= vel_factor;
    disk->vel_y[i] *= vel_factor;
    disk->vel_z[i] *= vel_factor;

    // Reduce parent pump_scale (ENERGY COST from pump energy)
    float pump_fraction = (E_removed * 0.3f) / fmaxf(S_parent, 0.001f);  // 30% from pump
    pump_fraction = fminf(pump_fraction, 0.3f);  // Cap at 30% pump loss
    disk->pump_scale[i] *= (1.0f - pump_fraction);
    disk->pump_history[i] *= (1.0f - pump_fraction * 0.5f);  // History decays slower

    // Generate small position offset (so offspring doesn't overlap parent)
    rng = rng * 1664525u + 1013904223u;
    float dx = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    rng = rng * 1664525u + 1013904223u;
    float dy = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.5f;
    rng = rng * 1664525u + 1013904223u;
    float dz = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;

    // Initialize offspring position
    disk->pos_x[new_idx] = disk->pos_x[i] + dx;
    disk->pos_y[new_idx] = disk->pos_y[i] + dy;
    disk->pos_z[new_idx] = disk->pos_z[i] + dz;

    // Child velocity: fraction of parent's (reduced) velocity with perturbation
    // Child gets sqrt(E_child_kinetic) worth of velocity
    float child_vel_scale = sqrtf(E_child_raw * 0.7f / fmaxf(T_parent, 0.001f));
    child_vel_scale = fminf(child_vel_scale, 0.5f);  // Cap at 50% of parent velocity

    rng = rng * 1664525u + 1013904223u;
    float vx_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;
    rng = rng * 1664525u + 1013904223u;
    float vy_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;
    rng = rng * 1664525u + 1013904223u;
    float vz_pert = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 0.05f;

    // Child velocity from parent's ORIGINAL velocity (before reduction)
    disk->vel_x[new_idx] = vx * child_vel_scale + vx_pert;
    disk->vel_y[new_idx] = vy * child_vel_scale + vy_pert;
    disk->vel_z[new_idx] = vz * child_vel_scale + vz_pert;

    // NOTE: disk_r, disk_phi, temp, in_disk no longer stored — computed on-demand

    // Child pump state: fraction of parent's energy goes to pump
    float child_pump_scale = scale * sqrtf(E_child_raw * 0.3f / fmaxf(S_parent, 0.001f));
    child_pump_scale = fminf(child_pump_scale, scale * 0.5f);  // Cap at 50% of parent scale

    disk->pump_state[new_idx] = 0;  // IDLE
    disk->pump_scale[new_idx] = fmaxf(child_pump_scale, 0.5f);  // Min 0.5
    disk->pump_coherent[new_idx] = 0;
    disk->pump_seam[new_idx] = 0;
    disk->pump_residual[new_idx] = 0.0f;
    disk->pump_work[new_idx] = 0.0f;
    disk->pump_history[new_idx] = history * 0.3f;  // Inherit 30% of parent's history
    disk->jet_phase[new_idx] = disk->jet_phase[i];  // Inherit parent's phase coherence

    // Kuramoto state: child inherits parent's theta + random offset, same ω
    // (keeps children loosely phase-coupled to parent, lets drift decorrelate)
    rng = rng * 1664525u + 1013904223u;
    float theta_offset = ((float)(rng & 0xFFFF) / 65536.0f) * 6.28318f;
    float parent_theta = disk->theta[i];
    disk->theta[new_idx] = fmodf(parent_theta + theta_offset, 6.28318f);
    disk->omega_nat[new_idx] = disk->omega_nat[i];

    // Activate the new particle
    disk->flags[new_idx] = PFLAG_ACTIVE;  // active, not ejected

    // V8-style: only count SUCCESSFUL spawns (after all writes complete)
    // This ensures spawn_success == actual initialized particles
    atomicAdd(spawn_success, 1);
}

// Inject high-entropy star cluster to test thread coherence breakdown
__global__ void injectEntropyCluster(GPUDisk* disk, int N, float sim_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Only inject 10,000 particles starting from a high index
    int inject_start = N - 10000;
    if (idx < inject_start) return;

    // Spawn particles in a spherical cluster at r ~ 200 (outside main disk)
    // Random positions on sphere
    unsigned int seed = idx + (unsigned int)(sim_time * 1000.0f);
    float theta = 2.0f * 3.14159f * (float)(seed % 1000) / 1000.0f;
    float phi = acosf(2.0f * (float)((seed / 1000) % 1000) / 1000.0f - 1.0f);

    float r_inject = 200.0f;
    float x = r_inject * sinf(phi) * cosf(theta);
    float y = r_inject * sinf(phi) * sinf(theta);
    float z = r_inject * cosf(phi);

    // Random inward velocities (thermal + infall)
    float v_thermal = 0.3f;
    float v_infall = -0.5f;  // Aimed at core

    disk->pos_x[idx] = x;
    disk->pos_y[idx] = y;
    disk->pos_z[idx] = z;
    disk->vel_x[idx] = v_infall * x / r_inject + v_thermal * (2.0f * (float)((seed * 7) % 1000) / 1000.0f - 1.0f);
    disk->vel_y[idx] = v_infall * y / r_inject + v_thermal * (2.0f * (float)((seed * 11) % 1000) / 1000.0f - 1.0f);
    disk->vel_z[idx] = v_infall * z / r_inject + v_thermal * (2.0f * (float)((seed * 13) % 1000) / 1000.0f - 1.0f);

    // High entropy state: random pump states, high residual
    disk->pump_state[idx] = seed % 4;  // Random IDLE/UP/DOWN/FULL
    disk->pump_scale[idx] = 0.5f + 1.5f * (float)((seed * 17) % 1000) / 1000.0f;  // 0.5-2.0 (chaotic)
    disk->pump_residual[idx] = 0.8f + 0.4f * (float)((seed * 19) % 1000) / 1000.0f;  // High stress
    disk->pump_work[idx] = 0.0f;
    disk->pump_history[idx] = 0.5f;  // Incoherent
    disk->pump_coherent[idx] = 0;
    disk->pump_seam[idx] = 0x00;  // Closed (will be forced open by stress)

    disk->flags[idx] = PFLAG_ACTIVE;  // active, not ejected
    // NOTE: disk_r, disk_phi, temp, in_disk no longer stored — computed on-demand
}

// End of spawn.cuh
