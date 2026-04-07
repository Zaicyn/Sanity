// cell_grid_kernels.cuh — Cell Grid Physics Kernels (DNA/RNA Streaming)
// =====================================================================
// Three-pass model for forward-only grid physics:
//   Pass 1: scatterParticlesToCells — particles → cells (atomic accumulation)
//   Pass 2: computeCellFields — cells → cells (6-neighbor stencil)
//   Pass 3: gatherCellForcesToParticles — cells → particles (direct gather)
//
// Also includes clearCellGrid for grid state reset.
//
// Dependencies (must be included before this header):
//   - disk.cuh (GPUDisk, particle_active, BH_MASS, ISCO_R, TWO_PI)
//   - cell_grid.cuh (GRID_HALF_SIZE, CellGrid)
//   - cuda_lut.cuh (cuda_lut_sin, cuda_lut_cos)
//   - forces.cuh (apply_keplerian_substrate_linear)
//   - vram_config.cuh (d_grid_* device constants)
//   - Device functions cellIndexFromPos, cellCoords (in blackhole_v20.cu)
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Clear cell grid state before scatter pass (used for cadence mode)
__global__ void clearCellGrid(
    float* density,
    float* momentum_x,
    float* momentum_y,
    float* momentum_z,
    float* phase_sin,
    float* phase_cos,
    float* pressure_x,
    float* pressure_y,
    float* pressure_z,
    float* vorticity_x,
    float* vorticity_y,
    float* vorticity_z
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= d_grid_cells) return;

    density[cell] = 0.0f;
    momentum_x[cell] = 0.0f;
    momentum_y[cell] = 0.0f;
    momentum_z[cell] = 0.0f;
    phase_sin[cell] = 0.0f;
    phase_cos[cell] = 0.0f;
    pressure_x[cell] = 0.0f;
    pressure_y[cell] = 0.0f;
    pressure_z[cell] = 0.0f;
    vorticity_x[cell] = 0.0f;
    vorticity_y[cell] = 0.0f;
    vorticity_z[cell] = 0.0f;
}

// Pass 1: Scatter particles to cells (forward-only atomic accumulation)
// Each particle computes its cell via O(1) hash and accumulates state
__global__ void scatterParticlesToCells(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip inactive particles (but still mark their cell as invalid)
    if (!particle_active(disk, i)) {
        particle_cell[i] = UINT32_MAX;
        return;
    }

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    // O(1) cell assignment — no Morton sort, no binary search
    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;  // Store for Pass 3

    // Atomic accumulation into cell state
    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    // Phase state for Kuramoto coupling (math.md Step 8)
    // Uses the dedicated theta[] field — a continuous rotation, not a pulse.
    // phase_sin/phase_cos accumulate ∑sin(θ_i) and ∑cos(θ_i) per cell, so
    // that the gather kernel can read the mean-field ⟨e^{iθ}⟩ as R_local
    // and apply Kuramoto coupling K·sin(θ_cell − θ_i).
    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Pass 2: Compute cell fields using fixed 6-neighbor stencil
// Central difference for gradients, curl for vorticity — no binary search
__global__ void computeCellFields(
    const float* __restrict__ density,
    const float* __restrict__ momentum_x,
    const float* __restrict__ momentum_y,
    const float* __restrict__ momentum_z,
    float* __restrict__ pressure_x,
    float* __restrict__ pressure_y,
    float* __restrict__ pressure_z,
    float* __restrict__ vorticity_x,
    float* __restrict__ vorticity_y,
    float* __restrict__ vorticity_z,
    float pressure_k,
    float vorticity_k
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= d_grid_cells) return;

    float rho = density[cell];
    if (rho < 0.5f) return;  // Skip nearly empty cells

    // Extract cell coordinates
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);

    // Normalize accumulated momentum to get average velocity
    float inv_rho = 1.0f / rho;
    // ========================================================================
    // PRESSURE GRADIENT: ∇ρ via central difference
    // ========================================================================
    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
    const float inv_2h = 1.0f / (2.0f * d_grid_cell_size);

    // X gradient: (ρ[x+1] - ρ[x-1]) / 2h
    if (cx > 0 && cx < d_grid_dim - 1) {
        float rho_px = density[cell + 1];
        float rho_mx = density[cell - 1];
        grad_x = (rho_px - rho_mx) * inv_2h;
    }

    // Y gradient
    if (cy > 0 && cy < d_grid_dim - 1) {
        float rho_py = density[cell + d_grid_stride_y];
        float rho_my = density[cell - d_grid_stride_y];
        grad_y = (rho_py - rho_my) * inv_2h;
    }

    // Z gradient
    if (cz > 0 && cz < d_grid_dim - 1) {
        float rho_pz = density[cell + d_grid_stride_z];
        float rho_mz = density[cell - d_grid_stride_z];
        grad_z = (rho_pz - rho_mz) * inv_2h;
    }

    // Pressure force = -k * ∇ρ / ρ (pushes toward lower density)
    float pressure_scale = -pressure_k * inv_rho;
    pressure_x[cell] = grad_x * pressure_scale;
    pressure_y[cell] = grad_y * pressure_scale;
    pressure_z[cell] = grad_z * pressure_scale;

    // ========================================================================
    // VORTICITY: ω = ∇ × v (curl of velocity field)
    // ========================================================================
    // ω_x = ∂vz/∂y - ∂vy/∂z
    // ω_y = ∂vx/∂z - ∂vz/∂x
    // ω_z = ∂vy/∂x - ∂vx/∂y

    float dvz_dy = 0.0f, dvy_dz = 0.0f;
    float dvx_dz = 0.0f, dvz_dx = 0.0f;
    float dvy_dx = 0.0f, dvx_dy = 0.0f;

    // Velocity derivatives via central difference on neighbor cells
    // Note: we read accumulated momentum and divide by neighbor density

    if (cy > 0 && cy < d_grid_dim - 1) {
        uint32_t cell_py = cell + d_grid_stride_y;
        uint32_t cell_my = cell - d_grid_stride_y;
        float rho_py = density[cell_py];
        float rho_my = density[cell_my];
        if (rho_py > 0.5f && rho_my > 0.5f) {
            float vz_py = momentum_z[cell_py] / rho_py;
            float vz_my = momentum_z[cell_my] / rho_my;
            dvz_dy = (vz_py - vz_my) * inv_2h;
        }
    }

    if (cz > 0 && cz < d_grid_dim - 1) {
        uint32_t cell_pz = cell + d_grid_stride_z;
        uint32_t cell_mz = cell - d_grid_stride_z;
        float rho_pz = density[cell_pz];
        float rho_mz = density[cell_mz];
        if (rho_pz > 0.5f && rho_mz > 0.5f) {
            float vy_pz = momentum_y[cell_pz] / rho_pz;
            float vy_mz = momentum_y[cell_mz] / rho_mz;
            dvy_dz = (vy_pz - vy_mz) * inv_2h;

            float vx_pz = momentum_x[cell_pz] / rho_pz;
            float vx_mz = momentum_x[cell_mz] / rho_mz;
            dvx_dz = (vx_pz - vx_mz) * inv_2h;
        }
    }

    if (cx > 0 && cx < d_grid_dim - 1) {
        uint32_t cell_px = cell + 1;
        uint32_t cell_mx = cell - 1;
        float rho_px = density[cell_px];
        float rho_mx = density[cell_mx];
        if (rho_px > 0.5f && rho_mx > 0.5f) {
            float vz_px = momentum_z[cell_px] / rho_px;
            float vz_mx = momentum_z[cell_mx] / rho_mx;
            dvz_dx = (vz_px - vz_mx) * inv_2h;

            float vy_px = momentum_y[cell_px] / rho_px;
            float vy_mx = momentum_y[cell_mx] / rho_mx;
            dvy_dx = (vy_px - vy_mx) * inv_2h;

            float vx_px = momentum_x[cell_px] / rho_px;
            float vx_mx = momentum_x[cell_mx] / rho_mx;
            dvx_dy = (vx_px - vx_mx) * inv_2h;  // Reusing for symmetry
        }
    }

    // Compute curl components
    float omega_x = dvz_dy - dvy_dz;
    float omega_y = dvx_dz - dvz_dx;
    float omega_z = dvy_dx - dvx_dy;

    // Scale vorticity force
    vorticity_x[cell] = omega_x * vorticity_k;
    vorticity_y[cell] = omega_y * vorticity_k;
    vorticity_z[cell] = omega_z * vorticity_k;
}

// Pass 3: Gather cell forces to particles (direct O(1) lookup)
// Each particle reads its cell's pressure/vorticity and updates velocity
__global__ void gatherCellForcesToParticles(
    GPUDisk* __restrict__ disk,
    const float* __restrict__ density,
    const float* __restrict__ pressure_x,
    const float* __restrict__ pressure_y,
    const float* __restrict__ pressure_z,
    const float* __restrict__ vorticity_x,
    const float* __restrict__ vorticity_y,
    const float* __restrict__ vorticity_z,
    const float* __restrict__ phase_sin,
    const float* __restrict__ phase_cos,
    const uint32_t* __restrict__ particle_cell,
    const uint8_t* __restrict__ in_active_region,  // Step 3: skip passive particles (may be nullptr if disabled)
    uint32_t N,
    float dt,
    float substrate_k,  // Keplerian substrate coupling (competes with Kuramoto)
    float shear_k,      // Phase-misalignment shear (non-monotonic magnetic friction)
    float rho_ref,      // Reference density for shear normalization (mean × 8)
    float kuramoto_k,   // Kuramoto phase coupling strength (0 = free-running only)
    int   use_n12,      // 1 = apply N12 mixer envelope to coupling, 0 = constant K
    float envelope_scale // Harmonic index multiplier: cos(3s·θ)·cos(4s·θ). s=1 is N12 baseline.
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t cell = particle_cell[i];
    if (cell == UINT32_MAX) return;  // Inactive particle

    // Step 3: passive particles get their physics from advectPassiveParticles.
    // Applying pressure/vorticity/Kuramoto here would corrupt their velocity
    // (passive kernel advances pos azimuthally but doesn't rewrite vel_x/vel_z,
    // so gather increments to stale velocity would cause unbounded divergence).
    if (in_active_region && !in_active_region[i]) return;

    // O(1) direct read — no binary search!
    float press_x = pressure_x[cell];
    float press_y = pressure_y[cell];
    float press_z = pressure_z[cell];
    float ox = vorticity_x[cell];
    float oy = vorticity_y[cell];
    float oz = vorticity_z[cell];

    // Particle position (for substrate torque)
    float pos_x = disk->pos_x[i];
    float pos_z = disk->pos_z[i];

    // Current velocity
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    // Apply pressure force
    vx += press_x * dt;
    vy += press_y * dt;
    vz += press_z * dt;

    // Apply vorticity force: F_ω = ω × v (cross product)
    // rsqrtf pattern: get both omega_mag and inv_omega from single SFU call
    float omega_sq = ox*ox + oy*oy + oz*oz;
    if (omega_sq > 1e-8f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float nx = ox * inv_omega;
        float ny = oy * inv_omega;
        float nz = oz * inv_omega;

        // Cross product: omega_hat × v
        float cross_x = ny * vz - nz * vy;
        float cross_y = nz * vx - nx * vz;
        float cross_z = nx * vy - ny * vx;

        vx += cross_x * omega_mag * dt;
        vy += cross_y * omega_mag * dt;
        vz += cross_z * omega_mag * dt;
    }

    // === FRICTIONAL SHEAR — DENSITY × ANGULAR HYBRID ===
    // Gu/Lüders/Bechinger arXiv:2602.11526v1 continuum analog.
    //   effective_k = shear_k × rho_factor × angular_profile
    //
    // rho_factor ∈ [0, 2]: density-weighting (Gemini) — only particles inside
    //   dense shell regions feel friction; inter-shell vacuum stays superfluid.
    //   Scale-invariant: rho_ref is passed from host as mean_density × 8.
    //
    // angular_profile = |sin(2φ)| where φ = atan2(|v_θ − v_Kep|, |v_r|)
    //   (Deepseek's non-monotonic profile) — friction vanishes for pure
    //   circular orbits (φ=0) AND pure radial infall (φ=90°), peaks at 45°
    //   where motion is mixed. This recovers the paper's hysteretic
    //   non-monotonic behavior using a signal that stays nonzero in steady
    //   state: v_r and v_θ − v_Kep are both small but nonzero everywhere.
    //
    // Combined: friction only inside shells AND only where motion is shear-mixed.
    if (shear_k > 0.0f) {
        float rho_cell = density[cell];
        float rho_factor = rho_cell / rho_ref;
        if (rho_factor > 2.0f) rho_factor = 2.0f;

        if (rho_factor > 0.0f) {
            float r2 = pos_x * pos_x + pos_z * pos_z + 1e-6f;
            float inv_r = rsqrtf(r2);
            float r_cyl = r2 * inv_r;
            float rx_hat = pos_x * inv_r;
            float rz_hat = pos_z * inv_r;
            float tx_kep = -rz_hat;  // prograde tangent in XZ
            float tz_kep =  rx_hat;

            float v_theta = vx * tx_kep + vz * tz_kep;
            float v_r = vx * rx_hat + vz * rz_hat;

            // Keplerian orbital speed at this radius: v_K = √(M/r)
            float v_kep = (r_cyl > ISCO_R * 0.5f) ? sqrtf(BH_MASS * inv_r) : 0.0f;
            float dv_tan = fabsf(v_theta - v_kep);
            float abs_vr = fabsf(v_r);

            // Angular profile: sin(2φ) = 2 sin(φ) cos(φ) = 2·dv_tan·|v_r|/(dv_tan² + v_r²)
            // Computed directly from components, no atan2 needed.
            float denom = dv_tan * dv_tan + abs_vr * abs_vr + 1e-8f;
            float angular_profile = 2.0f * dv_tan * abs_vr / denom;  // ∈ [0, 1]

            float drag = shear_k * rho_factor * angular_profile * dt;
            if (drag > 0.5f) drag = 0.5f;  // stability clamp
            float dv = v_theta * drag;
            vx -= dv * tx_kep;
            vz -= dv * tz_kep;
        }
    }

    // === KEPLERIAN SUBSTRATE TORQUE (Competing Interaction) ===
    // Based on magnetic friction paper: friction peaks when competing
    // interactions FRUSTRATE the system. Kuramoto wants phase alignment,
    // substrate wants Keplerian differential rotation. When balanced,
    // neither wins → hysteresis → dissipation → interesting dynamics.
    if (substrate_k > 0.0f) {
        // Use linear version for stability, can switch to sinusoidal later
        apply_keplerian_substrate_linear(pos_x, pos_z, vx, vz, substrate_k);
    }

    // Damping (same as octree path)
    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    disk->vel_x[i] = vx;
    disk->vel_y[i] = vy;
    disk->vel_z[i] = vz;

    // === KURAMOTO PHASE UPDATE (math.md Step 8, Step 10, Step 11) ===
    // Classical Kuramoto mean-field coupling via the grid phase_sin/cos:
    //   dθ/dt = ω_i + K · envelope · R_local · sin(θ_cell − θ_i)
    // The R_local · sin(Δθ) factor comes for free from the cell-averaged
    // phase sums because (mean_sin · cos θ_i − mean_cos · sin θ_i) already
    // has magnitude R_local = |⟨e^{iθ}⟩|. The N12 envelope (math.md Step 11,
    // period LCM(3,4) = 12) modulates coupling strength.
    {
        float theta_i = disk->theta[i];
        float omega_i = disk->omega_nat[i];
        float dtheta = omega_i;

        if (kuramoto_k > 0.0f) {
            float rho_cell = density[cell];
            if (rho_cell > 0.5f) {
                float ps = phase_sin[cell];
                float pc = phase_cos[cell];
                float inv_rho = 1.0f / rho_cell;
                float mean_sin = ps * inv_rho;  // ⟨sin θ⟩ over cell
                float mean_cos = pc * inv_rho;  // ⟨cos θ⟩ over cell
                // coupling = R_local · sin(θ_cell − θ_i)
                float sin_i = cuda_lut_sin(theta_i);
                float cos_i = cuda_lut_cos(theta_i);
                float coupling = mean_sin * cos_i - mean_cos * sin_i;

                // N12 envelope: 0.5 + 0.5·cos(3θ)·cos(4θ), period LCM(3,4)=12
                float envelope = 1.0f;
                if (use_n12) {
                    // envelope_scale controls the harmonic indices; s=1 is N12 baseline.
                    // s=2 → period halves → predicts optimal ω doubles (GPT test).
                    float c3 = cuda_lut_cos(3.0f * envelope_scale * theta_i);
                    float c4 = cuda_lut_cos(4.0f * envelope_scale * theta_i);
                    envelope = 0.5f + 0.5f * c3 * c4;
                }

                dtheta += kuramoto_k * envelope * coupling;
            }
        }

        // Advance and wrap to [0, 2π)
        theta_i += dtheta * dt;
        theta_i = fmodf(theta_i, TWO_PI);
        if (theta_i < 0.0f) theta_i += TWO_PI;
        disk->theta[i] = theta_i;
    }
}

// End of cell_grid_kernels.cuh
