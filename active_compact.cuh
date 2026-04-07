// active_compact.cuh — Active Particle Compaction + Sparse Tile Flags
// ====================================================================
// Active particle compaction (scatter-skip optimization):
//   - computeParticleActivityMask, compactActiveParticles
//   - scatterActiveParticles, scatterStaticParticles
//   - gatherToActiveParticles, copyCurrentToPrevCell, copyStaticToWorkingGrid
//
// Hierarchical tiled flags (methylation pattern):
//   - scatterWithTileFlags, compactActiveTiles, compactCellsInTiles
//   - sparseClearTileAndCellFlags, decayAndComputePressure
//
// Dependencies: disk.cuh, cell_grid.cuh, vram_config.cuh, cuda_lut.cuh, forces.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

__global__ void computeParticleActivityMask(
    const GPUDisk* __restrict__ disk,
    const uint32_t* __restrict__ curr_cell,
    const uint32_t* __restrict__ prev_cell,
    uint8_t* __restrict__ active_mask,
    uint32_t N,
    float velocity_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (!particle_active(disk, i)) {
        active_mask[i] = 0;
        return;
    }

    // Check 1: Did cell change?
    bool cell_changed = (curr_cell[i] != prev_cell[i]);

    // Check 2: Is velocity above threshold?
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];
    float v2 = vx*vx + vy*vy + vz*vz;
    bool moving = (v2 > velocity_threshold * velocity_threshold);

    active_mask[i] = (cell_changed || moving) ? 1 : 0;
}

// Compact active particles using atomic counter (simple but effective)
// For N=10M with ~10% active, atomics are fine (not a bottleneck)
__global__ void compactActiveParticles(
    const uint8_t* __restrict__ active_mask,
    uint32_t* __restrict__ active_list,
    uint32_t* __restrict__ active_count,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (active_mask[i]) {
        uint32_t idx = atomicAdd(active_count, 1);
        active_list[idx] = i;
    }
}

// Scatter ONLY active particles (uses compacted list)
__global__ void scatterActiveParticles(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    const uint32_t* __restrict__ active_list,
    uint32_t active_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int i = active_list[idx];  // Get actual particle index

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;

    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Scatter STATIC particles to bake grid (called once when lock engages)
__global__ void scatterStaticParticles(
    const GPUDisk* __restrict__ disk,
    const uint8_t* __restrict__ active_mask,  // Inverted: scatter where mask=0
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

    if (!particle_active(disk, i) || active_mask[i]) return;  // Skip inactive or active particles

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];

    uint32_t cell = cellIndexFromPos(px, py, pz);
    particle_cell[i] = cell;

    atomicAdd(&density[cell], 1.0f);
    atomicAdd(&momentum_x[cell], disk->vel_x[i]);
    atomicAdd(&momentum_y[cell], disk->vel_y[i]);
    atomicAdd(&momentum_z[cell], disk->vel_z[i]);

    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase));
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase));
}

// Gather forces ONLY to active particles
__global__ void gatherToActiveParticles(
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
    const uint32_t* __restrict__ active_list,
    uint32_t active_count,
    float dt,
    float substrate_k,
    float shear_k,
    float rho_ref,
    float kuramoto_k,
    int   use_n12,
    float envelope_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;

    int i = active_list[idx];
    uint32_t cell = particle_cell[i];
    if (cell == UINT32_MAX) return;

    float press_x = pressure_x[cell];
    float press_y = pressure_y[cell];
    float press_z = pressure_z[cell];
    float ox = vorticity_x[cell];
    float oy = vorticity_y[cell];
    float oz = vorticity_z[cell];

    float pos_x = disk->pos_x[i];
    float pos_z = disk->pos_z[i];
    float vx = disk->vel_x[i];
    float vy = disk->vel_y[i];
    float vz = disk->vel_z[i];

    vx += press_x * dt;
    vy += press_y * dt;
    vz += press_z * dt;

    float omega_sq = ox*ox + oy*oy + oz*oz;
    if (omega_sq > 1e-8f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float nx = ox * inv_omega;
        float ny = oy * inv_omega;
        float nz = oz * inv_omega;
        float cross_x = ny * vz - nz * vy;
        float cross_y = nz * vx - nx * vz;
        float cross_z = nx * vy - ny * vx;
        vx += cross_x * omega_mag * dt;
        vy += cross_y * omega_mag * dt;
        vz += cross_z * omega_mag * dt;
    }

    // === FRICTIONAL SHEAR — DENSITY × ANGULAR HYBRID (see gatherCellForcesToParticles) ===
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
            float tx_kep = -rz_hat;
            float tz_kep =  rx_hat;
            float v_theta = vx * tx_kep + vz * tz_kep;
            float v_r = vx * rx_hat + vz * rz_hat;
            float v_kep = (r_cyl > ISCO_R * 0.5f) ? sqrtf(BH_MASS * inv_r) : 0.0f;
            float dv_tan = fabsf(v_theta - v_kep);
            float abs_vr = fabsf(v_r);
            float denom = dv_tan * dv_tan + abs_vr * abs_vr + 1e-8f;
            float angular_profile = 2.0f * dv_tan * abs_vr / denom;
            float drag = shear_k * rho_factor * angular_profile * dt;
            if (drag > 0.5f) drag = 0.5f;
            float dv = v_theta * drag;
            vx -= dv * tx_kep;
            vz -= dv * tz_kep;
        }
    }

    if (substrate_k > 0.0f) {
        apply_keplerian_substrate_linear(pos_x, pos_z, vx, vz, substrate_k);
    }

    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    disk->vel_x[i] = vx;
    disk->vel_y[i] = vy;
    disk->vel_z[i] = vz;

    // === KURAMOTO PHASE UPDATE (see gatherCellForcesToParticles) ===
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
                float mean_sin = ps * inv_rho;
                float mean_cos = pc * inv_rho;
                float sin_i = cuda_lut_sin(theta_i);
                float cos_i = cuda_lut_cos(theta_i);
                float coupling = mean_sin * cos_i - mean_cos * sin_i;
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

        theta_i += dtheta * dt;
        theta_i = fmodf(theta_i, TWO_PI);
        if (theta_i < 0.0f) theta_i += TWO_PI;
        disk->theta[i] = theta_i;
    }
}

// Copy previous cell indices for next frame's activity detection
__global__ void copyCurrentToPrevCell(
    const uint32_t* __restrict__ curr_cell,
    uint32_t* __restrict__ prev_cell,
    uint32_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    prev_cell[i] = curr_cell[i];
}

// Merge static grid + active scatter into working grid
// dst = static_base (copy first, then active particles atomicAdd on top)
__global__ void copyStaticToWorkingGrid(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[i];
}

// ============================================================================
// FLAGS + COMPACTION KERNELS — O(n) Transcription Pattern
// ============================================================================
// CUB DeviceSelect::Flagged for O(n) compaction (parallel prefix sum, no atomics)
// Sparse clear for O(active_count) flag reset instead of O(GRID_CELLS) memset

// ============================================================================
// HIERARCHICAL TILED FLAGS — "Methylation" Pattern
// ============================================================================
// Two-level flags: tile flags (4096) + cell flags within active tiles
// Scatter marks both tile AND cell, compaction scans tiles then cells

// Scatter particles to cells AND mark tile flags — O(particles)
__global__ void scatterWithTileFlags(
    const GPUDisk* __restrict__ disk,
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    uint32_t* __restrict__ particle_cell,
    uint8_t* __restrict__ cell_flags,
    uint8_t* __restrict__ tile_flags,
    uint32_t N, float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (!particle_active(disk, i)) {
        particle_cell[i] = UINT32_MAX;
        return;
    }

    // Compute cell index
    uint32_t cell = cellIndexFromPos(disk->pos_x[i], disk->pos_y[i], disk->pos_z[i]);
    particle_cell[i] = cell;

    // Atomically accumulate to cell
    atomicAdd(&density[cell], alpha);
    atomicAdd(&momentum_x[cell], disk->vel_x[i] * alpha);
    atomicAdd(&momentum_y[cell], disk->vel_y[i] * alpha);
    atomicAdd(&momentum_z[cell], disk->vel_z[i] * alpha);

    // Phase: encode as sin/cos for proper averaging
    float phase = disk->theta[i];
    atomicAdd(&phase_sin[cell], cuda_lut_sin(phase) * alpha);
    atomicAdd(&phase_cos[cell], cuda_lut_cos(phase) * alpha);

    // Mark cell flag (duplicates collapse to same value)
    cell_flags[cell] = FLAG_INITIAL_VALUE;

    // Mark tile flag (coarse level)
    uint32_t tile = cellToTile(cell);
    tile_flags[tile] = FLAG_INITIAL_VALUE;
}

// Compact active tiles — O(NUM_TILES) = O(4096), much smaller than O(2M)
__global__ void compactActiveTiles(
    const uint8_t* __restrict__ tile_flags,
    uint32_t* __restrict__ active_tiles,
    uint32_t* __restrict__ tile_count
) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= NUM_TILES) return;

    if (tile_flags[tile] > 0) {
        uint32_t idx = atomicAdd(tile_count, 1);
        active_tiles[idx] = tile;
    }
}

// Compact cells within active tiles — O(active_tiles × 512)
// This is the key optimization: only scan cells in active tiles
__global__ void compactCellsInTiles(
    const uint8_t* __restrict__ cell_flags,
    const uint32_t* __restrict__ active_tiles,
    uint32_t num_active_tiles,
    uint32_t* __restrict__ active_cells,
    uint32_t* __restrict__ cell_count
) {
    // Each block processes one tile
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_active_tiles) return;

    uint32_t tile = active_tiles[tile_idx];

    // Compute tile origin (use runtime tiles_per_dim, not compile-time TILES_PER_DIM)
    int tiles_per_dim = d_grid_dim / TILE_DIM;
    uint32_t tx = tile % tiles_per_dim;
    uint32_t ty = (tile / tiles_per_dim) % tiles_per_dim;
    uint32_t tz = tile / (tiles_per_dim * tiles_per_dim);

    // Each thread in block handles one cell within the tile
    int local_cell = threadIdx.x;
    if (local_cell >= CELLS_PER_TILE) return;

    // Convert local cell to global cell index
    int lx = local_cell % TILE_DIM;
    int ly = (local_cell / TILE_DIM) % TILE_DIM;
    int lz = local_cell / (TILE_DIM * TILE_DIM);

    uint32_t cx = tx * TILE_DIM + lx;
    uint32_t cy = ty * TILE_DIM + ly;
    uint32_t cz = tz * TILE_DIM + lz;
    uint32_t cell = cx + cy * d_grid_stride_y + cz * d_grid_stride_z;

    if (cell_flags[cell] > 0) {
        uint32_t idx = atomicAdd(cell_count, 1);
        active_cells[idx] = cell;
    }
}

// Clear tile and cell flags for active cells — O(active_cells)
__global__ void sparseClearTileAndCellFlags(
    uint8_t* __restrict__ cell_flags,
    uint8_t* __restrict__ tile_flags,
    const uint32_t* __restrict__ active_cells,
    uint32_t num_active_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active_cells) return;

    uint32_t cell = active_cells[i];
    cell_flags[cell] = 0;

    // Also clear tile flag (may write multiple times, that's fine)
    uint32_t tile = cellToTile(cell);
    tile_flags[tile] = 0;
}

// Decay and compute pressure for active cells only — O(active_count), ~4k cells
// No flag propagation: particles scatter to new cells each frame naturally
__global__ void decayAndComputePressure(
    float* __restrict__ density,
    float* __restrict__ momentum_x,
    float* __restrict__ momentum_y,
    float* __restrict__ momentum_z,
    float* __restrict__ phase_sin,
    float* __restrict__ phase_cos,
    float* __restrict__ pressure_x,
    float* __restrict__ pressure_y,
    float* __restrict__ pressure_z,
    const uint32_t* __restrict__ active_list,
    uint32_t active_count,
    float decay,
    float pressure_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active_count) return;

    uint32_t cell = active_list[i];

    // Decay accumulated state
    density[cell] *= decay;
    momentum_x[cell] *= decay;
    momentum_y[cell] *= decay;
    momentum_z[cell] *= decay;
    phase_sin[cell] *= decay;
    phase_cos[cell] *= decay;

    // Reset pressure (will be computed below)
    pressure_x[cell] = 0.0f;
    pressure_y[cell] = 0.0f;
    pressure_z[cell] = 0.0f;

    float my_rho = density[cell];
    if (my_rho < 0.1f) return;

    // Extract cell coordinates for neighbor lookup
    uint32_t cx, cy, cz;
    cellCoords(cell, &cx, &cy, &cz);

    // Shell modulation: compute distance from grid center
    // Viviani period-4 interference: constructive at 0,4,8..., destructive at 2,6,10...
    float half_dim = (float)d_grid_dim * 0.5f;
    float rx = ((float)cx - half_dim) * d_grid_cell_size;
    float ry = ((float)cy - half_dim) * d_grid_cell_size;
    float rz = ((float)cz - half_dim) * d_grid_cell_size;
    float r = sqrtf(rx*rx + ry*ry + rz*rz);
    uint32_t shell = cuda_lut_shell_index(r, LAMBDA_OCTREE);
    float shell_mod = cuda_lut_shell_factor(shell);

    // 6-neighbor pressure computation (face neighbors only)
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int n = 0; n < 6; n++) {
        int nx = (int)cx + dx[n];
        int ny = (int)cy + dy[n];
        int nz = (int)cz + dz[n];

        if (nx < 0 || nx >= d_grid_dim || ny < 0 || ny >= d_grid_dim || nz < 0 || nz >= d_grid_dim) continue;

        uint32_t neighbor = nx + ny * d_grid_stride_y + nz * d_grid_stride_z;
        float neighbor_rho = density[neighbor];

        // Compute gradient and pressure force with shell modulation
        // Shell factor: 1.0 at constructive peaks, 0.2 at destructive troughs
        float gradient = (neighbor_rho - my_rho) / d_grid_cell_size;
        float force = -gradient * pressure_k * shell_mod / (my_rho + 0.01f);

        atomicAdd(&pressure_x[cell], force * (float)dx[n]);
        atomicAdd(&pressure_y[cell], force * (float)dy[n]);
        atomicAdd(&pressure_z[cell], force * (float)dz[n]);
    }
}

// ============================================================================
// PRESSURE + VORTICITY FORCE KERNEL
// ============================================================================
// Applies three forces:
//   1. Pressure: F_p = -k_p ∇ρ  (pushes toward lower density)
//   2. Vorticity: F_ω = k_ω (ω × v)  (induces rotation/spiral structure)
//   3. Phase coherence: modulates pressure by neighbor phase alignment
//
// Together these create a self-organizing medium with radial balance,
// rotational structure (spiral arms), and temporal coherence (standing waves).


// End of active_compact.cuh
