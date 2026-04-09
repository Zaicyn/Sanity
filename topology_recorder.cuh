// topology_recorder.cuh — Field-Based Topology Ring Buffer
// =========================================================
//
// Stores downsampled m-field (unit vector field) on a 64³ grid instead of
// full particle states. This is 100x more memory efficient while preserving
// exactly the information needed for Hopf invariant computation.
//
// Key insight: The Hopf invariant Q = ∫ A·B d³x operates on FIELDS, not particles.
// Particles are just Monte Carlo samples of the underlying topology.
//
// Memory comparison (128 frames):
//   - Particle-based: 20M × 32 bytes × 128 = 81.9 GB (impossible)
//   - Field-based:    64³ × 12 bytes × 128 = 402 MB (trivial)
//
// Usage:
//   1. Call topology_recorder_init() once at startup
//   2. Call topology_recorder_update() every N frames (e.g., every 4 frames)
//   3. When crystallization detected, call topology_recorder_dump()
//
// Math reference:
//   m(x) = normalized velocity field at grid position x
//   B(x) = m · (∂m/∂x × ∂m/∂y) = topological charge density
//   Q = (1/4π²) ∫ A · B d³x = Hopf invariant (integer for solitons)

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

#define TOPO_GRID_DIM       64                          // 64³ grid
#define TOPO_GRID_CELLS     (TOPO_GRID_DIM * TOPO_GRID_DIM * TOPO_GRID_DIM)  // 262,144 cells
#define TOPO_HISTORY_FRAMES 128                         // ~2 seconds at 60 Hz recording rate
#define TOPO_RECORD_INTERVAL 1                          // Record every N simulation frames

// Inner region mask for Q computation (in cell units from grid center)
// Set to 0 to disable masking, or ~2 cells (r~15 world units) to exclude collapsing inner shells
// At 64³ grid over 500 units: 1 cell ≈ 7.8 world units, so 2 cells ≈ r=15
#define TOPO_Q_INNER_MASK_CELLS 2  // Exclude inner 2 cells (~r<15) from Q computation

// Crystal detection thresholds (strict - only true frozen states)
// NOTE: We compute actual average velocity from particle data, not E_kin (which is always 0)
#define CRYSTAL_AVG_VEL_THRESHOLD 0.01f   // Average velocity must be below this (truly frozen)
#define CRYSTAL_STABILITY_THRESHOLD 0.001f // Stability below 0.1% (extremely uniform shells)
#define CRYSTAL_Q_VARIANCE_THRESHOLD 0.01f // Q variance below this = topologically stable
#define CRYSTAL_HOLD_FRAMES       16       // ~0.25s at 60fps; gives 112 frames of pre-collapse history in 128-frame buffer

// ============================================================================
// Data Structures
// ============================================================================

// Per-frame metadata (lightweight, always recorded)
struct TopoFrameMeta {
    uint64_t frame_idx;         // Simulation frame number
    float    sim_time;          // Simulation time
    float    E_kin;             // Total kinetic energy
    float    stability;         // System stability metric (from telemetry)
    float    Q_estimate;        // Quick Hopf Q estimate (from grid field)
    float    mean_pump_scale;   // Average coherence
    float    mean_n;            // Average refractive index
    int      n_particles;       // Current particle count
    uint8_t  flags;             // Bit flags (see below)
    uint8_t  _pad[3];           // Alignment padding
};

// Flag bits for TopoFrameMeta.flags
#define TOPO_FLAG_CRYSTAL_CANDIDATE  0x01  // Met crystal thresholds this frame
#define TOPO_FLAG_CRYSTAL_CONFIRMED  0x02  // Sustained crystal state
#define TOPO_FLAG_Q_NEARLY_INTEGER   0x04  // |Q - round(Q)| < 0.05
#define TOPO_FLAG_EXPORTED           0x08  // This frame was part of an export

// Full field snapshot (64³ × float3 = 3.1 MB)
struct TopoFieldSnapshot {
    float3 m[TOPO_GRID_CELLS];  // Unit vector field (normalized velocity)
};

// Ring buffer state
struct TopologyRecorder {
    // Ring buffer of field snapshots (GPU memory)
    TopoFieldSnapshot* d_ring_buffer;    // [TOPO_HISTORY_FRAMES] snapshots on GPU

    // Ring buffer of metadata (CPU memory, lightweight)
    TopoFrameMeta h_meta_ring[TOPO_HISTORY_FRAMES];

    // Ring buffer indices
    int write_head;              // Next write position [0, TOPO_HISTORY_FRAMES)
    int filled_count;            // How many valid frames in buffer

    // Crystal detection state
    int crystal_candidate_streak;  // Consecutive frames meeting thresholds
    bool crystal_confirmed;        // Currently in confirmed crystal state
    bool crystal_exported;         // True after crystal dumped - detection permanently disabled
    bool awaiting_user_continue;   // True while waiting for user to press continue
    uint64_t last_export_frame;    // Frame number of last export

    // Temporary buffers for downsampling
    float3* d_accum_velocity;     // [TOPO_GRID_CELLS] accumulated velocity
    int*    d_accum_count;        // [TOPO_GRID_CELLS] particle count per cell

    // Configuration
    bool enabled;                 // Recording enabled
    int  record_interval;         // Record every N frames (dynamic, lock-aware)
    int  frames_since_record;     // Counter for interval

    // Lock-aware gating: topology is purely diagnostic, so we run it at
    // reduced cadence when the system is locked/stable.
    // Unlocked: every frame (need full resolution during formation)
    // Locked:   every 8 frames (~16 Hz at 130 FPS — plenty for stable topology)
    static constexpr int TOPO_STRIDE_UNLOCKED = 1;
    static constexpr int TOPO_STRIDE_LOCKED   = 8;
    int  topo_skipped;            // Frames skipped for logging
    int  topo_recorded;           // Frames recorded for logging
};

// Global instance (defined in .cu file)
extern TopologyRecorder g_topo_recorder;

// ============================================================================
// Host Functions
// ============================================================================

// Initialize the topology recorder (call once at startup)
inline bool topology_recorder_init() {
    TopologyRecorder& tr = g_topo_recorder;

    // Allocate GPU ring buffer (~400 MB for 128 frames)
    size_t ring_size = TOPO_HISTORY_FRAMES * sizeof(TopoFieldSnapshot);
    cudaError_t err = cudaMalloc(&tr.d_ring_buffer, ring_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[topo] Failed to allocate ring buffer (%zu MB): %s\n",
                ring_size / (1024*1024), cudaGetErrorString(err));
        return false;
    }
    cudaMemset(tr.d_ring_buffer, 0, ring_size);

    // Allocate temporary accumulation buffers
    err = cudaMalloc(&tr.d_accum_velocity, TOPO_GRID_CELLS * sizeof(float3));
    if (err != cudaSuccess) {
        fprintf(stderr, "[topo] Failed to allocate velocity accumulator\n");
        cudaFree(tr.d_ring_buffer);
        return false;
    }

    err = cudaMalloc(&tr.d_accum_count, TOPO_GRID_CELLS * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[topo] Failed to allocate count accumulator\n");
        cudaFree(tr.d_ring_buffer);
        cudaFree(tr.d_accum_velocity);
        return false;
    }

    // Initialize state
    tr.write_head = 0;
    tr.filled_count = 0;
    tr.crystal_candidate_streak = 0;
    tr.crystal_confirmed = false;
    tr.last_export_frame = 0;
    tr.enabled = true;
    tr.record_interval = TopologyRecorder::TOPO_STRIDE_UNLOCKED;
    tr.frames_since_record = 0;
    tr.topo_skipped = 0;
    tr.topo_recorded = 0;

    // Clear metadata ring
    memset(tr.h_meta_ring, 0, sizeof(tr.h_meta_ring));

    printf("[topo] Ring buffer initialized: %d frames × %.1f MB = %.0f MB total\n",
           TOPO_HISTORY_FRAMES,
           sizeof(TopoFieldSnapshot) / (1024.0f * 1024.0f),
           ring_size / (1024.0f * 1024.0f));

    return true;
}

// Cleanup (call at shutdown)
inline void topology_recorder_cleanup() {
    TopologyRecorder& tr = g_topo_recorder;
    if (tr.d_ring_buffer) cudaFree(tr.d_ring_buffer);
    if (tr.d_accum_velocity) cudaFree(tr.d_accum_velocity);
    if (tr.d_accum_count) cudaFree(tr.d_accum_count);
    tr.d_ring_buffer = nullptr;
    tr.d_accum_velocity = nullptr;
    tr.d_accum_count = nullptr;
    tr.enabled = false;
}

// Enable/disable recording
inline void topology_recorder_set_enabled(bool enabled) {
    g_topo_recorder.enabled = enabled;
}

// Called after user acknowledges crystal detection - disables detection permanently
inline void topology_recorder_acknowledge_crystal() {
    TopologyRecorder& tr = g_topo_recorder;
    tr.crystal_exported = true;       // Permanently disable detection for this run
    tr.awaiting_user_continue = false;
    printf("[topo] Crystal acknowledged. Detection disabled for remainder of run.\n");
}

// Check if simulation should be paused waiting for user
inline bool topology_recorder_awaiting_continue() {
    return g_topo_recorder.awaiting_user_continue;
}

// Get the latest Q estimate (for logging alongside shell radii)
inline float topology_recorder_get_latest_Q() {
    TopologyRecorder& tr = g_topo_recorder;
    if (tr.filled_count == 0) return 0.0f;
    // Get most recent slot (write_head - 1, wrapped)
    int latest_slot = (tr.write_head + TOPO_HISTORY_FRAMES - 1) % TOPO_HISTORY_FRAMES;
    return tr.h_meta_ring[latest_slot].Q_estimate;
}

// ============================================================================
// Device Kernels
// ============================================================================

// Kernel 1: Clear accumulation buffers
__global__ void topoKernel_clearAccum(float3* velocity, int* count, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    velocity[idx] = make_float3(0.0f, 0.0f, 0.0f);
    count[idx] = 0;
}

// Kernel 2: Scatter particles to grid (accumulate velocities)
__global__ void topoKernel_scatterParticles(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ vel_z,
    float3* __restrict__ accum_vel,
    int* __restrict__ accum_count,
    int n_particles,
    float grid_half_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float px = pos_x[i];
    float py = pos_y[i];
    float pz = pos_z[i];

    // Map position to grid cell
    float cell_size = (2.0f * grid_half_size) / TOPO_GRID_DIM;
    int cx = (int)fminf(fmaxf((px + grid_half_size) / cell_size, 0.0f), (float)(TOPO_GRID_DIM - 1));
    int cy = (int)fminf(fmaxf((py + grid_half_size) / cell_size, 0.0f), (float)(TOPO_GRID_DIM - 1));
    int cz = (int)fminf(fmaxf((pz + grid_half_size) / cell_size, 0.0f), (float)(TOPO_GRID_DIM - 1));

    int cell_idx = cx + cy * TOPO_GRID_DIM + cz * TOPO_GRID_DIM * TOPO_GRID_DIM;

    // Atomic accumulate velocity
    atomicAdd(&accum_vel[cell_idx].x, vel_x[i]);
    atomicAdd(&accum_vel[cell_idx].y, vel_y[i]);
    atomicAdd(&accum_vel[cell_idx].z, vel_z[i]);
    atomicAdd(&accum_count[cell_idx], 1);
}

// Kernel 3: Normalize accumulated velocities to unit vectors
__global__ void topoKernel_normalizeField(
    const float3* __restrict__ accum_vel,
    const int* __restrict__ accum_count,
    float3* __restrict__ m_field,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    int count = accum_count[idx];
    if (count == 0) {
        // Empty cell: use zero vector (no topology contribution)
        m_field[idx] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float3 v = accum_vel[idx];
    float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    if (mag < 1e-8f) {
        m_field[idx] = make_float3(0.0f, 0.0f, 0.0f);
    } else {
        float inv_mag = 1.0f / mag;
        m_field[idx] = make_float3(v.x * inv_mag, v.y * inv_mag, v.z * inv_mag);
    }
}

// Kernel 4: Compute average velocity magnitude (for motion detection)
// Uses the accumulated velocities before normalization to get actual speeds
__global__ void topoKernel_computeAvgVelocity(
    const float3* __restrict__ accum_vel,
    const int* __restrict__ accum_count,
    float* __restrict__ vel_sum,      // Output: sum of velocity magnitudes
    int* __restrict__ particle_count, // Output: total particles counted
    int n_cells
) {
    __shared__ float s_vel_sum[256];
    __shared__ int s_count[256];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_vel = 0.0f;
    int local_count = 0;

    if (idx < n_cells) {
        int count = accum_count[idx];
        if (count > 0) {
            float3 v = accum_vel[idx];
            // Magnitude of sum / count = average velocity magnitude in cell
            float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z) / count;
            local_vel = mag * count;  // Weight by particle count
            local_count = count;
        }
    }

    s_vel_sum[tx] = local_vel;
    s_count[tx] = local_count;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s_vel_sum[tx] += s_vel_sum[tx + stride];
            s_count[tx] += s_count[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(vel_sum, s_vel_sum[0]);
        atomicAdd(particle_count, s_count[0]);
    }
}

// Kernel 5: Compute full Hopf invariant Q via topological charge density
// Q = ∫ B d³x / (4π²) where B = m · (∂m/∂x × ∂m/∂y + ∂m/∂y × ∂m/∂z + ∂m/∂z × ∂m/∂x)
// All three cross product terms are needed for the full Jacobian determinant
// r_mask_cells: exclude cells within this radius from center (in cell units, 0 = no mask)
__global__ void topoKernel_computeB(
    const float3* __restrict__ m_field,
    float* __restrict__ B_sum,  // Single output
    int grid_dim,
    int r_mask_cells = 0  // Exclude inner region (in cell units from grid center)
) {
    // Shared memory for reduction
    __shared__ float s_sum[256];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = grid_dim * grid_dim * grid_dim;

    float local_B = 0.0f;

    if (idx < n_cells) {
        // Unpack cell coordinates
        int cx = idx % grid_dim;
        int cy = (idx / grid_dim) % grid_dim;
        int cz = idx / (grid_dim * grid_dim);

        // Skip boundary cells (need neighbors in all directions)
        if (cx > 0 && cx < grid_dim - 1 &&
            cy > 0 && cy < grid_dim - 1 &&
            cz > 0 && cz < grid_dim - 1) {

            // Optional inner radius mask (skip cells too close to center)
            if (r_mask_cells > 0) {
                int half = grid_dim / 2;
                int dx = cx - half;
                int dy = cy - half;
                int dz = cz - half;
                int r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < r_mask_cells * r_mask_cells) {
                    // Inside masked region - skip this cell
                    goto reduce;
                }
            }

            float3 m = m_field[idx];

            // Neighbor indices
            int idx_xp = (cx + 1) + cy * grid_dim + cz * grid_dim * grid_dim;
            int idx_xm = (cx - 1) + cy * grid_dim + cz * grid_dim * grid_dim;
            int idx_yp = cx + (cy + 1) * grid_dim + cz * grid_dim * grid_dim;
            int idx_ym = cx + (cy - 1) * grid_dim + cz * grid_dim * grid_dim;
            int idx_zp = cx + cy * grid_dim + (cz + 1) * grid_dim * grid_dim;
            int idx_zm = cx + cy * grid_dim + (cz - 1) * grid_dim * grid_dim;

            // Central differences for all three gradients
            float3 dm_dx = make_float3(
                (m_field[idx_xp].x - m_field[idx_xm].x) * 0.5f,
                (m_field[idx_xp].y - m_field[idx_xm].y) * 0.5f,
                (m_field[idx_xp].z - m_field[idx_xm].z) * 0.5f
            );
            float3 dm_dy = make_float3(
                (m_field[idx_yp].x - m_field[idx_ym].x) * 0.5f,
                (m_field[idx_yp].y - m_field[idx_ym].y) * 0.5f,
                (m_field[idx_yp].z - m_field[idx_ym].z) * 0.5f
            );
            float3 dm_dz = make_float3(
                (m_field[idx_zp].x - m_field[idx_zm].x) * 0.5f,
                (m_field[idx_zp].y - m_field[idx_zm].y) * 0.5f,
                (m_field[idx_zp].z - m_field[idx_zm].z) * 0.5f
            );

            // Three cross products for full topological charge density:
            // Term 1: dm_dx × dm_dy (z-contribution)
            float3 cross_xy;
            cross_xy.x = dm_dx.y * dm_dy.z - dm_dx.z * dm_dy.y;
            cross_xy.y = dm_dx.z * dm_dy.x - dm_dx.x * dm_dy.z;
            cross_xy.z = dm_dx.x * dm_dy.y - dm_dx.y * dm_dy.x;

            // Term 2: dm_dy × dm_dz (x-contribution)
            float3 cross_yz;
            cross_yz.x = dm_dy.y * dm_dz.z - dm_dy.z * dm_dz.y;
            cross_yz.y = dm_dy.z * dm_dz.x - dm_dy.x * dm_dz.z;
            cross_yz.z = dm_dy.x * dm_dz.y - dm_dy.y * dm_dz.x;

            // Term 3: dm_dz × dm_dx (y-contribution)
            float3 cross_zx;
            cross_zx.x = dm_dz.y * dm_dx.z - dm_dz.z * dm_dx.y;
            cross_zx.y = dm_dz.z * dm_dx.x - dm_dz.x * dm_dx.z;
            cross_zx.z = dm_dz.x * dm_dx.y - dm_dz.y * dm_dx.x;

            // Full B = m · (cross_xy + cross_yz + cross_zx)
            local_B = m.x * (cross_xy.x + cross_yz.x + cross_zx.x)
                    + m.y * (cross_xy.y + cross_yz.y + cross_zx.y)
                    + m.z * (cross_xy.z + cross_yz.z + cross_zx.z);
        }
    }

reduce:
    // Block reduction
    s_sum[tx] = local_B;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            s_sum[tx] += s_sum[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(B_sum, s_sum[0]);
    }
}

// ============================================================================
// Recording & Detection Functions
// ============================================================================

// Record a frame to the ring buffer
// Returns true if crystal state is detected (and should trigger full export)
inline bool topology_recorder_update(
    const float* d_pos_x, const float* d_pos_y, const float* d_pos_z,
    const float* d_vel_x, const float* d_vel_y, const float* d_vel_z,
    int n_particles,
    uint64_t frame_idx,
    float sim_time,
    float E_kin,
    float stability,
    float mean_pump_scale,
    float mean_n,
    float grid_half_size = 250.0f,
    bool harmonic_locked = false
) {
    TopologyRecorder& tr = g_topo_recorder;
    if (!tr.enabled || !tr.d_ring_buffer) {
        if (frame_idx == 100) printf("[topo] WARNING: Recorder not initialized!\n");
        return false;
    }

    // Crystal already exported - detection permanently disabled for this run
    if (tr.crystal_exported) return false;

    // Skip first 100 frames (warmup period for async stats to populate)
    if (frame_idx < 100) return false;

    // Log start of recording (once)
    static bool logged_start = false;
    if (!logged_start) {
        printf("[topo] Recording started at frame %lu\n", frame_idx);
        logged_start = true;
    }

    // Lock-aware gating: adjust stride based on harmonic lock state
    // Topology is purely diagnostic — safe to skip when stable
    int target_stride = harmonic_locked ?
        TopologyRecorder::TOPO_STRIDE_LOCKED :
        TopologyRecorder::TOPO_STRIDE_UNLOCKED;
    tr.record_interval = target_stride;

    // Rate limiting
    tr.frames_since_record++;
    if (tr.frames_since_record < tr.record_interval) {
        tr.topo_skipped++;
        return false;
    }
    tr.frames_since_record = 0;
    tr.topo_recorded++;

    // Get current write position
    int slot = tr.write_head;

    // Launch kernels to build m-field
    int block_size = 256;
    int grid_cells = TOPO_GRID_CELLS;
    int particle_blocks = (n_particles + block_size - 1) / block_size;
    int cell_blocks = (grid_cells + block_size - 1) / block_size;

    // Clear accumulators
    topoKernel_clearAccum<<<cell_blocks, block_size>>>(
        tr.d_accum_velocity, tr.d_accum_count, grid_cells);

    // Scatter particles to grid
    topoKernel_scatterParticles<<<particle_blocks, block_size>>>(
        d_pos_x, d_pos_y, d_pos_z,
        d_vel_x, d_vel_y, d_vel_z,
        tr.d_accum_velocity, tr.d_accum_count,
        n_particles, grid_half_size);

    // === COMPUTE AVERAGE VELOCITY (for motion detection) ===
    // Do this BEFORE normalizing, as we need raw velocity magnitudes
    static float* d_vel_sum = nullptr;
    static int* d_vel_count = nullptr;
    static float* h_vel_pinned = nullptr;
    static int* h_count_pinned = nullptr;
    static float h_avg_vel_cached = 1.0f;  // Start high so we don't false-trigger
    static cudaStream_t vel_stream = nullptr;
    static cudaEvent_t vel_event = nullptr;
    static bool vel_copy_pending = false;

    if (!d_vel_sum) {
        cudaMalloc(&d_vel_sum, sizeof(float));
        cudaMalloc(&d_vel_count, sizeof(int));
        cudaMallocHost(&h_vel_pinned, sizeof(float));
        cudaMallocHost(&h_count_pinned, sizeof(int));
        cudaStreamCreate(&vel_stream);
        cudaEventCreate(&vel_event);
    }

    // Check if previous velocity copy is done
    if (vel_copy_pending) {
        if (cudaEventQuery(vel_event) == cudaSuccess) {
            if (*h_count_pinned > 0) {
                h_avg_vel_cached = *h_vel_pinned / *h_count_pinned;
            }
            vel_copy_pending = false;
        }
    }

    // Launch velocity computation (async)
    if (!vel_copy_pending) {
        cudaMemsetAsync(d_vel_sum, 0, sizeof(float), vel_stream);
        cudaMemsetAsync(d_vel_count, 0, sizeof(int), vel_stream);
        topoKernel_computeAvgVelocity<<<cell_blocks, block_size, 0, vel_stream>>>(
            tr.d_accum_velocity, tr.d_accum_count, d_vel_sum, d_vel_count, grid_cells);
        cudaMemcpyAsync(h_vel_pinned, d_vel_sum, sizeof(float),
                        cudaMemcpyDeviceToHost, vel_stream);
        cudaMemcpyAsync(h_count_pinned, d_vel_count, sizeof(int),
                        cudaMemcpyDeviceToHost, vel_stream);
        cudaEventRecord(vel_event, vel_stream);
        vel_copy_pending = true;
    }

    // Normalize to unit vectors and store in ring buffer
    float3* d_m_field = tr.d_ring_buffer[slot].m;
    topoKernel_normalizeField<<<cell_blocks, block_size>>>(
        tr.d_accum_velocity, tr.d_accum_count, d_m_field, grid_cells);

    // === COMPUTE Q ESTIMATE (async) ===
    static float* d_B_sum = nullptr;
    static float* h_B_sum_pinned = nullptr;
    static float h_B_sum_cached = 0.0f;
    static cudaStream_t topo_stream = nullptr;
    static cudaEvent_t topo_event = nullptr;
    static bool copy_pending = false;

    if (!d_B_sum) {
        cudaMalloc(&d_B_sum, sizeof(float));
        cudaMallocHost(&h_B_sum_pinned, sizeof(float));
        cudaStreamCreate(&topo_stream);
        cudaEventCreate(&topo_event);
    }

    // Check if previous async copy is done (non-blocking)
    if (copy_pending) {
        if (cudaEventQuery(topo_event) == cudaSuccess) {
            h_B_sum_cached = *h_B_sum_pinned;
            copy_pending = false;
        }
        // If not done, keep using cached value
    }

    // Launch compute on stream (async)
    cudaMemsetAsync(d_B_sum, 0, sizeof(float), topo_stream);
    topoKernel_computeB<<<cell_blocks, block_size, 0, topo_stream>>>(
        d_m_field, d_B_sum, TOPO_GRID_DIM, TOPO_Q_INNER_MASK_CELLS);

    // Start async copy (only if previous one is done)
    if (!copy_pending) {
        cudaMemcpyAsync(h_B_sum_pinned, d_B_sum, sizeof(float),
                        cudaMemcpyDeviceToHost, topo_stream);
        cudaEventRecord(topo_event, topo_stream);
        copy_pending = true;
    }

    // Use cached Q estimate (1-2 frames stale, fine for detection)
    float cell_size = (2.0f * grid_half_size) / TOPO_GRID_DIM;
    float Q_estimate = h_B_sum_cached * cell_size * cell_size * cell_size / (4.0f * 3.14159f * 3.14159f);
    float avg_velocity = h_avg_vel_cached;

    // Build metadata
    TopoFrameMeta& meta = tr.h_meta_ring[slot];
    meta.frame_idx = frame_idx;
    meta.sim_time = sim_time;
    meta.E_kin = avg_velocity;  // Store avg_velocity in E_kin field (repurposed)
    meta.stability = stability;
    meta.Q_estimate = Q_estimate;
    meta.mean_pump_scale = mean_pump_scale;
    meta.mean_n = mean_n;
    meta.n_particles = n_particles;
    meta.flags = 0;

    // Check for nearly-integer Q
    float delta_Q = fabsf(Q_estimate - roundf(Q_estimate));
    if (delta_Q < 0.05f) {
        meta.flags |= TOPO_FLAG_Q_NEARLY_INTEGER;
    }

    // Crystal detection - use actual average velocity, not E_kin
    // A crystal must have:
    // 1. Very low average velocity (particles essentially stationary)
    // 2. Uniform shell structure (low stability variance)
    bool vel_pass = (avg_velocity < CRYSTAL_AVG_VEL_THRESHOLD);
    bool stability_pass = (stability < CRYSTAL_STABILITY_THRESHOLD);
    bool is_candidate = vel_pass && stability_pass;

    if (is_candidate) {
        meta.flags |= TOPO_FLAG_CRYSTAL_CANDIDATE;
        tr.crystal_candidate_streak++;

        if (tr.crystal_candidate_streak >= CRYSTAL_HOLD_FRAMES) {
            meta.flags |= TOPO_FLAG_CRYSTAL_CONFIRMED;
            tr.crystal_confirmed = true;
        }
    } else {
        // Soft decay: decrement by 1 instead of hard reset to 0.
        // A single noisy frame shouldn't erase 15 frames of crystal evidence.
        // The streak is an analog nullable — 0 = no crystal evidence (null guard).
        if (tr.crystal_candidate_streak > 0) {
            tr.crystal_candidate_streak--;
        }
        // Only un-confirm if streak drops to zero (sustained noise, not a blip)
        if (tr.crystal_candidate_streak == 0) {
            tr.crystal_confirmed = false;
        }
    }

    // Advance ring buffer
    tr.write_head = (tr.write_head + 1) % TOPO_HISTORY_FRAMES;
    if (tr.filled_count < TOPO_HISTORY_FRAMES) {
        tr.filled_count++;
    }

    // Periodic status logging (every 500 frames after warmup)
    if (frame_idx % 500 == 0) {
        printf("[topo] frame=%lu | buf=%d/%d | Q≈%.2f | avg_vel=%.3f | stability=%.3f | streak=%d | stride=%d (rec=%d skip=%d)%s\n",
               frame_idx, tr.filled_count, TOPO_HISTORY_FRAMES,
               Q_estimate, avg_velocity, stability, tr.crystal_candidate_streak,
               tr.record_interval, tr.topo_recorded, tr.topo_skipped,
               tr.crystal_confirmed ? " CRYSTAL!" : "");
    }

    // Return true ONCE when crystal first confirmed (triggers pause + dump)
    // Only one crystal can be detected per run
    if (tr.crystal_confirmed && !tr.awaiting_user_continue) {
        tr.awaiting_user_continue = true;
        return true;
    }

    return false;
}

// Dump ring buffer to disk
inline void topology_recorder_dump(const char* prefix = "crystal") {
    TopologyRecorder& tr = g_topo_recorder;
    if (!tr.d_ring_buffer || tr.filled_count == 0) {
        printf("[topo] Nothing to dump\n");
        return;
    }

    // Create output directory
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p topology_dumps/");
    system(cmd);

    // Get timestamp
    uint64_t latest_frame = tr.h_meta_ring[(tr.write_head + TOPO_HISTORY_FRAMES - 1) % TOPO_HISTORY_FRAMES].frame_idx;

    // Dump metadata CSV
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "topology_dumps/%s_%lu_meta.csv", prefix, latest_frame);
    FILE* meta_file = fopen(meta_path, "w");
    if (meta_file) {
        fprintf(meta_file, "frame,sim_time,E_kin,stability,Q_estimate,mean_pump_scale,mean_n,n_particles,flags\n");

        // Write in chronological order
        for (int i = 0; i < tr.filled_count; i++) {
            int idx = (tr.write_head - tr.filled_count + i + TOPO_HISTORY_FRAMES) % TOPO_HISTORY_FRAMES;
            TopoFrameMeta& m = tr.h_meta_ring[idx];
            fprintf(meta_file, "%lu,%.4f,%.6e,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                    m.frame_idx, m.sim_time, m.E_kin, m.stability,
                    m.Q_estimate, m.mean_pump_scale, m.mean_n, m.n_particles, m.flags);
            m.flags |= TOPO_FLAG_EXPORTED;
        }
        fclose(meta_file);
        printf("[topo] Saved metadata: %s\n", meta_path);
    }

    // Dump field snapshots (binary)
    char field_path[256];
    snprintf(field_path, sizeof(field_path), "topology_dumps/%s_%lu_fields.bin", prefix, latest_frame);
    FILE* field_file = fopen(field_path, "wb");
    if (field_file) {
        // Header
        uint32_t magic = 0x544F504F;  // "TOPO"
        uint32_t version = 1;
        uint32_t grid_dim = TOPO_GRID_DIM;
        uint32_t n_frames = tr.filled_count;
        fwrite(&magic, sizeof(uint32_t), 1, field_file);
        fwrite(&version, sizeof(uint32_t), 1, field_file);
        fwrite(&grid_dim, sizeof(uint32_t), 1, field_file);
        fwrite(&n_frames, sizeof(uint32_t), 1, field_file);

        // Allocate host buffer for one frame
        TopoFieldSnapshot h_snapshot;

        // Write frames in chronological order
        for (int i = 0; i < tr.filled_count; i++) {
            int idx = (tr.write_head - tr.filled_count + i + TOPO_HISTORY_FRAMES) % TOPO_HISTORY_FRAMES;

            // Copy from GPU
            cudaMemcpy(&h_snapshot, &tr.d_ring_buffer[idx], sizeof(TopoFieldSnapshot), cudaMemcpyDeviceToHost);

            // Write to file
            fwrite(h_snapshot.m, sizeof(float3), TOPO_GRID_CELLS, field_file);
        }

        fclose(field_file);
        printf("[topo] Saved fields: %s (%.1f MB)\n", field_path,
               (float)(sizeof(TopoFieldSnapshot) * tr.filled_count) / (1024.0f * 1024.0f));
    }

    tr.last_export_frame = latest_frame;
    printf("[topo] Dump complete: %d frames from %lu\n", tr.filled_count, latest_frame);
}

// Get current status string
inline void topology_recorder_status(char* buf, size_t buf_size) {
    TopologyRecorder& tr = g_topo_recorder;
    if (!tr.enabled) {
        snprintf(buf, buf_size, "[topo] Disabled");
        return;
    }

    int latest_idx = (tr.write_head + TOPO_HISTORY_FRAMES - 1) % TOPO_HISTORY_FRAMES;
    TopoFrameMeta& m = tr.h_meta_ring[latest_idx];

    snprintf(buf, buf_size,
             "[topo] %d/%d frames | Q≈%.2f | E_kin=%.2e | streak=%d%s",
             tr.filled_count, TOPO_HISTORY_FRAMES,
             m.Q_estimate, m.E_kin, tr.crystal_candidate_streak,
             tr.crystal_confirmed ? " CRYSTAL!" : "");
}

// End of topology_recorder.cuh
