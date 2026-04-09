// sim_init.cuh — Simulation Initialization Functions
// ===================================================
// Extracts initialization code from blackhole_v20.cu into reusable functions.
// Each function populates a subsystem of the SimulationContext.
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstdio>
#include "sim_context.h"
#include "disk.cuh"
#include "kuramoto.cuh"              // PHASE_HIST_BINS

// ============================================================================
// initParticles — Host-side particle setup + GPU allocation + upload
// ============================================================================
// Creates the GPUDisk struct on device, initializes N particles with
// position, velocity, pump state, Kuramoto phase, and hopfion topo_state.
// Stores the device pointer in ctx.particles.buf_disk.
//
// Returns the GPUDisk* for backward compatibility with existing code.

inline GPUDisk* initParticles(SimulationContext& ctx, int N, unsigned int seed) {
    std::vector<float> h_px(N), h_py(N), h_pz(N);
    std::vector<float> h_vx(N), h_vy(N), h_vz(N);
    std::vector<int> h_state(N);
    std::vector<float> h_scale(N);
    std::vector<int> h_coherent(N);
    std::vector<uint8_t> h_seam(N);
    std::vector<float> h_residual(N), h_work(N);
    std::vector<float> h_history(N);
    std::vector<float> h_theta(N);
    std::vector<float> h_omega_nat(N);
    std::vector<uint8_t> h_flags(N);
    std::vector<uint8_t> h_topo(N);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> runif(0.0f, 1.0f);
    std::uniform_real_distribution<float> rphase(0, TWO_PI);
    std::normal_distribution<float> rnorm(0.0f, 1.0f);

    // Host-side copy of shell radii for init (device __constant__ not readable on host)
    static const float h_shell_radii[8] = {
        6.0f, 9.7f, 15.7f, 25.4f, 41.1f, 66.5f, 107.5f, 174.0f
    };

    float box_half = DISK_OUTER_R;
    float box_height = box_half * 0.3f;

    for (int i = 0; i < N; i++) {
        float x, y, z;

        if (g_shell_init) {
            // Shell-aware initialization: distribute across 8 shells weighted by 1/r
            float weight_sum = 0.0f;
            for (int s = 0; s < 8; s++) weight_sum += 1.0f / h_shell_radii[s];
            float pick = runif(rng) * weight_sum;
            float accum = 0.0f;
            int shell = 7;
            for (int s = 0; s < 8; s++) {
                accum += 1.0f / h_shell_radii[s];
                if (pick <= accum) { shell = s; break; }
            }
            float r_shell = h_shell_radii[shell];
            float r_jitter = r_shell + rnorm(rng) * 2.0f;
            if (r_jitter < ISCO_R * 0.8f) r_jitter = ISCO_R * 0.8f;

            float phi = rphase(rng);
            x = r_jitter * cosf(phi);
            z = r_jitter * sinf(phi);
            y = rnorm(rng) * r_shell * 0.2f;
        } else {
            // Uniform box initialization
            x = (runif(rng) * 2.0f - 1.0f) * box_half;
            y = (runif(rng) * 2.0f - 1.0f) * box_height;
            z = (runif(rng) * 2.0f - 1.0f) * box_half;

            float r = sqrtf(x*x + y*y + z*z);
            if (r < SCHW_R * 3.0f) {
                float scale = (SCHW_R * 3.0f) / r;
                x *= scale; y *= scale; z *= scale;
            }
        }

        h_px[i] = x;
        h_py[i] = y;
        h_pz[i] = z;

        // Full Keplerian tangential velocity
        float r_xz = sqrtf(x*x + z*z);
        if (r_xz > 0.1f) {
            float rot_sign = g_retrograde_init ? -1.0f : 1.0f;
            float v_rot = rot_sign * sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            float thermal = 0.02f;
            h_vx[i] = -v_rot * (z / r_xz) + rnorm(rng) * thermal;
            h_vy[i] = rnorm(rng) * 0.005f;
            h_vz[i] = v_rot * (x / r_xz) + rnorm(rng) * thermal;
        } else {
            h_vx[i] = rnorm(rng) * 0.05f;
            h_vy[i] = rnorm(rng) * 0.05f;
            h_vz[i] = rnorm(rng) * 0.05f;
        }

        // Pump state: IDLE
        h_state[i] = 0;
        h_scale[i] = 1.0f;
        h_coherent[i] = 0;
        h_seam[i] = 0x00;
        h_residual[i] = 0.0f;
        h_work[i] = 0.0f;
        h_history[i] = 1.0f;

        // Kuramoto: uniform random phase, Gaussian natural frequency
        h_theta[i] = rphase(rng);
        h_omega_nat[i] = g_omega_base + g_omega_spread * rnorm(rng);

        h_flags[i] = PFLAG_ACTIVE;

        // Hopfion: random single-axis-active state (Q=0)
        int topo_axis = (int)(runif(rng) * 4.0f) % 4;
        int topo_sign = (runif(rng) < 0.5f) ? 1 : -1;
        h_topo[i] = topo_set_axis(0, topo_axis, topo_sign);
    }

    // Verification
    int active_count = 0;
    for (int i = 0; i < N; i++) if (h_flags[i] & PFLAG_ACTIVE) active_count++;
    printf("[lattice] %d particles initialized in %.0f×%.0f×%.0f box (active=%d)\n",
           N, box_half*2, box_height*2, box_half*2, active_count);
    printf("[kuramoto-init] theta uniform[0,2π), omega_nat ~ N(%.2f, %.2f)\n",
           g_omega_base, g_omega_spread);

    int topo_dist[8] = {};
    int topo_Q_sum = 0;
    for (int i = 0; i < N; i++) {
        for (int a = 0; a < 4; a++) {
            int s = topo_get_axis(h_topo[i], a);
            if (s == 1) topo_dist[a * 2]++;
            else if (s == -1) topo_dist[a * 2 + 1]++;
        }
        topo_Q_sum += topo_compute_Q(h_topo[i]);
    }
    printf("[hopfion-init] topo_state distribution (axis+/axis-): "
           "e1(%d/%d) e2(%d/%d) e3(%d/%d) e4(%d/%d) Q_sum=%d\n",
           topo_dist[0], topo_dist[1], topo_dist[2], topo_dist[3],
           topo_dist[4], topo_dist[5], topo_dist[6], topo_dist[7], topo_Q_sum);

    // GPU allocation and upload
    GPUDisk* d_disk;
    cudaMalloc(&d_disk, sizeof(GPUDisk));
    cudaMemset(d_disk, 0, sizeof(GPUDisk));

    #define UPLOAD(field, hvec) \
    cudaMemcpy((char*)d_disk + offsetof(GPUDisk, field), hvec.data(), N*sizeof(hvec[0]), cudaMemcpyHostToDevice)

    UPLOAD(pos_x, h_px);    UPLOAD(pos_y, h_py);    UPLOAD(pos_z, h_pz);
    UPLOAD(vel_x, h_vx);    UPLOAD(vel_y, h_vy);    UPLOAD(vel_z, h_vz);
    UPLOAD(pump_state, h_state);
    UPLOAD(pump_scale, h_scale);
    UPLOAD(pump_coherent, h_coherent);
    UPLOAD(pump_seam, h_seam);
    UPLOAD(pump_residual, h_residual);
    UPLOAD(pump_work, h_work);
    UPLOAD(pump_history, h_history);
    UPLOAD(theta, h_theta);
    UPLOAD(omega_nat, h_omega_nat);
    UPLOAD(flags, h_flags);
    UPLOAD(topo_state, h_topo);
    #undef UPLOAD

    // Wire into context
    ctx.particles.buf_disk = d_disk;
    ctx.particles.N_current = N;

    return d_disk;
}

// ============================================================================
// initDiagnostics — Stress counters, Kuramoto reduction, phase histogram,
//                   spawn counters, sampling, async streams/events
// ============================================================================
// Allocates all diagnostic and async infrastructure buffers.
// Host-side readback vectors (h_kr_*, h_phase_*) remain in the caller
// since they're read every stats frame — only device pointers go in ctx.

struct DiagnosticLocals {
    StressCounters* d_stress;
    StressCounters* d_stress_async;
    float* d_kr_sin_sum;
    float* d_kr_cos_sum;
    int* d_kr_count;
    int kr_max_blocks;
    int* d_phase_hist;
    float* d_phase_omega_sum;
    float* d_phase_omega_sq;
    unsigned int* d_spawn_idx;
    unsigned int* d_spawn_success;
    int* d_sample_indices;
    SampleMetrics* d_sample_metrics[2];
    SampleMetrics* h_sample_metrics;
    StressCounters h_stats_cache;
    cudaStream_t sample_stream;
    cudaStream_t stats_stream;
    cudaEvent_t stats_ready;
    cudaStream_t spawn_stream;
    cudaEvent_t spawn_ready;
    unsigned int* h_spawn_pinned;
};

inline DiagnosticLocals initDiagnostics(SimulationContext& ctx, int N) {
    DiagnosticLocals d = {};

    // Stress counters
    cudaMalloc(&d.d_stress, sizeof(StressCounters));
    cudaMalloc(&d.d_stress_async, sizeof(StressCounters));

    // Kuramoto reduction buffers
    const int KR_THREADS = 256;
    d.kr_max_blocks = (ctx.particles.particle_cap + KR_THREADS - 1) / KR_THREADS;
    cudaMalloc(&d.d_kr_sin_sum, d.kr_max_blocks * sizeof(float));
    cudaMalloc(&d.d_kr_cos_sum, d.kr_max_blocks * sizeof(float));
    cudaMalloc(&d.d_kr_count, d.kr_max_blocks * sizeof(int));

    // Phase histogram
    cudaMalloc(&d.d_phase_hist, PHASE_HIST_BINS * sizeof(int));
    cudaMalloc(&d.d_phase_omega_sum, PHASE_HIST_BINS * sizeof(float));
    cudaMalloc(&d.d_phase_omega_sq, PHASE_HIST_BINS * sizeof(float));

    // Spawn counters
    cudaMalloc(&d.d_spawn_idx, sizeof(unsigned int));
    cudaMalloc(&d.d_spawn_success, sizeof(unsigned int));
    cudaMemset(d.d_spawn_idx, 0, sizeof(unsigned int));
    cudaMemset(d.d_spawn_success, 0, sizeof(unsigned int));

    // Stratified sampling
    std::vector<int> h_sample_indices(SAMPLE_COUNT);
    for (int i = 0; i < SAMPLE_COUNT; i++)
        h_sample_indices[i] = (i * N) / SAMPLE_COUNT;
    cudaMalloc(&d.d_sample_indices, SAMPLE_COUNT * sizeof(int));
    cudaMemcpy(d.d_sample_indices, h_sample_indices.data(),
               SAMPLE_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    // Double-buffered sample metrics
    cudaMalloc(&d.d_sample_metrics[0], sizeof(SampleMetrics));
    cudaMalloc(&d.d_sample_metrics[1], sizeof(SampleMetrics));
    cudaMallocHost(&d.h_sample_metrics, sizeof(SampleMetrics));
    d.h_sample_metrics->avg_scale = 1.0f;
    d.h_sample_metrics->avg_residual = 0.0f;
    d.h_sample_metrics->total_work = 0.0f;

    // Async streams and events
    cudaStreamCreate(&d.sample_stream);
    cudaStreamCreate(&d.stats_stream);
    cudaEventCreate(&d.stats_ready);
    cudaStreamCreate(&d.spawn_stream);
    cudaEventCreate(&d.spawn_ready);
    cudaMallocHost(&d.h_spawn_pinned, sizeof(unsigned int));
    *d.h_spawn_pinned = 0;

    d.h_stats_cache = {};

    // Topology ring buffer
    if (!topology_recorder_init()) {
        fprintf(stderr, "[topo] WARNING: Failed to initialize topology recorder\n");
    }

    // Wire into context
    ctx.diagnostics.buf_stress = d.d_stress;
    ctx.diagnostics.buf_stress_async = d.d_stress_async;
    ctx.diagnostics.buf_kr_sin_sum = d.d_kr_sin_sum;
    ctx.diagnostics.buf_kr_cos_sum = d.d_kr_cos_sum;
    ctx.diagnostics.buf_kr_count = d.d_kr_count;
    ctx.diagnostics.buf_phase_hist = d.d_phase_hist;
    ctx.diagnostics.buf_phase_omega_sum = d.d_phase_omega_sum;
    ctx.diagnostics.buf_phase_omega_sq = d.d_phase_omega_sq;
    ctx.diagnostics.buf_spawn_idx = d.d_spawn_idx;
    ctx.diagnostics.buf_spawn_success = d.d_spawn_success;
    ctx.diagnostics.buf_sample_indices = d.d_sample_indices;
    ctx.diagnostics.buf_sample_metrics[0] = d.d_sample_metrics[0];
    ctx.diagnostics.buf_sample_metrics[1] = d.d_sample_metrics[1];
    ctx.diagnostics.kr_max_blocks = d.kr_max_blocks;
    ctx.async.sample_stream = &d.sample_stream;
    ctx.async.stats_stream = &d.stats_stream;
    ctx.async.stats_ready_event = &d.stats_ready;
    ctx.async.spawn_stream = &d.spawn_stream;
    ctx.async.spawn_ready_event = &d.spawn_ready;
    ctx.async.pinned_sample_metrics = d.h_sample_metrics;
    ctx.async.pinned_spawn_count = d.h_spawn_pinned;

    return d;
}

// ============================================================================
// initOctree — Morton-sorted spatial tree (conditionally allocated)
// ============================================================================
// Only allocates if --octree-rebuild or --octree-render was passed.
// Builds frozen analytic tree at init. Returns locals needed by main loop.

struct OctreeLocals {
    uint64_t* d_morton_keys;
    uint32_t* d_xor_corners;
    uint32_t* d_particle_ids;
    OctreeNode* d_octree_nodes;
    uint32_t* d_node_count;
    uint32_t* d_leaf_counts;
    uint32_t* d_leaf_counts_culled;
    uint32_t* d_leaf_offsets;
    uint32_t* d_leaf_node_indices;
    uint32_t* d_leaf_node_count;
    float* d_leaf_vel_x;
    float* d_leaf_vel_y;
    float* d_leaf_vel_z;
    float* d_leaf_phase;
    float* d_leaf_frequency;
    float* d_leaf_coherence;
    uint64_t* d_leaf_hash_keys;
    uint32_t* d_leaf_hash_values;
    bool octreeEnabled;
    bool useOctreeTraversal;
    bool useOctreePhysics;
    int morton_capacity;
    uint32_t h_leaf_hash_size;
    uint32_t h_analytic_node_count;
    uint32_t h_total_node_count;
    uint32_t h_leaf_node_count;
    uint32_t h_cached_total_particles;
    uint32_t h_culled_total_particles;
    uint32_t h_num_active;
    float pressure_k;
    float vorticity_k;
    float substrate_k;
    float phase_coupling_k;
};

inline OctreeLocals initOctree(SimulationContext& ctx) {
    OctreeLocals o = {};

    extern bool g_octree_rebuild;
    extern bool g_octree_render;
    extern bool g_octree_physics;

    o.octreeEnabled = (g_octree_rebuild || g_octree_render);
    o.useOctreeTraversal = g_octree_render;
    o.useOctreePhysics = g_octree_physics;
    o.pressure_k = 0.03f;
    o.vorticity_k = 0.01f;
    o.substrate_k = 0.05f;
    o.phase_coupling_k = 0.05f;

    o.morton_capacity = g_octree_particle_cap;
    if (o.morton_capacity > MAX_DISK_PTS) o.morton_capacity = MAX_DISK_PTS;

    if (o.octreeEnabled) {
        cudaMalloc(&o.d_morton_keys, o.morton_capacity * sizeof(uint64_t));
        cudaMalloc(&o.d_xor_corners, o.morton_capacity * sizeof(uint32_t));
        cudaMalloc(&o.d_particle_ids, o.morton_capacity * sizeof(uint32_t));
        cudaMalloc(&o.d_octree_nodes, OCTREE_MAX_NODES * sizeof(OctreeNode));
        cudaMalloc(&o.d_node_count, sizeof(uint32_t));
        cudaMemset(o.d_node_count, 0, sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_counts, OCTREE_MAX_NODES * sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_counts_culled, OCTREE_MAX_NODES * sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_offsets, OCTREE_MAX_NODES * sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_node_indices, OCTREE_MAX_NODES * sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_node_count, sizeof(uint32_t));
        cudaMalloc(&o.d_leaf_vel_x, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&o.d_leaf_vel_y, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&o.d_leaf_vel_z, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&o.d_leaf_phase, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&o.d_leaf_frequency, OCTREE_MAX_NODES * sizeof(float));
        cudaMalloc(&o.d_leaf_coherence, OCTREE_MAX_NODES * sizeof(float));
        cudaMemset(o.d_leaf_phase, 0, OCTREE_MAX_NODES * sizeof(float));
        cudaMemset(o.d_leaf_frequency, 0, OCTREE_MAX_NODES * sizeof(float));
        cudaMemset(o.d_leaf_coherence, 0, OCTREE_MAX_NODES * sizeof(float));

        o.h_leaf_hash_size = 262144;
        cudaMalloc(&o.d_leaf_hash_keys, o.h_leaf_hash_size * sizeof(uint64_t));
        cudaMalloc(&o.d_leaf_hash_values, o.h_leaf_hash_size * sizeof(uint32_t));
        cudaMemset(o.d_leaf_hash_keys, 0xFF, o.h_leaf_hash_size * sizeof(uint64_t));

        printf("[octree] Allocated: morton cap=%d (%.1f MB), nodes=%zuMB\n",
               o.morton_capacity, o.morton_capacity * 16 / 1e6,
               OCTREE_MAX_NODES * sizeof(OctreeNode) / (1024 * 1024));
        printf("[octree] Hash: %u entries (%.1f MB) — L2 resident, max %u leaves\n",
               o.h_leaf_hash_size, o.h_leaf_hash_size * 12 / 1e6, o.h_leaf_hash_size / 2);

        // Build frozen analytic tree
        float boxSize = 500.0f;
        int maxNodesAtLevel5 = 1 << (ANALYTIC_MAX_LEVEL * 3);
        int analyticBlocks = (maxNodesAtLevel5 + 255) / 256;
        buildAnalyticTree<<<analyticBlocks, 256>>>(
            o.d_octree_nodes, o.d_node_count, boxSize, 0.0f, ANALYTIC_MAX_LEVEL
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&o.h_analytic_node_count, o.d_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("[octree] Frozen analytic tree: %u nodes (levels 0-%d)\n",
               o.h_analytic_node_count, ANALYTIC_MAX_LEVEL);
    }

    // Wire into context
    ctx.octree.buf_morton_keys = o.d_morton_keys;
    ctx.octree.buf_xor_corners = o.d_xor_corners;
    ctx.octree.buf_particle_ids = o.d_particle_ids;
    ctx.octree.buf_octree_nodes = o.d_octree_nodes;
    ctx.octree.buf_node_count = o.d_node_count;
    ctx.octree.buf_leaf_counts = o.d_leaf_counts;
    ctx.octree.buf_leaf_counts_culled = o.d_leaf_counts_culled;
    ctx.octree.buf_leaf_offsets = o.d_leaf_offsets;
    ctx.octree.buf_leaf_node_indices = o.d_leaf_node_indices;
    ctx.octree.buf_leaf_node_count = o.d_leaf_node_count;
    ctx.octree.buf_leaf_vel_x = o.d_leaf_vel_x;
    ctx.octree.buf_leaf_vel_y = o.d_leaf_vel_y;
    ctx.octree.buf_leaf_vel_z = o.d_leaf_vel_z;
    ctx.octree.buf_leaf_phase = o.d_leaf_phase;
    ctx.octree.buf_leaf_frequency = o.d_leaf_frequency;
    ctx.octree.buf_leaf_coherence = o.d_leaf_coherence;
    ctx.octree.buf_leaf_hash_keys = o.d_leaf_hash_keys;
    ctx.octree.buf_leaf_hash_values = o.d_leaf_hash_values;
    ctx.octree.enabled = o.octreeEnabled;
    ctx.octree.use_traversal = o.useOctreeTraversal;
    ctx.octree.use_physics = o.useOctreePhysics;
    ctx.octree.morton_capacity = o.morton_capacity;
    ctx.octree.h_leaf_hash_size = o.h_leaf_hash_size;
    ctx.octree.h_analytic_node_count = o.h_analytic_node_count;

    return o;
}

// ============================================================================
// initGrid — Cell grid + sparse flags + active compaction buffers
// ============================================================================

struct GridLocals {
    float* d_grid_density;
    float* d_grid_momentum_x, *d_grid_momentum_y, *d_grid_momentum_z;
    float* d_grid_phase_sin, *d_grid_phase_cos;
    float* d_grid_pressure_x, *d_grid_pressure_y, *d_grid_pressure_z;
    float* d_grid_vorticity_x, *d_grid_vorticity_y, *d_grid_vorticity_z;
    float* d_grid_R_cell;
    float* d_rc_bin_R, *d_rc_bin_W, *d_rc_bin_N;
    uint32_t* d_particle_cell;
    // Sparse flags
    uint8_t* d_active_flags;
    uint8_t* d_tile_flags;
    uint32_t* d_compact_active_list;
    uint32_t* d_compact_active_count;
    uint32_t* d_active_tiles;
    uint32_t* d_active_tile_count;
};

inline GridLocals initGrid(SimulationContext& ctx) {
    GridLocals g = {};

    extern bool g_grid_physics;
    extern bool g_grid_flags;
    extern bool g_active_compaction;
    extern ActiveParticleState g_active_particles;

    if (g_grid_physics) {
        size_t cell_array_size = g_grid_cells * sizeof(float);
        size_t particle_cell_size = (size_t)g_runtime_particle_cap * sizeof(uint32_t);

        cudaMalloc(&g.d_grid_density, cell_array_size);
        cudaMalloc(&g.d_grid_momentum_x, cell_array_size);
        cudaMalloc(&g.d_grid_momentum_y, cell_array_size);
        cudaMalloc(&g.d_grid_momentum_z, cell_array_size);
        cudaMalloc(&g.d_grid_phase_sin, cell_array_size);
        cudaMalloc(&g.d_grid_phase_cos, cell_array_size);
        cudaMalloc(&g.d_grid_pressure_x, cell_array_size);
        cudaMalloc(&g.d_grid_pressure_y, cell_array_size);
        cudaMalloc(&g.d_grid_pressure_z, cell_array_size);
        cudaMalloc(&g.d_grid_vorticity_x, cell_array_size);
        cudaMalloc(&g.d_grid_vorticity_y, cell_array_size);
        cudaMalloc(&g.d_grid_vorticity_z, cell_array_size);
        cudaMalloc(&g.d_grid_R_cell, cell_array_size);
        cudaMalloc(&g.d_rc_bin_R, 16 * sizeof(float));
        cudaMalloc(&g.d_rc_bin_W, 16 * sizeof(float));
        cudaMalloc(&g.d_rc_bin_N, 16 * sizeof(float));
        cudaMalloc(&g.d_particle_cell, particle_cell_size);

        size_t total_grid_mem = 12 * cell_array_size + particle_cell_size;
        printf("[grid] DNA/RNA streaming grid allocated: %zuMB\n",
               total_grid_mem / (1024 * 1024));
        printf("[grid] Grid: %dx%dx%d cells (%.2f units/cell)\n",
               g_grid_dim, g_grid_dim, g_grid_dim, g_grid_cell_size);

        if (!mip_tree_init(g_grid_dim, 0.1f)) {
            fprintf(stderr, "[mip] WARNING: Failed to initialize mip-tree\n");
        }
    }

    // Sparse flags
    if (g_grid_flags && g_grid_physics) {
        size_t cell_flags_size = g_grid_cells * sizeof(uint8_t);
        size_t tile_flags_size = NUM_TILES * sizeof(uint8_t);
        size_t active_list_size = g_grid_cells * sizeof(uint32_t);
        size_t tile_list_size = NUM_TILES * sizeof(uint32_t);

        cudaMalloc(&g.d_active_flags, cell_flags_size);
        cudaMalloc(&g.d_tile_flags, tile_flags_size);
        cudaMalloc(&g.d_compact_active_list, active_list_size);
        cudaMalloc(&g.d_compact_active_count, sizeof(uint32_t));
        cudaMalloc(&g.d_active_tiles, tile_list_size);
        cudaMalloc(&g.d_active_tile_count, sizeof(uint32_t));
        cudaMemset(g.d_active_flags, 0, cell_flags_size);
        cudaMemset(g.d_tile_flags, 0, tile_flags_size);
        cudaMemset(g.d_compact_active_count, 0, sizeof(uint32_t));
        cudaMemset(g.d_active_tile_count, 0, sizeof(uint32_t));

        size_t total_mem = cell_flags_size + tile_flags_size + active_list_size + tile_list_size;
        printf("[flags+tiles] Hierarchical tiled compaction: %.1fMB\n", total_mem / (1024.0 * 1024.0));
    }

    // Active compaction
    if (g_active_compaction && g_grid_physics) {
        size_t particle_cap = (size_t)g_runtime_particle_cap;
        size_t cell_array_size = g_grid_cells * sizeof(float);

        cudaMalloc(&g_active_particles.d_prev_cell, particle_cap * sizeof(uint32_t));
        cudaMalloc(&g_active_particles.d_active_mask, particle_cap * sizeof(uint8_t));
        cudaMalloc(&g_active_particles.d_active_list, particle_cap * sizeof(uint32_t));
        cudaMalloc(&g_active_particles.d_active_count, sizeof(uint32_t));
        cudaMalloc(&g_active_particles.d_static_density, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_x, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_y, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_momentum_z, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_phase_sin, cell_array_size);
        cudaMalloc(&g_active_particles.d_static_phase_cos, cell_array_size);
        cudaMemset(g_active_particles.d_prev_cell, 0xFF, particle_cap * sizeof(uint32_t));
        cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));
        g_active_particles.initialized = true;
        g_active_particles.static_baked = false;
        g_active_particles.h_active_count = 0;

        size_t total_active_mem = particle_cap * (sizeof(uint32_t) * 2 + sizeof(uint8_t)) + 6 * cell_array_size;
        printf("[active-compact] Active particle compaction: %.1fMB\n", total_active_mem / (1024.0 * 1024.0));
    }

    // Wire into context
    ctx.grid.buf_density = g.d_grid_density;
    ctx.grid.buf_momentum_x = g.d_grid_momentum_x;
    ctx.grid.buf_momentum_y = g.d_grid_momentum_y;
    ctx.grid.buf_momentum_z = g.d_grid_momentum_z;
    ctx.grid.buf_phase_sin = g.d_grid_phase_sin;
    ctx.grid.buf_phase_cos = g.d_grid_phase_cos;
    ctx.grid.buf_pressure_x = g.d_grid_pressure_x;
    ctx.grid.buf_pressure_y = g.d_grid_pressure_y;
    ctx.grid.buf_pressure_z = g.d_grid_pressure_z;
    ctx.grid.buf_vorticity_x = g.d_grid_vorticity_x;
    ctx.grid.buf_vorticity_y = g.d_grid_vorticity_y;
    ctx.grid.buf_vorticity_z = g.d_grid_vorticity_z;
    ctx.grid.buf_R_cell = g.d_grid_R_cell;
    ctx.grid.buf_particle_cell = g.d_particle_cell;
    ctx.grid.buf_active_flags = g.d_active_flags;
    ctx.grid.buf_tile_flags = g.d_tile_flags;
    ctx.grid.buf_compact_active_list = g.d_compact_active_list;
    ctx.grid.buf_compact_active_count = g.d_compact_active_count;
    ctx.grid.buf_active_tiles = g.d_active_tiles;
    ctx.grid.buf_active_tile_count = g.d_active_tile_count;
    ctx.grid.enabled = g_grid_physics;
    ctx.grid.flags_enabled = g_grid_flags;
    ctx.active_compact.buf_prev_cell = g_active_particles.d_prev_cell;
    ctx.active_compact.buf_active_mask = g_active_particles.d_active_mask;
    ctx.active_compact.buf_active_list = g_active_particles.d_active_list;
    ctx.active_compact.buf_active_count = g_active_particles.d_active_count;
    ctx.active_compact.buf_static_density = g_active_particles.d_static_density;
    ctx.active_compact.buf_static_momentum_x = g_active_particles.d_static_momentum_x;
    ctx.active_compact.buf_static_momentum_y = g_active_particles.d_static_momentum_y;
    ctx.active_compact.buf_static_momentum_z = g_active_particles.d_static_momentum_z;
    ctx.active_compact.buf_static_phase_sin = g_active_particles.d_static_phase_sin;
    ctx.active_compact.buf_static_phase_cos = g_active_particles.d_static_phase_cos;
    ctx.active_compact.enabled = g_active_compaction && g_grid_physics;

    return g;
}

// ============================================================================
// initTopology — Passive advection mask, hopfion enforcement, ActiveRegion
// ============================================================================

struct TopologyLocals {
    uint8_t* d_in_active_region;
    int* d_Q_sum;
    int* d_operator_counts;
    int* d_cell_topo_s;
    int* d_cell_topo_cnt;
    ActiveRegion* d_active_regions;
    int h_num_active_regions;
};

inline TopologyLocals initTopology(SimulationContext& ctx) {
    TopologyLocals t = {};

    // Passive/active region mask
    size_t in_active_region_size = (size_t)g_runtime_particle_cap * sizeof(uint8_t);
    cudaMalloc(&t.d_in_active_region, in_active_region_size);
    cudaMemset(t.d_in_active_region, 0xFF, in_active_region_size);
    printf("[passive] d_in_active_region allocated: %zu bytes, init=all-in-region\n",
           in_active_region_size);

    // Hopfion enforcement
    cudaMalloc(&t.d_Q_sum, sizeof(int));
    cudaMalloc(&t.d_operator_counts, 5 * sizeof(int));
    cudaMemset(t.d_Q_sum, 0, sizeof(int));
    cudaMemset(t.d_operator_counts, 0, 5 * sizeof(int));
    cudaMalloc(&t.d_cell_topo_s, 4 * g_grid_cells * sizeof(int));
    cudaMalloc(&t.d_cell_topo_cnt, g_grid_cells * sizeof(int));
    printf("[hopfion] Enforcement buffers allocated: Q_sum + 4 counters + cell topo (%.1f MB)\n",
           (4 * g_grid_cells * sizeof(int) + g_grid_cells * sizeof(int)) / 1e6);

    // Bootstrap ActiveRegion (all-encompassing)
    cudaMalloc(&t.d_active_regions, MAX_ACTIVE_REGIONS * sizeof(ActiveRegion));
    cudaMemset(t.d_active_regions, 0, MAX_ACTIVE_REGIONS * sizeof(ActiveRegion));
    ActiveRegion h_bootstrap = {};
    h_bootstrap.gate_positions[0] = make_float3(-500.0f, -500.0f, -500.0f);
    h_bootstrap.gate_positions[1] = make_float3( 500.0f,  500.0f,  500.0f);
    h_bootstrap.gate_positions[2] = make_float3(0.0f, 0.0f, 0.0f);
    h_bootstrap.parent_shell = -1;
    h_bootstrap.birth_frame = 0;
    h_bootstrap.stability_integral = 0.0f;
    h_bootstrap.state = REGION_STATE_ACTIVE;
    cudaMemcpy(t.d_active_regions, &h_bootstrap, sizeof(ActiveRegion),
               cudaMemcpyHostToDevice);
    t.h_num_active_regions = 1;
    printf("[passive] ActiveRegion bootstrap: 1 all-encompassing region seeded\n");

    // Wire into context
    ctx.topology.buf_in_active_region = t.d_in_active_region;
    ctx.topology.buf_Q_sum = t.d_Q_sum;
    ctx.topology.buf_operator_counts = t.d_operator_counts;
    ctx.topology.buf_cell_topo_s = t.d_cell_topo_s;
    ctx.topology.buf_cell_topo_cnt = t.d_cell_topo_cnt;
    ctx.topology.buf_active_regions = t.d_active_regions;
    ctx.topology.h_num_active_regions = t.h_num_active_regions;
    ctx.topology.hopfion_flip_scale = 1.0f;

    return t;
}
