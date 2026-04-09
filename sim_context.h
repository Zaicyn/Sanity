// sim_context.h — Backend-Agnostic Simulation Context
// ====================================================
// Bundles all simulation state into a single struct for portable dispatch.
// All buffer pointers are void* to decouple from CUDA/Vulkan/OpenCL.
// Cast to typed pointers at the dispatch layer boundary.
//
// Naming convention: buf_ prefix (not d_) — no backend assumption.
// Rendering buffers are NOT included (they belong to the graphics layer).
//
// Analog nullable conventions (see nullable.h):
//   void* buf_* = nullptr  → buffer not allocated (subsystem disabled)
//   float fields = 0.0f    → uninitialized / inactive (exact 0 is collision-free
//                             in continuous simulation — only occurs by explicit assignment)
//   int indices = -1       → unassigned / no parent
//
// Usage:
//   SimulationContext ctx = {};
//   initParticles(ctx, ...);
//   initDiagnostics(ctx);
//   ...
//   while (running) { simulationStep(ctx, dt); }
//   cleanupSimulation(ctx);

#pragma once
#include <cstdint>

struct SimulationContext {

    // ================================================================
    // PARTICLES — The main SoA particle data
    // ================================================================
    struct {
        void* buf_disk;              // GPUDisk* — SoA struct with pos/vel/pump/phase arrays
        int   N_seed;                // Initial particle count
        int   N_current;             // Current count (grows via spawning)
        int   particle_cap;          // VRAM-safe upper bound
    } particles;

    // ================================================================
    // DIAGNOSTICS — Stress, sampling, spawn counters, Kuramoto reduction
    // ================================================================
    struct {
        void* buf_stress;            // StressCounters*
        void* buf_stress_async;      // StressCounters* (async copy target)
        void* buf_kr_sin_sum;        // float* — Kuramoto reduction sin
        void* buf_kr_cos_sum;        // float* — Kuramoto reduction cos
        void* buf_kr_count;          // int*   — Kuramoto reduction count
        void* buf_phase_hist;        // int*   — phase histogram bins
        void* buf_phase_omega_sum;   // float* — omega per bin
        void* buf_phase_omega_sq;    // float* — omega² per bin
        void* buf_sample_indices;    // int*   — stratified sample indices
        void* buf_sample_metrics[2]; // SampleMetrics* — double-buffered
        void* buf_spawn_idx;         // unsigned int* — spawn slot counter
        void* buf_spawn_success;     // unsigned int* — successful spawn count
        int   kr_max_blocks;
        int   current_buffer;        // 0 or 1 for double-buffering
    } diagnostics;

    // ================================================================
    // OCTREE — Morton-sorted spatial tree (conditionally allocated)
    // ================================================================
    struct {
        void* buf_morton_keys;
        void* buf_xor_corners;
        void* buf_particle_ids;
        void* buf_octree_nodes;
        void* buf_node_count;
        void* buf_leaf_counts;
        void* buf_leaf_counts_culled;
        void* buf_leaf_offsets;
        void* buf_leaf_node_indices;
        void* buf_leaf_node_count;
        void* buf_leaf_vel_x;
        void* buf_leaf_vel_y;
        void* buf_leaf_vel_z;
        void* buf_leaf_phase;
        void* buf_leaf_frequency;
        void* buf_leaf_coherence;
        void* buf_leaf_hash_keys;
        void* buf_leaf_hash_values;
        bool  enabled;
        bool  use_traversal;
        bool  use_physics;
        int   morton_capacity;
        uint32_t h_leaf_hash_size;
        uint32_t h_analytic_node_count;
        uint32_t h_total_node_count;
        uint32_t h_leaf_node_count;
        uint32_t h_cached_total_particles;
        uint32_t h_culled_total_particles;
    } octree;

    // ================================================================
    // GRID — Cell grid physics (DNA/RNA streaming)
    // ================================================================
    struct {
        void* buf_density;
        void* buf_momentum_x;
        void* buf_momentum_y;
        void* buf_momentum_z;
        void* buf_phase_sin;
        void* buf_phase_cos;
        void* buf_pressure_x;
        void* buf_pressure_y;
        void* buf_pressure_z;
        void* buf_vorticity_x;
        void* buf_vorticity_y;
        void* buf_vorticity_z;
        void* buf_R_cell;
        void* buf_rc_bin_R;
        void* buf_rc_bin_W;
        void* buf_rc_bin_N;
        void* buf_particle_cell;
        // Sparse flags
        void* buf_active_flags;
        void* buf_tile_flags;
        void* buf_compact_active_list;
        void* buf_compact_active_count;
        void* buf_active_tiles;
        void* buf_active_tile_count;
        bool  enabled;
        bool  flags_enabled;
        uint32_t h_compact_active_count;
        uint32_t h_active_tile_count;
    } grid;

    // ================================================================
    // ACTIVE COMPACTION — Scatter-skip optimization
    // ================================================================
    struct {
        void* buf_prev_cell;
        void* buf_active_mask;
        void* buf_active_list;
        void* buf_active_count;
        void* buf_static_density;
        void* buf_static_momentum_x;
        void* buf_static_momentum_y;
        void* buf_static_momentum_z;
        void* buf_static_phase_sin;
        void* buf_static_phase_cos;
        bool  enabled;
        bool  initialized;
        bool  static_baked;
        int   bake_frame;
        uint32_t h_active_count;
    } active_compact;

    // ================================================================
    // TOPOLOGY — Hopfion enforcement, active regions, passive advection
    // ================================================================
    struct {
        void* buf_in_active_region;
        void* buf_Q_sum;
        void* buf_Q_delta_sum;       // Writer monad: per-frame Q conservation delta
        void* buf_operator_counts;
        void* buf_cell_topo_s;
        void* buf_cell_topo_cnt;
        void* buf_active_regions;
        int   h_Q_sum;
        int   h_operator_counts[5];
        int   Q_target;
        float hopfion_flip_scale;
        int   h_num_active_regions;
    } topology;

    // ================================================================
    // ACCUMULATORS — Sparse photonic representations of coherent clusters
    // ================================================================
    struct {
        void* buf_accumulators;      // PhotonAccumulator* [ACCUMULATOR_MAX_COUNT]
        void* buf_count;             // int* (atomic counter, reset each frame)
        int   h_count;               // Host-side readback of accumulator count
    } accumulators;

    // ================================================================
    // ASYNC — Streams, events, pinned host memory
    // ================================================================
    struct {
        void* sample_stream;
        void* stats_stream;
        void* spawn_stream;
        void* stats_ready_event;
        void* spawn_ready_event;
        void* pinned_sample_metrics;
        void* pinned_spawn_count;
        bool  stats_pending;
        bool  spawn_pending;
    } async;

    // ================================================================
    // TIMING — Simulation clock
    // ================================================================
    struct {
        float sim_time;
        int   frame;
        int   threads;               // Block size (typically 256)
    } timing;
};
