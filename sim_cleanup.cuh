// sim_cleanup.cuh — Simulation Cleanup Functions
// ================================================
// Frees all CUDA buffers and streams allocated by sim_init.cuh functions.
// Rendering cleanup (Vulkan/OpenGL) stays in main — it's backend-specific.
#pragma once

#include "V21/core/v21_mem.h"
#include <cstdio>
#include "sim_context.h"
#include "sim_init.cuh"  // DiagnosticLocals, OctreeLocals, etc.

// ============================================================================
// cleanupSimulation — Free all CUDA resources from the SimulationContext
// ============================================================================
// Call after the main loop exits, AFTER rendering cleanup.
// Frees streams, events, device buffers, pinned host memory.

inline void cleanupSimulation(
    SimulationContext& ctx,
    DiagnosticLocals& diag,
    OctreeLocals& octree)
{
    // 1. Destroy streams and events
    V21_STREAM_DESTROY(diag.sample_stream);
    V21_STREAM_DESTROY(diag.stats_stream);
    V21_EVENT_DESTROY(diag.stats_ready);
    V21_STREAM_DESTROY(diag.spawn_stream);
    V21_EVENT_DESTROY(diag.spawn_ready);
    V21_FREE(diag.d_stress_async);
    printf("[shutdown] CUDA streams destroyed\n");

    // 2. Free main particle data
    V21_FREE(ctx.particles.buf_disk);
    printf("[shutdown] Main particle data freed\n");

    // 3. Free topology ring buffer and mip-tree
    topology_recorder_cleanup();
    printf("[shutdown] Topology ring buffer freed\n");
    mip_tree_cleanup();
    printf("[shutdown] Mip-tree hierarchy freed\n");

    // 4. Free diagnostic buffers
    V21_FREE(diag.d_stress);
    V21_FREE(diag.d_sample_indices);
    V21_FREE(diag.d_sample_metrics[0]);
    V21_FREE(diag.d_sample_metrics[1]);
    V21_FREE_HOST(diag.h_sample_metrics);
    V21_FREE_HOST(diag.h_spawn_pinned);
    V21_FREE(diag.d_kr_sin_sum);
    V21_FREE(diag.d_kr_cos_sum);
    V21_FREE(diag.d_kr_count);
    V21_FREE(diag.d_phase_hist);
    V21_FREE(diag.d_phase_omega_sum);
    V21_FREE(diag.d_phase_omega_sq);
    V21_FREE(diag.d_spawn_idx);
    V21_FREE(diag.d_spawn_success);
    printf("[shutdown] Diagnostic buffers freed\n");

    // 5. Free octree allocations
    if (octree.octreeEnabled) {
        V21_FREE(octree.d_morton_keys);
        V21_FREE(octree.d_xor_corners);
        V21_FREE(octree.d_particle_ids);
        V21_FREE(octree.d_octree_nodes);
        V21_FREE(octree.d_node_count);
        V21_FREE(octree.d_leaf_counts);
        V21_FREE(octree.d_leaf_counts_culled);
        V21_FREE(octree.d_leaf_offsets);
        V21_FREE(octree.d_leaf_node_indices);
        V21_FREE(octree.d_leaf_node_count);
        V21_FREE(octree.d_leaf_vel_x);
        V21_FREE(octree.d_leaf_vel_y);
        V21_FREE(octree.d_leaf_vel_z);
        V21_FREE(octree.d_leaf_phase);
        V21_FREE(octree.d_leaf_frequency);
        V21_FREE(octree.d_leaf_coherence);
        V21_FREE(octree.d_leaf_hash_keys);
        V21_FREE(octree.d_leaf_hash_values);
        printf("[shutdown] Octree data freed\n");
    }

    // 6. Free topology buffers
    V21_FREE(ctx.topology.buf_in_active_region);
    V21_FREE(ctx.topology.buf_Q_sum);
    V21_FREE(ctx.topology.buf_Q_delta_sum);
    V21_FREE(ctx.topology.buf_operator_counts);
    V21_FREE(ctx.topology.buf_cell_topo_s);
    V21_FREE(ctx.topology.buf_cell_topo_cnt);
    V21_FREE(ctx.topology.buf_active_regions);

    // 7. Free accumulators
    V21_FREE(ctx.accumulators.buf_accumulators);
    V21_FREE(ctx.accumulators.buf_count);

    // 8. Free grid buffers (null-safe — V21_FREE(nullptr) is a no-op)
    V21_FREE(ctx.grid.buf_density);
    V21_FREE(ctx.grid.buf_momentum_x);
    V21_FREE(ctx.grid.buf_momentum_y);
    V21_FREE(ctx.grid.buf_momentum_z);
    V21_FREE(ctx.grid.buf_phase_sin);
    V21_FREE(ctx.grid.buf_phase_cos);
    V21_FREE(ctx.grid.buf_pressure_x);
    V21_FREE(ctx.grid.buf_pressure_y);
    V21_FREE(ctx.grid.buf_pressure_z);
    V21_FREE(ctx.grid.buf_vorticity_x);
    V21_FREE(ctx.grid.buf_vorticity_y);
    V21_FREE(ctx.grid.buf_vorticity_z);
    V21_FREE(ctx.grid.buf_R_cell);
    V21_FREE(ctx.grid.buf_particle_cell);
    V21_FREE(ctx.grid.buf_active_flags);
    V21_FREE(ctx.grid.buf_tile_flags);
    V21_FREE(ctx.grid.buf_compact_active_list);
    V21_FREE(ctx.grid.buf_compact_active_count);
    V21_FREE(ctx.grid.buf_active_tiles);
    V21_FREE(ctx.grid.buf_active_tile_count);

    printf("[shutdown] Cleanup complete!\n");
}
