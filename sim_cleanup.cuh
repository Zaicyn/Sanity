// sim_cleanup.cuh — Simulation Cleanup Functions
// ================================================
// Frees all CUDA buffers and streams allocated by sim_init.cuh functions.
// Rendering cleanup (Vulkan/OpenGL) stays in main — it's backend-specific.
#pragma once

#include <cuda_runtime.h>
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
    cudaStreamDestroy(diag.sample_stream);
    cudaStreamDestroy(diag.stats_stream);
    cudaEventDestroy(diag.stats_ready);
    cudaStreamDestroy(diag.spawn_stream);
    cudaEventDestroy(diag.spawn_ready);
    cudaFree(diag.d_stress_async);
    printf("[shutdown] CUDA streams destroyed\n");

    // 2. Free main particle data
    cudaFree(ctx.particles.buf_disk);
    printf("[shutdown] Main particle data freed\n");

    // 3. Free topology ring buffer and mip-tree
    topology_recorder_cleanup();
    printf("[shutdown] Topology ring buffer freed\n");
    mip_tree_cleanup();
    printf("[shutdown] Mip-tree hierarchy freed\n");

    // 4. Free diagnostic buffers
    cudaFree(diag.d_stress);
    cudaFree(diag.d_sample_indices);
    cudaFree(diag.d_sample_metrics[0]);
    cudaFree(diag.d_sample_metrics[1]);
    cudaFreeHost(diag.h_sample_metrics);
    cudaFreeHost(diag.h_spawn_pinned);
    cudaFree(diag.d_kr_sin_sum);
    cudaFree(diag.d_kr_cos_sum);
    cudaFree(diag.d_kr_count);
    cudaFree(diag.d_phase_hist);
    cudaFree(diag.d_phase_omega_sum);
    cudaFree(diag.d_phase_omega_sq);
    cudaFree(diag.d_spawn_idx);
    cudaFree(diag.d_spawn_success);
    printf("[shutdown] Diagnostic buffers freed\n");

    // 5. Free octree allocations
    if (octree.octreeEnabled) {
        cudaFree(octree.d_morton_keys);
        cudaFree(octree.d_xor_corners);
        cudaFree(octree.d_particle_ids);
        cudaFree(octree.d_octree_nodes);
        cudaFree(octree.d_node_count);
        cudaFree(octree.d_leaf_counts);
        cudaFree(octree.d_leaf_counts_culled);
        cudaFree(octree.d_leaf_offsets);
        cudaFree(octree.d_leaf_node_indices);
        cudaFree(octree.d_leaf_node_count);
        cudaFree(octree.d_leaf_vel_x);
        cudaFree(octree.d_leaf_vel_y);
        cudaFree(octree.d_leaf_vel_z);
        cudaFree(octree.d_leaf_phase);
        cudaFree(octree.d_leaf_frequency);
        cudaFree(octree.d_leaf_coherence);
        cudaFree(octree.d_leaf_hash_keys);
        cudaFree(octree.d_leaf_hash_values);
        printf("[shutdown] Octree data freed\n");
    }

    // 6. Free topology buffers
    cudaFree(ctx.topology.buf_in_active_region);
    cudaFree(ctx.topology.buf_Q_sum);
    cudaFree(ctx.topology.buf_Q_delta_sum);
    cudaFree(ctx.topology.buf_operator_counts);
    cudaFree(ctx.topology.buf_cell_topo_s);
    cudaFree(ctx.topology.buf_cell_topo_cnt);
    cudaFree(ctx.topology.buf_active_regions);

    // 7. Free grid buffers (null-safe — cudaFree(nullptr) is a no-op)
    cudaFree(ctx.grid.buf_density);
    cudaFree(ctx.grid.buf_momentum_x);
    cudaFree(ctx.grid.buf_momentum_y);
    cudaFree(ctx.grid.buf_momentum_z);
    cudaFree(ctx.grid.buf_phase_sin);
    cudaFree(ctx.grid.buf_phase_cos);
    cudaFree(ctx.grid.buf_pressure_x);
    cudaFree(ctx.grid.buf_pressure_y);
    cudaFree(ctx.grid.buf_pressure_z);
    cudaFree(ctx.grid.buf_vorticity_x);
    cudaFree(ctx.grid.buf_vorticity_y);
    cudaFree(ctx.grid.buf_vorticity_z);
    cudaFree(ctx.grid.buf_R_cell);
    cudaFree(ctx.grid.buf_particle_cell);
    cudaFree(ctx.grid.buf_active_flags);
    cudaFree(ctx.grid.buf_tile_flags);
    cudaFree(ctx.grid.buf_compact_active_list);
    cudaFree(ctx.grid.buf_compact_active_count);
    cudaFree(ctx.grid.buf_active_tiles);
    cudaFree(ctx.grid.buf_active_tile_count);

    printf("[shutdown] Cleanup complete!\n");
}
