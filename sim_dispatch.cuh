// sim_dispatch.cuh — Simulation Dispatch Layer
// ==============================================
// Thin orchestrator functions that cast void* to typed pointers and
// launch kernels. This is the ONLY file that touches CUDA launch syntax.
// When a CPU backend is added, a parallel sim_dispatch_cpu.h replaces
// the kernel launches with OpenMP parallel_for loops.
#pragma once

#include <cuda_runtime.h>
#include "sim_context.h"
#include "disk.cuh"
#include "physics_constants.cuh"
#include "active_region.cuh"
#include "hopfion.cuh"
#include "spawn.cuh"
#include "passive_advection.cuh"

// ============================================================================
// dispatchCorePhysics — Active region mask → Siphon → Passive → Hopfion → Spawn
// ============================================================================
// The core physics pipeline, executed once per frame. All other subsystems
// (octree, grid, diagnostics) are dispatched separately.

inline void dispatchCorePhysics(
    SimulationContext& ctx,
    float sim_time, float dt,
    int frame,
    uint8_t seam_bits, float bias,
    // Spawn state (mutated)
    int& N_current, int& spawn_blocks, int threads,
    bool& spawn_pending,
    // Hopfion state
    float g_hopfion_flip_scale)
{
    GPUDisk* d_disk = static_cast<GPUDisk*>(ctx.particles.buf_disk);
    uint8_t* d_in_active_region = static_cast<uint8_t*>(ctx.topology.buf_in_active_region);
    ActiveRegion* d_active_regions = static_cast<ActiveRegion*>(ctx.topology.buf_active_regions);
    int* d_Q_sum = static_cast<int*>(ctx.topology.buf_Q_sum);
    int* d_operator_counts = static_cast<int*>(ctx.topology.buf_operator_counts);
    int* d_cell_topo_s = static_cast<int*>(ctx.topology.buf_cell_topo_s);
    int* d_cell_topo_cnt = static_cast<int*>(ctx.topology.buf_cell_topo_cnt);
    unsigned int* d_spawn_idx = static_cast<unsigned int*>(ctx.diagnostics.buf_spawn_idx);
    unsigned int* d_spawn_success = static_cast<unsigned int*>(ctx.diagnostics.buf_spawn_success);
    unsigned int* h_spawn_pinned = static_cast<unsigned int*>(ctx.async.pinned_spawn_count);
    float* d_grid_density = static_cast<float*>(ctx.grid.buf_density);
    cudaStream_t spawn_stream = *static_cast<cudaStream_t*>(ctx.async.spawn_stream);
    cudaEvent_t spawn_ready = *static_cast<cudaEvent_t*>(ctx.async.spawn_ready_event);

    spawn_blocks = (N_current + threads - 1) / threads;

    // 1. Classify active/passive particles
#if ENABLE_PASSIVE_ADVECTION
    extern float g_corner_threshold;
    computeInActiveRegionMask<<<spawn_blocks, threads>>>(
        d_disk, d_active_regions, ctx.topology.h_num_active_regions,
        d_in_active_region, N_current, g_corner_threshold);
#endif

    // 2. Full siphon physics on active particles
    siphonDiskKernel<<<spawn_blocks, threads>>>(
        d_disk, d_in_active_region, N_current,
        sim_time, dt * 2.0f, seam_bits, bias);

    // 3. Cheap Keplerian advection on passive particles
#if ENABLE_PASSIVE_ADVECTION
    extern float g_passive_residual_tau;
    advectPassiveParticles<<<spawn_blocks, threads>>>(
        d_disk, d_in_active_region, N_current,
        dt * 2.0f, g_passive_residual_tau);
#endif

    // 4. Hopfion topological enforcement
    {
        int clear_blocks = (g_grid_cells + threads - 1) / threads;
        clearTopoBuffers<<<clear_blocks, threads>>>(
            d_cell_topo_s, d_cell_topo_cnt, g_grid_cells);
        scatterTopoToCells<<<spawn_blocks, threads>>>(
            d_disk, N_current, d_cell_topo_s, d_cell_topo_cnt);
        cudaMemset(d_Q_sum, 0, sizeof(int));
        cudaMemset(d_operator_counts, 0, 5 * sizeof(int));
        hopfionEnforceKernel<<<spawn_blocks, threads>>>(
            d_disk, d_in_active_region, N_current,
            d_cell_topo_s, d_cell_topo_cnt,
            d_Q_sum, d_operator_counts, sim_time, g_hopfion_flip_scale);
    }

    // 5. Natural growth (Toomre instability spawning)
    if (SPAWN_ENABLE && N_current < MAX_DISK_PTS) {
        // Apply previous frame's spawns (async, one-frame lag)
        if (spawn_pending) {
            cudaError_t status = cudaEventQuery(spawn_ready);
            if (status == cudaSuccess) {
                unsigned int h_spawned = *h_spawn_pinned;
                spawn_pending = false;

                if (h_spawned > 0) {
                    int new_total = N_current + (int)h_spawned;
                    if (new_total > RUNTIME_PARTICLE_CAP) {
                        static bool oom_warned = false;
                        if (!oom_warned) {
                            printf("[OOM PROTECTION] Particle cap reached: %d (limit %d)\n",
                                   N_current, RUNTIME_PARTICLE_CAP);
                            oom_warned = true;
                        }
                        h_spawned = 0;
                    }
                    N_current += h_spawned;
                    spawn_blocks = (N_current + threads - 1) / threads;

                    static int last_log_frame = 0;
                    if (frame - last_log_frame >= 10000) {
                        printf("[growth] Frame %d: %d particles (%.1f%% capacity)\n",
                               frame, N_current,
                               100.0f * N_current / (float)MAX_DISK_PTS);
                        last_log_frame = frame;
                    }
                }
            }
        }

        // Launch spawn for this frame
        extern bool g_spawn_enabled;
        if (g_spawn_enabled) {
            cudaMemsetAsync(d_spawn_idx, 0, sizeof(unsigned int), spawn_stream);
            cudaMemsetAsync(d_spawn_success, 0, sizeof(unsigned int), spawn_stream);

            unsigned int spawn_seed = (unsigned int)(frame * 12345 + (int)(sim_time * 1000));
            spawnParticlesKernel<<<spawn_blocks, threads, 0, spawn_stream>>>(
                d_disk, N_current, MAX_DISK_PTS, d_spawn_idx, d_spawn_success,
                sim_time, spawn_seed, d_in_active_region, d_grid_density);

            cudaMemcpyAsync(h_spawn_pinned, d_spawn_success, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost, spawn_stream);
            cudaEventRecord(spawn_ready, spawn_stream);
            spawn_pending = true;
        }
    }

    // Update context particle count
    ctx.particles.N_current = N_current;
}
