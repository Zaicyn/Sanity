// frame_export.cuh — Binary frame export for offline validation
// ==============================================================
//
// Exports particle state to binary files for the SuperValidator pipeline.
// Binary format per particle (32 bytes):
//   [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, pump_scale, pump_residual]
//
// Usage:
//   - Press 'X' to export current frame
//   - Or enable continuous export with g_validation_export_enabled = true
//
// Python validator reads these with:
//   data = np.fromfile("frame_0000.bin", dtype=np.float32).reshape(-1, 8)
//
// Math.md mapping:
//   - Positions feed into Hopf invariant Q = integral(A . F)
//   - Velocities become vector potential A via NGP deposition
//   - pump_scale encodes local phase coherence (topology proxy)
//   - pump_residual tracks entropy leakage (w-drift indicator)

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Forward declaration - GPUDisk defined in disk.cuh which is included before this header
struct GPUDisk;

// ============================================================================
// Configuration
// ============================================================================

#ifndef VALIDATOR_FRAME_DIR
#define VALIDATOR_FRAME_DIR "frames/"
#endif

#ifndef VALIDATOR_MAX_PARTICLES_EXPORT
#define VALIDATOR_MAX_PARTICLES_EXPORT 10000000  // 10M max per frame (320 MB)
#endif

// Global state for validation mode
static bool g_validation_export_enabled = false;
static int  g_validation_frame_counter = 0;
static int  g_validation_export_interval = 100;  // Export every N frames when continuous

// Stack capture mode: dump 64 consecutive frames then stop
static bool g_stack_capture_active = false;
static int  g_stack_capture_remaining = 0;
static const int STACK_CAPTURE_FRAMES = 64;  // Full phase stack (2 samples per 32 complexity layers)

// ============================================================================
// Host-side frame export
// ============================================================================

// Export frame to binary file
// Returns true on success, false on error
inline bool exportFrameBinary(
    GPUDisk* d_disk,           // Device pointer to disk
    int N_particles,           // Number of active particles
    int frame_id,              // Frame number for filename
    const char* output_dir = VALIDATOR_FRAME_DIR)
{
    // Clamp particle count
    if (N_particles > VALIDATOR_MAX_PARTICLES_EXPORT) {
        printf("[validator] Warning: Clamping export from %d to %d particles\n",
               N_particles, VALIDATOR_MAX_PARTICLES_EXPORT);
        N_particles = VALIDATOR_MAX_PARTICLES_EXPORT;
    }

    // Allocate host buffers
    size_t float_size = N_particles * sizeof(float);
    float* h_pos_x = (float*)malloc(float_size);
    float* h_pos_y = (float*)malloc(float_size);
    float* h_pos_z = (float*)malloc(float_size);
    float* h_vel_x = (float*)malloc(float_size);
    float* h_vel_y = (float*)malloc(float_size);
    float* h_vel_z = (float*)malloc(float_size);
    float* h_pump_scale = (float*)malloc(float_size);
    float* h_pump_residual = (float*)malloc(float_size);

    if (!h_pos_x || !h_pos_y || !h_pos_z || !h_vel_x || !h_vel_y || !h_vel_z ||
        !h_pump_scale || !h_pump_residual) {
        printf("[validator] Error: Failed to allocate host buffers for %d particles\n", N_particles);
        free(h_pos_x); free(h_pos_y); free(h_pos_z);
        free(h_vel_x); free(h_vel_y); free(h_vel_z);
        free(h_pump_scale); free(h_pump_residual);
        return false;
    }

    // Copy from device (async would be nice but we need sync for file write)
    cudaMemcpy(h_pos_x, d_disk->pos_x, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_y, d_disk->pos_y, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_z, d_disk->pos_z, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_x, d_disk->vel_x, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_y, d_disk->vel_y, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_z, d_disk->vel_z, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pump_scale, d_disk->pump_scale, float_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pump_residual, d_disk->pump_residual, float_size, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[validator] Error: CUDA memcpy failed: %s\n", cudaGetErrorString(err));
        free(h_pos_x); free(h_pos_y); free(h_pos_z);
        free(h_vel_x); free(h_vel_y); free(h_vel_z);
        free(h_pump_scale); free(h_pump_residual);
        return false;
    }

    // Build filename
    char filename[256];
    snprintf(filename, sizeof(filename), "%sframe_%04d.bin", output_dir, frame_id);

    // Open file
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("[validator] Error: Failed to open %s for writing\n", filename);
        free(h_pos_x); free(h_pos_y); free(h_pos_z);
        free(h_vel_x); free(h_vel_y); free(h_vel_z);
        free(h_pump_scale); free(h_pump_residual);
        return false;
    }

    // Write header: magic + particle count
    uint32_t magic = 0x48505846;  // "HPFX" (HoPFion eXport)
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&N_particles, sizeof(int), 1, f);

    // Write interleaved data: [px, py, pz, vx, vy, vz, scale, residual] per particle
    for (int i = 0; i < N_particles; i++) {
        float row[8] = {
            h_pos_x[i], h_pos_y[i], h_pos_z[i],
            h_vel_x[i], h_vel_y[i], h_vel_z[i],
            h_pump_scale[i], h_pump_residual[i]
        };
        fwrite(row, sizeof(float), 8, f);
    }

    fclose(f);

    // Cleanup
    free(h_pos_x); free(h_pos_y); free(h_pos_z);
    free(h_vel_x); free(h_vel_y); free(h_vel_z);
    free(h_pump_scale); free(h_pump_residual);

    printf("[validator] Exported frame %d: %d particles to %s (%.1f MB)\n",
           frame_id, N_particles, filename, (N_particles * 8 * 4 + 12) / 1e6);

    return true;
}

// Export metadata file with simulation parameters
inline void exportValidationMetadata(
    int frame_id,
    int N_particles,
    float sim_time,
    float heartbeat,
    float avg_scale,
    float avg_residual,
    int grid_dim,
    float L,
    const char* output_dir = VALIDATOR_FRAME_DIR)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "%smeta_%04d.txt", output_dir, frame_id);

    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "# Hopfion Validation Metadata\n");
    fprintf(f, "frame_id=%d\n", frame_id);
    fprintf(f, "n_particles=%d\n", N_particles);
    fprintf(f, "sim_time=%.6f\n", sim_time);
    fprintf(f, "heartbeat=%.6f\n", heartbeat);
    fprintf(f, "avg_scale=%.6f\n", avg_scale);
    fprintf(f, "avg_residual=%.6f\n", avg_residual);
    fprintf(f, "grid_dim=%d\n", grid_dim);
    fprintf(f, "L=%.1f\n", L);
    fprintf(f, "# Derived:\n");
    fprintf(f, "dx=%.6f\n", L / grid_dim);

    fclose(f);
}

// ============================================================================
// Stack capture mode (64 consecutive frames)
// ============================================================================

// Start a stack capture - clears frames/ and exports 64 consecutive frames
inline void startStackCapture() {
    system("rm -rf " VALIDATOR_FRAME_DIR "frame_*.bin " VALIDATOR_FRAME_DIR "meta_*.txt");
    system("mkdir -p " VALIDATOR_FRAME_DIR);
    g_stack_capture_active = true;
    g_stack_capture_remaining = STACK_CAPTURE_FRAMES;
    g_validation_frame_counter = 0;
    printf("[validator] Stack capture started: dumping %d consecutive frames\n", STACK_CAPTURE_FRAMES);
}

// Check if stack capture is in progress (simulation should sync/wait during capture)
inline bool isStackCaptureActive() {
    return g_stack_capture_active;
}

// Export frame during stack capture - returns true if capture still active
inline bool maybeExportStackFrame(
    GPUDisk* d_disk,
    int N_particles,
    float sim_time,
    float heartbeat,
    float avg_scale,
    float avg_residual,
    int grid_dim,
    float L)
{
    if (!g_stack_capture_active) return false;

    // Export this frame
    exportFrameBinary(d_disk, N_particles, g_validation_frame_counter);
    exportValidationMetadata(g_validation_frame_counter, N_particles, sim_time,
                             heartbeat, avg_scale, avg_residual, grid_dim, L);

    g_validation_frame_counter++;
    g_stack_capture_remaining--;

    if (g_stack_capture_remaining <= 0) {
        g_stack_capture_active = false;
        printf("[validator] Stack capture complete: %d frames saved to %s\n",
               STACK_CAPTURE_FRAMES, VALIDATOR_FRAME_DIR);
        return false;  // Capture finished
    }

    // Print progress every 8 frames
    if (g_validation_frame_counter % 8 == 0) {
        printf("[validator] Stack capture: %d/%d frames\n",
               g_validation_frame_counter, STACK_CAPTURE_FRAMES);
    }

    return true;  // Capture still active
}

// ============================================================================
// Continuous export mode (legacy - exports every N frames)
// ============================================================================

inline void maybeExportFrame(
    GPUDisk* d_disk,
    int N_particles,
    int frame_number,
    float sim_time,
    float heartbeat,
    float avg_scale,
    float avg_residual,
    int grid_dim,
    float L)
{
    if (!g_validation_export_enabled) return;
    if (frame_number % g_validation_export_interval != 0) return;

    // Create frames directory if needed (first export)
    static bool dir_created = false;
    if (!dir_created) {
        system("mkdir -p " VALIDATOR_FRAME_DIR);
        dir_created = true;
    }

    exportFrameBinary(d_disk, N_particles, g_validation_frame_counter);
    exportValidationMetadata(g_validation_frame_counter, N_particles, sim_time,
                             heartbeat, avg_scale, avg_residual, grid_dim, L);
    g_validation_frame_counter++;
}
