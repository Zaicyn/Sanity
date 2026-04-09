// spirv_runtime.h — SPIRV-First Compute Runtime
// ===============================================
// The core representation is SPIRV. Hardware is an implementation detail.
// Vendors (NVIDIA, AMD, Intel) are compile-time extensions that translate
// SPIRV to native ISA. If no vendor is enabled, SPIRV runs via Vulkan
// (which every GPU driver already supports).
//
// Architecture:
//   GLSL source → glslc → SPIRV binary (core, ships with app)
//   SPIRV binary → vendor JIT → native ISA (extension, optional)
//   SPIRV binary → Vulkan driver → native execution (fallback, universal)
//
// This file defines the SPIRV kernel registry and dispatch interface.
// sim_backend.h provides the low-level alloc/free/sync operations.
// This layer adds kernel management on top.

#pragma once

#include <stdint.h>
#include <stddef.h>

// Maximum kernels that can be registered
#define SPIRV_MAX_KERNELS 32

// Maximum push constant size (Vulkan minimum guarantee is 128 bytes)
#define SPIRV_MAX_PUSH_CONSTANTS 128

// Kernel descriptor — loaded from .spv file
typedef struct {
    const char* name;           // "siphon", "passive_advection", etc.
    uint32_t*   spirv_code;     // SPIRV binary (owned, freed on cleanup)
    size_t      spirv_size;     // Size in bytes
    int         num_bindings;   // Number of SSBO bindings
    int         push_size;      // Size of push constants struct
    void*       pipeline;       // Vendor-specific compiled pipeline (opaque)
} SPIRVKernel;

// Kernel registry — all loaded compute shaders
typedef struct {
    SPIRVKernel kernels[SPIRV_MAX_KERNELS];
    int         count;
    const char* backend_name;   // "vulkan", "nvidia", "amd", "intel"
} SPIRVRegistry;

// ============================================================================
// API — Kernel lifecycle
// ============================================================================

// Initialize empty registry
static inline void spirv_registry_init(SPIRVRegistry* reg, const char* backend) {
    reg->count = 0;
    reg->backend_name = backend;
}

// Load a SPIRV binary from file and register it
// Returns kernel index, or -1 on failure
// The caller is responsible for loading the file; this just registers the binary.
static inline int spirv_register_kernel(
    SPIRVRegistry* reg,
    const char* name,
    uint32_t* spirv_code,
    size_t spirv_size,
    int num_bindings,
    int push_size)
{
    if (reg->count >= SPIRV_MAX_KERNELS) return -1;

    int idx = reg->count++;
    reg->kernels[idx].name = name;
    reg->kernels[idx].spirv_code = spirv_code;
    reg->kernels[idx].spirv_size = spirv_size;
    reg->kernels[idx].num_bindings = num_bindings;
    reg->kernels[idx].push_size = push_size;
    reg->kernels[idx].pipeline = NULL;  // Compiled lazily by backend

    return idx;
}

// Find kernel by name
static inline int spirv_find_kernel(const SPIRVRegistry* reg, const char* name) {
    for (int i = 0; i < reg->count; i++) {
        // Simple string compare (kernels are registered once at startup)
        const char* a = reg->kernels[i].name;
        const char* b = name;
        while (*a && *b && *a == *b) { a++; b++; }
        if (*a == 0 && *b == 0) return i;
    }
    return -1;
}

// ============================================================================
// Dispatch descriptor — what to execute
// ============================================================================

typedef struct {
    int         kernel_idx;     // Index into registry
    int         grid_x, grid_y, grid_z;
    int         block_x, block_y, block_z;
    void**      buffers;        // Array of buffer pointers (one per binding)
    int         num_buffers;
    void*       push_constants; // Push constant data
    int         push_size;
    void*       stream;         // Async stream (NULL = synchronous)
} SPIRVDispatch;
