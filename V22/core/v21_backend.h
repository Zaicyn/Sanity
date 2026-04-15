// v21_backend.h — Abstract Compute Backend Interface
// ===================================================
// Decouples simulation logic from GPU vendor.
// Primary backend: Vulkan Compute (vendor-neutral)
// Optional extension: CUDA (compile with -DENABLE_CUDA=ON)
//
// All simulation code calls backend->alloc(), backend->launch(), etc.
// The backend implementation handles the vendor-specific dispatch.
//
// Usage:
//   V21Backend* backend = create_vulkan_backend(vkCtx);
//   // or: V21Backend* backend = create_cuda_backend();
//
//   void* buf = backend->alloc(1024);
//   backend->launch("siphonDiskKernel", num_blocks, 256, args, num_args);
//   backend->sync();
//   backend->free(buf);

#ifndef V21_BACKEND_H
#define V21_BACKEND_H
#include <stddef.h>
#include <stdint.h>

// Memory transfer direction (matches cudaMemcpyKind ordering)
enum V21CopyDir {
    V21_COPY_HOST_TO_DEVICE = 1,
    V21_COPY_DEVICE_TO_HOST = 2,
    V21_COPY_DEVICE_TO_DEVICE = 3
};

// Abstract compute backend — function pointer table
// Each backend (Vulkan, CUDA, CPU) provides its own implementation.
typedef struct V21Backend {
    // === Memory management ===
    void* (*alloc)(size_t size);
    void  (*free)(void* ptr);
    void  (*memset)(void* ptr, int value, size_t size);
    void  (*copy)(void* dst, const void* src, size_t size, enum V21CopyDir dir);

    // === Pinned host memory (for async transfers) ===
    void* (*alloc_host)(size_t size);
    void  (*free_host)(void* ptr);

    // === Synchronization ===
    void  (*sync)(void);                       // Global device sync
    void* (*stream_create)(void);              // Create async stream
    void  (*stream_destroy)(void* stream);     // Destroy async stream
    void  (*stream_sync)(void* stream);        // Sync one stream

    // === Kernel dispatch ===
    // kernel_name: string identifier (maps to .spv or __global__ function)
    // grid/block: dispatch dimensions
    // args: array of pointers to kernel arguments
    // num_args: length of args array
    // stream: async stream (NULL = default/synchronous)
    void  (*launch)(const char* kernel_name,
                    int grid_x, int grid_y, int grid_z,
                    int block_x, int block_y, int block_z,
                    void** args, int num_args,
                    void* stream);

    // === Backend info ===
    const char* name;           // "vulkan", "cuda", "cpu"
    int         device_index;   // GPU device index
    size_t      total_memory;   // Total device memory (bytes)
    size_t      free_memory;    // Available device memory (bytes)

    // === Backend-specific context (opaque) ===
    void* ctx;
} V21Backend;

// Convenience macros for common 1D dispatch pattern
#define V21_LAUNCH_1D(backend, name, n, block_size, args, nargs) \
    (backend)->launch((name), \
        ((n) + (block_size) - 1) / (block_size), 1, 1, \
        (block_size), 1, 1, \
        (args), (nargs), NULL)

#define V21_LAUNCH_1D_STREAM(backend, name, n, block_size, args, nargs, stream) \
    (backend)->launch((name), \
        ((n) + (block_size) - 1) / (block_size), 1, 1, \
        (block_size), 1, 1, \
        (args), (nargs), (stream))
#endif /* V21_BACKEND_H */
