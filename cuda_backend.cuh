// cuda_backend.cuh — CUDA Implementation of SimBackend
// ====================================================
// Wraps existing cudaMalloc/Free/<<<>>> behind the abstract interface.
// Zero behavioral change — just indirection through function pointers.
//
// This is the OPTIONAL extension backend. Vulkan Compute is the default.
// Compile with -DENABLE_CUDA=ON to include this.
#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include "sim_backend.h"

// === Memory management ===

static void* cuda_alloc(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

static void cuda_free(void* ptr) {
    cudaFree(ptr);
}

static void cuda_memset(void* ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

static void cuda_copy(void* dst, const void* src, size_t size, enum SimCopyDir dir) {
    cudaMemcpyKind kind;
    switch (dir) {
        case SIM_COPY_HOST_TO_DEVICE:   kind = cudaMemcpyHostToDevice; break;
        case SIM_COPY_DEVICE_TO_HOST:   kind = cudaMemcpyDeviceToHost; break;
        case SIM_COPY_DEVICE_TO_DEVICE: kind = cudaMemcpyDeviceToDevice; break;
        default: kind = cudaMemcpyDefault; break;
    }
    cudaMemcpy(dst, src, size, kind);
}

// === Pinned host memory ===

static void* cuda_alloc_host(size_t size) {
    void* ptr = nullptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

static void cuda_free_host(void* ptr) {
    cudaFreeHost(ptr);
}

// === Synchronization ===

static void cuda_sync(void) {
    cudaDeviceSynchronize();
}

static void* cuda_stream_create(void) {
    cudaStream_t* stream = new cudaStream_t;
    cudaStreamCreate(stream);
    return stream;
}

static void cuda_stream_destroy(void* stream) {
    cudaStreamDestroy(*(cudaStream_t*)stream);
    delete (cudaStream_t*)stream;
}

static void cuda_stream_sync(void* stream) {
    cudaStreamSynchronize(*(cudaStream_t*)stream);
}

// === Kernel dispatch ===
// Note: CUDA kernel dispatch uses <<<>>> syntax which can't be called
// through a function pointer. The launch() function here is a stub —
// actual CUDA kernel launches go through sim_dispatch.cuh directly.
// This exists for interface completeness and future migration.

static void cuda_launch(const char* kernel_name,
                        int gx, int gy, int gz,
                        int bx, int by, int bz,
                        void** args, int num_args,
                        void* stream) {
    // CUDA kernels are dispatched directly via <<<>>> in sim_dispatch.cuh.
    // This stub exists for interface parity with Vulkan backend.
    // When migrating a kernel to the abstract interface, replace the
    // direct <<<>>> call with cuLaunchKernel here.
    (void)kernel_name; (void)gx; (void)gy; (void)gz;
    (void)bx; (void)by; (void)bz;
    (void)args; (void)num_args; (void)stream;
}

// === Backend creation ===

static inline SimBackend create_cuda_backend() {
    SimBackend b = {};
    b.alloc = cuda_alloc;
    b.free = cuda_free;
    b.memset = cuda_memset;
    b.copy = cuda_copy;
    b.alloc_host = cuda_alloc_host;
    b.free_host = cuda_free_host;
    b.sync = cuda_sync;
    b.stream_create = cuda_stream_create;
    b.stream_destroy = cuda_stream_destroy;
    b.stream_sync = cuda_stream_sync;
    b.launch = cuda_launch;
    b.name = "cuda";
    b.device_index = 0;

    // Query device memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    b.total_memory = total_mem;
    b.free_memory = free_mem;

    return b;
}

#endif // __CUDACC__
