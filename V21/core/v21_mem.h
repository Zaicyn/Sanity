/*
 * V21 MEMORY OPERATIONS — Backend-routed allocation macros
 * ========================================================
 *
 * Drop-in replacements for cudaMalloc/Free/Memset/Memcpy that route
 * through the V21 backend interface. When compiling with CUDA, these
 * resolve to the CUDA calls directly (zero overhead). When compiling
 * without CUDA, they use the backend function pointers.
 *
 * Usage in sim_init / sim_cleanup / sim_dispatch:
 *   Replace: cudaMalloc(&ptr, size)
 *   With:    V21_MALLOC(&ptr, size)
 *
 * The backend must be initialized before any V21_MALLOC call:
 *   ctx.backend = create_cuda_backend();  // or vulkan, or cpu
 *
 * License: Public domain / CC0
 */

#ifndef V21_MEM_H
#define V21_MEM_H

#include "v21_backend.h"

/*
 * Global backend pointer — set once at startup, used by all V21_M* macros.
 * This avoids passing the backend through every init/cleanup function.
 */
static V21Backend* v21_active_backend = NULL;

static inline void v21_set_backend(V21Backend* b) {
    v21_active_backend = b;
}

/* ========================================================================
 * ALLOCATION — routes through backend or falls back to CUDA
 * ======================================================================== */

#ifdef __CUDACC__
    /* When compiled by nvcc, use CUDA directly (zero overhead path) */
    #include <cuda_runtime.h>
    #define V21_MALLOC(ptr, size)      cudaMalloc((void**)(ptr), (size))
    #define V21_FREE(ptr)             cudaFree((ptr))
    #define V21_MEMSET(ptr, val, sz)  cudaMemset((ptr), (val), (sz))
    #define V21_MEMCPY(dst, src, sz, dir) cudaMemcpy((dst), (src), (sz), (dir))
    #define V21_MEMCPY_H2D            cudaMemcpyHostToDevice
    #define V21_MEMCPY_D2H            cudaMemcpyDeviceToHost
    #define V21_MEMCPY_D2D            cudaMemcpyDeviceToDevice
    #define V21_MALLOC_HOST(ptr, sz)  cudaMallocHost((void**)(ptr), (sz))
    #define V21_FREE_HOST(ptr)        cudaFreeHost((ptr))
    #define V21_SYNC()                cudaDeviceSynchronize()
    #define V21_STREAM_CREATE(ptr)    cudaStreamCreate((cudaStream_t*)(ptr))
    #define V21_STREAM_DESTROY(s)     cudaStreamDestroy((cudaStream_t)(s))
    #define V21_EVENT_CREATE(ptr)     cudaEventCreate((cudaEvent_t*)(ptr))
    #define V21_EVENT_DESTROY(e)      cudaEventDestroy((cudaEvent_t)(e))
#else
    /* Non-CUDA: route through backend function pointers */
    #define V21_MALLOC(ptr, size)     do { *(void**)(ptr) = v21_active_backend->alloc((size)); } while(0)
    #define V21_FREE(ptr)            v21_active_backend->free((ptr))
    #define V21_MEMSET(ptr, val, sz) v21_active_backend->memset((ptr), (val), (sz))
    #define V21_MEMCPY(dst, src, sz, dir) v21_active_backend->copy((dst), (src), (sz), (dir))
    #define V21_MEMCPY_H2D            V21_COPY_HOST_TO_DEVICE
    #define V21_MEMCPY_D2H            V21_COPY_DEVICE_TO_HOST
    #define V21_MEMCPY_D2D            V21_COPY_DEVICE_TO_DEVICE
    #define V21_MALLOC_HOST(ptr, sz)  do { *(void**)(ptr) = v21_active_backend->alloc_host((sz)); } while(0)
    #define V21_FREE_HOST(ptr)        v21_active_backend->free_host((ptr))
    #define V21_SYNC()                v21_active_backend->sync()
    #define V21_STREAM_CREATE(ptr)    do { *(void**)(ptr) = v21_active_backend->stream_create(); } while(0)
    #define V21_STREAM_DESTROY(s)     v21_active_backend->stream_destroy((s))
    #define V21_EVENT_CREATE(ptr)     do { *(void**)(ptr) = NULL; } while(0)  /* Events abstracted away */
    #define V21_EVENT_DESTROY(e)      ((void)0)
#endif

#endif /* V21_MEM_H */
