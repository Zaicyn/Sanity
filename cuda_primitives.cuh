// cuda_primitives.cuh — Native GPU Primitives
// =============================================
//
// No external dependencies (V8 philosophy):
//   "Everything correct stays untouched forever."
//
// These are warp/block-cooperative primitives built from first principles:
//   - Warp/block reduce (sum)
//   - Global reduce kernel
//   - Blelloch exclusive scan (work-efficient)
//   - Count elements less than threshold
//   - Hash-based deduplication (O(N) without sorting)
//
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Warp-Level Primitives
// ============================================================================

__device__ __forceinline__ uint32_t warpReduceSum(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceSumF(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Block-Level Primitives
// ============================================================================

__device__ __forceinline__ uint32_t blockReduceSum(uint32_t val) {
    __shared__ uint32_t shared[32];  // One per warp
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp does final reduction
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// ============================================================================
// Global Reduce Kernel
// ============================================================================

__global__ void reduceKernel(const uint32_t* d_in, uint32_t* d_out, int N) {
    uint32_t sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        sum += d_in[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) atomicAdd(d_out, sum);
}

// Host wrapper: reduce sum
static uint32_t* g_reduce_out = nullptr;

inline uint32_t gpuReduceSum(uint32_t* d_in, int N) {
    if (!g_reduce_out) cudaMalloc(&g_reduce_out, sizeof(uint32_t));
    cudaMemset(g_reduce_out, 0, sizeof(uint32_t));

    int blocks = min((N + 255) / 256, 256);
    reduceKernel<<<blocks, 256>>>(d_in, g_reduce_out, N);

    uint32_t result;
    cudaMemcpy(&result, g_reduce_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================================
// Blelloch Exclusive Scan (work-efficient)
// ============================================================================

// Phase 1: Up-sweep (reduce)
__global__ void scanUpSweep(uint32_t* d_data, int N, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x + 1) * stride * 2 - 1;
    if (idx < N) {
        d_data[idx] += d_data[idx - stride];
    }
}

// Phase 2: Down-sweep
__global__ void scanDownSweep(uint32_t* d_data, int N, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x + 1) * stride * 2 - 1;
    if (idx < N) {
        uint32_t temp = d_data[idx - stride];
        d_data[idx - stride] = d_data[idx];
        d_data[idx] += temp;
    }
}

// Simple single-block scan for small arrays (< 1024 elements)
__global__ void scanSmallKernel(uint32_t* d_in, uint32_t* d_out, int N) {
    __shared__ uint32_t temp[1024];
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2*tid] = (2*tid < N) ? d_in[2*tid] : 0;
    temp[2*tid+1] = (2*tid+1 < N) ? d_in[2*tid+1] : 0;

    // Up-sweep (reduce) phase
    for (int d = 512; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Set last element to 0 for exclusive scan
    if (tid == 0) temp[1023] = 0;

    // Down-sweep phase
    for (int d = 1; d < 1024; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid+1) - 1;
            int bi = offset * (2*tid+2) - 1;
            uint32_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output
    if (2*tid < N) d_out[2*tid] = temp[2*tid];
    if (2*tid+1 < N) d_out[2*tid+1] = temp[2*tid+1];
}

// Host wrapper: exclusive scan
inline void gpuExclusiveScan(uint32_t* d_in, uint32_t* d_out, int N) {
    if (N <= 1024) {
        // Small array: single-block scan
        scanSmallKernel<<<1, 512>>>(d_in, d_out, N);
    } else {
        // Large array: multi-pass Blelloch
        // Copy input to output (in-place scan)
        cudaMemcpy(d_out, d_in, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // Up-sweep
        for (int stride = 1; stride < N; stride <<= 1) {
            int threads = N / (stride * 2);
            if (threads > 0) {
                int blocks = (threads + 255) / 256;
                scanUpSweep<<<blocks, min(threads, 256)>>>(d_out, N, stride);
            }
        }

        // Set last element to 0
        cudaMemset(d_out + N - 1, 0, sizeof(uint32_t));

        // Down-sweep
        for (int stride = N/2; stride >= 1; stride >>= 1) {
            int threads = N / (stride * 2);
            if (threads > 0) {
                int blocks = (threads + 255) / 256;
                scanDownSweep<<<blocks, min(threads, 256)>>>(d_out, N, stride);
            }
        }
    }
}

// ============================================================================
// Count Elements Less Than Threshold
// ============================================================================

__global__ void countLessThanKernel(const uint64_t* keys, int N, uint64_t threshold, uint32_t* count) {
    uint32_t local_count = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        if (keys[i] < threshold) local_count++;
    }

    local_count = blockReduceSum(local_count);
    if (threadIdx.x == 0) atomicAdd(count, local_count);
}

// Host wrapper: count less than
static uint32_t* g_count_out = nullptr;

inline uint32_t gpuCountLessThan(uint64_t* d_keys, int N, uint64_t threshold) {
    if (!g_count_out) cudaMalloc(&g_count_out, sizeof(uint32_t));
    cudaMemset(g_count_out, 0, sizeof(uint32_t));

    int blocks = min((N + 255) / 256, 256);
    countLessThanKernel<<<blocks, 256>>>(d_keys, N, threshold, g_count_out);

    uint32_t result;
    cudaMemcpy(&result, g_count_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================================
// Hash-Based Deduplication (O(N) without sorting)
// ============================================================================
// Replaces thrust::sort + thrust::unique
// Uses atomic CAS for lock-free hash table insertion

__global__ void hashDedupKernel(uint32_t* input, int N, uint32_t* output,
                                 uint32_t* out_count, uint32_t* hash_table, int table_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint32_t val = input[tid];

    // Simple hash: value mod table_size
    uint32_t hash = val % table_size;

    // Linear probe to find empty slot or existing value
    for (int probe = 0; probe < 32; probe++) {  // Max 32 probes
        uint32_t slot = (hash + probe) % table_size;
        uint32_t old = atomicCAS(&hash_table[slot], 0xFFFFFFFF, val);

        if (old == 0xFFFFFFFF) {
            // Successfully inserted — add to output
            uint32_t idx = atomicAdd(out_count, 1);
            output[idx] = val;
            return;
        } else if (old == val) {
            // Already exists — duplicate, skip
            return;
        }
        // Collision with different value — continue probing
    }
    // Too many collisions — add anyway (rare, allows duplicates)
    uint32_t idx = atomicAdd(out_count, 1);
    output[idx] = val;
}

// Pre-allocated static buffers to avoid malloc/free thrashing
static uint32_t* g_dedup_hash_table = nullptr;
static uint32_t* g_dedup_count = nullptr;
static int g_dedup_table_size = 0;

inline void ensureDedupBuffers(int table_size) {
    if (g_dedup_table_size < table_size) {
        if (g_dedup_hash_table) cudaFree(g_dedup_hash_table);
        if (g_dedup_count) cudaFree(g_dedup_count);
        g_dedup_table_size = table_size;
        cudaMalloc(&g_dedup_hash_table, table_size * sizeof(uint32_t));
        cudaMalloc(&g_dedup_count, sizeof(uint32_t));
    }
}

// Returns number of unique elements (output in d_output)
inline uint32_t hashDedup(uint32_t* d_input, int N, uint32_t* d_output, int max_output) {
    if (N == 0) return 0;

    // Use fixed table size (2M entries = 8MB)
    int table_size = 2 * 1024 * 1024;
    ensureDedupBuffers(table_size);

    // Initialize hash table to "empty" marker (0xFFFFFFFF)
    cudaMemset(g_dedup_hash_table, 0xFF, table_size * sizeof(uint32_t));
    cudaMemset(g_dedup_count, 0, sizeof(uint32_t));

    int blocks = (N + 255) / 256;
    hashDedupKernel<<<blocks, 256>>>(d_input, N, d_output, g_dedup_count, g_dedup_hash_table, table_size);

    uint32_t result;
    cudaMemcpy(&result, g_dedup_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (result > (uint32_t)max_output) result = max_output;
    return result;
}
