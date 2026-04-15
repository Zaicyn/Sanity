/*
 * V21 CPU ALLOCATOR — Viviani-Grounded Memory Management
 * ======================================================
 *
 * Extracted from Resonance V9 (aizawa.cuh). Lock-free, forward-bias
 * allocator grounded in microtubule geometry.
 *
 * Features:
 *   - 16 size classes (64B to 64MB, period-4 quaternary encoding)
 *   - Shadow stride = 13 (one per microtubule protofilament span)
 *   - Viviani curve offset calculation (5D recirculation)
 *   - Hopf Q topological invariant for integrity checking
 *   - Seam-aware phase shifts at protofilament crossings
 *   - Lock-free flag queue for defect reporting
 *   - Free-list cache for fast reuse (viral hijacking)
 *
 * All CUDA-specific code stripped. Uses V21_ATOMIC_* macros from v21_types.h.
 *
 * License: Public domain / CC0
 */

#ifndef V21_ALLOC_CPU_H
#define V21_ALLOC_CPU_H

#include "v21_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* CPU allocator always uses host-side atomics, even when compiled by CUDA.
 * Override the V21_ATOMIC macros that would otherwise resolve to __device__ atomics. */
#ifdef __CUDACC__
  #undef V21_ATOMIC_ADD
  #undef V21_ATOMIC_AND
  #undef V21_ATOMIC_OR
  #undef V21_ATOMIC_XOR
  #undef V21_ATOMIC_CAS
  #undef V21_ATOMIC_LOAD
  #undef V21_ATOMIC_STORE
  /* Use GCC builtins on host side (nvcc compiles host code with GCC/Clang) */
  #define V21_ATOMIC_ADD(p, v)       __sync_fetch_and_add((p), (v))
  #define V21_ATOMIC_AND(p, v)       __sync_fetch_and_and((p), (v))
  #define V21_ATOMIC_OR(p, v)        __sync_fetch_and_or((p), (v))
  #define V21_ATOMIC_XOR(p, v)       __sync_fetch_and_xor((p), (v))
  #define V21_ATOMIC_CAS(p, cmp, v)  __sync_val_compare_and_swap((p), (cmp), (v))
  #define V21_ATOMIC_LOAD(p)         __sync_val_compare_and_swap((p), 0, 0)
  #define V21_ATOMIC_STORE(p, v)     __sync_lock_test_and_set((p), (v))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * POOL CONSTANTS
 * ======================================================================== */

#define V21_POOL_SIZE       (64ULL * 1024 * 1024)   /* 64 MB base pool */
#define V21_BLOCK_SIZE      64                       /* Cache-line aligned */
#define V21_UNIT_SIZE       256                      /* 4 blocks = 1 Hopfion unit */
#define V21_SHADOW_STRIDE   V21_PROTOFILAMENTS       /* 13 */
#define V21_FLAG_RING_SIZE  512                      /* Power-of-2 ring buffer */
#define V21_MAX_DEFECTS     (2 * V21_INTERFERENCE_CYCLE)  /* 52 */
#define V21_BIN_COUNT       16
#define V21_FREELIST_SLOTS  8
#define V21_FRACTAL_DEPTH   4
#define V21_MAX_EJECTED     4096
#define V21_EJECTED_EMPTY   0xFFFFFFFFu
#define V21_SEAM_DEFECT_THRESHOLD 5

/* Period-4 bin sizes (64B to 64MB) */
static const size_t V21_BIN_SIZES[V21_BIN_COUNT] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192,
    32768, 65536, 131072, 262144,
    1048576, 4194304, 16777216, 67108864
};

/* ========================================================================
 * TYPES
 * ======================================================================== */

/* v21_vec3_t defined in v21_types.h or v21_geometry.h — avoid duplicate */
#ifndef V21_VEC3_DEFINED
#define V21_VEC3_DEFINED
typedef struct { float x, y, z; } v21_vec3_t;
#endif

typedef struct {
    uint64_t primary;
    uint64_t shadow;
} v21_shadow_pair_t;

typedef struct {
    uint32_t chunk_id;
    uint32_t version;
    uint8_t  defect_type;   /* 0=parity, 1=topo drift, 2=overflow, 3=hopf, 4=seam */
    uint8_t  priority;
    uint16_t delta_hint;
} v21_flag_entry_t;

typedef struct {
    v21_flag_entry_t entries[V21_FLAG_RING_SIZE];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
} v21_flag_queue_t;

typedef struct {
    void*    slots[V21_FREELIST_SLOTS];
    uint32_t count;
} v21_freelist_cache_t;

/* ========================================================================
 * VIVIANI GEOMETRY (from Resonance V9 — unchanged)
 * ======================================================================== */

static inline v21_vec3_t v21_viviani_normal(float theta) {
    float s  = sinf(theta),        c  = cosf(theta);
    float s3 = sinf(3.0f * theta), c3 = cosf(3.0f * theta);
    float x = s  - 0.5f * s3;
    float y = -c + 0.5f * c3;
    float z = c  * c3;
    float norm = sqrtf(x*x + y*y + z*z);
    if (norm < 1e-6f) norm = 1.0f;
    return (v21_vec3_t){ x/norm, y/norm, z/norm };
}

static inline int v21_compute_hopf_q(size_t block_idx, size_t total_blocks,
                                      const size_t* offset_table, uint32_t num_units) {
    if (num_units == 0 || !offset_table || block_idx >= total_blocks) return 0;
    int unit_idx = (int)(block_idx % num_units);
    int prev = (unit_idx - 1 + (int)num_units) % (int)num_units;
    int next = (unit_idx + 1) % (int)num_units;
    long long dp = (long long)offset_table[unit_idx] - (long long)offset_table[prev];
    long long dn = (long long)offset_table[next]     - (long long)offset_table[unit_idx];
    int q = (int)(((llabs(dp) + llabs(dn)) / V21_BLOCK_SIZE) % V21_5D_MODULUS);
    return q % 4;
}

static inline int v21_fractal_bin(int base_bin, int level) {
    if (level > V21_FRACTAL_DEPTH || base_bin < 0 || base_bin >= V21_BIN_COUNT) return base_bin;
    return (base_bin + (level % 4)) % V21_BIN_COUNT;
}

static inline size_t v21_viviani_offset(int unit_index, int total_units) {
    if (total_units == 0) return 0;
    float theta = V21_TWO_PI * (float)unit_index / (float)total_units;
    v21_vec3_t n = v21_viviani_normal(theta);
    float projection = fabsf(n.z) * V21_HOPF_Q;
    size_t base   = (size_t)unit_index * V21_UNIT_SIZE;
    size_t offset = (size_t)(projection * V21_UNIT_SIZE) % V21_POOL_SIZE;
    int q = (int)(projection * (float)V21_5D_MODULUS) % V21_5D_MODULUS;
    size_t recirc = ((size_t)q * V21_BLOCK_SIZE) % V21_POOL_SIZE;
    return (base + offset + recirc) % V21_POOL_SIZE;
}

/* Period-4 protection */
static inline int v21_is_protected(int count, int bin) {
    return ((count + (bin / 4) * 4) % 4) == 0;
}

/* ========================================================================
 * SHADOW INVARIANT (integrity checking)
 * ======================================================================== */

static inline uint64_t v21_compute_invariant(const uint8_t* data, size_t size) {
    uint64_t inv = 0;
    const uint64_t* ptr = (const uint64_t*)data;
    size_t words = size / sizeof(uint64_t);
    for (size_t i = 0; i < words; i++) inv ^= ptr[i];
    inv ^= (inv >> 32);
    inv ^= (inv >> 16);
    inv ^= (inv >> 8);
    return inv;
}

static inline uint64_t v21_viviani_invariant(const uint8_t* data, size_t size,
                                              int block_index, int total_blocks,
                                              const size_t* offset_table, uint32_t num_units) {
    uint64_t base = v21_compute_invariant(data, size);
    float theta = V21_TWO_PI * (float)block_index / (float)total_blocks;
    v21_vec3_t n = v21_viviani_normal(theta);
    int q = v21_compute_hopf_q((size_t)block_index, (size_t)total_blocks, offset_table, num_units);
    uint64_t geo  = (uint64_t)(fabsf(n.z) * 255.0f);
    uint64_t hopf = ((uint64_t)q & 0x03ULL) << 56;
    return (base & 0xFF) | (geo << 8) | hopf;
}

/* ========================================================================
 * FLAG QUEUE (lock-free MPSC ring)
 * ======================================================================== */

static inline void v21_flag_queue_init(v21_flag_queue_t* fq) {
    memset(fq->entries, 0, sizeof(fq->entries));
    fq->head = 0; fq->tail = 0; fq->count = 0;
}

static inline int v21_flag_queue_push(v21_flag_queue_t* fq, v21_flag_entry_t entry) {
    uint32_t count = V21_ATOMIC_LOAD(&fq->count);
    if (count >= V21_FLAG_RING_SIZE - 1) return 0;
    uint32_t head = V21_ATOMIC_LOAD(&fq->head);
    fq->entries[head] = entry;
    V21_ATOMIC_STORE(&fq->head, (head + 1) % V21_FLAG_RING_SIZE);
    V21_ATOMIC_ADD(&fq->count, 1);
    return 1;
}

static inline int v21_flag_queue_pop(v21_flag_queue_t* fq, v21_flag_entry_t* out) {
    uint32_t count = V21_ATOMIC_LOAD(&fq->count);
    if (count == 0) return 0;
    uint32_t tail = V21_ATOMIC_LOAD(&fq->tail);
    *out = fq->entries[tail];
    V21_ATOMIC_STORE(&fq->tail, (tail + 1) % V21_FLAG_RING_SIZE);
    V21_ATOMIC_ADD(&fq->count, (uint32_t)-1);  /* decrement */
    return 1;
}

/* ========================================================================
 * FREE-LIST CACHE (viral hijacking)
 * ======================================================================== */

static inline void v21_freelist_init(v21_freelist_cache_t* fc) {
    memset(fc->slots, 0, sizeof(fc->slots));
    fc->count = 0;
}

static inline int v21_freelist_push(v21_freelist_cache_t* fc, void* ptr) {
    uint32_t c = V21_ATOMIC_LOAD(&fc->count);
    if (c >= V21_FREELIST_SLOTS) return 0;
    uint32_t old = V21_ATOMIC_CAS(&fc->count, c, c + 1);
    if (old == c) { fc->slots[c] = ptr; return 1; }
    return 0;
}

static inline void* v21_freelist_pop(v21_freelist_cache_t* fc) {
    uint32_t c = V21_ATOMIC_LOAD(&fc->count);
    if (c == 0) return NULL;
    uint32_t old = V21_ATOMIC_CAS(&fc->count, c, c - 1);
    if (old == c) { void* p = fc->slots[c - 1]; fc->slots[c - 1] = NULL; return p; }
    return NULL;
}

/* ========================================================================
 * MAIN ALLOCATOR STATE
 * ======================================================================== */

typedef struct {
    uint8_t*  arena;
    size_t    arena_size;
    size_t    fork_position;

    v21_shadow_pair_t*  shadows;
    size_t              shadow_count;

    v21_flag_queue_t    flag_queue;
    v21_freelist_cache_t freelist_caches[V21_BIN_COUNT];

    uint32_t allocated_blocks;
    uint32_t defect_count;
    uint32_t flip_state;
    uint32_t version;

    size_t*       offset_table;
    v21_vec3_t*   normal_table;
    uint32_t      num_units;

    /* Seam pathology counters */
    uint32_t seam_defect_count;
    uint32_t seam_catastrophe_count;
    uint32_t event_count;
} v21_cpu_allocator_t;

/* ========================================================================
 * LIFECYCLE
 * ======================================================================== */

static inline void v21_cpu_init(v21_cpu_allocator_t* va, size_t arena_size) {
    memset(va, 0, sizeof(*va));
    va->arena_size = (arena_size / 4096) * 4096;
    va->arena = (uint8_t*)aligned_alloc(4096, va->arena_size);
    if (!va->arena) { fprintf(stderr, "V21: arena alloc failed\n"); return; }
    memset(va->arena, 0, va->arena_size);

    va->num_units = (uint32_t)(va->arena_size / V21_UNIT_SIZE);
    va->offset_table = (size_t*)malloc(va->num_units * sizeof(size_t));
    va->normal_table = (v21_vec3_t*)malloc(va->num_units * sizeof(v21_vec3_t));

    for (uint32_t i = 0; i < va->num_units; i++) {
        va->offset_table[i] = v21_viviani_offset(i, va->num_units);
        float theta = V21_TWO_PI * (float)i / (float)va->num_units;
        va->normal_table[i] = v21_viviani_normal(theta);
    }

    va->shadow_count = va->arena_size / (V21_BLOCK_SIZE * V21_SHADOW_STRIDE);
    va->shadows = (v21_shadow_pair_t*)calloc(va->shadow_count, sizeof(v21_shadow_pair_t));

    v21_flag_queue_init(&va->flag_queue);
    for (int i = 0; i < V21_BIN_COUNT; i++)
        v21_freelist_init(&va->freelist_caches[i]);

    printf("[v21-cpu] Arena: %zu MB, %u units, %zu shadows (stride=%d)\n",
           va->arena_size / (1024*1024), va->num_units, va->shadow_count, V21_SHADOW_STRIDE);
}

static inline void v21_cpu_destroy(v21_cpu_allocator_t* va) {
    free(va->arena);
    free(va->offset_table);
    free(va->normal_table);
    free(va->shadows);
    memset(va, 0, sizeof(*va));
}

/* ========================================================================
 * BIN SELECTION
 * ======================================================================== */

static inline int v21_cpu_bin_index(v21_cpu_allocator_t* va, size_t size) {
    int base = V21_BIN_COUNT - 1;
    for (int i = 0; i < V21_BIN_COUNT; i++) {
        if (size <= V21_BIN_SIZES[i]) { base = i; break; }
    }
    return v21_fractal_bin(base, va->flip_state);
}

/* ========================================================================
 * ALLOCATION
 * ======================================================================== */

static inline void* v21_cpu_alloc(v21_cpu_allocator_t* va, size_t size) {
    int bin = v21_cpu_bin_index(va, size);
    size_t actual_size = V21_BIN_SIZES[bin];

    /* Try free-list cache first */
    void* cached = v21_freelist_pop(&va->freelist_caches[bin]);
    if (cached) {
        V21_ATOMIC_ADD(&va->allocated_blocks, 1);
        V21_ATOMIC_ADD(&va->event_count, 1);
        return cached;
    }

    /* Bump allocator */
    size_t blocks = (actual_size + V21_BLOCK_SIZE - 1) / V21_BLOCK_SIZE;
    size_t alloc_size = blocks * V21_BLOCK_SIZE;
    size_t old_fork = V21_ATOMIC_ADD(&va->fork_position, (uint32_t)alloc_size);

    if (old_fork + alloc_size > va->arena_size) {
        V21_ATOMIC_ADD(&va->fork_position, (uint32_t)(-(int32_t)alloc_size));
        return NULL;  /* Arena exhausted */
    }

    void* ptr = va->arena + old_fork;
    size_t block_idx = old_fork / V21_BLOCK_SIZE;

    /* Shadow invariant at stride boundaries */
    if (block_idx % V21_SHADOW_STRIDE == 0) {
        size_t shadow_idx = block_idx / V21_SHADOW_STRIDE;
        if (shadow_idx < va->shadow_count) {
            uint64_t inv = v21_viviani_invariant(
                (uint8_t*)ptr, actual_size,
                (int)block_idx, (int)(va->arena_size / V21_BLOCK_SIZE),
                va->offset_table, va->num_units);
            va->shadows[shadow_idx].primary = inv;
            /* Seam positions: store inverse-shifted */
            va->shadows[shadow_idx].shadow =
                (shadow_idx % V21_PROTOFILAMENTS == 0)
                ? v21_seam_inverse_shift(inv)
                : inv;
        }
    }

    V21_ATOMIC_ADD(&va->allocated_blocks, 1);
    V21_ATOMIC_ADD(&va->event_count, 1);
    return ptr;
}

/* ========================================================================
 * DEALLOCATION
 * ======================================================================== */

static inline void v21_cpu_free(v21_cpu_allocator_t* va, void* ptr, size_t size) {
    if (!ptr) return;
    int bin = v21_cpu_bin_index(va, size);

    if (v21_freelist_push(&va->freelist_caches[bin], ptr)) {
        V21_ATOMIC_ADD(&va->allocated_blocks, (uint32_t)-1);
        V21_ATOMIC_ADD(&va->event_count, 1);
        return;
    }

    /* Protected mode: flag for deferred reconciliation */
    if (v21_is_protected(va->allocated_blocks, bin)) {
        v21_flag_entry_t entry = {
            .chunk_id = (uint32_t)(((uint8_t*)ptr - va->arena) / V21_BLOCK_SIZE),
            .version  = va->version,
            .defect_type = 2,
            .priority = 100,
            .delta_hint = (uint16_t)(size / V21_BLOCK_SIZE)
        };
        v21_flag_queue_push(&va->flag_queue, entry);
    }

    V21_ATOMIC_ADD(&va->allocated_blocks, (uint32_t)-1);
    V21_ATOMIC_ADD(&va->event_count, 1);
}

#ifdef __cplusplus
}
#endif

#endif /* V21_ALLOC_CPU_H */
