/*
 * V21 TYPES — Portable Type Definitions and Atomic Abstractions
 * ==============================================================
 *
 * The foundation layer for V21's vendor-neutral compute backend.
 * All code above this layer uses V21_ATOMIC_* macros and v21_* types.
 * The macros resolve to platform-native atomics at compile time:
 *
 *   CUDA:    __device__ atomicAdd/And/Or/CAS/Xor
 *   C11:     stdatomic (CPU path)
 *   GLSL:    native atomicAdd/atomicCompSwap (SPIRV path)
 *   Win32:   Interlocked* (fallback)
 *
 * Ground truth: 13-protofilament microtubule (from Resonance V9).
 * Every timing and stride constant flows from this. No magic numbers.
 *
 * License: Public domain / CC0
 */

#ifndef V21_TYPES_H
#define V21_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * MICROTUBULE-GROUNDED CONSTANTS
 *
 * Ground truth: 13-protofilament microtubule (12 parallel + 1 seam).
 * From Resonance V9 / Squaragon — identical constants in both.
 * ======================================================================== */

#define V21_PROTOFILAMENTS      13   /* Total protofilaments */
#define V21_PARALLEL            12   /* In-phase strands (bulk flow) */
#define V21_SEAM                 1   /* Quadrature strand (half-step trigger) */

/* Half-step fires after 12 parallel operations (one seam crossing) */
#define V21_HALFSTEP_INTERVAL   V21_PARALLEL   /* 12 */

/* Full interference cycle: leading(12) + seam(1) + lagging(12) + seam(1) = 26 */
#define V21_INTERFERENCE_CYCLE  (2 * (V21_PARALLEL + V21_SEAM))  /* 26 */

/* Window: smallest power-of-2 container for one cycle */
#define V21_INTERFERENCE_WINDOW 32   /* 2^5 >= 26 */

/* Seam phase shift: π/2 scaled by topological defect strength */
#define V21_HOPF_Q              1.97f
#define V21_SEAM_STRENGTH       0.03f   /* 2.0 - HOPF_Q */
#define V21_SEAM_PHASE_SHIFT    2       /* round(64 * SEAM_STRENGTH) = 2 bits */

/* ========================================================================
 * HARMONIC CONSTANTS (from Viviani curve)
 * ======================================================================== */

#define V21_PHI          1.6180339887498948f  /* Golden ratio */
#define V21_SCALE_RATIO  1.6875f              /* 27/16 coherent scaling */
#define V21_BIAS         0.75f                /* Pump bias (25% dissonance) */
#define V21_INV_SQRT2    0.7071067811865475f
#define V21_TWO_PI       6.28318530717959f

/* Viviani 5D modulus */
#define V21_5D_MODULUS   8

/* ========================================================================
 * SLAB ALLOCATOR CONSTANTS
 * ======================================================================== */

#define V21_SLAB_CLASSES     3    /* 64B, 128B, 256B */
#define V21_SLAB_CLASS_64    0
#define V21_SLAB_CLASS_128   1
#define V21_SLAB_CLASS_256   2

#define V21_SUPERBLOCK_SIZE  4096  /* Bytes per superblock */
#define V21_SUPERBLOCK_ALIGN 4096  /* Alignment */

/* Slots per superblock per class */
#define V21_SLOTS_64    (V21_SUPERBLOCK_SIZE / 64)   /* 64 */
#define V21_SLOTS_128   (V21_SUPERBLOCK_SIZE / 128)  /* 32 */
#define V21_SLOTS_256   (V21_SUPERBLOCK_SIZE / 256)  /* 16 */

/* Shell count for half-step phase alternation */
#define V21_NUM_SHELLS  2

/* Warp/workgroup sizes */
#define V21_WARP_SIZE       32   /* NVIDIA/AMD common */
#define V21_WORKGROUP_SIZE  256  /* Default compute workgroup */

/* ========================================================================
 * ATOMIC ABSTRACTIONS
 *
 * Platform-specific atomics resolved at compile time.
 * GLSL compute shaders use native atomics directly (no macro needed).
 * ======================================================================== */

#ifdef __CUDACC__
    /* CUDA device atomics */
    #define V21_ATOMIC_ADD(p, v)       atomicAdd((p), (v))
    #define V21_ATOMIC_ADD_U64(p, v)   atomicAdd((unsigned long long*)(p), (unsigned long long)(v))
    #define V21_ATOMIC_AND(p, v)       atomicAnd((uint32_t*)(p), (v))
    #define V21_ATOMIC_OR(p, v)        atomicOr((uint32_t*)(p), (v))
    #define V21_ATOMIC_XOR(p, v)       atomicXor((uint32_t*)(p), (v))
    #define V21_ATOMIC_CAS(p, cmp, v)  atomicCAS((uint32_t*)(p), (cmp), (v))
    #define V21_ATOMIC_LOAD(p)         (*(volatile uint32_t*)(p))
    #define V21_ATOMIC_STORE(p, v)     (*(volatile uint32_t*)(p) = (v))

#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    /* C11 stdatomic (CPU path) */
    #include <stdatomic.h>

    #define V21_ATOMIC_ADD(p, v)       atomic_fetch_add((_Atomic uint32_t*)(p), (v))
    #define V21_ATOMIC_ADD_U64(p, v)   atomic_fetch_add((_Atomic uint64_t*)(p), (v))
    #define V21_ATOMIC_AND(p, v)       atomic_fetch_and((_Atomic uint32_t*)(p), (v))
    #define V21_ATOMIC_OR(p, v)        atomic_fetch_or((_Atomic uint32_t*)(p), (v))
    #define V21_ATOMIC_XOR(p, v)       atomic_fetch_xor((_Atomic uint32_t*)(p), (v))
    #define V21_ATOMIC_CAS(p, cmp, v)  v21_atomic_cas_c11((uint32_t*)(p), (cmp), (v))
    #define V21_ATOMIC_LOAD(p)         atomic_load((_Atomic uint32_t*)(p))
    #define V21_ATOMIC_STORE(p, v)     atomic_store((_Atomic uint32_t*)(p), (v))

    static inline uint32_t v21_atomic_cas_c11(uint32_t* ptr, uint32_t expected, uint32_t desired) {
        uint32_t exp = expected;
        atomic_compare_exchange_strong((_Atomic uint32_t*)ptr, &exp, desired);
        return exp;  /* Returns old value (matches CUDA semantics) */
    }

#elif defined(_MSC_VER)
    /* Win32 Interlocked (MSVC) */
    #include <intrin.h>
    #define V21_ATOMIC_ADD(p, v)       _InterlockedExchangeAdd((volatile long*)(p), (long)(v))
    #define V21_ATOMIC_AND(p, v)       _InterlockedAnd((volatile long*)(p), (long)(v))
    #define V21_ATOMIC_OR(p, v)        _InterlockedOr((volatile long*)(p), (long)(v))
    #define V21_ATOMIC_XOR(p, v)       _InterlockedXor((volatile long*)(p), (long)(v))
    #define V21_ATOMIC_CAS(p, cmp, v)  _InterlockedCompareExchange((volatile long*)(p), (long)(v), (long)(cmp))
    #define V21_ATOMIC_LOAD(p)         (*(volatile uint32_t*)(p))
    #define V21_ATOMIC_STORE(p, v)     (*(volatile uint32_t*)(p) = (v))

#else
    /* GCC/Clang builtins (fallback) */
    #define V21_ATOMIC_ADD(p, v)       __sync_fetch_and_add((p), (v))
    #define V21_ATOMIC_AND(p, v)       __sync_fetch_and_and((p), (v))
    #define V21_ATOMIC_OR(p, v)        __sync_fetch_and_or((p), (v))
    #define V21_ATOMIC_XOR(p, v)       __sync_fetch_and_xor((p), (v))
    #define V21_ATOMIC_CAS(p, cmp, v)  __sync_val_compare_and_swap((p), (cmp), (v))
    #define V21_ATOMIC_LOAD(p)         __sync_val_compare_and_swap((p), 0, 0)
    #define V21_ATOMIC_STORE(p, v)     __sync_lock_test_and_set((p), (v))
#endif

/* ========================================================================
 * SEAM PHASE SHIFT (from V19 / squaragon)
 *
 * 2-bit rotation in 64-bit invariants at seam crossings.
 * Identical implementation in aizawa.cuh and squaragon.h.
 * ======================================================================== */

static inline uint64_t v21_seam_forward_shift(uint64_t v) {
    return (v >> V21_SEAM_PHASE_SHIFT) | (v << (64 - V21_SEAM_PHASE_SHIFT));
}

static inline uint64_t v21_seam_inverse_shift(uint64_t v) {
    return (v << V21_SEAM_PHASE_SHIFT) | (v >> (64 - V21_SEAM_PHASE_SHIFT));
}

/* ========================================================================
 * ANALOG NULLABLE CONVENTIONS (from Sanity/nullable.h)
 *
 * In continuous simulation, exact 0.0 only occurs by explicit assignment.
 * ======================================================================== */

#define V21_IS_NULL_F(val)      ((val) == 0.0f)
#define V21_IS_PRESENT_F(val)   ((val) != 0.0f)
#define V21_INDEX_NULL          (-1)
#define V21_IS_INDEX_NULL(i)    ((i) == V21_INDEX_NULL)

#ifdef __cplusplus
}
#endif

#endif /* V21_TYPES_H */
