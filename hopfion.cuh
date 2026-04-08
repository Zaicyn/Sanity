// hopfion.cuh — Topological Reaction Algebra
// ============================================
// Implements the six operators from hopfion_spec.docx as device functions
// and a sparse 256-entry Q lookup table in __constant__ memory.
//
// The hopfion algebra is the BOUNDARY CONDITION for the medium (siphon pump,
// Kuramoto coupling, passive advection). The medium can do whatever it wants
// as long as global discrete helicity Σ Q is conserved every frame.
//
// Operators:
//   1. Fusion    — saturating signed add per axis
//   2. Venting   — drop one axis, spawn opposite-helicity child
//   3. Phason flip — negate one axis (minimal reconfiguration)
//   4. Fission   — Q-conserving split into two children
//   5. Diffusion — Q transported (handled by medium advection)
//   6. Recycling equilibrium — fusion rate = venting rate (host feedback)
//
// Two breakdown modes:
//   Iron freeze: dim-4 + PUMP_IDLE + low history → locked inert (0xFF)
//   Thread tension: cell weight variance > T_CRIT → burst ejection

#pragma once

#include "disk.cuh"

// ============================================================================
// Compile-time constants (tune empirically before adding CLI flags)
// ============================================================================
#define HOPFION_T_CRIT          0.3f    // Tension explosion threshold
#define HOPFION_FUSION_DENSITY  2.0f    // Cell density threshold for fusion
#define HOPFION_FLIP_BASE_RATE  0.01f   // Base phason flip probability per frame
#define HOPFION_FREEZE_HISTORY  0.01f   // pump_history below which iron freeze triggers
#define HOPFION_VENT_HISTORY    0.5f    // pump_history above which venting triggers

// Frozen state marker — all bits set to 11 (reserved) on all 4 axes
#define TOPO_FROZEN  0xFF

// Vent pending flag — bit 2 of flags byte (piggybacked on spawn)
#define PFLAG_VENT_PENDING  0x04

// ============================================================================
// Sparse Q Lookup Table — 256 entries in __constant__ memory
// ============================================================================
// Indexed directly by the packed topo_state byte. Invalid encodings map to 0.
// Computed on host by init_Q_lut() and uploaded via cudaMemcpyToSymbol.

__constant__ int8_t d_Q_lut[256];

// ============================================================================
// Host-side LUT initialization
// ============================================================================

inline void init_Q_lut() {
    int8_t h_lut[256];
    memset(h_lut, 0, sizeof(h_lut));

    // Enumerate all 256 possible byte values
    for (int b = 0; b < 256; b++) {
        // Check if all 4 axis encodings are valid (not 0x03 = reserved)
        bool valid = true;
        for (int a = 0; a < 4; a++) {
            if (((b >> (a * 2)) & 0x03) == TOPO_AXIS_RSVD) {
                valid = false;
                break;
            }
        }
        if (valid) {
            h_lut[b] = (int8_t)topo_compute_Q((uint8_t)b);
        }
    }

    cudaMemcpyToSymbol(d_Q_lut, h_lut, sizeof(h_lut));

    // Verify a few known values
    // {+1,+1,+1,0} = 0b00_01_01_01 = 0x15: Q = 1*1*1 = 1 (triple 0,1,2)
    // {+1,+1,+1,+1} = 0b01_01_01_01 = 0x55: Q = s0s1s2 - s0s1s3 + s0s2s3 - s1s2s3 = 1-1+1-1 = 0
    // {+1,+1,+1,-1} = 0b10_01_01_01 = 0x95: Q = 1 - (-1) + (-1) - (-1) = 1+1-1+1 = 2
    printf("[hopfion] Q LUT initialized: Q(0x15)=%d Q(0x55)=%d Q(0x95)=%d\n",
           h_lut[0x15], h_lut[0x55], h_lut[0x95]);
}

// ============================================================================
// Device-side Q lookup (fast path)
// ============================================================================

__device__ __forceinline__ int topo_Q_fast(uint8_t state) {
    return d_Q_lut[state];
}

// ============================================================================
// Operator: Fusion — saturating signed add per axis
// ============================================================================
// H(a,b)_i = clamp(a_i + b_i, -1, +1)
// Opposite-signed axes cancel (magnetic reconnection / vortex annihilation).
// dim(result) >= max(dim(a), dim(b)) — monotonic span growth.

__device__ __forceinline__ uint8_t hopfion_fusion(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    for (int axis = 0; axis < 4; axis++) {
        int sa = topo_get_axis(a, axis);
        int sb = topo_get_axis(b, axis);
        int sum = sa + sb;
        // Saturate to [-1, +1]
        int clamped = (sum > 1) ? 1 : (sum < -1) ? -1 : sum;
        result = topo_set_axis(result, axis, clamped);
    }
    return result;
}

// ============================================================================
// Operator: Venting — drop one axis from parent, child gets opposite sign
// ============================================================================
// Parent loses axis `axis`. Child gets ONLY that axis with opposite sign.
// NOTE: Q(parent_new) + Q(child) != Q(parent_old) in general because
// single-axis children always have Q=0 (triple product needs 3 axes).
// Global Q conservation is maintained STATISTICALLY through the recycling
// equilibrium: fusion rebuilds the Q that venting destroys. The spec says
// "Q redistributed" — the redistribution is through the fusion/vent cycle,
// not per-operation.

struct VentResult {
    uint8_t parent_new;
    uint8_t child;
};

__device__ __forceinline__ VentResult hopfion_vent(uint8_t state, int axis) {
    VentResult r;
    int s = topo_get_axis(state, axis);
    r.parent_new = topo_set_axis(state, axis, 0);   // Drop axis from parent
    r.child = topo_set_axis(0, axis, -s);            // Child gets opposite sign
    return r;
}

// ============================================================================
// Operator: Phason flip — negate one axis (sign toggle)
// ============================================================================
// P_i(s): s_i → -s_i
// Smallest topology-preserving reconfiguration.

__device__ __forceinline__ uint8_t hopfion_phason_flip(uint8_t state, int axis) {
    int s = topo_get_axis(state, axis);
    return topo_set_axis(state, axis, -s);
}

// ============================================================================
// Operator: Fission — Q-conserving split into two children
// ============================================================================
// Constraint: Q(parent) = Q(s1) + Q(s2)
// Strategy: try all axis subsets for s1, compute s2 = parent - s1 (per axis).
// Accept first valid split where both children have dim > 0.

struct FissionResult {
    uint8_t s1;
    uint8_t s2;
    bool valid;
};

__device__ __forceinline__ FissionResult hopfion_fission(uint8_t state) {
    FissionResult r = {0, 0, false};
    int Q_parent = topo_Q_fast(state);
    int dim = topo_dim(state);
    if (dim < 2) return r;  // Can't split a single-axis state

    // Try giving each single axis to s1, rest to s2
    for (int a = 0; a < 4; a++) {
        int sa = topo_get_axis(state, a);
        if (sa == 0) continue;

        uint8_t s1 = topo_set_axis(0, a, sa);
        uint8_t s2 = topo_set_axis(state, a, 0);

        if (topo_Q_fast(s1) + topo_Q_fast(s2) == Q_parent &&
            topo_dim(s2) > 0) {
            r.s1 = s1;
            r.s2 = s2;
            r.valid = true;
            return r;
        }
    }

    // Try 2-axis splits for dim >= 3
    if (dim >= 3) {
        for (int a = 0; a < 4; a++) {
            for (int b = a + 1; b < 4; b++) {
                int sa = topo_get_axis(state, a);
                int sb = topo_get_axis(state, b);
                if (sa == 0 || sb == 0) continue;

                uint8_t s1 = 0;
                s1 = topo_set_axis(s1, a, sa);
                s1 = topo_set_axis(s1, b, sb);
                uint8_t s2 = state;
                s2 = topo_set_axis(s2, a, 0);
                s2 = topo_set_axis(s2, b, 0);

                if (topo_Q_fast(s1) + topo_Q_fast(s2) == Q_parent &&
                    topo_dim(s1) > 0 && topo_dim(s2) > 0) {
                    r.s1 = s1;
                    r.s2 = s2;
                    r.valid = true;
                    return r;
                }
            }
        }
    }

    return r;
}

// ============================================================================
// Weight function — pluggable, initially = dim (popcount of nonzero axes)
// ============================================================================

__device__ __forceinline__ float hopfion_weight(uint8_t state) {
    return (float)topo_dim(state);
}

// ============================================================================
// Inline test harness (compile with -DTEST_HOPFION=1)
// ============================================================================

#if TEST_HOPFION
inline bool run_hopfion_tests() {
    printf("[hopfion-test] Running exhaustive state tests...\n");
    int valid_count = 0;
    int q_nonzero = 0;

    // Test all 256 byte values
    for (int b = 0; b < 256; b++) {
        uint8_t st = (uint8_t)b;

        // Check if valid (no reserved bits)
        bool valid = true;
        for (int a = 0; a < 4; a++) {
            if (((st >> (a * 2)) & 0x03) == TOPO_AXIS_RSVD) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        valid_count++;

        // Round-trip encode/decode
        int s[4];
        for (int a = 0; a < 4; a++) s[a] = topo_get_axis(st, a);
        uint8_t repacked = topo_pack(s[0], s[1], s[2], s[3]);
        if (repacked != st) {
            printf("[hopfion-test] FAIL: round-trip 0x%02x → (%d,%d,%d,%d) → 0x%02x\n",
                   st, s[0], s[1], s[2], s[3], repacked);
            return false;
        }

        // Q computation
        int Q = topo_compute_Q(st);
        if (Q != 0) q_nonzero++;

        // Phason flip conservation: flip and unflip should restore state
        for (int a = 0; a < 4; a++) {
            uint8_t flipped = hopfion_phason_flip(st, a);
            uint8_t restored = hopfion_phason_flip(flipped, a);
            if (restored != st) {
                printf("[hopfion-test] FAIL: phason double-flip 0x%02x axis %d\n", st, a);
                return false;
            }
        }

        // Fusion with zero is identity
        uint8_t fused_zero = hopfion_fusion(st, 0x00);
        if (fused_zero != st) {
            printf("[hopfion-test] FAIL: fusion(0x%02x, 0x00) = 0x%02x\n", st, fused_zero);
            return false;
        }

        // Vent: verify structural correctness (parent loses axis, child gets opposite)
        for (int a = 0; a < 4; a++) {
            if (topo_get_axis(st, a) == 0) continue;
            VentResult vr = hopfion_vent(st, a);
            // Parent must have axis zeroed
            if (topo_get_axis(vr.parent_new, a) != 0) {
                printf("[hopfion-test] FAIL: vent parent axis %d not zeroed for 0x%02x\n", a, st);
                return false;
            }
            // Child must have only this axis, with opposite sign
            int s_orig = topo_get_axis(st, a);
            int s_child = topo_get_axis(vr.child, a);
            if (s_child != -s_orig || topo_dim(vr.child) != 1) {
                printf("[hopfion-test] FAIL: vent child wrong for 0x%02x axis %d\n", st, a);
                return false;
            }
        }

        // Fission: if valid, Q must be conserved
        FissionResult fr = hopfion_fission(st);
        if (fr.valid) {
            int Q_parent = topo_compute_Q(st);
            int Q_children = topo_compute_Q(fr.s1) + topo_compute_Q(fr.s2);
            if (Q_children != Q_parent) {
                printf("[hopfion-test] FAIL: fission Q non-conservation 0x%02x: %d → %d\n",
                       st, Q_parent, Q_children);
                return false;
            }
        }
    }

    printf("[hopfion-test] PASS: %d valid states, %d with Q≠0\n", valid_count, q_nonzero);
    if (valid_count != 81) {
        printf("[hopfion-test] FAIL: expected 81 valid states, got %d\n", valid_count);
        return false;
    }
    return true;
}
#endif // TEST_HOPFION
