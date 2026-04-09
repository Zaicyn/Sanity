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
#include "cell_grid.cuh"  // cellIndexFromPos, d_grid_cells, GRID_HALF_SIZE

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
    // Analog nullable: 0x00 = unoccupied (Q=0), 0xFF = frozen (Q=0)
    // Any other byte with a reserved axis encoding (0x03) is invalid —
    // the LUT maps it to 0 silently. In debug builds, flag it.
#if defined(DEBUG_HOPFION) && DEBUG_HOPFION
    if (state != 0x00 && state != 0xFF) {
        for (int a = 0; a < 4; a++) {
            if (((state >> (a * 2)) & 0x03) == TOPO_AXIS_RSVD) {
                printf("[hopfion] WARNING: invalid topo_state 0x%02x (axis %d = reserved)\n", state, a);
                break;
            }
        }
    }
#endif
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

// ============================================================================
// Cell-level topo scatter — accumulate per-axis state sums per grid cell
// ============================================================================
// Same pattern as scatterParticlesToCells. Each particle atomically adds
// its axis values to the cell's topo accumulators. The enforcement kernel
// reads these to compute cell-average state for fusion and weight variance
// for tension explosion.

__global__ void scatterTopoToCells(
    const GPUDisk* __restrict__ disk,
    int N,
    int* __restrict__ cell_topo_s,   // 4 arrays of GRID_CELLS ints, interleaved: [axis][cell]
    int* __restrict__ cell_topo_cnt  // particle count per cell
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !particle_active(disk, i)) return;
    if (disk->flags[i] & PFLAG_EJECTED) return;

    uint8_t state = disk->topo_state[i];
    if (state == TOPO_FROZEN) return;  // frozen particles don't contribute

    uint32_t cell = cellIndexFromPos(disk->pos_x[i], disk->pos_y[i], disk->pos_z[i]);
    if (cell >= (uint32_t)d_grid_cells) return;

    for (int a = 0; a < 4; a++) {
        int s = topo_get_axis(state, a);
        if (s != 0) atomicAdd(&cell_topo_s[a * d_grid_cells + cell], s);
    }
    atomicAdd(&cell_topo_cnt[cell], 1);
}

// Clear cell topo buffers
__global__ void clearTopoBuffers(
    int* __restrict__ cell_topo_s,
    int* __restrict__ cell_topo_cnt,
    int total_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_cells) return;
    cell_topo_s[0 * total_cells + i] = 0;
    cell_topo_s[1 * total_cells + i] = 0;
    cell_topo_s[2 * total_cells + i] = 0;
    cell_topo_s[3 * total_cells + i] = 0;
    cell_topo_cnt[i] = 0;
}

// ============================================================================
// Enforcement Kernel — topological boundary conditions on the medium
// ============================================================================
// Runs AFTER siphonDiskKernel and advectPassiveParticles, BEFORE spawn.
// Default stream, sequential with medium physics.
//
// Per-particle operators applied:
//   - Phason flip: stochastic, rate ∝ |pump_residual|. Relaxation only
//     (accept if dim doesn't increase).
//   - Iron freeze: dim-4 + PUMP_IDLE + low history → locked (0xFF).
//
// Fusion and tension explosion are deferred to H5 (need cell-level topo).
// Venting is deferred to H6 (needs spawn piggybacking).
//
// Global Q is accumulated via warp-level reduction + atomicAdd.

__global__ void hopfionEnforceKernel(
    GPUDisk* disk,
    const uint8_t* __restrict__ in_active_region,
    int N,
    const int* __restrict__ cell_topo_s,   // 4 × GRID_CELLS axis sums (nullptr before H5)
    const int* __restrict__ cell_topo_cnt, // particle count per cell (nullptr before H5)
    int* __restrict__ d_Q_sum,
    int* __restrict__ d_Q_delta_sum,     // Writer monad: tracks per-particle Q changes for conservation
    int* __restrict__ d_operator_counts,  // [0]=flips, [1]=freezes, [2]=fusions, [3]=tensions, [4]=vents
    float sim_time,
    float flip_rate_scale  // host-side recycling equilibrium multiplier
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip dead/ejected/passive particles
    uint8_t flags = disk->flags[i];
    if (!(flags & PFLAG_ACTIVE)) return;
    if (flags & PFLAG_EJECTED) return;
    if (!in_active_region[i]) return;

    uint8_t state = disk->topo_state[i];
    int Q_before = (state == TOPO_FROZEN) ? 0 : topo_Q_fast(state);  // Writer monad: capture Q before ops

    // Simple LCG RNG seeded from thread index + time
    unsigned int rng = (unsigned int)(i * 2654435761u + __float_as_uint(sim_time));

    // Skip frozen particles
    if (state == TOPO_FROZEN) {
        goto reduce;
    }

    // --- Fusion: saturating add with cell-average topo state ---
    if (cell_topo_cnt != nullptr) {
        uint32_t cell = cellIndexFromPos(disk->pos_x[i], disk->pos_y[i], disk->pos_z[i]);
        if (cell < (uint32_t)d_grid_cells) {
            int cnt = cell_topo_cnt[cell];
            if (cnt >= 2) {
                // Compute cell-average topo state (rounded to nearest {-1,0,+1})
                uint8_t cell_avg = 0;
                for (int a = 0; a < 4; a++) {
                    int sum = cell_topo_s[a * d_grid_cells + cell];
                    // Round: if |sum/cnt| > 0.5, the axis is active in the cell average
                    int avg = (2 * abs(sum) > cnt) ? (sum > 0 ? 1 : -1) : 0;
                    cell_avg = topo_set_axis(cell_avg, a, avg);
                }

                // Fuse if cell average differs from current state AND result
                // has at least as many axes (monotonic span growth)
                uint8_t fused = hopfion_fusion(state, cell_avg);
                if (fused != state && topo_dim(fused) >= topo_dim(state)) {
                    // Stochastic: probability ∝ density / FUSION_DENSITY threshold
                    float density = (float)cnt;
                    float fuse_prob = fminf(density / (HOPFION_FUSION_DENSITY * 100.0f), 0.1f);
                    rng = rng * 1664525u + 1013904223u;
                    float r = (float)(rng & 0xFFFF) / 65536.0f;
                    if (r < fuse_prob) {
                        state = fused;
                        atomicAdd(&d_operator_counts[2], 1);
                    }
                }

                // --- Tension explosion: weight variance too high ---
                // Approximate variance from axis sums: if axes are polarized
                // (|sum| close to cnt) the cell is coherent; if near 0 despite
                // high count, the cell has canceling orientations = tension
                float coherence = 0.0f;
                for (int a = 0; a < 4; a++) {
                    int sum = cell_topo_s[a * d_grid_cells + cell];
                    coherence += (float)(sum * sum) / (float)(cnt * cnt);
                }
                coherence *= 0.25f;  // normalize to [0,1]

                float tension = 1.0f - coherence;  // high tension = low coherence
                if (tension > HOPFION_T_CRIT && topo_dim(state) >= 3) {
                    // Rate-limit: use cell-level atomic to allow max 1 ejection per cell
                    // We don't have a per-cell flag array, so use stochastic rate-limiting
                    rng = rng * 1664525u + 1013904223u;
                    float r2 = (float)(rng & 0xFFFF) / 65536.0f;
                    if (r2 < 0.001f) {  // Very low probability per particle → ~1 per cell
                        disk->flags[i] |= PFLAG_EJECTED;
                        atomicAdd(&d_operator_counts[3], 1);
                    }
                }
            }
        }
    }

    // --- Phason flip: stochastic relaxation ---
    {
        float residual = fabsf(disk->pump_residual[i]);
        float flip_prob = HOPFION_FLIP_BASE_RATE * residual * flip_rate_scale;

        rng = rng * 1664525u + 1013904223u;
        float r = (float)(rng & 0xFFFF) / 65536.0f;

        if (r < flip_prob) {
            int axis = (int)((rng >> 16) & 0x03);
            int s = topo_get_axis(state, axis);
            if (s != 0) {
                uint8_t flipped = hopfion_phason_flip(state, axis);
                if (topo_dim(flipped) <= topo_dim(state)) {
                    state = flipped;
                    atomicAdd(&d_operator_counts[0], 1);
                }
            }
        }
    }

    // --- Venting: dim-4 + high history → mark for spawn with opposite-helicity child ---
    // Piggybacked on spawn infrastructure via PFLAG_VENT_PENDING. The spawn kernel
    // checks this flag and uses hopfion_vent() for the child's topo_state.
    if (topo_dim(state) == 4 &&
        disk->pump_history[i] > HOPFION_VENT_HISTORY &&
        !(disk->flags[i] & PFLAG_VENT_PENDING)) {
        disk->flags[i] |= PFLAG_VENT_PENDING;
        atomicAdd(&d_operator_counts[4], 1);
    }

    // --- Iron freeze: dim-4 + idle + exhausted history ---
    if (topo_dim(state) == 4 &&
        disk->pump_state[i] == 0 &&  // PUMP_IDLE
        disk->pump_history[i] < HOPFION_FREEZE_HISTORY) {
        state = TOPO_FROZEN;
        atomicAdd(&d_operator_counts[1], 1);
    }

    disk->topo_state[i] = state;

reduce:
    // Writer monad: accumulate Q delta (conservation audit)
    int Q_after = (state == TOPO_FROZEN) ? 0 : topo_Q_fast(state);
    int Q_delta = Q_after - Q_before;

    // Warp-level Q reduction
    int q = Q_after;
    int dq = Q_delta;
    for (int offset = 16; offset > 0; offset >>= 1) {
        q += __shfl_down_sync(0xFFFFFFFF, q, offset);
        dq += __shfl_down_sync(0xFFFFFFFF, dq, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(d_Q_sum, q);
        if (d_Q_delta_sum) atomicAdd(d_Q_delta_sum, dq);
    }
}
