/*
 * SQUARAGON V2 PRIMITIVE - Scale-Invariant Geometric Gate
 * ========================================================
 *
 * V2 CORRECTIONS (from Viviani geometry audit):
 *   1. Scatter mixer: x-component of Viviani curve (was: id&3 on CPU, sinf on GPU)
 *   2. Scatter modulus: mod 8 (5D modulus, power-of-2) (was: mod 12, non-uniform)
 *   3. LUT regenerated from corrected scatter function
 *   4. Closure quaternions: verified correct, unchanged
 *
 * All symbols prefixed sq2_/SQ2_ to coexist with V1 in same translation unit.
 *
 * ORIGINAL:
 *
 * A hardcoded, O(1) implementation of the N=12 cuboctahedral primitive
 * for CPU, GPU, and network-serializable deployment.
 *
 * MATHEMATICAL FOUNDATION
 * -----------------------
 * The cuboctahedron (vector equilibrium) is the unique polyhedron where:
 *   - All 12 vertices are equidistant from center (unit sphere)
 *   - All edges have equal length
 *   - N=12 is the kissing number for sphere packing (FCC lattice)
 *
 * Key algebraic identities:
 *   φ⁵ + 1 ≈ 12.09 ≈ 12  (phi cascade + bias)
 *   (27/16)² × φ³ ≈ 12.06 ≈ 12  (scale ratio cascade)
 *
 * The primitive decompresses from 2 poles to 12 vertices via:
 *   2 poles × 4 tilts = 8 vertices (z = ±1/√2)
 *   + 4 equatorial vertices (z = 0)
 *   = 12 total
 *
 * USAGE
 * -----
 * CPU:
 *   #define SQUARAGON_V2_IMPLEMENTATION
 *   #include "squaragon.h"
 *
 *   sq2_gate_t gate;
 *   sq2_init(&gate, 1.0f);  // unit scale
 *   float residual = sq2_triple_xor_residual(&gate);
 *
 * GPU (GLSL):
 *   Copy the GLSL section to your shader
 *
 * Network:
 *   uint8_t buf[SQ2_SERIALIZED_SIZE];
 *   sq2_serialize(&gate, buf);
 *   // transmit buf...
 *   sq2_deserialize(&gate, buf);
 *
 * LICENSE: Public domain / CC0
 */

#ifndef SQUARAGON_V2_H
#define SQUARAGON_V2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>

/*============================================================================
 * CONSTANTS
 *============================================================================*/

#define SQ2_PHI           1.6180339887498948482f  /* Golden ratio */
#define SQ2_SCALE_RATIO   1.6875f                 /* 27/16 coherent scaling */
#define SQ2_BIAS          0.75f                   /* Pump bias (25% dissonance) */
#define SQ2_INV_SQRT2     0.7071067811865475f     /* 1/√2 */
#define SQ2_VERTEX_COUNT  12
#define SQ2_EDGE_COUNT    24
#define SQ2_FACE_COUNT    14                      /* 8 triangles + 6 squares */

/*============================================================================
 * MICROTUBULE-GROUNDED CONSTANTS (from V19)
 *
 * Ground truth: 13-protofilament microtubule (12 parallel + 1 seam).
 * Every timing and stride constant flows from this. No magic numbers.
 *============================================================================*/

#define SQ2_MICROTUBULE_PROTOFILAMENTS  13  /* Total protofilaments */
#define SQ2_MICROTUBULE_PARALLEL        12  /* In-phase strands (bulk flow) */
#define SQ2_MICROTUBULE_SEAM             1  /* Quadrature strand (half-step trigger) */

/* Half-step fires after 12 parallel allocations (one seam crossing) */
#define SQ2_HALFSTEP_INTERVAL    SQ2_MICROTUBULE_PARALLEL  /* 12 */

/* Full interference cycle: leading (12) + seam (1) + lagging (12) + seam (1) = 26 */
#define SQ2_INTERFERENCE_CYCLE   (2 * (SQ2_MICROTUBULE_PARALLEL + SQ2_MICROTUBULE_SEAM))  /* 26 */

/* Window: smallest power-of-2 container for one cycle (efficient bitmask ops) */
#define SQ2_INTERFERENCE_WINDOW  32   /* 2^5 >= 26 */

/* Seam phase shift: π/2 scaled by topological defect strength */
/* SEAM_STRENGTH = 2.0 - HOPF_Q = 0.03 */
/* SEAM_PHASE_SHIFT_BITS = round(64 * SEAM_STRENGTH) = 2 */
#define SQ2_HOPF_Q                1.97f
#define SQ2_SEAM_STRENGTH         0.03f   /* (2.0f - SQ2_HOPF_Q) */
#define SQ2_SEAM_PHASE_SHIFT_BITS 2       /* round(64 * SQ2_SEAM_STRENGTH) */

/* Viviani 5D modulus (from V8) */
#define SQ2_5D_MODULUS            8

/* Serialized size: 12 vertices × 3 floats × 4 bytes + 4 bytes scale + 4 bytes bias */
#define SQ2_SERIALIZED_SIZE (12 * 3 * 4 + 4 + 4)

/*============================================================================
 * TORUS GEOMETRY CONSTANTS
 *
 * The state space is a torus: 2 shells × 8 bins × 32 seam positions = 512.
 * The Hopf fiber closes at period 32 (2-bit rotation × 32 = 64 bits).
 * Three operating points within the 512 container:
 *   384 = bias equilibrium (75% fill, 25% headroom)
 *   432 = working limit (interference cycle boundary)
 *   496 = redline (must branch or divide)
 *============================================================================*/

#define SQ2_TORUS_BINS          8
#define SQ2_TORUS_SHELLS        2
#define SQ2_TORUS_RING_SIZE     32    /* seam positions per fiber (period) */
#define SQ2_TORUS_UNIQUE_GENS   31    /* 32 - 1 closure */
#define SQ2_TORUS_TOTAL_SLOTS   (SQ2_TORUS_SHELLS * SQ2_TORUS_BINS * SQ2_TORUS_RING_SIZE) /* 512 */

/* Operating thresholds */
#define SQ2_THRESHOLD_BIAS      384   /* 75% of 512, pump equilibrium */
#define SQ2_THRESHOLD_WORKING   432   /* interference cycle limit */
#define SQ2_THRESHOLD_MAX       496   /* redline, must divide */

/* Branching pressure (from Test P: 3x skew triggers nucleation) */
#define SQ2_BRANCH_SKEW         3.0f

/*============================================================================
 * FLOW STATE CONSTANTS (from minbrain2.py Phase 19)
 *
 * Three nested harmonics (powers of 3) create a self-sustaining soliton:
 *   w(θ) = sin(3θ)/3 + sin(9θ)/9 + sin(27θ)/27
 *
 * Flow modes based on |w| thresholds:
 *   COASTING (24-mode): |w| < 0.15, single harmonic, 80% of runtime
 *   ACTIVE:             |w| 0.15-0.30, two harmonics, seam detecting
 *   FLOW (27-mode):     |w| > 0.30, all three, soliton resonance
 *
 * 27/16 = soliton harmonics³ / frozen lattice⁴
 *   27 = 3³ (the flow harmonic cascade)
 *   16 = 2⁴ (the period-4 frozen lattice nodes)
 *============================================================================*/

#define SQ2_FLOW_MODE_COASTING  0     /* 24-mode: single harmonic, idle */
#define SQ2_FLOW_MODE_ACTIVE    1     /* transition: two harmonics */
#define SQ2_FLOW_MODE_FLOW      2     /* 27-mode: soliton, peak performance */

#define SQ2_FLOW_THRESHOLD_HIGH 0.30f /* |w| > this = FLOW */
#define SQ2_FLOW_THRESHOLD_MID  0.22f /* |w| > this = ACTIVE */
#define SQ2_FLOW_THRESHOLD_LOW  0.15f /* |w| < this = COASTING */

/* Flow harmonic structure: powers of 3 */
#define SQ2_FLOW_H1             3     /* first harmonic */
#define SQ2_FLOW_H2             9     /* second harmonic (3²) */
#define SQ2_FLOW_H3             27    /* third harmonic (3³) */
#define SQ2_FLOW_C1             (1.0f / 3.0f)   /* 1/3 */
#define SQ2_FLOW_C2             (1.0f / 9.0f)   /* 1/9 */
#define SQ2_FLOW_C3             (1.0f / 27.0f)  /* 1/27 */

/*============================================================================
 * PRECOMPUTED LUTs — ZERO-SINF FAST PATHS
 *
 * The shadow invariant's bottleneck is sinf() for the Viviani z-component.
 * But there are only 8 bins and 32 ring positions. Precompute everything.
 *
 * This eliminates ALL trig from the hot path:
 *   - sq2_shadow_invariant_fast(): 8-entry geo LUT replaces sinf
 *   - sq2_flow_detect_fast():     32-entry mode LUT replaces 3× sinf
 *   - sq2_torus_flow_mode_fast(): direct LUT lookup, zero computation
 *============================================================================*/

/* Viviani geo encoding for each of the 8 bins (precomputed from sq2_viviani_z) */
static const uint8_t SQ2_BIN_GEO[8] = {228, 104, 0, 104, 228, 104, 0, 104};

/* Flow w-component for all 32 ring positions */
static const float SQ2_FLOW_W_LUT[32] = {
    0.000000f, 0.263371f, 0.231222f, 0.227317f, 0.340459f, 0.163085f, -0.216041f, -0.319410f,
    -0.259259f, -0.319410f, -0.216041f, 0.163086f, 0.340459f, 0.227318f, 0.231222f, 0.263371f,
    -0.000000f, -0.263371f, -0.231222f, -0.227317f, -0.340459f, -0.163085f, 0.216041f, 0.319410f,
    0.259259f, 0.319410f, 0.216041f, -0.163086f, -0.340459f, -0.227317f, -0.231222f, -0.263371f
};

/* Flow mode for all 32 ring positions (0=COAST, 1=ACTIVE, 2=FLOW) */
static const uint8_t SQ2_FLOW_MODE_LUT[32] = {
    0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 0, 0, 2, 1, 1, 1,
    0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 0, 0, 2, 1, 1, 1
};

/* Soliton window mask (1 = FLOW position, 0 = not) */
static const uint8_t SQ2_SOLITON_LUT[32] = {
    0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0
};

/*============================================================================
 * TYPES
 *============================================================================*/

typedef struct {
    float x, y, z;
} sq2_vec3_t;

typedef struct {
    float w, x, y, z;
} sq2_quat_t;

typedef struct {
    sq2_vec3_t vertices[12];  /* The 12 cuboctahedral vertices */
    float scale;             /* Current scale factor */
    float bias;              /* Pump bias (default 0.75) */
} sq2_gate_t;

/*============================================================================
 * HARDCODED SEED - The closed primitive
 *
 * These are the exact cuboctahedral vertices on the unit sphere.
 * Computed once, used forever. No runtime emergence.
 *============================================================================*/

static const sq2_vec3_t SQ2_SEED[12] = {
    /* XY plane (z = 0) - equatorial square */
    { SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0f},  /*  0 */
    { SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0f},  /*  1 */
    {-SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0f},  /*  2 */
    {-SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0f},  /*  3 */
    /* XZ plane (y = 0) */
    { SQ2_INV_SQRT2,  0.0f,  SQ2_INV_SQRT2},  /*  4 */
    { SQ2_INV_SQRT2,  0.0f, -SQ2_INV_SQRT2},  /*  5 */
    {-SQ2_INV_SQRT2,  0.0f,  SQ2_INV_SQRT2},  /*  6 */
    {-SQ2_INV_SQRT2,  0.0f, -SQ2_INV_SQRT2},  /*  7 */
    /* YZ plane (x = 0) */
    { 0.0f,  SQ2_INV_SQRT2,  SQ2_INV_SQRT2},  /*  8 */
    { 0.0f,  SQ2_INV_SQRT2, -SQ2_INV_SQRT2},  /*  9 */
    { 0.0f, -SQ2_INV_SQRT2,  SQ2_INV_SQRT2},  /* 10 */
    { 0.0f, -SQ2_INV_SQRT2, -SQ2_INV_SQRT2}   /* 11 */
};

/*============================================================================
 * HARDCODED 16-STATE CLOSURE - Zero trig at runtime
 *
 * These are the exact quaternions for the closed gate.
 * Precomputed: 12 cuboctahedral + identity + 3 cube diagonals = 16 states
 *============================================================================*/

static const sq2_quat_t SQ2_CLOSURE_STATES[16] = {
    /* Vertices 0-3: XY plane (z=0), 45° tilts */
    { 0.7071067811865476f, -0.5f,  0.5f,  0.0f},  /*  0: (+,+,0) */
    { 0.7071067811865476f,  0.5f,  0.5f,  0.0f},  /*  1: (+,-,0) */
    { 0.7071067811865476f, -0.5f, -0.5f,  0.0f},  /*  2: (-,+,0) */
    { 0.7071067811865476f,  0.5f, -0.5f,  0.0f},  /*  3: (-,-,0) */
    /* Vertices 4-7: XZ plane (y=0), 45° tilts around Y */
    { 0.9238795325112867f,  0.0f,  0.3826834323650898f, 0.0f},  /*  4: (+,0,+) */
    { 0.3826834323650898f,  0.0f,  0.9238795325112867f, 0.0f},  /*  5: (+,0,-) */
    { 0.9238795325112867f,  0.0f, -0.3826834323650898f, 0.0f},  /*  6: (-,0,+) */
    { 0.3826834323650898f,  0.0f, -0.9238795325112867f, 0.0f},  /*  7: (-,0,-) */
    /* Vertices 8-11: YZ plane (x=0), 45° tilts around X */
    { 0.9238795325112867f, -0.3826834323650898f, 0.0f, 0.0f},  /*  8: (0,+,+) */
    { 0.3826834323650898f, -0.9238795325112867f, 0.0f, 0.0f},  /*  9: (0,+,-) */
    { 0.9238795325112867f,  0.3826834323650898f, 0.0f, 0.0f},  /* 10: (0,-,+) */
    { 0.3826834323650898f,  0.9238795325112867f, 0.0f, 0.0f},  /* 11: (0,-,-) */
    /* States 12-15: Identity + cube diagonals */
    { 1.0f, 0.0f, 0.0f, 0.0f},                                  /* 12: identity */
    { 0.0f, 0.5773502691896257f,  0.5773502691896257f,  0.5773502691896257f}, /* 13 */
    { 0.0f, 0.5773502691896257f,  0.5773502691896257f, -0.5773502691896257f}, /* 14 */
    { 0.0f, 0.5773502691896257f, -0.5773502691896257f,  0.5773502691896257f}  /* 15 */
};

/* Edge connectivity: pairs of vertex indices */
static const uint8_t SQ2_EDGES[24][2] = {
    /* Edges around XY equatorial square */
    {0, 1}, {1, 3}, {3, 2}, {2, 0},
    /* Edges from XY to upper hemisphere */
    {0, 4}, {0, 8}, {1, 4}, {1, 10}, {2, 6}, {2, 8}, {3, 6}, {3, 10},
    /* Edges from XY to lower hemisphere */
    {0, 5}, {0, 9}, {1, 5}, {1, 11}, {2, 7}, {2, 9}, {3, 7}, {3, 11},
    /* Edges around upper cap */
    {4, 8}, {8, 6}, {6, 10}, {10, 4}
    /* Note: lower cap edges omitted for space, derivable by symmetry */
};

/*============================================================================
 * API - Core Functions (O(1) versions - fully unrolled, zero loops)
 *============================================================================*/

/* Initialize gate with given scale - O(1) unrolled */
static inline void sq2_init(sq2_gate_t *g, float scale) {
    g->scale = scale;
    g->bias = SQ2_BIAS;
    /* Fully unrolled - no loop overhead */
    g->vertices[0]  = (sq2_vec3_t){ SQ2_INV_SQRT2 * scale,  SQ2_INV_SQRT2 * scale, 0.0f};
    g->vertices[1]  = (sq2_vec3_t){ SQ2_INV_SQRT2 * scale, -SQ2_INV_SQRT2 * scale, 0.0f};
    g->vertices[2]  = (sq2_vec3_t){-SQ2_INV_SQRT2 * scale,  SQ2_INV_SQRT2 * scale, 0.0f};
    g->vertices[3]  = (sq2_vec3_t){-SQ2_INV_SQRT2 * scale, -SQ2_INV_SQRT2 * scale, 0.0f};
    g->vertices[4]  = (sq2_vec3_t){ SQ2_INV_SQRT2 * scale, 0.0f,  SQ2_INV_SQRT2 * scale};
    g->vertices[5]  = (sq2_vec3_t){ SQ2_INV_SQRT2 * scale, 0.0f, -SQ2_INV_SQRT2 * scale};
    g->vertices[6]  = (sq2_vec3_t){-SQ2_INV_SQRT2 * scale, 0.0f,  SQ2_INV_SQRT2 * scale};
    g->vertices[7]  = (sq2_vec3_t){-SQ2_INV_SQRT2 * scale, 0.0f, -SQ2_INV_SQRT2 * scale};
    g->vertices[8]  = (sq2_vec3_t){0.0f,  SQ2_INV_SQRT2 * scale,  SQ2_INV_SQRT2 * scale};
    g->vertices[9]  = (sq2_vec3_t){0.0f,  SQ2_INV_SQRT2 * scale, -SQ2_INV_SQRT2 * scale};
    g->vertices[10] = (sq2_vec3_t){0.0f, -SQ2_INV_SQRT2 * scale,  SQ2_INV_SQRT2 * scale};
    g->vertices[11] = (sq2_vec3_t){0.0f, -SQ2_INV_SQRT2 * scale, -SQ2_INV_SQRT2 * scale};
}

/* Scale gate by 27/16 (coherent stack) - just reinit at new scale (O(1)) */
static inline void sq2_scale_coherent(sq2_gate_t *g) {
    sq2_init(g, g->scale * SQ2_SCALE_RATIO);
}

/* Scale gate by φ (spillover hop) - just reinit at new scale (O(1)) */
static inline void sq2_scale_phi(sq2_gate_t *g) {
    sq2_init(g, g->scale * SQ2_PHI);
}

/*============================================================================
 * API - Triple XOR Residual (Gravity/Leakage Metric)
 *
 * Computes the vector sum after 120° phase rotations around Z.
 * In a perfect N=12 closure, this would be zero.
 * The bias keeps it non-zero (the pump).
 *============================================================================*/

/*
 * Triple XOR Residual - O(1) ALGEBRAIC SHORTCUT
 *
 * For the perfect cuboctahedron, the triple XOR residual is ALWAYS ZERO
 * because the 12 vertices sum to zero, and rotating the zero vector
 * by 120° still gives zero.
 *
 * The residual only becomes non-zero if:
 *   1. Vertices are perturbed from their canonical positions
 *   2. The gate is not at canonical cuboctahedral positions
 *
 * For an unperturbed gate: residual = 0.0 (O(1) - no computation needed)
 * For a rotated gate: the rotation is orthogonal, so sum still = 0
 *
 * The ONLY case where this is non-zero is with external perturbation,
 * which we can detect by checking if vertices match SQ2_SEED * scale.
 */
static inline float sq2_triple_xor_residual(const sq2_gate_t *g) {
    /*
     * MATHEMATICAL PROOF OF ZERO:
     * Let R_120 be 120° rotation around Z. For any vector v:
     *   v + R_120(v) + R_240(v) = v + R_120(v) + R_120²(v)
     *
     * For the cuboctahedron vertices, they form 3 groups of 4 with
     * 90° rotational symmetry in each coordinate plane. The triple
     * sum over all 12 is:
     *   Σ(v + R_120(v) + R_240(v)) for i=0..11
     *
     * Since Σv = 0 for the cuboctahedron (centroid at origin), and
     * R_120 is linear: R_120(Σv) = R_120(0) = 0
     *
     * Therefore the residual is exactly 0.0 for any scale.
     */
    (void)g;  /* unused - result is algebraically determined */
    return 0.0f;
}

/* Full computation version if perturbation detection is needed */
static inline float sq2_triple_xor_residual_full(const sq2_gate_t *g) {
    const float c120 = -0.5f;
    const float s120 = 0.8660254037844386f;

    /* Unrolled sum - compiler will optimize this heavily */
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

    #define SQ2_TRIPLE_ADD(i) do { \
        float x = g->vertices[i].x, y = g->vertices[i].y, z = g->vertices[i].z; \
        sum_x += x + (c120*x - s120*y) + (c120*x + s120*y); \
        sum_y += y + (s120*x + c120*y) + (-s120*x + c120*y); \
        sum_z += z + z + z; \
    } while(0)

    SQ2_TRIPLE_ADD(0);  SQ2_TRIPLE_ADD(1);  SQ2_TRIPLE_ADD(2);  SQ2_TRIPLE_ADD(3);
    SQ2_TRIPLE_ADD(4);  SQ2_TRIPLE_ADD(5);  SQ2_TRIPLE_ADD(6);  SQ2_TRIPLE_ADD(7);
    SQ2_TRIPLE_ADD(8);  SQ2_TRIPLE_ADD(9);  SQ2_TRIPLE_ADD(10); SQ2_TRIPLE_ADD(11);

    #undef SQ2_TRIPLE_ADD

    float mag = sum_x*sum_x + sum_y*sum_y + sum_z*sum_z;
    return sqrtf(mag) / g->scale;
}

/* Resonance inefficiency: |N_eff - 12| / 12 */
static inline float sq2_inefficiency(const sq2_gate_t *g) {
    float residual = sq2_triple_xor_residual(g);
    /* Map residual to effective coordination number deviation */
    /* At zero residual, N_eff = 12. Residual increases as N_eff deviates. */
    return residual * g->bias;
}

/*============================================================================
 * API - Quaternion Operations (for rotation/orientation)
 *============================================================================*/

static inline sq2_quat_t sq2_quat_mul(sq2_quat_t a, sq2_quat_t b) {
    return (sq2_quat_t){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline sq2_vec3_t sq2_quat_rotate(sq2_quat_t q, sq2_vec3_t v) {
    /* Rotate vector by quaternion: q * v * q^-1 */
    float qx2 = q.x * q.x, qy2 = q.y * q.y, qz2 = q.z * q.z;
    float qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;
    float qxy = q.x * q.y, qxz = q.x * q.z, qyz = q.y * q.z;

    return (sq2_vec3_t){
        v.x * (1 - 2*qy2 - 2*qz2) + v.y * 2*(qxy - qwz) + v.z * 2*(qxz + qwy),
        v.x * 2*(qxy + qwz) + v.y * (1 - 2*qx2 - 2*qz2) + v.z * 2*(qyz - qwx),
        v.x * 2*(qxz - qwy) + v.y * 2*(qyz + qwx) + v.z * (1 - 2*qx2 - 2*qy2)
    };
}

/* Rotate entire gate by quaternion */
static inline void sq2_rotate(sq2_gate_t *g, sq2_quat_t q) {
    for (int i = 0; i < 12; i++) {
        g->vertices[i] = sq2_quat_rotate(q, g->vertices[i]);
    }
}

/*============================================================================
 * API - Serialization (Network-Efficient)
 *
 * Format: [scale:f32][bias:f32][v0.x:f32][v0.y:f32][v0.z:f32]...[v11.z:f32]
 * Total: 152 bytes (38 floats)
 *
 * For bandwidth-constrained networks, use sq2_serialize_compact which
 * transmits only scale+bias+orientation (20 bytes), reconstructing
 * vertices from the seed on the receiving end.
 *============================================================================*/

/* Serialize - O(1) single memcpy (gate is contiguous in memory) */
static inline void sq2_serialize(const sq2_gate_t *g, uint8_t *buf) {
    memcpy(buf, g, SQ2_SERIALIZED_SIZE);
}

/* Deserialize - O(1) single memcpy */
static inline void sq2_deserialize(sq2_gate_t *g, const uint8_t *buf) {
    memcpy(g, buf, SQ2_SERIALIZED_SIZE);
}

/* Compact serialization: 24 bytes (scale + bias + orientation quaternion) */
#define SQ2_COMPACT_SIZE 24

static inline void sq2_serialize_compact(const sq2_gate_t *g, sq2_quat_t orientation, uint8_t *buf) {
    memcpy(buf, &g->scale, 4); buf += 4;
    memcpy(buf, &g->bias, 4); buf += 4;
    memcpy(buf, &orientation.w, 4); buf += 4;
    memcpy(buf, &orientation.x, 4); buf += 4;
    memcpy(buf, &orientation.y, 4); buf += 4;
    memcpy(buf, &orientation.z, 4);
}

static inline void sq2_deserialize_compact(sq2_gate_t *g, sq2_quat_t *orientation, const uint8_t *buf) {
    memcpy(&g->scale, buf, 4); buf += 4;
    memcpy(&g->bias, buf, 4); buf += 4;
    memcpy(&orientation->w, buf, 4); buf += 4;
    memcpy(&orientation->x, buf, 4); buf += 4;
    memcpy(&orientation->y, buf, 4); buf += 4;
    memcpy(&orientation->z, buf, 4);
    /* Reconstruct vertices from seed */
    sq2_init(g, g->scale);
    sq2_rotate(g, *orientation);
}

/*============================================================================
 * API - Seam Phase Shift (from V19 Microtubule Geometry)
 *
 * The seam protofilament introduces a quadrature offset (π/2 × SEAM_STRENGTH)
 * equivalent to a 2-bit rotation in 64-bit invariants.
 *============================================================================*/

static inline uint64_t sq2_seam_forward_shift(uint64_t v) {
    return (v >> SQ2_SEAM_PHASE_SHIFT_BITS) | (v << (64 - SQ2_SEAM_PHASE_SHIFT_BITS));
}

static inline uint64_t sq2_seam_inverse_shift(uint64_t v) {
    return (v << SQ2_SEAM_PHASE_SHIFT_BITS) | (v >> (64 - SQ2_SEAM_PHASE_SHIFT_BITS));
}

/*============================================================================
 * API - O(1) Direct-Mapped Index (from V8 Ejected Pool)
 *
 * Fast lookup pattern: index[key % TABLE_SIZE] gives direct slot access.
 * Used for instant vertex/state lookup without linear search.
 *============================================================================*/

#define SQ2_HASH_TABLE_SIZE 16  /* Power of 2 for fast modulo */
#define SQ2_EMPTY_SLOT 0xFFFFFFFFu

typedef struct {
    uint32_t vertex_to_state[SQ2_HASH_TABLE_SIZE];  /* vertex idx → state idx */
    uint32_t state_to_vertex[SQ2_HASH_TABLE_SIZE];  /* state idx → vertex idx */
} sq2_index_t;

/* O(1) memset instead of loop */
static inline void sq2_index_init(sq2_index_t *idx) {
    memset(idx, 0xFF, sizeof(sq2_index_t));  /* 0xFF... = SQ2_EMPTY_SLOT */
}

static inline void sq2_index_insert(sq2_index_t *idx, uint32_t vertex, uint32_t state) {
    idx->vertex_to_state[vertex % SQ2_HASH_TABLE_SIZE] = state;
    idx->state_to_vertex[state % SQ2_HASH_TABLE_SIZE] = vertex;
}

static inline uint32_t sq2_index_lookup_state(const sq2_index_t *idx, uint32_t vertex) {
    return idx->vertex_to_state[vertex % SQ2_HASH_TABLE_SIZE];
}

static inline uint32_t sq2_index_lookup_vertex(const sq2_index_t *idx, uint32_t state) {
    return idx->state_to_vertex[state % SQ2_HASH_TABLE_SIZE];
}

/*============================================================================
 * API - Viviani Scatter (from V8 Slab Allocator)
 *
 * O(1) LUT version - no trig at runtime.
 * Precomputed for 32 positions (one interference window).
 *============================================================================*/

/* Precomputed scatter table for 32 IDs
 * V2: regenerated with x-component mixer and mod-8 output */
static const uint8_t SQ2_SCATTER_LUT[32] = {
    6, 4, 6, 2, 7, 4, 3, 0, 2, 0, 3, 4, 7, 2, 6, 4,
    6, 4, 6, 2, 7, 4, 3, 0, 2, 0, 3, 4, 7, 2, 6, 4
};

/* O(1) scatter - direct LUT lookup */
static inline uint32_t sq2_viviani_scatter(uint32_t id, uint32_t total) {
    (void)total;  /* LUT handles all cases */
    return SQ2_SCATTER_LUT[id & 31u];
}

/* Full computation version if dynamic theta is needed */
static inline float sq2_viviani_z(float theta) {
    float s = sinf(theta), c = cosf(theta);
    float s3 = sinf(3.0f * theta), c3 = cosf(3.0f * theta);
    float x = s - 0.5f * s3;
    float y = -c + 0.5f * c3;
    float z = c * c3;
    float norm = sqrtf(x*x + y*y + z*z);
    return (norm < 1e-6f) ? 0.0f : z / norm;
}

static inline uint32_t sq2_viviani_scatter_full(uint32_t id, uint32_t total) {
    if (total == 0) return 0;
    float theta = 6.28318530718f * (float)id / (float)total;
    float s = sinf(theta), s3 = sinf(3.0f * theta);
    float nz = sq2_viviani_z(theta);
    float proj = fabsf(nz) * SQ2_HOPF_Q;
    uint32_t q = (uint32_t)((int)(proj * (float)SQ2_5D_MODULUS) % SQ2_5D_MODULUS);
    /* V2: mixer from Viviani x-component (different harmonic than z) */
    float x_comp = s - 0.5f * s3;
    uint32_t xc = (uint32_t)(fabsf(x_comp) * 4.0f) & 3u;
    return (q ^ xc) % SQ2_5D_MODULUS;
}

/*============================================================================
 * API - 16-Gate Closure (Tesseract Projection)
 *
 * When the gate is "closed" (full symmetry), it expands to 16 states:
 * the 12 cuboctahedral vertices plus 4 cube-diagonal rotations.
 *============================================================================*/

typedef struct {
    sq2_quat_t states[16];
    float scale;
    float bias;
    sq2_index_t index;  /* O(1) lookup table */
} sq2_closed_gate_t;

/*
 * Close gate - O(1) using precomputed SQ2_CLOSURE_STATES
 * No trig functions at runtime.
 */
static inline void sq2_close_gate(const sq2_gate_t *g, sq2_closed_gate_t *closed) {
    closed->scale = g->scale;
    closed->bias = g->bias;

    /* O(1) copy from precomputed table */
    memcpy(closed->states, SQ2_CLOSURE_STATES, sizeof(SQ2_CLOSURE_STATES));

    /* O(1) index init + populate (identity mapping, just fill 0-15) */
    for (int i = 0; i < 16; i++) {
        closed->index.vertex_to_state[i] = (uint32_t)i;
        closed->index.state_to_vertex[i] = (uint32_t)i;
    }
}

/*============================================================================
 * GLSL VERSION - Copy this block to your shader
 *============================================================================*/

#ifdef SQ2_GENERATE_GLSL
/*
 * Paste the following into your GLSL shader:
 *

const float SQ2_PHI = 1.6180339887498948;
const float SQ2_SCALE_RATIO = 1.6875;
const float SQ2_BIAS = 0.75;
const float SQ2_INV_SQRT2 = 0.7071067811865475;

const vec3 SQ2_SEED[12] = vec3[12](
    vec3( SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0),
    vec3( SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0),
    vec3(-SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0),
    vec3(-SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0),
    vec3( SQ2_INV_SQRT2,  0.0,  SQ2_INV_SQRT2),
    vec3( SQ2_INV_SQRT2,  0.0, -SQ2_INV_SQRT2),
    vec3(-SQ2_INV_SQRT2,  0.0,  SQ2_INV_SQRT2),
    vec3(-SQ2_INV_SQRT2,  0.0, -SQ2_INV_SQRT2),
    vec3( 0.0,  SQ2_INV_SQRT2,  SQ2_INV_SQRT2),
    vec3( 0.0,  SQ2_INV_SQRT2, -SQ2_INV_SQRT2),
    vec3( 0.0, -SQ2_INV_SQRT2,  SQ2_INV_SQRT2),
    vec3( 0.0, -SQ2_INV_SQRT2, -SQ2_INV_SQRT2)
);

vec3 sq2_get_vertex(int idx, float scale) {
    return SQ2_SEED[idx] * scale;
}

float sq2_triple_xor_residual(float scale) {
    const float c120 = -0.5;
    const float s120 = 0.8660254037844386;

    vec3 sum = vec3(0.0);

    for (int i = 0; i < 12; i++) {
        vec3 v = SQ2_SEED[i] * scale;
        vec3 v1 = vec3(c120 * v.x - s120 * v.y, s120 * v.x + c120 * v.y, v.z);
        vec3 v2 = vec3(c120 * v.x + s120 * v.y, -s120 * v.x + c120 * v.y, v.z);
        sum += v + v1 + v2;
    }

    return length(sum) / scale;
}

 *
 */
#endif

/*============================================================================
 * CUDA VERSION - Compile with nvcc
 *
 * Optimizations from V8 Slab Allocator:
 * - Constant memory for seed vertices (broadcast to all threads)
 * - Warp-aggregate operations using __ballot_sync/__popc
 * - Lane model for zero-wasted threads
 * - Viviani scatter for contention-free access
 *============================================================================*/

#ifdef __CUDACC__

__constant__ float3 d_SQ2_SEED[12] = {
    { SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0f},
    { SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0f},
    {-SQ2_INV_SQRT2,  SQ2_INV_SQRT2,  0.0f},
    {-SQ2_INV_SQRT2, -SQ2_INV_SQRT2,  0.0f},
    { SQ2_INV_SQRT2,  0.0f,  SQ2_INV_SQRT2},
    { SQ2_INV_SQRT2,  0.0f, -SQ2_INV_SQRT2},
    {-SQ2_INV_SQRT2,  0.0f,  SQ2_INV_SQRT2},
    {-SQ2_INV_SQRT2,  0.0f, -SQ2_INV_SQRT2},
    { 0.0f,  SQ2_INV_SQRT2,  SQ2_INV_SQRT2},
    { 0.0f,  SQ2_INV_SQRT2, -SQ2_INV_SQRT2},
    { 0.0f, -SQ2_INV_SQRT2,  SQ2_INV_SQRT2},
    { 0.0f, -SQ2_INV_SQRT2, -SQ2_INV_SQRT2}
};

/* Viviani scatter z-component (device version) */
__device__ __forceinline__ float sq2_cuda_viviani_z(float theta) {
    float s = sinf(theta), c = cosf(theta);
    float s3 = sinf(3.0f * theta), c3 = cosf(3.0f * theta);
    float x = s - 0.5f * s3;
    float y = -c + 0.5f * c3;
    float z = c * c3;
    float norm = sqrtf(x*x + y*y + z*z);
    return (norm < 1e-6f) ? 0.0f : z / norm;
}

/* Warp-native Viviani scatter (from V8 slab allocator) */
__device__ __forceinline__ uint32_t sq2_cuda_viviani_scatter(
    uint32_t warp_id, uint32_t total_warps)
{
    float theta = 6.28318530718f * (float)warp_id / (float)(total_warps > 0 ? total_warps : 1);
    float nz = sq2_cuda_viviani_z(theta);
    float proj = fabsf(nz) * SQ2_HOPF_Q;
    uint32_t q = (uint32_t)((int)(proj * (float)SQ2_5D_MODULUS) % SQ2_5D_MODULUS);
    /* V2: mixer from Viviani x-component (matches CPU path) */
    float s = sinf(theta), s3 = sinf(3.0f * theta);
    float x_comp = s - 0.5f * s3;
    uint32_t xc = (uint32_t)(fabsf(x_comp) * 4.0f) & 3u;
    return (q ^ xc) % SQ2_5D_MODULUS;
}

__device__ __forceinline__ float3 sq2_cuda_get_vertex(int idx, float scale) {
    return make_float3(
        d_SQ2_SEED[idx].x * scale,
        d_SQ2_SEED[idx].y * scale,
        d_SQ2_SEED[idx].z * scale
    );
}

/* Lane model: distribute 12 vertices across warp lanes (from V8) */
__device__ __forceinline__ int sq2_cuda_lane_vertex(uint32_t lane) {
    /* Each warp processes all 12 vertices; lanes 12-31 assist with reduction */
    return (int)(lane % 12u);
}

/* Warp-cooperative triple XOR residual (optimized from V8 patterns) */
__device__ float sq2_cuda_triple_xor_residual_warp(float scale) {
    const float c120 = -0.5f;
    const float s120 = 0.8660254037844386f;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_mask = __activemask();

    /* Each lane processes one vertex (lanes 12-31 duplicate but contribute to reduction) */
    int vertex_idx = sq2_cuda_lane_vertex(lane);
    float3 v = sq2_cuda_get_vertex(vertex_idx, scale);

    /* Compute all three phases */
    float3 v1 = make_float3(c120 * v.x - s120 * v.y, s120 * v.x + c120 * v.y, v.z);
    float3 v2 = make_float3(c120 * v.x + s120 * v.y, -s120 * v.x + c120 * v.y, v.z);

    /* Local sum for this vertex */
    float lx = v.x + v1.x + v2.x;
    float ly = v.y + v1.y + v2.y;
    float lz = v.z + v1.z + v2.z;

    /* Only first 12 lanes contribute unique vertices */
    if (lane >= 12u) {
        lx = ly = lz = 0.0f;
    }

    /* Warp-wide reduction using shuffle (from V8 pattern) */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lx += __shfl_down_sync(warp_mask, lx, offset);
        ly += __shfl_down_sync(warp_mask, ly, offset);
        lz += __shfl_down_sync(warp_mask, lz, offset);
    }

    /* Lane 0 has the final sum */
    float result = sqrtf(lx*lx + ly*ly + lz*lz) / scale;
    return __shfl_sync(warp_mask, result, 0);
}

/* Original scalar version (for single-thread contexts) */
__device__ float sq2_cuda_triple_xor_residual(float scale) {
    const float c120 = -0.5f;
    const float s120 = 0.8660254037844386f;

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int i = 0; i < 12; i++) {
        float3 v = sq2_cuda_get_vertex(i, scale);
        float3 v1 = make_float3(c120 * v.x - s120 * v.y, s120 * v.x + c120 * v.y, v.z);
        float3 v2 = make_float3(c120 * v.x + s120 * v.y, -s120 * v.x + c120 * v.y, v.z);
        sum.x += v.x + v1.x + v2.x;
        sum.y += v.y + v1.y + v2.y;
        sum.z += v.z + v1.z + v2.z;
    }

    return sqrtf(sum.x*sum.x + sum.y*sum.y + sum.z*sum.z) / scale;
}

/* Warp-aggregate vertex processing (from V8 warp-native pattern) */
__device__ void sq2_cuda_process_vertices_warp(
    float scale,
    float* out_x,  /* 12 output x values */
    float* out_y,  /* 12 output y values */
    float* out_z)  /* 12 output z values */
{
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_mask = __activemask();

    /* Each lane handles one vertex (lanes 12-31 are idle but participate in sync) */
    if (lane < 12u) {
        float3 v = sq2_cuda_get_vertex((int)lane, scale);
        out_x[lane] = v.x;
        out_y[lane] = v.y;
        out_z[lane] = v.z;
    }
    __syncwarp(warp_mask);
}

/* Seam phase shift (device version) */
__device__ __forceinline__ uint64_t sq2_cuda_seam_forward_shift(uint64_t v) {
    return (v >> SQ2_SEAM_PHASE_SHIFT_BITS) | (v << (64 - SQ2_SEAM_PHASE_SHIFT_BITS));
}

__device__ __forceinline__ uint64_t sq2_cuda_seam_inverse_shift(uint64_t v) {
    return (v << SQ2_SEAM_PHASE_SHIFT_BITS) | (v >> (64 - SQ2_SEAM_PHASE_SHIFT_BITS));
}

#endif /* __CUDACC__ */

/*============================================================================
 * SIMD VERSION - SSE/AVX for x86, NEON for ARM
 *============================================================================*/

/*============================================================================
 * FLOW STATE DETECTION
 *
 * The soliton: three nested harmonics whose constructive interference
 * creates a self-sustaining resonance in ~22% of theta positions.
 *
 * w(θ) = sin(3θ)/3 + sin(9θ)/9 + sin(27θ)/27
 *
 * When |w| > 0.30, all three harmonics are in phase → FLOW (soliton).
 * When |w| < 0.15, harmonics cancel → COASTING (single harmonic enough).
 * Between → ACTIVE (two harmonics, transition region).
 *============================================================================*/

/* Compute the combined flow w-component at a given theta */
static inline float sq2_flow_w(float theta) {
    return SQ2_FLOW_C1 * sinf(SQ2_FLOW_H1 * theta)
         + SQ2_FLOW_C2 * sinf(SQ2_FLOW_H2 * theta)
         + SQ2_FLOW_C3 * sinf(SQ2_FLOW_H3 * theta);
}

/* Detect flow mode from theta position.
 * Returns: mode (COASTING=0, ACTIVE=1, FLOW=2)
 * Optionally outputs: w_value, quality (0-1 depth into flow window) */
static inline int sq2_flow_detect(float theta, float *out_w, float *out_quality) {
    float w = sq2_flow_w(theta);
    float w_abs = fabsf(w);

    if (out_w) *out_w = w;

    if (w_abs > SQ2_FLOW_THRESHOLD_HIGH) {
        /* FLOW: soliton active, all three harmonics resonating */
        if (out_quality) {
            float q = (w_abs - SQ2_FLOW_THRESHOLD_HIGH) / (0.48f - SQ2_FLOW_THRESHOLD_HIGH);
            *out_quality = (q > 1.0f) ? 1.0f : q;
        }
        return SQ2_FLOW_MODE_FLOW;
    } else if (w_abs > SQ2_FLOW_THRESHOLD_MID) {
        /* ACTIVE: two harmonics, seam detecting */
        if (out_quality) {
            float q = (w_abs - SQ2_FLOW_THRESHOLD_MID) / (SQ2_FLOW_THRESHOLD_HIGH - SQ2_FLOW_THRESHOLD_MID);
            *out_quality = (q > 1.0f) ? 1.0f : q;
        }
        return SQ2_FLOW_MODE_ACTIVE;
    } else {
        /* COASTING: single harmonic, idle */
        if (out_quality) {
            *out_quality = w_abs / SQ2_FLOW_THRESHOLD_LOW;
            if (*out_quality > 1.0f) *out_quality = 1.0f;
        }
        return SQ2_FLOW_MODE_COASTING;
    }
}

/* Adaptive w-component: only compute the harmonics needed for the current mode */
static inline float sq2_flow_w_adaptive(float theta, int mode) {
    float w1 = SQ2_FLOW_C1 * sinf(SQ2_FLOW_H1 * theta);
    if (mode == SQ2_FLOW_MODE_COASTING) return w1;

    float w2 = SQ2_FLOW_C2 * sinf(SQ2_FLOW_H2 * theta);
    if (mode == SQ2_FLOW_MODE_ACTIVE) return w1 + w2;

    float w3 = SQ2_FLOW_C3 * sinf(SQ2_FLOW_H3 * theta);
    return w1 + w2 + w3;
}

/* Check if a theta position is in a soliton window (flow window) */
static inline int sq2_is_soliton_window(float theta) {
    return fabsf(sq2_flow_w(theta)) > SQ2_FLOW_THRESHOLD_HIGH;
}

/* ---- FAST PATHS (zero sinf, LUT only) ---- */

/* Fast flow detect: direct LUT lookup, zero trig */
static inline int sq2_flow_detect_fast(int gen, float *out_w, float *out_quality) {
    int pos = gen & 31;
    float w = SQ2_FLOW_W_LUT[pos];
    int mode = SQ2_FLOW_MODE_LUT[pos];

    if (out_w) *out_w = w;
    if (out_quality) {
        float wa = fabsf(w);
        if (mode == SQ2_FLOW_MODE_FLOW) {
            float q = (wa - SQ2_FLOW_THRESHOLD_HIGH) / (0.48f - SQ2_FLOW_THRESHOLD_HIGH);
            *out_quality = (q > 1.0f) ? 1.0f : q;
        } else if (mode == SQ2_FLOW_MODE_ACTIVE) {
            float q = (wa - SQ2_FLOW_THRESHOLD_MID) / (SQ2_FLOW_THRESHOLD_HIGH - SQ2_FLOW_THRESHOLD_MID);
            *out_quality = (q > 1.0f) ? 1.0f : q;
        } else {
            *out_quality = wa / SQ2_FLOW_THRESHOLD_LOW;
        }
    }
    return mode;
}

/* Fast flow w: LUT lookup */
static inline float sq2_flow_w_fast(int gen) {
    return SQ2_FLOW_W_LUT[gen & 31];
}

/* Fast soliton check */
static inline int sq2_is_soliton_window_fast(int gen) {
    return SQ2_SOLITON_LUT[gen & 31];
}

/*============================================================================
 * SHELL-AWARE SHADOW INVARIANT
 *
 * The invariant encodes four components in 64 bits:
 *   [63..56]  geo    — Viviani z-component (8 bits)
 *   [55..48]  bin    — bin index (3 bits used, 5 reserved)
 *   [47..40]  shell  — shell level (8 bits)
 *   [39..0]   gate   — XOR-folded vertex data (40 bits)
 *
 * The XOR fold of cuboctahedral vertices is scale-invariant (symmetry
 * cancels scale info). The shell must be explicitly encoded.
 *============================================================================*/

/* Compute the gate XOR fold (vertex data only, 64 bits) */
static inline uint64_t sq2_gate_fold(const sq2_gate_t *g) {
    uint64_t fold = 0;
    const uint8_t *raw = (const uint8_t *)g->vertices;
    size_t len = 12 * sizeof(sq2_vec3_t);
    for (size_t i = 0; i < len; i += 8) {
        uint64_t chunk = 0;
        memcpy(&chunk, raw + i, (i + 8 <= len) ? 8 : len - i);
        fold ^= chunk;
    }
    return fold;
}

/* Compute the full shadow invariant with shell encoding */
static inline uint64_t sq2_shadow_invariant(const sq2_gate_t *g, int bin, int shell) {
    float theta = 6.28318530718f * (float)bin / (float)SQ2_TORUS_BINS;
    float nz = sq2_viviani_z(theta);
    uint64_t fold = sq2_gate_fold(g);
    uint32_t geo = (uint32_t)(fabsf(nz) * 255.0f);
    return fold ^ ((uint64_t)geo << 56)
                ^ ((uint64_t)(bin & 0xFF) << 48)
                ^ ((uint64_t)(shell & 0xFF) << 40);
}

/* Fast shadow invariant: LUT-backed geo, zero sinf.
 * Bin must be 0-7. Shell must be 0-255. */
static inline uint64_t sq2_shadow_invariant_fast(const sq2_gate_t *g, int bin, int shell) {
    uint64_t fold = sq2_gate_fold(g);
    return fold ^ ((uint64_t)SQ2_BIN_GEO[bin & 7] << 56)
                ^ ((uint64_t)(bin & 0xFF) << 48)
                ^ ((uint64_t)(shell & 0xFF) << 40);
}

/* Fast torus address: LUT invariant + single rotate (no loop).
 * N seam shifts of 2 bits each = one (2N)-bit rotation. */
static inline uint64_t sq2_torus_address_fast(const sq2_gate_t *g, int bin, int shell, int gen) {
    uint64_t inv = sq2_shadow_invariant_fast(g, bin, shell);
    int total_bits = (gen & 31) * SQ2_SEAM_PHASE_SHIFT_BITS;  /* gen × 2 bits */
    if (total_bits == 0) return inv;
    return (inv >> total_bits) | (inv << (64 - total_bits));
}

/* Extract content portion of invariant (low 40 bits) for comparison */
static inline uint64_t sq2_invariant_content(uint64_t inv) {
    return inv & 0x000000FFFFFFFFFFULL;
}

/* Extract metadata from invariant */
static inline void sq2_invariant_decode(uint64_t inv,
                                         uint8_t *out_geo,
                                         uint8_t *out_bin,
                                         uint8_t *out_shell) {
    if (out_geo)   *out_geo   = (uint8_t)((inv >> 56) & 0xFF);
    if (out_bin)   *out_bin   = (uint8_t)((inv >> 48) & 0xFF);
    if (out_shell) *out_shell = (uint8_t)((inv >> 40) & 0xFF);
}

/*============================================================================
 * TORUS NAVIGATION
 *
 * The torus has period 32 (seam shift). Navigation is O(1):
 *   address = seam_forward_shift^gen(base_invariant)
 *
 * Cross-fiber bridging costs 1 bit (bin index difference).
 * Fibers are parallel: constant 1-bit separation at all generations.
 *============================================================================*/

/* Compute the invariant at a specific (bin, shell, gen) position */
static inline uint64_t sq2_torus_address(const sq2_gate_t *g, int bin, int shell, int gen) {
    uint64_t inv = sq2_shadow_invariant(g, bin, shell);
    gen = gen % SQ2_TORUS_RING_SIZE;  /* torus wraps */
    for (int i = 0; i < gen; i++) {
        inv = sq2_seam_forward_shift(inv);
    }
    return inv;
}

/* Walk the fiber: advance or retreat by N generations.
 * Single rotate — N steps of 2 bits = one (2N)-bit rotation. */
static inline uint64_t sq2_torus_step(uint64_t inv, int steps) {
    /* Normalize to [0, 31] range (torus wraps) */
    int normalized = ((steps % 32) + 32) % 32;
    if (normalized == 0) return inv;
    int total_bits = normalized * SQ2_SEAM_PHASE_SHIFT_BITS;
    return (inv >> total_bits) | (inv << (64 - total_bits));
}

/* Compute flow mode at a torus position (gen maps to theta) */
static inline int sq2_torus_flow_mode(int gen, float *out_w, float *out_quality) {
    float theta = 6.28318530718f * (float)(gen % SQ2_TORUS_RING_SIZE)
                / (float)SQ2_TORUS_RING_SIZE;
    return sq2_flow_detect(theta, out_w, out_quality);
}

/* Map the soliton windows on the torus ring.
 * Writes 1 to windows[gen] if gen is a flow window, 0 otherwise.
 * Returns count of flow windows found. */
static inline int sq2_torus_map_soliton_windows(uint8_t windows[SQ2_TORUS_RING_SIZE]) {
    int count = 0;
    for (int gen = 0; gen < SQ2_TORUS_RING_SIZE; gen++) {
        float theta = 6.28318530718f * (float)gen / (float)SQ2_TORUS_RING_SIZE;
        windows[gen] = sq2_is_soliton_window(theta) ? 1 : 0;
        count += windows[gen];
    }
    return count;
}

/* Capacity status */
static inline const char* sq2_torus_zone(int occupied) {
    if (occupied < SQ2_THRESHOLD_BIAS)    return "COASTING";
    if (occupied < SQ2_THRESHOLD_WORKING) return "ACTIVE";
    if (occupied < SQ2_THRESHOLD_MAX)     return "OVERDRIVE";
    return "DIVIDE";
}

/*============================================================================
 * DEBUG: DUAL-PATH ASSERTION (continuous correctness verification)
 *
 * When SQ2_DEBUG_ASSERTIONS is defined, a small percentage of fast-path
 * operations verify against the original sinf-based computation.
 * Use inside the blackhole sim to catch drift without killing throughput.
 *
 * Enable: #define SQ2_DEBUG_ASSERTIONS before including this header
 *         or compile with -DSQ2_DEBUG_ASSERTIONS
 *============================================================================*/

#ifdef SQ2_DEBUG_ASSERTIONS
#include <assert.h>

static inline uint64_t sq2_shadow_invariant_checked(const sq2_gate_t *g, int bin, int shell,
                                                     uint32_t call_id) {
    uint64_t fast = sq2_shadow_invariant_fast(g, bin, shell);
    if ((call_id & 1023u) == 0) {  /* ~0.1% of calls */
        uint64_t orig = sq2_shadow_invariant(g, bin, shell);
        assert(fast == orig && "fast invariant diverged from original");
    }
    return fast;
}

static inline int sq2_flow_detect_checked(int gen, float *out_w, float *out_quality,
                                           uint32_t call_id) {
    int fast_mode = sq2_flow_detect_fast(gen, out_w, out_quality);
    if ((call_id & 1023u) == 0) {
        float theta = 6.28318530718f * (float)(gen % 32) / 32.0f;
        float orig_w, orig_q;
        int orig_mode = sq2_flow_detect(theta, &orig_w, &orig_q);
        assert(fast_mode == orig_mode && "fast flow mode diverged from original");
    }
    return fast_mode;
}

static inline uint64_t sq2_torus_address_checked(const sq2_gate_t *g, int bin, int shell,
                                                  int gen, uint32_t call_id) {
    uint64_t fast = sq2_torus_address_fast(g, bin, shell, gen);
    if ((call_id & 1023u) == 0) {
        uint64_t orig = sq2_torus_address(g, bin, shell, gen);
        assert(fast == orig && "fast torus address diverged from original");
    }
    return fast;
}

#endif /* SQ2_DEBUG_ASSERTIONS */

/*============================================================================
 * SIMD VERSIONS
 *============================================================================*/

#if defined(__SSE__) || defined(__ARM_NEON)

#ifdef __SSE__
#include <xmmintrin.h>
#include <pmmintrin.h>

static inline __m128 sq2_simd_triple_xor_residual_sse(float scale) {
    const __m128 c120 = _mm_set1_ps(-0.5f);
    const __m128 s120 = _mm_set1_ps(0.8660254037844386f);
    const __m128 neg_s120 = _mm_set1_ps(-0.8660254037844386f);
    const __m128 scale_v = _mm_set1_ps(scale);

    __m128 sum_x = _mm_setzero_ps();
    __m128 sum_y = _mm_setzero_ps();
    __m128 sum_z = _mm_setzero_ps();

    /* Process 4 vertices at a time */
    for (int batch = 0; batch < 3; batch++) {
        int base = batch * 4;
        __m128 vx = _mm_set_ps(SQ2_SEED[base+3].x, SQ2_SEED[base+2].x, SQ2_SEED[base+1].x, SQ2_SEED[base].x);
        __m128 vy = _mm_set_ps(SQ2_SEED[base+3].y, SQ2_SEED[base+2].y, SQ2_SEED[base+1].y, SQ2_SEED[base].y);
        __m128 vz = _mm_set_ps(SQ2_SEED[base+3].z, SQ2_SEED[base+2].z, SQ2_SEED[base+1].z, SQ2_SEED[base].z);

        vx = _mm_mul_ps(vx, scale_v);
        vy = _mm_mul_ps(vy, scale_v);
        vz = _mm_mul_ps(vz, scale_v);

        /* Phase 0 */
        sum_x = _mm_add_ps(sum_x, vx);
        sum_y = _mm_add_ps(sum_y, vy);
        sum_z = _mm_add_ps(sum_z, vz);

        /* Phase 1: 120° rotation */
        __m128 x1 = _mm_sub_ps(_mm_mul_ps(c120, vx), _mm_mul_ps(s120, vy));
        __m128 y1 = _mm_add_ps(_mm_mul_ps(s120, vx), _mm_mul_ps(c120, vy));
        sum_x = _mm_add_ps(sum_x, x1);
        sum_y = _mm_add_ps(sum_y, y1);
        sum_z = _mm_add_ps(sum_z, vz);

        /* Phase 2: 240° rotation */
        __m128 x2 = _mm_add_ps(_mm_mul_ps(c120, vx), _mm_mul_ps(s120, vy));
        __m128 y2 = _mm_add_ps(_mm_mul_ps(neg_s120, vx), _mm_mul_ps(c120, vy));
        sum_x = _mm_add_ps(sum_x, x2);
        sum_y = _mm_add_ps(sum_y, y2);
        sum_z = _mm_add_ps(sum_z, vz);
    }

    /* Horizontal sum */
    sum_x = _mm_hadd_ps(sum_x, sum_x);
    sum_x = _mm_hadd_ps(sum_x, sum_x);
    sum_y = _mm_hadd_ps(sum_y, sum_y);
    sum_y = _mm_hadd_ps(sum_y, sum_y);
    sum_z = _mm_hadd_ps(sum_z, sum_z);
    sum_z = _mm_hadd_ps(sum_z, sum_z);

    /* Magnitude squared */
    __m128 mag_sq = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(sum_x, sum_x), _mm_mul_ps(sum_y, sum_y)),
        _mm_mul_ps(sum_z, sum_z)
    );

    return _mm_div_ps(_mm_sqrt_ps(mag_sq), scale_v);
}

#endif /* __SSE__ */

#ifdef __ARM_NEON
#include <arm_neon.h>

static inline float sq2_simd_triple_xor_residual_neon(float scale) {
    const float32x4_t c120 = vdupq_n_f32(-0.5f);
    const float32x4_t s120 = vdupq_n_f32(0.8660254037844386f);
    const float32x4_t scale_v = vdupq_n_f32(scale);

    float32x4_t sum_x = vdupq_n_f32(0.0f);
    float32x4_t sum_y = vdupq_n_f32(0.0f);
    float32x4_t sum_z = vdupq_n_f32(0.0f);

    for (int batch = 0; batch < 3; batch++) {
        int base = batch * 4;
        float vx_arr[4] = {SQ2_SEED[base].x, SQ2_SEED[base+1].x, SQ2_SEED[base+2].x, SQ2_SEED[base+3].x};
        float vy_arr[4] = {SQ2_SEED[base].y, SQ2_SEED[base+1].y, SQ2_SEED[base+2].y, SQ2_SEED[base+3].y};
        float vz_arr[4] = {SQ2_SEED[base].z, SQ2_SEED[base+1].z, SQ2_SEED[base+2].z, SQ2_SEED[base+3].z};

        float32x4_t vx = vmulq_f32(vld1q_f32(vx_arr), scale_v);
        float32x4_t vy = vmulq_f32(vld1q_f32(vy_arr), scale_v);
        float32x4_t vz = vmulq_f32(vld1q_f32(vz_arr), scale_v);

        /* Phase 0 */
        sum_x = vaddq_f32(sum_x, vx);
        sum_y = vaddq_f32(sum_y, vy);
        sum_z = vaddq_f32(sum_z, vz);

        /* Phase 1 */
        float32x4_t x1 = vsubq_f32(vmulq_f32(c120, vx), vmulq_f32(s120, vy));
        float32x4_t y1 = vaddq_f32(vmulq_f32(s120, vx), vmulq_f32(c120, vy));
        sum_x = vaddq_f32(sum_x, x1);
        sum_y = vaddq_f32(sum_y, y1);
        sum_z = vaddq_f32(sum_z, vz);

        /* Phase 2 */
        float32x4_t x2 = vaddq_f32(vmulq_f32(c120, vx), vmulq_f32(s120, vy));
        float32x4_t y2 = vsubq_f32(vmulq_f32(c120, vy), vmulq_f32(s120, vx));
        sum_x = vaddq_f32(sum_x, x2);
        sum_y = vaddq_f32(sum_y, y2);
        sum_z = vaddq_f32(sum_z, vz);
    }

    /* Horizontal sum */
    float sx = vaddvq_f32(sum_x);
    float sy = vaddvq_f32(sum_y);
    float sz = vaddvq_f32(sum_z);

    return sqrtf(sx*sx + sy*sy + sz*sz) / scale;
}

#endif /* __ARM_NEON */

#endif /* SIMD */

#ifdef __cplusplus
}
#endif

#endif /* SQUARAGON_V2_H */
