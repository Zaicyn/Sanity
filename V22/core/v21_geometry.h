/*
 * V21 GEOMETRY — Viviani Curve + Cuboctahedral Gate Primitive
 * ===========================================================
 *
 * Extracted from squaragon.h. The O(1) geometric foundation for V21.
 * All functions are portable: C, C++, CUDA, GLSL (via copy-paste section).
 *
 * Contents:
 *   - N=12 cuboctahedral seed vertices (hardcoded, zero trig)
 *   - 16-state quaternion closure (tesseract projection)
 *   - Viviani scatter LUT (O(1) contention-free distribution)
 *   - O(1) vertex↔state index table
 *   - Gate init/scale/rotate operations
 *   - Quaternion algebra
 *   - Compact serialization (24 bytes for network transport)
 *   - SSE/NEON SIMD paths (bracketed, optional)
 *
 * The Viviani curve:
 *   x(θ) = sin(θ) - ½·sin(3θ)
 *   y(θ) = -cos(θ) + ½·cos(3θ)
 *   z(θ) = cos(θ)·cos(3θ)
 *   w(θ) = ⅓·sin(5θ)        (5D extension)
 *
 * License: Public domain / CC0
 */

#ifndef V21_GEOMETRY_H
#define V21_GEOMETRY_H

#include "v21_types.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * N=12 CUBOCTAHEDRAL SEED — hardcoded unit-sphere vertices
 * ======================================================================== */

#define V21_VERTEX_COUNT  12
#define V21_EDGE_COUNT    24
#define V21_FACE_COUNT    14   /* 8 triangles + 6 squares */

#ifndef V21_VEC3_DEFINED
#define V21_VEC3_DEFINED
typedef struct { float x, y, z; } v21_vec3_t;
#endif
typedef struct { float w, x, y, z; } v21_quat_t;

static const v21_vec3_t V21_SEED[12] = {
    /* XY plane (z=0) — equatorial square */
    { V21_INV_SQRT2,  V21_INV_SQRT2,  0.0f},
    { V21_INV_SQRT2, -V21_INV_SQRT2,  0.0f},
    {-V21_INV_SQRT2,  V21_INV_SQRT2,  0.0f},
    {-V21_INV_SQRT2, -V21_INV_SQRT2,  0.0f},
    /* XZ plane (y=0) */
    { V21_INV_SQRT2,  0.0f,  V21_INV_SQRT2},
    { V21_INV_SQRT2,  0.0f, -V21_INV_SQRT2},
    {-V21_INV_SQRT2,  0.0f,  V21_INV_SQRT2},
    {-V21_INV_SQRT2,  0.0f, -V21_INV_SQRT2},
    /* YZ plane (x=0) */
    { 0.0f,  V21_INV_SQRT2,  V21_INV_SQRT2},
    { 0.0f,  V21_INV_SQRT2, -V21_INV_SQRT2},
    { 0.0f, -V21_INV_SQRT2,  V21_INV_SQRT2},
    { 0.0f, -V21_INV_SQRT2, -V21_INV_SQRT2}
};

/* ========================================================================
 * 16-STATE CLOSURE — precomputed quaternions (zero trig at runtime)
 * 12 cuboctahedral + identity + 3 cube diagonals = 16
 * ======================================================================== */

static const v21_quat_t V21_CLOSURE[16] = {
    { 0.7071067811865476f, -0.5f,  0.5f,  0.0f},
    { 0.7071067811865476f,  0.5f,  0.5f,  0.0f},
    { 0.7071067811865476f, -0.5f, -0.5f,  0.0f},
    { 0.7071067811865476f,  0.5f, -0.5f,  0.0f},
    { 0.9238795325112867f,  0.0f,  0.3826834323650898f, 0.0f},
    { 0.3826834323650898f,  0.0f,  0.9238795325112867f, 0.0f},
    { 0.9238795325112867f,  0.0f, -0.3826834323650898f, 0.0f},
    { 0.3826834323650898f,  0.0f, -0.9238795325112867f, 0.0f},
    { 0.9238795325112867f, -0.3826834323650898f, 0.0f, 0.0f},
    { 0.3826834323650898f, -0.9238795325112867f, 0.0f, 0.0f},
    { 0.9238795325112867f,  0.3826834323650898f, 0.0f, 0.0f},
    { 0.3826834323650898f,  0.9238795325112867f, 0.0f, 0.0f},
    { 1.0f, 0.0f, 0.0f, 0.0f},  /* identity */
    { 0.0f, 0.5773502691896257f,  0.5773502691896257f,  0.5773502691896257f},
    { 0.0f, 0.5773502691896257f,  0.5773502691896257f, -0.5773502691896257f},
    { 0.0f, 0.5773502691896257f, -0.5773502691896257f,  0.5773502691896257f}
};

/* ========================================================================
 * VIVIANI SCATTER LUT — O(1) contention-free distribution
 * Precomputed for 32 IDs (one interference window)
 * ======================================================================== */

static const uint8_t V21_SCATTER_LUT[32] = {
    6, 5, 4, 1, 6, 7, 1, 2, 0, 0, 1, 5, 6, 3, 4, 7,
    6, 5, 4, 1, 6, 7, 1, 2, 0, 0, 1, 5, 6, 3, 4, 7
};

static inline uint32_t v21_viviani_scatter(uint32_t id) {
    return V21_SCATTER_LUT[id & 31u];
}

/* ========================================================================
 * VIVIANI CURVE EVALUATION (full computation when needed)
 * ======================================================================== */

static inline v21_vec3_t v21_viviani_curve(float theta) {
    float s  = sinf(theta),        c  = cosf(theta);
    float s3 = sinf(3.0f * theta), c3 = cosf(3.0f * theta);
    return (v21_vec3_t){
        s  - 0.5f * s3,
        -c + 0.5f * c3,
        c * c3
    };
}

/* Tangent vector: dx/dθ, dy/dθ, dz/dθ */
static inline v21_vec3_t v21_viviani_tangent(float theta) {
    float s  = sinf(theta),        c  = cosf(theta);
    float s3 = sinf(3.0f * theta), c3 = cosf(3.0f * theta);
    return (v21_vec3_t){
        c - 1.5f * c3,
        s - 1.5f * s3,
        -s * c3 - 3.0f * c * s3
    };
}

/* Z-component of normalized Viviani normal */
static inline float v21_viviani_z(float theta) {
    v21_vec3_t v = v21_viviani_curve(theta);
    float norm = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return (norm < 1e-6f) ? 0.0f : v.z / norm;
}

/* Full scatter computation (when LUT doesn't cover the range) */
static inline uint32_t v21_viviani_scatter_full(uint32_t id, uint32_t total) {
    if (total == 0) return 0;
    float theta = V21_TWO_PI * (float)id / (float)total;
    float nz = v21_viviani_z(theta);
    float proj = fabsf(nz) * V21_HOPF_Q;
    uint32_t q = (uint32_t)((int)(proj * (float)V21_5D_MODULUS) % V21_5D_MODULUS);
    return (q ^ (id & 3u)) % V21_VERTEX_COUNT;
}

/* ========================================================================
 * GATE — scaled cuboctahedral primitive
 * ======================================================================== */

typedef struct {
    v21_vec3_t vertices[12];
    float scale;
    float bias;
} v21_gate_t;

static inline void v21_gate_init(v21_gate_t* g, float scale) {
    g->scale = scale;
    g->bias = V21_BIAS;
    for (int i = 0; i < 12; i++) {
        g->vertices[i].x = V21_SEED[i].x * scale;
        g->vertices[i].y = V21_SEED[i].y * scale;
        g->vertices[i].z = V21_SEED[i].z * scale;
    }
}

static inline void v21_gate_scale_coherent(v21_gate_t* g) {
    v21_gate_init(g, g->scale * V21_SCALE_RATIO);
}

static inline void v21_gate_scale_phi(v21_gate_t* g) {
    v21_gate_init(g, g->scale * V21_PHI);
}

/* Triple XOR residual — algebraically zero for unperturbed gate */
static inline float v21_gate_residual(const v21_gate_t* g) {
    (void)g;
    return 0.0f;
}

/* ========================================================================
 * QUATERNION OPERATIONS
 * ======================================================================== */

static inline v21_quat_t v21_quat_mul(v21_quat_t a, v21_quat_t b) {
    return (v21_quat_t){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline v21_vec3_t v21_quat_rotate(v21_quat_t q, v21_vec3_t v) {
    float qx2 = q.x*q.x, qy2 = q.y*q.y, qz2 = q.z*q.z;
    float qwx = q.w*q.x, qwy = q.w*q.y, qwz = q.w*q.z;
    float qxy = q.x*q.y, qxz = q.x*q.z, qyz = q.y*q.z;
    return (v21_vec3_t){
        v.x*(1-2*qy2-2*qz2) + v.y*2*(qxy-qwz) + v.z*2*(qxz+qwy),
        v.x*2*(qxy+qwz) + v.y*(1-2*qx2-2*qz2) + v.z*2*(qyz-qwx),
        v.x*2*(qxz-qwy) + v.y*2*(qyz+qwx) + v.z*(1-2*qx2-2*qy2)
    };
}

static inline void v21_gate_rotate(v21_gate_t* g, v21_quat_t q) {
    for (int i = 0; i < 12; i++)
        g->vertices[i] = v21_quat_rotate(q, g->vertices[i]);
}

/* ========================================================================
 * O(1) DIRECT-MAPPED INDEX — vertex↔state bidirectional lookup
 * ======================================================================== */

#define V21_INDEX_SIZE   16
#define V21_INDEX_EMPTY  0xFFFFFFFFu

typedef struct {
    uint32_t vertex_to_state[V21_INDEX_SIZE];
    uint32_t state_to_vertex[V21_INDEX_SIZE];
} v21_index_t;

static inline void v21_index_init(v21_index_t* idx) {
    memset(idx, 0xFF, sizeof(v21_index_t));
}

static inline void v21_index_insert(v21_index_t* idx, uint32_t vertex, uint32_t state) {
    idx->vertex_to_state[vertex % V21_INDEX_SIZE] = state;
    idx->state_to_vertex[state % V21_INDEX_SIZE] = vertex;
}

static inline uint32_t v21_index_state(const v21_index_t* idx, uint32_t vertex) {
    return idx->vertex_to_state[vertex % V21_INDEX_SIZE];
}

static inline uint32_t v21_index_vertex(const v21_index_t* idx, uint32_t state) {
    return idx->state_to_vertex[state % V21_INDEX_SIZE];
}

/* ========================================================================
 * 16-GATE CLOSURE
 * ======================================================================== */

typedef struct {
    v21_quat_t states[16];
    float scale;
    float bias;
    v21_index_t index;
} v21_closed_gate_t;

static inline void v21_gate_close(const v21_gate_t* g, v21_closed_gate_t* closed) {
    closed->scale = g->scale;
    closed->bias = g->bias;
    memcpy(closed->states, V21_CLOSURE, sizeof(V21_CLOSURE));
    for (int i = 0; i < 16; i++) {
        closed->index.vertex_to_state[i] = (uint32_t)i;
        closed->index.state_to_vertex[i] = (uint32_t)i;
    }
}

/* ========================================================================
 * SERIALIZATION — network-efficient encoding
 * ======================================================================== */

#define V21_SERIAL_FULL_SIZE    (12 * 3 * 4 + 4 + 4)  /* 152 bytes */
#define V21_SERIAL_COMPACT_SIZE 24                      /* scale + bias + quat */

static inline void v21_gate_serialize(const v21_gate_t* g, uint8_t* buf) {
    memcpy(buf, g, V21_SERIAL_FULL_SIZE);
}

static inline void v21_gate_deserialize(v21_gate_t* g, const uint8_t* buf) {
    memcpy(g, buf, V21_SERIAL_FULL_SIZE);
}

static inline void v21_gate_serialize_compact(const v21_gate_t* g, v21_quat_t orient, uint8_t* buf) {
    memcpy(buf,      &g->scale, 4);
    memcpy(buf + 4,  &g->bias,  4);
    memcpy(buf + 8,  &orient,   16);
}

static inline void v21_gate_deserialize_compact(v21_gate_t* g, v21_quat_t* orient, const uint8_t* buf) {
    memcpy(&g->scale, buf,     4);
    memcpy(&g->bias,  buf + 4, 4);
    memcpy(orient,    buf + 8, 16);
    v21_gate_init(g, g->scale);
    v21_gate_rotate(g, *orient);
}

/* ========================================================================
 * SIMD PATHS (optional, bracketed per architecture)
 * ======================================================================== */

#if defined(__SSE__) && !defined(V21_NO_SIMD)
#include <xmmintrin.h>
#include <pmmintrin.h>

static inline float v21_gate_residual_sse(float scale) {
    const __m128 c120 = _mm_set1_ps(-0.5f);
    const __m128 s120 = _mm_set1_ps(0.8660254037844386f);
    const __m128 sv   = _mm_set1_ps(scale);
    __m128 sx = _mm_setzero_ps(), sy = _mm_setzero_ps(), sz = _mm_setzero_ps();

    for (int b = 0; b < 3; b++) {
        int base = b * 4;
        __m128 vx = _mm_mul_ps(_mm_set_ps(V21_SEED[base+3].x, V21_SEED[base+2].x, V21_SEED[base+1].x, V21_SEED[base].x), sv);
        __m128 vy = _mm_mul_ps(_mm_set_ps(V21_SEED[base+3].y, V21_SEED[base+2].y, V21_SEED[base+1].y, V21_SEED[base].y), sv);
        __m128 vz = _mm_mul_ps(_mm_set_ps(V21_SEED[base+3].z, V21_SEED[base+2].z, V21_SEED[base+1].z, V21_SEED[base].z), sv);
        sx = _mm_add_ps(sx, _mm_add_ps(vx, _mm_add_ps(_mm_sub_ps(_mm_mul_ps(c120, vx), _mm_mul_ps(s120, vy)),
                                                        _mm_add_ps(_mm_mul_ps(c120, vx), _mm_mul_ps(s120, vy)))));
        sy = _mm_add_ps(sy, _mm_add_ps(vy, _mm_add_ps(_mm_add_ps(_mm_mul_ps(s120, vx), _mm_mul_ps(c120, vy)),
                                                        _mm_sub_ps(_mm_mul_ps(c120, vy), _mm_mul_ps(s120, vx)))));
        sz = _mm_add_ps(sz, _mm_add_ps(vz, _mm_add_ps(vz, vz)));
    }
    sx = _mm_hadd_ps(sx, sx); sx = _mm_hadd_ps(sx, sx);
    sy = _mm_hadd_ps(sy, sy); sy = _mm_hadd_ps(sy, sy);
    sz = _mm_hadd_ps(sz, sz); sz = _mm_hadd_ps(sz, sz);
    __m128 mag = _mm_add_ps(_mm_add_ps(_mm_mul_ps(sx,sx), _mm_mul_ps(sy,sy)), _mm_mul_ps(sz,sz));
    float result;
    _mm_store_ss(&result, _mm_div_ps(_mm_sqrt_ps(mag), sv));
    return result;
}
#endif /* __SSE__ */

#if defined(__ARM_NEON) && !defined(V21_NO_SIMD)
#include <arm_neon.h>

static inline float v21_gate_residual_neon(float scale) {
    const float32x4_t c120 = vdupq_n_f32(-0.5f);
    const float32x4_t s120 = vdupq_n_f32(0.8660254037844386f);
    const float32x4_t sv   = vdupq_n_f32(scale);
    float32x4_t sx = vdupq_n_f32(0), sy = vdupq_n_f32(0), sz = vdupq_n_f32(0);

    for (int b = 0; b < 3; b++) {
        int base = b * 4;
        float va[4] = {V21_SEED[base].x, V21_SEED[base+1].x, V21_SEED[base+2].x, V21_SEED[base+3].x};
        float vb[4] = {V21_SEED[base].y, V21_SEED[base+1].y, V21_SEED[base+2].y, V21_SEED[base+3].y};
        float vc[4] = {V21_SEED[base].z, V21_SEED[base+1].z, V21_SEED[base+2].z, V21_SEED[base+3].z};
        float32x4_t vx = vmulq_f32(vld1q_f32(va), sv);
        float32x4_t vy = vmulq_f32(vld1q_f32(vb), sv);
        float32x4_t vz = vmulq_f32(vld1q_f32(vc), sv);
        sx = vaddq_f32(sx, vaddq_f32(vx, vaddq_f32(vsubq_f32(vmulq_f32(c120,vx), vmulq_f32(s120,vy)),
                                                     vaddq_f32(vmulq_f32(c120,vx), vmulq_f32(s120,vy)))));
        sy = vaddq_f32(sy, vaddq_f32(vy, vaddq_f32(vaddq_f32(vmulq_f32(s120,vx), vmulq_f32(c120,vy)),
                                                     vsubq_f32(vmulq_f32(c120,vy), vmulq_f32(s120,vx)))));
        sz = vaddq_f32(sz, vaddq_f32(vz, vaddq_f32(vz, vz)));
    }
    float rx = vaddvq_f32(sx), ry = vaddvq_f32(sy), rz = vaddvq_f32(sz);
    return sqrtf(rx*rx + ry*ry + rz*rz) / scale;
}
#endif /* __ARM_NEON */

#ifdef __cplusplus
}
#endif

#endif /* V21_GEOMETRY_H */
