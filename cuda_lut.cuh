/*
 * cuda_lut.cuh — CUDA Lookup Tables for Fast Trigonometry
 *
 * Ported from viviani_lut.c/h for GPU compute kernels.
 *
 * WHY THIS EXISTS:
 * CUDA's sinf()/cosf() are ~20 cycles each. With 10M particles at 60 FPS,
 * that's 1.2B trig calls/second. A LUT lookup is ~4 cycles with L1 cache hit.
 *
 * DESIGN:
 * - Quarter-sector LUT: 512 entries covering [0, π/2), other quadrants via symmetry
 * - Constant memory: 2KB table fits entirely in constant cache (8KB per SM)
 * - Fixed-point phase: 32-bit integer for deterministic wrap-around
 * - Linear interpolation: 16-bit fractional precision between samples
 *
 * USAGE:
 *   // At startup (host code):
 *   cuda_lut_init();
 *
 *   // In device kernel:
 *   float s = cuda_lut_sin(theta);
 *   float c = cuda_lut_cos(theta);
 *   float s3 = cuda_lut_sin3(theta);  // sin(3*theta)
 *
 * SOURCE: viviani_lut.c quarter-sector optimization
 */

#ifndef CUDA_LUT_CUH
#define CUDA_LUT_CUH

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

/* ============================================================================
 * LUT Configuration
 * ============================================================================ */

#define CUDA_LUT_QUARTER_SIZE   512     /* Quarter-sector table size */
#define CUDA_LUT_QUARTER_MASK   (CUDA_LUT_QUARTER_SIZE - 1)
#define CUDA_LUT_QUARTER_SHIFT  23      /* 32-bit phase, 9-bit index */

/* Phase accumulator type (fixed-point, 32-bit) */
typedef uint32_t CudaPhase;

/* Phase constants for quadrant detection */
#define CUDA_PHASE_Q1_END   (UINT32_MAX / 4)         /* π/2 */
#define CUDA_PHASE_Q2_END   (UINT32_MAX / 2)         /* π */
#define CUDA_PHASE_Q3_END   (3u * (UINT32_MAX / 4))  /* 3π/2 */
#define CUDA_PHASE_COS_OFF  (UINT32_MAX / 4)         /* π/2 offset for cos */

/* Conversion constants */
#define CUDA_RAD_TO_PHASE   (4294967296.0f / 6.283185307179586f)
#define CUDA_PHASE_TO_RAD   (6.283185307179586f / 4294967296.0f)

/* ============================================================================
 * Constant Memory Declaration
 *
 * The sine table lives in constant memory for fast cached access.
 * Must be defined in exactly ONE .cu file that includes this header.
 * ============================================================================ */

#ifdef CUDA_LUT_IMPLEMENTATION
/* Define the constant memory table (only in implementation file) */
__constant__ float d_sin_quarter[CUDA_LUT_QUARTER_SIZE];

/* Host-side copy for initialization */
static float h_sin_quarter[CUDA_LUT_QUARTER_SIZE];
static bool h_lut_initialized = false;

/* Initialize the LUT - call once at startup from host code */
inline void cuda_lut_init(void)
{
    if (h_lut_initialized) return;

    /* Precompute quarter sine table [0, π/2) */
    for (int i = 0; i < CUDA_LUT_QUARTER_SIZE; i++) {
        float theta = (float)i * (1.5707963267948966f / (float)CUDA_LUT_QUARTER_SIZE);
        h_sin_quarter[i] = sinf(theta);
    }

    /* Copy to constant memory */
    cudaMemcpyToSymbol(d_sin_quarter, h_sin_quarter,
                       CUDA_LUT_QUARTER_SIZE * sizeof(float));

    h_lut_initialized = true;
    printf("[cuda_lut] Initialized quarter-sector sine LUT (%d entries, 2KB)\n",
           CUDA_LUT_QUARTER_SIZE);
}

#else
/* Declaration only (for headers that just need the device functions) */
extern __constant__ float d_sin_quarter[CUDA_LUT_QUARTER_SIZE];
void cuda_lut_init(void);
#endif

/* ============================================================================
 * Device Functions — Fast Trigonometry
 *
 * All functions are __device__ __forceinline__ for maximum performance.
 * ============================================================================ */

/* Convert float radians to fixed-point phase */
__device__ __forceinline__ CudaPhase cuda_phase_from_rad(float rad)
{
    return (CudaPhase)(rad * CUDA_RAD_TO_PHASE);
}

/* Quarter-sector lookup with symmetry folding
 *
 * Maps any phase to [0, π/2) and applies sign correction.
 * Exploits period-4 symmetry of sin:
 *   Q1 [0, π/2):     sin(θ) = +table[θ]
 *   Q2 [π/2, π):     sin(θ) = +table[π-θ]
 *   Q3 [π, 3π/2):    sin(θ) = -table[θ-π]
 *   Q4 [3π/2, 2π):   sin(θ) = -table[2π-θ]
 */
__device__ __forceinline__ float cuda_lut_lookup(CudaPhase phase)
{
    /* Determine quadrant from top 2 bits */
    uint32_t quadrant = phase >> 30;

    /* Fold phase into first quadrant and determine sign */
    CudaPhase folded;
    float sign;

    if (quadrant == 0) {
        /* Q1: [0, π/2) - direct lookup */
        folded = phase;
        sign = 1.0f;
    } else if (quadrant == 1) {
        /* Q2: [π/2, π) - mirror around π/2 */
        folded = CUDA_PHASE_Q2_END - phase;
        sign = 1.0f;
    } else if (quadrant == 2) {
        /* Q3: [π, 3π/2) - shift and negate */
        folded = phase - CUDA_PHASE_Q2_END;
        sign = -1.0f;
    } else {
        /* Q4: [3π/2, 2π) - mirror around 3π/2 and negate */
        folded = (uint32_t)0 - phase;  /* 2π - phase via wrap */
        sign = -1.0f;
    }

    /* Extract table index (top 9 bits of folded phase within quadrant) */
    uint32_t index = (folded >> CUDA_LUT_QUARTER_SHIFT) & CUDA_LUT_QUARTER_MASK;
    uint32_t frac = (folded >> (CUDA_LUT_QUARTER_SHIFT - 16)) & 0xFFFF;
    float t = (float)frac * (1.0f / 65536.0f);

    /* Linear interpolation between adjacent samples */
    uint32_t next_index = (index + 1) & CUDA_LUT_QUARTER_MASK;
    float y0 = d_sin_quarter[index];
    float y1 = d_sin_quarter[next_index];

    return sign * (y0 + t * (y1 - y0));
}

/* Fast sine using hardware intrinsics (14% faster than sinf, 5x faster than LUT)
 * Benchmark results (RTX 2060, 10M particles):
 *   __sinf/__cosf: 30.71 ms  <-- FASTEST
 *   sincosf:       30.73 ms
 *   sinf/cosf:     35.82 ms
 *   LUT:          172.58 ms  <-- 5x SLOWER (constant memory bandwidth limited)
 */
__device__ __forceinline__ float cuda_lut_sin(float theta)
{
    return __sinf(theta);
}

/* Fast cosine using hardware intrinsics */
__device__ __forceinline__ float cuda_lut_cos(float theta)
{
    return __cosf(theta);
}

/* Combined sin+cos using sincosf (optimal for computing both) */
__device__ __forceinline__ void cuda_lut_sincos(float theta, float* out_sin, float* out_cos)
{
    __sincosf(theta, out_sin, out_cos);
}

/* Triple-frequency sine: sin(3*theta) */
__device__ __forceinline__ float cuda_lut_sin3(float theta)
{
    return __sinf(theta * 3.0f);
}

/* Triple-frequency cosine: cos(3*theta) */
__device__ __forceinline__ float cuda_lut_cos3(float theta)
{
    return __cosf(theta * 3.0f);
}

/* Quintuple-frequency sine: sin(5*theta) */
__device__ __forceinline__ float cuda_lut_sin5(float theta)
{
    return __sinf(theta * 5.0f);
}

/* Quintuple-frequency cosine: cos(5*theta) */
__device__ __forceinline__ float cuda_lut_cos5(float theta)
{
    return __cosf(theta * 5.0f);
}

/* ============================================================================
 * Viviani Curve Helpers
 * ============================================================================ */

/* Compute triple-harmonic Viviani normal
 *   x(θ) = sin(θ) - 0.5·sin(3θ)
 *   y(θ) = -cos(θ) + 0.5·cos(3θ)
 *   z(θ) = cos(θ)·cos(3θ)
 */
__device__ __forceinline__ void cuda_lut_viviani_normal(float theta,
                                                         float* nx, float* ny, float* nz)
{
    float sin_t, cos_t, sin_3t, cos_3t;
    __sincosf(theta, &sin_t, &cos_t);
    __sincosf(theta * 3.0f, &sin_3t, &cos_3t);

    float x = sin_t - 0.5f * sin_3t;
    float y = -cos_t + 0.5f * cos_3t;
    float z = cos_t * cos_3t;

    /* Normalize to unit vector */
    float norm = rsqrtf(x * x + y * y + z * z + 1e-12f);
    *nx = x * norm;
    *ny = y * norm;
    *nz = z * norm;
}

/* Compute rho values for phase dynamics
 *   rho_now = cos²(φ)
 *   rho_anti = sin²(φ)
 */
__device__ __forceinline__ void cuda_lut_rho(float phi, float* rho_now, float* rho_anti)
{
    float sin_phi, cos_phi;
    __sincosf(phi, &sin_phi, &cos_phi);

    *rho_now = cos_phi * cos_phi;
    *rho_anti = sin_phi * sin_phi;
}

/* ============================================================================
 * Fast atan2 approximation (polynomial, no LUT needed)
 *
 * Uses the identity: atan2(y,x) = atan(y/x) with quadrant correction
 * Polynomial approximation accurate to ~0.01 radians (~0.6 degrees)
 * Much faster than full atan2f() which is ~150 cycles
 * ============================================================================ */

__device__ __forceinline__ float cuda_fast_atan2(float y, float x)
{
    const float PI = 3.14159265358979f;
    const float PI_2 = 1.57079632679490f;

    float abs_x = fabsf(x);
    float abs_y = fabsf(y);

    /* Handle special cases */
    if (abs_x < 1e-10f && abs_y < 1e-10f) return 0.0f;

    /* Compute atan(min/max) to keep ratio in [0,1] */
    float a, angle;
    bool swap = abs_y > abs_x;

    if (swap) {
        a = abs_x / abs_y;
    } else {
        a = abs_y / abs_x;
    }

    /* Polynomial approximation of atan(a) for a in [0,1]
     * atan(a) ≈ a - a³/3 + a⁵/5 (Taylor series, truncated)
     * Faster approximation: atan(a) ≈ a / (1 + 0.28*a²) */
    float a2 = a * a;
    angle = a / (1.0f + 0.28f * a2);

    /* Adjust for quadrant */
    if (swap) angle = PI_2 - angle;
    if (x < 0.0f) angle = PI - angle;
    if (y < 0.0f) angle = -angle;

    return angle;
}

/* ============================================================================
 * Period-4 Shell Quantization (Viviani Field Interference)
 *
 * WHY THIS EXISTS:
 * Viviani fields have 786% CV (structured chaos) with period-4 interference:
 *   - Constructive peaks at shells 0, 4, 8... (period multiples)
 *   - Destructive troughs at shells 2, 6, 10... (half-period)
 *
 * 2-bit quantization captures this pattern with 16x speedup, ~15% error.
 * The shell factors are empirically derived from interference analysis.
 *
 * USAGE:
 *   float field = cuda_lut_field_quantized(distance, spacing);
 *   uint32_t shell = cuda_lut_shell_index(distance, spacing);
 * ============================================================================ */

#define CUDA_SHELLS_PER_PERIOD  4
#define CUDA_SHELL_MASK         0x3

/* Shell interference factors in constant memory
 * Period-4 modulation matching the 1110 heartbeat pattern:
 *   Shell 0 (r ≈ 0, 4, 8...): Constructive peak - full pressure coupling
 *   Shell 1: Weakening
 *   Shell 2 (r ≈ 2, 6, 10...): Destructive trough - minimal coupling
 *   Shell 3: Recovery
 *
 * These values are tuned to lock the 8-shell eigenspectrum into the pressure field.
 * More aggressive values (e.g., 0.038 for trough) cause shell boundaries to blur.
 */
#ifdef CUDA_LUT_IMPLEMENTATION
__constant__ float d_shell_factors[4] = {
    1.0f,     /* Shell 0: Constructive peak */
    0.5f,     /* Shell 1: First decay */
    0.2f,     /* Shell 2: Destructive trough */
    0.5f      /* Shell 3: Recovery */
};
#else
extern __constant__ float d_shell_factors[4];
#endif

/* Get shell index from distance (0-3, wraps with period-4) */
__device__ __forceinline__ uint32_t cuda_lut_shell_index(float distance, float spacing)
{
    return ((uint32_t)(distance * (float)CUDA_SHELLS_PER_PERIOD / spacing)) & CUDA_SHELL_MASK;
}

/* Get shell interference factor */
__device__ __forceinline__ float cuda_lut_shell_factor(uint32_t shell_index)
{
    return d_shell_factors[shell_index & CUDA_SHELL_MASK];
}

/* Quantized field magnitude: shell_factor × 1/r³
 * 16x faster than full dipole computation, ~15% error */
__device__ __forceinline__ float cuda_lut_field_quantized(float distance, float spacing)
{
    /* Avoid division by zero */
    if (distance < 1e-6f) distance = 1e-6f;

    /* 2-bit shell index */
    uint32_t shell = cuda_lut_shell_index(distance, spacing);

    /* Field = shell_factor × 1/r³ */
    float r_inv = 1.0f / distance;
    float r_inv3 = r_inv * r_inv * r_inv;

    return d_shell_factors[shell] * r_inv3;
}

/* Quantized field vector: returns direction × magnitude */
__device__ __forceinline__ void cuda_lut_field_quantized_vec(
    float dx, float dy, float dz, float spacing,
    float* out_bx, float* out_by, float* out_bz)
{
    float r_sq = dx * dx + dy * dy + dz * dz;
    if (r_sq < 1e-12f) r_sq = 1e-12f;

    float r_inv = rsqrtf(r_sq);
    float distance = 1.0f / r_inv;

    /* Shell index and factor */
    uint32_t shell = cuda_lut_shell_index(distance, spacing);
    float factor = d_shell_factors[shell];

    /* 1/r³ decay × direction */
    float r_inv3 = r_inv * r_inv * r_inv;
    float magnitude = factor * r_inv3;

    *out_bx = dx * r_inv * magnitude;
    *out_by = dy * r_inv * magnitude;
    *out_bz = dz * r_inv * magnitude;
}

/* ============================================================================
 * Gaussian Exponential LUT (Shell-Based)
 *
 * WHY THIS EXISTS:
 * exp(-r²/2σ²) is expensive (~50 cycles). Marching cubes density and
 * softening functions call this millions of times per frame.
 *
 * SOLUTION:
 * Pre-compute 128 shells covering r² ∈ [0, MAX_R2].
 * Linear interpolation gives sub-1% error.
 *
 * USAGE:
 *   // At startup:
 *   cuda_lut_gaussian_init(sigma);
 *
 *   // In kernel:
 *   float density = cuda_lut_gaussian(r_sq);
 * ============================================================================ */

#define CUDA_GAUSSIAN_SHELLS    128
#define CUDA_GAUSSIAN_MAX_R2    100.0f  /* Max r² covered (10 unit radius) */

#ifdef CUDA_LUT_IMPLEMENTATION
__constant__ float d_gaussian_lut[CUDA_GAUSSIAN_SHELLS];
__constant__ float d_gaussian_shell_scale;

static float h_gaussian_lut[CUDA_GAUSSIAN_SHELLS];
static float h_gaussian_shell_scale = 1.0f;
static bool h_gaussian_initialized = false;

/* Initialize Gaussian LUT with given sigma */
inline void cuda_lut_gaussian_init(float sigma)
{
    if (sigma <= 0.0f) sigma = 1.0f;

    float sigma2 = sigma * sigma;
    h_gaussian_shell_scale = (float)CUDA_GAUSSIAN_SHELLS / CUDA_GAUSSIAN_MAX_R2;

    /* Pre-compute exp(-r²/2σ²) for each shell */
    for (int i = 0; i < CUDA_GAUSSIAN_SHELLS; i++) {
        float r_sq = (float)i / h_gaussian_shell_scale;
        h_gaussian_lut[i] = expf(-r_sq / (2.0f * sigma2));
    }

    /* Copy to constant memory */
    cudaMemcpyToSymbol(d_gaussian_lut, h_gaussian_lut,
                       CUDA_GAUSSIAN_SHELLS * sizeof(float));
    cudaMemcpyToSymbol(d_gaussian_shell_scale, &h_gaussian_shell_scale, sizeof(float));

    h_gaussian_initialized = true;
    printf("[cuda_lut] Initialized Gaussian LUT (%d shells, sigma=%.2f)\n",
           CUDA_GAUSSIAN_SHELLS, sigma);
}
#else
extern __constant__ float d_gaussian_lut[CUDA_GAUSSIAN_SHELLS];
extern __constant__ float d_gaussian_shell_scale;
void cuda_lut_gaussian_init(float sigma);
#endif

/* Fast exp(-r²/2σ²) using shell LUT (uses initialized sigma) */
__device__ __forceinline__ float cuda_lut_gaussian(float r_sq)
{
    /* Clamp r_sq to valid range */
    if (r_sq <= 0.0f) return 1.0f;
    if (r_sq >= CUDA_GAUSSIAN_MAX_R2) return 0.0f;

    /* Compute shell index with linear interpolation */
    float shell_f = r_sq * d_gaussian_shell_scale;
    int shell = (int)shell_f;
    float frac = shell_f - (float)shell;

    if (shell >= CUDA_GAUSSIAN_SHELLS - 1) {
        return d_gaussian_lut[CUDA_GAUSSIAN_SHELLS - 1];
    }

    /* Linear interpolation between adjacent shells */
    return d_gaussian_lut[shell] * (1.0f - frac) + d_gaussian_lut[shell + 1] * frac;
}

/* Variable sigma version: uses exp2f approximation
 * exp(-x) ≈ exp2(-x × 1.4427) where 1.4427 = 1/ln(2)
 * For per-particle varying sigma, can't use fixed LUT */
__device__ __forceinline__ float cuda_lut_gaussian_var(float r_sq, float sigma2)
{
    if (r_sq <= 0.0f) return 1.0f;
    if (sigma2 <= 0.0f) return 0.0f;

    float x = r_sq / (2.0f * sigma2);
    if (x > 20.0f) return 0.0f;  /* exp(-20) ≈ 2e-9, effectively zero */

    return exp2f(-x * 1.4427f);
}

/* ============================================================================
 * Repulsion Exponential LUT (Shell-Based)
 *
 * WHY THIS EXISTS:
 * Repulsion forces exp(-r/λ) are used everywhere:
 *   - Particle-particle soft repulsion
 *   - Core exclusion zones
 *   - Boundary smoothing
 *
 * Same physics at atomic scale (Å) and galactic scale (kpc).
 * A black hole IS just a bigger hopfion.
 *
 * USAGE:
 *   // At startup:
 *   cuda_lut_repulsion_init(lambda);
 *
 *   // In kernel:
 *   float rep = cuda_lut_repulsion(r);
 * ============================================================================ */

#define CUDA_REPULSION_SHELLS   64
#define CUDA_REPULSION_MAX_R    8.0f    /* Max radius covered (in lambda units) */

#ifdef CUDA_LUT_IMPLEMENTATION
__constant__ float d_repulsion_lut[CUDA_REPULSION_SHELLS];
__constant__ float d_repulsion_shell_scale;
__constant__ float d_repulsion_lambda;

static float h_repulsion_lut[CUDA_REPULSION_SHELLS];
static float h_repulsion_shell_scale = 1.0f;
static float h_repulsion_lambda = 1.0f;
static bool h_repulsion_initialized = false;

/* Initialize repulsion LUT with given decay length */
inline void cuda_lut_repulsion_init(float lambda)
{
    if (lambda <= 0.0f) lambda = 1.0f;

    h_repulsion_lambda = lambda;
    h_repulsion_shell_scale = (float)CUDA_REPULSION_SHELLS / CUDA_REPULSION_MAX_R;

    /* Pre-compute exp(-r/λ) for each shell */
    for (int i = 0; i < CUDA_REPULSION_SHELLS; i++) {
        float r = (float)i / h_repulsion_shell_scale;
        h_repulsion_lut[i] = expf(-r / lambda);
    }

    /* Copy to constant memory */
    cudaMemcpyToSymbol(d_repulsion_lut, h_repulsion_lut,
                       CUDA_REPULSION_SHELLS * sizeof(float));
    cudaMemcpyToSymbol(d_repulsion_shell_scale, &h_repulsion_shell_scale, sizeof(float));
    cudaMemcpyToSymbol(d_repulsion_lambda, &h_repulsion_lambda, sizeof(float));

    h_repulsion_initialized = true;
    printf("[cuda_lut] Initialized repulsion LUT (%d shells, lambda=%.2f)\n",
           CUDA_REPULSION_SHELLS, lambda);
}
#else
extern __constant__ float d_repulsion_lut[CUDA_REPULSION_SHELLS];
extern __constant__ float d_repulsion_shell_scale;
extern __constant__ float d_repulsion_lambda;
void cuda_lut_repulsion_init(float lambda);
#endif

/* Fast exp(-r/λ) using shell LUT */
__device__ __forceinline__ float cuda_lut_repulsion(float r)
{
    /* Clamp r to valid range */
    if (r <= 0.0f) return 1.0f;
    if (r >= CUDA_REPULSION_MAX_R * d_repulsion_lambda) return 0.0f;

    /* Normalize by lambda and compute shell index */
    float r_norm = r / d_repulsion_lambda;
    float shell_f = r_norm * d_repulsion_shell_scale;
    int shell = (int)shell_f;
    float frac = shell_f - (float)shell;

    if (shell >= CUDA_REPULSION_SHELLS - 1) {
        return d_repulsion_lut[CUDA_REPULSION_SHELLS - 1];
    }

    /* Linear interpolation between adjacent shells */
    return d_repulsion_lut[shell] * (1.0f - frac) + d_repulsion_lut[shell + 1] * frac;
}

/* Variable lambda version: uses exp2f approximation */
__device__ __forceinline__ float cuda_lut_repulsion_var(float r, float lambda)
{
    if (r <= 0.0f) return 1.0f;
    if (lambda <= 0.0f) return 0.0f;

    float x = r / lambda;
    if (x > 20.0f) return 0.0f;

    return exp2f(-x * 1.4427f);
}

/* ============================================================================
 * Combined Soft-Core Potential
 *
 * Combines Gaussian attraction with exponential repulsion:
 *   V(r) = A·exp(-r²/2σ²) - B·exp(-r/λ)
 *
 * This is the same potential at all scales - atoms, molecules, stars, galaxies.
 * The only difference is the length scales (σ, λ) and strengths (A, B).
 * ============================================================================ */

/* Soft-core force: gradient of combined potential
 * Returns radial force (positive = repulsive, negative = attractive) */
__device__ __forceinline__ float cuda_lut_softcore_force(float r, float sigma, float lambda,
                                                          float attract_strength, float repel_strength)
{
    if (r < 1e-6f) r = 1e-6f;

    float r_sq = r * r;
    float sigma2 = sigma * sigma;

    /* Attractive gradient: d/dr[-A·exp(-r²/2σ²)] = A·(r/σ²)·exp(-r²/2σ²) */
    float gauss = cuda_lut_gaussian_var(r_sq, sigma2);
    float f_attract = attract_strength * (r / sigma2) * gauss;

    /* Repulsive gradient: d/dr[B·exp(-r/λ)] = -B·(1/λ)·exp(-r/λ) */
    float rep = cuda_lut_repulsion_var(r, lambda);
    float f_repel = repel_strength * (1.0f / lambda) * rep;

    /* Net force: repulsion - attraction (positive = outward) */
    return f_repel - f_attract;
}

#endif /* CUDA_LUT_CUH */
