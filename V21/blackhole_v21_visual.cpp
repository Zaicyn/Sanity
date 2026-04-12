/*
 * BLACKHOLE V21 VISUAL — CPU Physics + Vulkan Rendering
 * ======================================================
 *
 * V21 CPU physics (pure C) driving V20's Vulkan rendering pipeline.
 * The GPU's framebuffer IS the harmonic accumulator.
 * Additive blending IS waveform superposition.
 *
 * Build: cmake .. && make blackhole_v21_visual
 * Run:   ./blackhole_v21_visual -n 10000 --frames 5000
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <vector>
#include <algorithm>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

/* V20 Vulkan infrastructure */
#include "../vulkan/vk_types.h"

/* Forward declarations for V20 Vulkan functions */
namespace vk {
    void createInstance(VulkanContext& ctx);
    void setupDebugMessenger(VulkanContext& ctx);
    void pickPhysicalDevice(VulkanContext& ctx);
    void createLogicalDevice(VulkanContext& ctx);
    void createSwapchain(VulkanContext& ctx);
    void createImageViews(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createDescriptorSetLayout(VulkanContext& ctx);
    void createGraphicsPipeline(VulkanContext& ctx);
    void createDepthResources(VulkanContext& ctx);
    void createFramebuffers(VulkanContext& ctx);
    void createCommandPool(VulkanContext& ctx);
    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);
    void cleanup(VulkanContext& ctx);
}

/* V20 rendering functions we need */
namespace vk {
    void updateUniformBuffer(VulkanContext& ctx, uint32_t currentImage);
    void recreateSwapchain(VulkanContext& ctx);
}

/* V20 globals that the rendering code references */
#include "../vulkan/vk_attractor.h"
AttractorPipeline g_attractor = {};

/* ========================================================================
 * Mouse-based camera controls
 * ======================================================================== */
struct CameraDrag {
    bool  left_down = false;
    double last_x = 0.0;
    double last_y = 0.0;
};
static CameraDrag g_cam_drag;

static void mouse_button_cb(GLFWwindow* w, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        g_cam_drag.left_down = true;
        glfwGetCursorPos(w, &g_cam_drag.last_x, &g_cam_drag.last_y);
    } else if (action == GLFW_RELEASE) {
        g_cam_drag.left_down = false;
    }
}

static void cursor_pos_cb(GLFWwindow* w, double x, double y) {
    if (!g_cam_drag.left_down) return;
    VulkanContext* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
    if (!ctx) return;
    double dx = x - g_cam_drag.last_x;
    double dy = y - g_cam_drag.last_y;
    g_cam_drag.last_x = x;
    g_cam_drag.last_y = y;
    const float sens = 0.005f;
    ctx->cameraYaw   += (float)dx * sens;
    ctx->cameraPitch += (float)dy * sens;
    /* Clamp pitch to avoid flipping over the poles */
    const float lim = 1.55f; /* ~89° */
    if (ctx->cameraPitch >  lim) ctx->cameraPitch =  lim;
    if (ctx->cameraPitch < -lim) ctx->cameraPitch = -lim;
}

static void scroll_cb(GLFWwindow* w, double /*xoff*/, double yoff) {
    VulkanContext* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
    if (!ctx) return;
    /* Exponential zoom — 10% per click. Positive = in, negative = out. */
    float factor = (yoff > 0.0) ? 0.9f : 1.1f;
    ctx->cameraRadius *= factor;
    if (ctx->cameraRadius < 10.0f)   ctx->cameraRadius = 10.0f;
    if (ctx->cameraRadius > 10000.0f) ctx->cameraRadius = 10000.0f;
}

/* V21 core (C headers) */
extern "C" {
#include "core/v21_types.h"
#include "core/v21_vertex_pack.h"
#include "core/v21_oracle.h"
}
#include "vk_compute.h"

/* ========================================================================
 * SIMULATION PARAMETERS
 * ======================================================================== */

#define DEFAULT_PARTICLES    10000
#define DEFAULT_DT           (1.0f / 60.0f)
#define BH_MASS              1.0f
#define ISCO_R               6.0f
#define DISK_OUTER_R         120.0f

/* ========================================================================
 * PARTICLE STATE (SoA)
 * ======================================================================== */

struct ParticleState {
    float *pos_x, *pos_y, *pos_z;
    float *vel_x, *vel_y, *vel_z;
    float *pump_scale, *pump_residual, *pump_history;
    int   *pump_state;
    float *theta, *omega_nat;
    uint8_t *flags, *topo_state;
    int N, capacity;
};

/* Init-time particle sort mode. Affects memory order of the initial
 * particle arrays — changes cache behavior of downstream per-particle
 * kernels without changing physics. */
enum InitSortMode {
    INIT_SORT_NONE    = 0,  /* original generation order (random) */
    INIT_SORT_CELL    = 1,  /* sorted by scatter.comp's cell index */
    INIT_SORT_VIVIANI = 2   /* sorted by Viviani curve parameter (atan2 in xz) */
};

/* Optional embedded rigid-body lattice for the constraint-solver experiment.
 * When enabled, the last N_rigid SoA slots hold a structured lattice of
 * particles bound by distance constraints, dropped into the galaxy to test
 * whether GPU constraint work disturbs the Viviani sort's cache coherence. */
enum RigidBodyMode {
    RIGID_BODY_OFF              = 0,  /* no lattice (default) */
    RIGID_BODY_CUBE1000         = 1,  /* 10×10×10 cube, 2700 distance constraints */
    RIGID_BODY_CUBE2_BALLSOCKET = 2,  /* 2 cubes + 1 ball-socket joint (Phase 2.3 MVP) */
    RIGID_BODY_CUBE2_HINGE      = 3,  /* 2 cubes + hinge (2 distance constraints along x-axis, Phase 2.3.1) */
    RIGID_BODY_CUBE2_COLLIDE    = 4   /* 2 antipodal cubes drifting inward, dynamic contact (Phase 2.2) */
};

/* Lattice data computed on the host side and uploaded to the GPU constraint
 * solver. Empty when rigid_body_mode == RIGID_BODY_OFF. */
struct ConstraintLattice {
    std::vector<uint32_t> pairs;              /* flattened uvec2 (size = 2 * M) */
    std::vector<float>    rest;               /* size = M */
    std::vector<float>    inv_m;              /* size = N_rigid */
    uint32_t              bucket_offsets[7];  /* start offset of each bucket in pairs[] */
    uint32_t              bucket_counts[7];   /* count per (axis × parity) bucket; [6] reserved for joint edges */
    uint32_t              base_index;         /* first lattice particle index */
    uint32_t              rigid_count;        /* = N_rigid */

    /* Phase 2.2 collision pipeline metadata. Empty unless the scenario
     * populates it (currently only cube2-collide). When non-empty, length
     * equals the total particle count N (host arrays know N from ParticleState).
     * Each entry is the rigid-body id: 0 for field particles, 1 for cube 0,
     * 2 for cube 1, etc. */
    std::vector<uint32_t> rigid_body_id;
};

/* N          = total allocation size (galaxy + optional trailing lattice slots)
 * n_generate = number of galaxy particles to generate into slots [0, n_generate).
 *              When equal to N (the default), behavior matches the original
 *              single-region init. When smaller, the trailing slots [n_generate, N)
 *              are left calloc-zeroed for a caller (e.g. init_rigid_body_cube) to
 *              fill — crucially, the Viviani sort applies only to the first
 *              n_generate particles, so those slots are bit-identical to a run
 *              with N == n_generate. */
static void init_particles(ParticleState& ps, int N, unsigned int seed,
                           InitSortMode sort_mode = INIT_SORT_NONE,
                           int n_generate = -1) {
    if (n_generate < 0 || n_generate > N) n_generate = N;
    ps.N = N;
    ps.capacity = N;
    ps.pos_x = (float*)calloc(N, sizeof(float));
    ps.pos_y = (float*)calloc(N, sizeof(float));
    ps.pos_z = (float*)calloc(N, sizeof(float));
    ps.vel_x = (float*)calloc(N, sizeof(float));
    ps.vel_y = (float*)calloc(N, sizeof(float));
    ps.vel_z = (float*)calloc(N, sizeof(float));
    ps.pump_scale = (float*)calloc(N, sizeof(float));
    ps.pump_residual = (float*)calloc(N, sizeof(float));
    ps.pump_history = (float*)calloc(N, sizeof(float));
    ps.pump_state = (int*)calloc(N, sizeof(int));
    ps.theta = (float*)calloc(N, sizeof(float));
    ps.omega_nat = (float*)calloc(N, sizeof(float));
    ps.flags = (uint8_t*)calloc(N, sizeof(uint8_t));
    ps.topo_state = (uint8_t*)calloc(N, sizeof(uint8_t));

    /* Temporary AoS buffer for generation + sort. Free before return. */
    struct Particle {
        float px, py, pz;
        float vx, vy, vz;
        float theta, omega_nat;
        uint8_t topo_state;
    };
    std::vector<Particle> tmp(n_generate);
    std::vector<uint64_t> sort_key(n_generate);

    srand(seed);
    for (int i = 0; i < n_generate; i++) {
        float x = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float y = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R * 0.3f;
        float z = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float r = sqrtf(x*x+y*y+z*z);
        if (r < 6.0f) { float s=6.0f/r; x*=s; y*=s; z*=s; }

        Particle p = {};
        p.px = x; p.py = y; p.pz = z;

        float r_xz = sqrtf(x*x+z*z);
        if (r_xz > 0.1f) {
            float v = sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            p.vx = -v*(z/r_xz);
            p.vz =  v*(x/r_xz);
        }
        p.theta = (float)rand()/RAND_MAX * 6.28318f;
        float r3d_init = sqrtf(x*x + y*y + z*z);
        float r_eff = fmaxf(r3d_init, ISCO_R);
        p.omega_nat = 0.377f / r_eff;
        int axis = rand()%4; int sign = (rand()%2)?1:-1;
        p.topo_state = (uint8_t)(((sign>0)?1:2) << (axis*2));
        tmp[i] = p;

        /* Compute sort key based on mode. Both cell and viviani produce
         * uint64 keys we can sort linearly. */
        if (sort_mode == INIT_SORT_CELL) {
            /* Match scatter.comp's cellIndex() math exactly — 64^3 grid
             * over [-250, +250]. Particles in the same cell get the same
             * key, so stable sort preserves intra-cell order. */
            const float half = 250.0f;
            const int   dim  = 64;
            const float inv_h = (float)dim / (2.0f * half);
            int cx = (int)fmaxf(0.0f, fminf((float)(dim-1), (x + half) * inv_h));
            int cy = (int)fmaxf(0.0f, fminf((float)(dim-1), (y + half) * inv_h));
            int cz = (int)fmaxf(0.0f, fminf((float)(dim-1), (z + half) * inv_h));
            sort_key[i] = (uint64_t)(cx + cy * dim + cz * dim * dim);
        } else if (sort_mode == INIT_SORT_VIVIANI) {
            /* Primary axis: Viviani curve parameter θ = atan2(pz, px)
             * mapped to [0, 2π). Inner particles (small r) get a lower-
             * radix tiebreak so each theta-slice is radially coherent. */
            float theta_v = atan2f(z, x) + 3.14159265f;   /* [0, 2π) */
            float r2d = sqrtf(x*x + z*z);
            /* Pack as uint64: high 32 bits = theta bucket (1024 buckets),
             * low 32 bits = radius bucket (16384 buckets). */
            uint32_t theta_bucket = (uint32_t)(theta_v * (1024.0f / 6.28318f));
            uint32_t r_bucket     = (uint32_t)fminf(16383.0f, r2d * (16384.0f / DISK_OUTER_R));
            sort_key[i] = ((uint64_t)theta_bucket << 32) | r_bucket;
        } else {
            sort_key[i] = (uint64_t)i;  /* identity */
        }
    }

    /* Build permutation indices and sort by key. */
    std::vector<uint32_t> perm(n_generate);
    for (int i = 0; i < n_generate; i++) perm[i] = i;
    if (sort_mode != INIT_SORT_NONE) {
        std::stable_sort(perm.begin(), perm.end(),
            [&sort_key](uint32_t a, uint32_t b) { return sort_key[a] < sort_key[b]; });
    }

    /* Scatter sorted AoS back into SoA arrays. Trailing slots [n_generate, N)
     * remain calloc-zeroed for init_rigid_body_cube or similar to populate. */
    for (int i = 0; i < n_generate; i++) {
        const Particle& p = tmp[perm[i]];
        ps.pos_x[i] = p.px; ps.pos_y[i] = p.py; ps.pos_z[i] = p.pz;
        ps.vel_x[i] = p.vx; ps.vel_y[i] = p.vy; ps.vel_z[i] = p.vz;
        ps.theta[i] = p.theta;
        ps.omega_nat[i] = p.omega_nat;
        ps.topo_state[i] = p.topo_state;
        ps.pump_scale[i] = 1.0f;
        ps.flags[i] = 0x01;
    }

    const char* sort_name =
        (sort_mode == INIT_SORT_CELL)    ? "cell"    :
        (sort_mode == INIT_SORT_VIVIANI) ? "viviani" : "none";
    if (n_generate == N) {
        printf("[init] %d particles generated, sort=%s\n", N, sort_name);
    } else {
        printf("[init] %d galaxy particles generated (+%d reserved), sort=%s\n",
               n_generate, N - n_generate, sort_name);
    }
}

/* ========================================================================
 * RIGID-BODY LATTICE (for constraint solver experiment)
 * ======================================================================== */

/* Fill SoA slots [n_galaxy, n_galaxy + 1000) with a 10×10×10 cube lattice
 * centered at (50, 0, 0) — inside the hot gather region of the galaxy disk.
 * Returns the constraint pair list bucketed into 6 vertex-disjoint groups
 * by (axis, parity-of-starting-endpoint) so the GPU solver can do true
 * Gauss-Seidel without atomics or graph coloring at runtime. */
static ConstraintLattice init_rigid_body_cube(ParticleState& ps,
                                              int n_galaxy,
                                              int n_cubes) {
    const int   LX = 10, LY = 10, LZ = 10;
    const int   N_LATTICE     = LX * LY * LZ;          /* 1000 per cube */
    const int   N_LATTICE_TOT = n_cubes * N_LATTICE;
    const float SPACING       = 0.5f;
    const float R_ORBIT       = 50.0f;

    if (n_galaxy + N_LATTICE_TOT > ps.N) {
        fprintf(stderr, "[rigid] ERROR: ps.N=%d insufficient for n_galaxy=%d + "
                        "%d cubes (%d lattice particles)\n",
                ps.N, n_galaxy, n_cubes, N_LATTICE_TOT);
        return ConstraintLattice{};
    }

    const uint32_t base = (uint32_t)n_galaxy;

    /* Per-cube edge buckets, concatenated into 6 lattice super-buckets in cube
     * order, plus a 7th reserved for cross-body joint edges (always empty in
     * this function; populated only by init_rigid_body_cube2_ballsocket and
     * similar joint-aware helpers). Within one super-bucket no two edges share
     * an endpoint because (a) each cube's own bucket k is vertex-disjoint by
     * the (axis × parity) coloring and (b) different cubes occupy disjoint
     * index ranges [base + c·1000, base + (c+1)·1000). So GPU Gauss-Seidel
     * holds without atomics, same as the single-cube MVP, at any cube count.
     * Total GPU dispatches per frame stays at 24 (4 iters × 6 buckets) when
     * bucket 6 is empty, 28 (4 × 7) when joints are enabled. */
    std::vector<std::pair<uint32_t, uint32_t>> super[7];
    for (int k = 0; k < 6; k++) {
        super[k].reserve((size_t)n_cubes * 450);
    }
    /* super[6] stays empty — no joint edges in the pure cube-scaling function. */

    for (int c = 0; c < n_cubes; c++) {
        /* Azimuthal placement on the r=50 circle. For N=1 this collapses to
         * theta=0 → (cx,cy,cz) = (50, 0, 0), bit-identical to the MVP. */
        const float theta = 2.0f * 3.14159265f * (float)c / (float)n_cubes;
        const float cx = R_ORBIT * cosf(theta);
        const float cy = 0.0f;
        const float cz = R_ORBIT * sinf(theta);

        /* Circular-orbit velocity tangent to this cube's own center.
         * Viviani field is rotational transport only (|v| invariant), so a
         * zero-v lattice would sit still and the constraint solver would have
         * nothing to push against. Each cube gets the velocity a galaxy
         * particle at its center would carry, so it orbits as a rigid body
         * while differential rotation tries to shear it. */
        const float r_center = sqrtf(cx * cx + cy * cy + cz * cz);
        const float v_orbit  = (r_center > 0.1f)
                             ? sqrtf(BH_MASS / fmaxf(r_center, ISCO_R))
                             : 0.0f;
        const float r_xz = sqrtf(cx * cx + cz * cz);
        const float vx0  = (r_xz > 0.1f) ? -v_orbit * (cz / r_xz) : 0.0f;
        const float vz0  = (r_xz > 0.1f) ?  v_orbit * (cx / r_xz) : 0.0f;

        const uint32_t cube_base = base + (uint32_t)(c * N_LATTICE);
        auto idx_of = [&](int ix, int iy, int iz) -> uint32_t {
            return cube_base + (uint32_t)(ix + iy * LX + iz * LX * LY);
        };

        /* Place this cube's 1000 particles. Same nested-loop order as MVP, so
         * for n_cubes==1 the SoA writes are bit-identical to commit eda621d. */
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t p = idx_of(ix, iy, iz);
            ps.pos_x[p] = cx + (ix - (LX - 1) * 0.5f) * SPACING;
            ps.pos_y[p] = cy + (iy - (LY - 1) * 0.5f) * SPACING;
            ps.pos_z[p] = cz + (iz - (LZ - 1) * 0.5f) * SPACING;
            ps.vel_x[p] = vx0;
            ps.vel_y[p] = 0.0f;
            ps.vel_z[p] = vz0;
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;             /* PFLAG_ACTIVE */
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;        /* any non-zero topo state */
        }

        /* Build this cube's 6 local buckets with the same (axis × parity)
         * coloring as the MVP. */
        std::vector<std::pair<uint32_t, uint32_t>> local[6];
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t a = idx_of(ix, iy, iz);
            if (ix + 1 < LX) {
                int bucket = 0 + (ix & 1);
                local[bucket].push_back({a, idx_of(ix + 1, iy, iz)});
            }
            if (iy + 1 < LY) {
                int bucket = 2 + (iy & 1);
                local[bucket].push_back({a, idx_of(ix, iy + 1, iz)});
            }
            if (iz + 1 < LZ) {
                int bucket = 4 + (iz & 1);
                local[bucket].push_back({a, idx_of(ix, iy, iz + 1)});
            }
        }

        /* Concat this cube's locals onto the super-buckets in cube order.
         * Cube 0 first, then cube 1, ... — deterministic ordering so N=1
         * produces a flattened pair list bit-identical to the MVP. */
        for (int k = 0; k < 6; k++) {
            super[k].insert(super[k].end(), local[k].begin(), local[k].end());
        }
    }

    ConstraintLattice L = {};   /* value-init so bucket slot [6] is zero */
    L.base_index  = base;
    L.rigid_count = (uint32_t)N_LATTICE_TOT;
    L.inv_m.assign(N_LATTICE_TOT, 1.0f);

    uint32_t running = 0;
    for (int k = 0; k < 7; k++) {
        L.bucket_offsets[k] = running;
        L.bucket_counts[k]  = (uint32_t)super[k].size();
        running += L.bucket_counts[k];
    }
    L.pairs.reserve(2u * running);
    L.rest.reserve(running);
    for (int k = 0; k < 7; k++) {
        for (auto& pr : super[k]) {
            L.pairs.push_back(pr.first);
            L.pairs.push_back(pr.second);
            L.rest.push_back(SPACING);      /* regular grid → uniform rest length */
        }
    }

    /* 6-bucket printf unchanged — bucket 6 is always 0 in this function and
     * printing it would break the Phase 2.1 regression-check format. The new
     * init_rigid_body_cube2_ballsocket function has its own 7-bucket printf. */
    printf("[rigid] cube1000 x%d on r=%.0f circle: %d lattice particles, "
           "%u constraints, buckets=[%u %u %u %u %u %u], base=%u\n",
           n_cubes, R_ORBIT, N_LATTICE_TOT, running,
           L.bucket_counts[0], L.bucket_counts[1], L.bucket_counts[2],
           L.bucket_counts[3], L.bucket_counts[4], L.bucket_counts[5], base);
    return L;
}

/* Phase 2.3 MVP: two cubes + one ball-socket joint.
 *
 * Builds two 10×10×10 lattices stacked vertically in y at (50, ±2.75, 0)
 * with a 0.5-unit gap at y=0. Adds a single zero-length distance constraint
 * between the center of cube 0's top face (iy=9) and the center of cube 1's
 * bottom face (iy=0). The joint constraint lives in super-bucket 6 — a 7th
 * bucket append to the normal 6 lattice buckets. Within super-bucket 6 the
 * MVP has exactly one edge, so vertex-disjointness is trivially preserved.
 *
 * Both cubes get the same circular-orbit velocity tangent to r=50 (not
 * each cube's own r — the y-offset makes their r slightly different, but
 * using a midpoint velocity is within the joint's correction envelope and
 * keeps the math simpler).
 *
 * This function does NOT share code with init_rigid_body_cube despite
 * superficial similarity — the contracts are different (fixed 2 bodies
 * + joint vs. N independent bodies on a circle), and the per-cube
 * placement duplication is ~20 lines and acceptable. */
static ConstraintLattice init_rigid_body_cube2_ballsocket(ParticleState& ps,
                                                          int n_galaxy) {
    const int   LX = 10, LY = 10, LZ = 10;
    const int   N_LATTICE     = LX * LY * LZ;          /* 1000 per cube */
    const int   N_LATTICE_TOT = 2 * N_LATTICE;         /* 2000 */
    const float SPACING       = 0.5f;
    const float R_MIDPOINT    = 50.0f;
    const float Y_OFFSET      = 2.75f;                 /* half-gap between cube centers */

    if (n_galaxy + N_LATTICE_TOT > ps.N) {
        fprintf(stderr, "[rigid] ERROR: ps.N=%d insufficient for n_galaxy=%d + "
                        "2 cubes + joint\n", ps.N, n_galaxy);
        return ConstraintLattice{};
    }

    /* Shared orbital velocity, computed at the midpoint (50, 0, 0).
     * Each cube's own center is at r ≈ 50.075; the 0.15% velocity mismatch
     * is well within the joint's correction envelope. */
    const float r_center = R_MIDPOINT;
    const float v_orbit  = sqrtf(BH_MASS / fmaxf(r_center, ISCO_R));
    const float vx0      = 0.0f;
    const float vz0      = v_orbit;   /* tangent to +x at (50, 0, 0) */

    const uint32_t base = (uint32_t)n_galaxy;

    /* 7 super-buckets: 0..5 lattice, 6 joint. */
    std::vector<std::pair<uint32_t, uint32_t>> super[7];
    for (int k = 0; k < 6; k++) super[k].reserve(900);
    super[6].reserve(1);

    const float cube_centers_y[2] = { -Y_OFFSET, +Y_OFFSET };
    uint32_t joint_anchor[2] = { 0, 0 };

    for (int c = 0; c < 2; c++) {
        const float cx = R_MIDPOINT;
        const float cy = cube_centers_y[c];
        const float cz = 0.0f;

        const uint32_t cube_base = base + (uint32_t)(c * N_LATTICE);
        auto idx_of = [&](int ix, int iy, int iz) -> uint32_t {
            return cube_base + (uint32_t)(ix + iy * LX + iz * LX * LY);
        };

        /* Place particles — same body as init_rigid_body_cube per-cube block. */
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t p = idx_of(ix, iy, iz);
            ps.pos_x[p] = cx + (ix - (LX - 1) * 0.5f) * SPACING;
            ps.pos_y[p] = cy + (iy - (LY - 1) * 0.5f) * SPACING;
            ps.pos_z[p] = cz + (iz - (LZ - 1) * 0.5f) * SPACING;
            ps.vel_x[p] = vx0;
            ps.vel_y[p] = 0.0f;
            ps.vel_z[p] = vz0;
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;             /* PFLAG_ACTIVE */
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;
        }

        /* Build this cube's 6 local buckets with the same coloring. */
        std::vector<std::pair<uint32_t, uint32_t>> local[6];
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t a = idx_of(ix, iy, iz);
            if (ix + 1 < LX) local[0 + (ix & 1)].push_back({a, idx_of(ix + 1, iy, iz)});
            if (iy + 1 < LY) local[2 + (iy & 1)].push_back({a, idx_of(ix, iy + 1, iz)});
            if (iz + 1 < LZ) local[4 + (iz & 1)].push_back({a, idx_of(ix, iy, iz + 1)});
        }
        for (int k = 0; k < 6; k++) {
            super[k].insert(super[k].end(), local[k].begin(), local[k].end());
        }

        /* Record this cube's joint anchor particle.
         * Cube 0: top face (iy = LY - 1), centered in (ix, iz).
         * Cube 1: bottom face (iy = 0), centered in (ix, iz). */
        int anchor_iy = (c == 0) ? (LY - 1) : 0;
        joint_anchor[c] = idx_of(LX / 2, anchor_iy, LZ / 2);
    }

    /* Joint edge: single distance constraint between the two anchor
     * particles, rest length 1.0 (the y-gap between the facing cube
     * faces: cube 0 top at y=-0.5, cube 1 bottom at y=+0.5). */
    super[6].push_back({joint_anchor[0], joint_anchor[1]});

    ConstraintLattice L = {};
    L.base_index  = base;
    L.rigid_count = (uint32_t)N_LATTICE_TOT;
    L.inv_m.assign(N_LATTICE_TOT, 1.0f);

    uint32_t running = 0;
    for (int k = 0; k < 7; k++) {
        L.bucket_offsets[k] = running;
        L.bucket_counts[k]  = (uint32_t)super[k].size();
        running += L.bucket_counts[k];
    }
    L.pairs.reserve(2u * running);
    L.rest.reserve(running);
    /* Lattice edges (rest = SPACING). */
    for (int k = 0; k < 6; k++) {
        for (auto& pr : super[k]) {
            L.pairs.push_back(pr.first);
            L.pairs.push_back(pr.second);
            L.rest.push_back(SPACING);
        }
    }
    /* Joint edges (rest = 1.0 = y-gap between facing cube faces). */
    for (auto& pr : super[6]) {
        L.pairs.push_back(pr.first);
        L.pairs.push_back(pr.second);
        L.rest.push_back(1.0f);
    }

    printf("[rigid] cube2-ballsocket: %d lattice particles, %u constraints "
           "(5400 lattice + %u joint), buckets=[%u %u %u %u %u %u %u], "
           "base=%u, joint_anchors=(%u, %u)\n",
           N_LATTICE_TOT, running, L.bucket_counts[6],
           L.bucket_counts[0], L.bucket_counts[1], L.bucket_counts[2],
           L.bucket_counts[3], L.bucket_counts[4], L.bucket_counts[5],
           L.bucket_counts[6], base, joint_anchor[0], joint_anchor[1]);
    return L;
}

/* Phase 2.3.1: two cubes + one hinge joint (2 distance constraints along
 * the x-axis). Same y-stacked placement as cube2-ballsocket: cubes at
 * (50, ±2.75, 0) with a 0.5-unit gap. The hinge has two anchor pairs,
 * one at each end of the desired rotation axis, both at iz=LZ/2:
 *   edge A: cube0.(ix=0,      iy=LY-1, iz=LZ/2) ↔ cube1.(ix=0,      iy=0, iz=LZ/2)
 *   edge B: cube0.(ix=LX-1,   iy=LY-1, iz=LZ/2) ↔ cube1.(ix=LX-1,   iy=0, iz=LZ/2)
 *
 * The two edges span the x-axis at y=0 between the cube centers, so the
 * hinge allows rotation around x while locking translation along y, z,
 * and x, plus rotation around y and z. Both edges go into super-bucket 6
 * and are vertex-disjoint (different ix values, different particle IDs).
 *
 * Optional spin_rate: if non-zero, cube 1 gets an initial angular velocity
 * of spin_rate rad/frame around the x-axis (the hinge axis). Applied as
 * omega × (p − cube1_center) where omega = (spin_rate, 0, 0). This
 * exercises the hinge's rotational DOF — a ball-socket couldn't absorb
 * this input, a hinge should. spin_rate=0 reproduces the minimal test. */
static ConstraintLattice init_rigid_body_cube2_hinge(ParticleState& ps,
                                                     int n_galaxy,
                                                     float spin_rate) {
    const int   LX = 10, LY = 10, LZ = 10;
    const int   N_LATTICE     = LX * LY * LZ;          /* 1000 per cube */
    const int   N_LATTICE_TOT = 2 * N_LATTICE;         /* 2000 */
    const float SPACING       = 0.5f;
    const float R_MIDPOINT    = 50.0f;
    const float Y_OFFSET      = 2.75f;                 /* half-gap between cube centers */

    if (n_galaxy + N_LATTICE_TOT > ps.N) {
        fprintf(stderr, "[rigid] ERROR: ps.N=%d insufficient for n_galaxy=%d + "
                        "2 cubes + hinge\n", ps.N, n_galaxy);
        return ConstraintLattice{};
    }

    /* Shared orbital velocity, computed at the midpoint. Same as cube2-ballsocket. */
    const float r_center = R_MIDPOINT;
    const float v_orbit  = sqrtf(BH_MASS / fmaxf(r_center, ISCO_R));
    const float vx0      = 0.0f;
    const float vz0      = v_orbit;

    const uint32_t base = (uint32_t)n_galaxy;

    /* 7 super-buckets: 0..5 lattice, 6 joint (2 hinge edges). */
    std::vector<std::pair<uint32_t, uint32_t>> super[7];
    for (int k = 0; k < 6; k++) super[k].reserve(900);
    super[6].reserve(2);

    const float cube_centers_y[2] = { -Y_OFFSET, +Y_OFFSET };
    uint32_t anchor_A[2] = { 0, 0 };   /* cube 0/1 anchor at ix=0 end of hinge axis */
    uint32_t anchor_B[2] = { 0, 0 };   /* cube 0/1 anchor at ix=LX-1 end of hinge axis */

    for (int c = 0; c < 2; c++) {
        const float cx = R_MIDPOINT;
        const float cy = cube_centers_y[c];
        const float cz = 0.0f;

        const uint32_t cube_base = base + (uint32_t)(c * N_LATTICE);
        auto idx_of = [&](int ix, int iy, int iz) -> uint32_t {
            return cube_base + (uint32_t)(ix + iy * LX + iz * LX * LY);
        };

        /* Place particles. For cube 1, optionally add omega × r angular
         * velocity around the x-axis (the hinge axis). omega = (R, 0, 0),
         *   omega × (dx, dy, dz) = (0, -R·dz, R·dy)
         * where (dx, dy, dz) is the offset from cube 1's center. Cube 0
         * is unspun so cube 1's rotation is a *relative* motion across
         * the hinge. */
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t p = idx_of(ix, iy, iz);
            float px = cx + (ix - (LX - 1) * 0.5f) * SPACING;
            float py = cy + (iy - (LY - 1) * 0.5f) * SPACING;
            float pz = cz + (iz - (LZ - 1) * 0.5f) * SPACING;
            ps.pos_x[p] = px;
            ps.pos_y[p] = py;
            ps.pos_z[p] = pz;

            float vy_extra = 0.0f, vz_extra = 0.0f;
            if (c == 1 && spin_rate != 0.0f) {
                float dy = py - cy;
                float dz = pz - cz;
                vy_extra = -spin_rate * dz;
                vz_extra =  spin_rate * dy;
            }

            ps.vel_x[p] = vx0;
            ps.vel_y[p] = 0.0f + vy_extra;
            ps.vel_z[p] = vz0  + vz_extra;
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;             /* PFLAG_ACTIVE */
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;
        }

        /* Build this cube's 6 local lattice buckets — same coloring as
         * cube2-ballsocket. */
        std::vector<std::pair<uint32_t, uint32_t>> local[6];
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t a = idx_of(ix, iy, iz);
            if (ix + 1 < LX) local[0 + (ix & 1)].push_back({a, idx_of(ix + 1, iy, iz)});
            if (iy + 1 < LY) local[2 + (iy & 1)].push_back({a, idx_of(ix, iy + 1, iz)});
            if (iz + 1 < LZ) local[4 + (iz & 1)].push_back({a, idx_of(ix, iy, iz + 1)});
        }
        for (int k = 0; k < 6; k++) {
            super[k].insert(super[k].end(), local[k].begin(), local[k].end());
        }

        /* Record this cube's two hinge anchors. For cube 0 use the top
         * face (iy = LY - 1); for cube 1 the bottom face (iy = 0). Both
         * at iz = LZ/2 (the midpoint of the z range), with ix = 0 and
         * ix = LX - 1 as the two ends of the hinge axis. */
        int anchor_iy = (c == 0) ? (LY - 1) : 0;
        anchor_A[c] = idx_of(0,          anchor_iy, LZ / 2);
        anchor_B[c] = idx_of(LX - 1,     anchor_iy, LZ / 2);
    }

    /* Two hinge edges in bucket 6. Vertex-disjoint by construction:
     * anchor_A uses ix=0, anchor_B uses ix=LX-1, so no shared endpoints. */
    super[6].push_back({anchor_A[0], anchor_A[1]});
    super[6].push_back({anchor_B[0], anchor_B[1]});

    ConstraintLattice L = {};
    L.base_index  = base;
    L.rigid_count = (uint32_t)N_LATTICE_TOT;
    L.inv_m.assign(N_LATTICE_TOT, 1.0f);

    uint32_t running = 0;
    for (int k = 0; k < 7; k++) {
        L.bucket_offsets[k] = running;
        L.bucket_counts[k]  = (uint32_t)super[k].size();
        running += L.bucket_counts[k];
    }
    L.pairs.reserve(2u * running);
    L.rest.reserve(running);
    for (int k = 0; k < 6; k++) {
        for (auto& pr : super[k]) {
            L.pairs.push_back(pr.first);
            L.pairs.push_back(pr.second);
            L.rest.push_back(SPACING);
        }
    }
    for (auto& pr : super[6]) {
        L.pairs.push_back(pr.first);
        L.pairs.push_back(pr.second);
        L.rest.push_back(1.0f);   /* same y-gap as ball-socket */
    }

    printf("[rigid] cube2-hinge: %d lattice particles, %u constraints "
           "(5400 lattice + %u joint), buckets=[%u %u %u %u %u %u %u], "
           "base=%u, hinge_A=(%u, %u), hinge_B=(%u, %u), spin_rate=%.4f rad/frame\n",
           N_LATTICE_TOT, running, L.bucket_counts[6],
           L.bucket_counts[0], L.bucket_counts[1], L.bucket_counts[2],
           L.bucket_counts[3], L.bucket_counts[4], L.bucket_counts[5],
           L.bucket_counts[6], base,
           anchor_A[0], anchor_A[1],
           anchor_B[0], anchor_B[1],
           spin_rate);
    return L;
}

/* Phase 2.2: two cubes on antipodal orbits, drifting inward toward each other.
 *
 * Builds two 10×10×10 lattices placed at (±R_ANTIPODE, 0, 0) on the orbital
 * plane. Each cube gets the orbital tangent velocity at its own location plus
 * a small inward radial drift so the two cubes close on the origin and
 * eventually collide. No joints — bucket 6 is empty.
 *
 *   Cube 0 at (+R_ANTIPODE, 0, 0)
 *     orbital tangent: (0, 0, +v_orbit)         (+ŷ × +x̂ = +ẑ)
 *     inward drift:    (-v_drift, 0, 0)
 *   Cube 1 at (-R_ANTIPODE, 0, 0)
 *     orbital tangent: (0, 0, -v_orbit)         (+ŷ × -x̂ = -ẑ)
 *     inward drift:    (+v_drift, 0, 0)
 *
 * Combined closing velocity along x = 2 * v_drift; with v_drift = 0.01 the
 * gap of 2*R_ANTIPODE = 100 closes at 0.02 per frame. Adjusting for cube
 * half-width (4.5 SPACING ≈ 2.25 each side), expected first contact frame is
 * roughly (100 - 4.5) / 0.02 ≈ 4775.
 *
 * Populates lattice.rigid_body_id with N entries:
 *   index 0..n_galaxy-1                              → 0 (field particles)
 *   index n_galaxy..n_galaxy+999                     → 1 (cube 0)
 *   index n_galaxy+1000..n_galaxy+1999               → 2 (cube 1)
 *
 * This is the data the C2 broadphase will use to filter cross-body pairs. C1
 * uploads it but does not yet read it from any kernel — the apply kernel only
 * touches vel_delta. */
static ConstraintLattice init_rigid_body_cube2_collide(ParticleState& ps,
                                                       int n_galaxy) {
    const int   LX = 10, LY = 10, LZ = 10;
    const int   N_LATTICE     = LX * LY * LZ;          /* 1000 per cube */
    const int   N_LATTICE_TOT = 2 * N_LATTICE;         /* 2000 */
    const float SPACING       = 0.5f;
    const float R_ANTIPODE    = 10.0f;                  /* orbital radius (close for quick contact) */
    const float V_DRIFT       = 0.5f;                  /* inward radial speed (units/sec, not /frame) */

    if (n_galaxy + N_LATTICE_TOT > ps.N) {
        fprintf(stderr, "[rigid] ERROR: ps.N=%d insufficient for n_galaxy=%d + "
                        "2 cubes (cube2-collide)\n", ps.N, n_galaxy);
        return ConstraintLattice{};
    }

    /* Pure head-on collision: no orbital tangent velocity, just inward drift.
     * With in_active_region=0 the lattice is excluded from the Viviani field,
     * so no orbit is needed. The cubes coast inertially along x. */
    const float v_orbit = sqrtf(BH_MASS / fmaxf(R_ANTIPODE, ISCO_R));  /* for info only */

    const uint32_t base = (uint32_t)n_galaxy;

    /* 7 super-buckets: 0..5 lattice, 6 joint (empty for cube2-collide). */
    std::vector<std::pair<uint32_t, uint32_t>> super[7];
    for (int k = 0; k < 6; k++) super[k].reserve(900);

    /* Per-cube center placement: cube 0 at +x, cube 1 at -x. */
    const float cube_centers_x[2] = { +R_ANTIPODE, -R_ANTIPODE };
    /* No tangent: pure head-on collision along x. */
    const float cube_tangent_z[2] = { 0.0f,         0.0f        };
    /* Inward drift x-component: -v at +x cube, +v at -x cube. */
    const float cube_drift_x[2]   = { -V_DRIFT,    +V_DRIFT    };

    for (int c = 0; c < 2; c++) {
        const float cx = cube_centers_x[c];
        const float cy = 0.0f;
        const float cz = 0.0f;

        const uint32_t cube_base = base + (uint32_t)(c * N_LATTICE);
        auto idx_of = [&](int ix, int iy, int iz) -> uint32_t {
            return cube_base + (uint32_t)(ix + iy * LX + iz * LX * LY);
        };

        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t p = idx_of(ix, iy, iz);
            ps.pos_x[p] = cx + (ix - (LX - 1) * 0.5f) * SPACING;
            ps.pos_y[p] = cy + (iy - (LY - 1) * 0.5f) * SPACING;
            ps.pos_z[p] = cz + (iz - (LZ - 1) * 0.5f) * SPACING;
            ps.vel_x[p] = cube_drift_x[c];
            ps.vel_y[p] = 0.0f;
            ps.vel_z[p] = cube_tangent_z[c];
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;             /* PFLAG_ACTIVE */
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;
        }

        /* Build this cube's 6 lattice buckets — same coloring as the
         * other cube2 modes. */
        std::vector<std::pair<uint32_t, uint32_t>> local[6];
        for (int iz = 0; iz < LZ; iz++)
        for (int iy = 0; iy < LY; iy++)
        for (int ix = 0; ix < LX; ix++) {
            uint32_t a = idx_of(ix, iy, iz);
            if (ix + 1 < LX) local[0 + (ix & 1)].push_back({a, idx_of(ix + 1, iy, iz)});
            if (iy + 1 < LY) local[2 + (iy & 1)].push_back({a, idx_of(ix, iy + 1, iz)});
            if (iz + 1 < LZ) local[4 + (iz & 1)].push_back({a, idx_of(ix, iy, iz + 1)});
        }
        for (int k = 0; k < 6; k++) {
            super[k].insert(super[k].end(), local[k].begin(), local[k].end());
        }
    }

    /* Bucket 6 stays empty: no joints for cube2-collide. */

    ConstraintLattice L = {};
    L.base_index  = base;
    L.rigid_count = (uint32_t)N_LATTICE_TOT;
    L.inv_m.assign(N_LATTICE_TOT, 1.0f);

    uint32_t running = 0;
    for (int k = 0; k < 7; k++) {
        L.bucket_offsets[k] = running;
        L.bucket_counts[k]  = (uint32_t)super[k].size();
        running += L.bucket_counts[k];
    }
    L.pairs.reserve(2u * running);
    L.rest.reserve(running);
    for (int k = 0; k < 6; k++) {
        for (auto& pr : super[k]) {
            L.pairs.push_back(pr.first);
            L.pairs.push_back(pr.second);
            L.rest.push_back(SPACING);
        }
    }
    /* Bucket 6 has zero edges; no rest lengths to push. */

    /* Populate the rigid_body_id table for the collision pipeline.
     * Field particles default to 0; cube 0 = 1; cube 1 = 2. */
    L.rigid_body_id.assign((size_t)ps.N, 0u);
    for (int k = 0; k < N_LATTICE; k++) {
        L.rigid_body_id[(size_t)base + (size_t)k] = 1u;
        L.rigid_body_id[(size_t)base + (size_t)N_LATTICE + (size_t)k] = 2u;
    }

    printf("[rigid] cube2-collide: %d lattice particles, %u constraints "
           "(5400 lattice + 0 joint), buckets=[%u %u %u %u %u %u %u], "
           "base=%u, R_ANTIPODE=%.1f, v_orbit=%.4f, v_drift=%.4f, "
           "expected_first_contact_frame≈%d\n",
           N_LATTICE_TOT, running,
           L.bucket_counts[0], L.bucket_counts[1], L.bucket_counts[2],
           L.bucket_counts[3], L.bucket_counts[4], L.bucket_counts[5],
           L.bucket_counts[6], base, R_ANTIPODE, v_orbit, V_DRIFT,
           (int)((2.0f * R_ANTIPODE - (LX - 1) * SPACING) / (2.0f * V_DRIFT)));
    return L;
}

/* ========================================================================
 * VIVIANI FIELD (same as blackhole_v21.c)
 * ======================================================================== */

static void physics_step(ParticleState& ps, float dt) {
    for (int i = 0; i < ps.N; i++) {
        if (!(ps.flags[i] & 0x01)) continue;
        float px=ps.pos_x[i], py=ps.pos_y[i], pz=ps.pos_z[i];
        float vx=ps.vel_x[i], vy=ps.vel_y[i], vz=ps.vel_z[i];

        float r3d = sqrtf(px*px+py*py+pz*pz);
        float inv_r = 1.0f/(r3d+1e-8f);
        float r_safe = fmaxf(r3d, 1.0f);

        /* Viviani field — Gemini's velocity projection-steering prescription.
         * Rotates velocity toward the topological template instead of pulling
         * position toward a point. Replaces the old curve-attracting centripetal
         * (which was a point-attractor producing the "dagger" collapse). */
        float theta_pos = atan2f(pz, px);
        float c1=cosf(theta_pos), s1=sinf(theta_pos);
        float c3=cosf(3*theta_pos), s3=sinf(3*theta_pos);
        /* Asymmetric Viviani curve position scaled to particle's radius */
        float cxp = s1 - 0.5f*s3;
        float cyp = -c1 + 0.5f*c3;
        float czp = c1*c3;
        float inv_cp = 1.0f/sqrtf(cxp*cxp + cyp*cyp + czp*czp + 1e-8f);
        cxp = cxp * inv_cp * r3d;
        cyp = cyp * inv_cp * r3d;
        czp = czp * inv_cp * r3d;

        /* Velocity projection-steering: drive (v·ĉ) toward cos(θ)
         * v_mag clamped at 0.5 (Gemini surgery step 1). */
        float v_mag_raw = sqrtf(vx*vx + vy*vy + vz*vz + 1e-8f);
        float v_mag = fmaxf(v_mag_raw, 0.5f);
        float dot_vc = (vx*cxp + vy*cyp + vz*czp) / (v_mag * r3d);
        float targetp = cosf(theta_pos);
        float steering = 0.01f * (dot_vc - targetp);  /* FIELD_STRENGTH = 0.01 */
        float inv_r_force = 1.0f / r3d;
        float ax = -steering * cxp * inv_r_force;
        float ay = -steering * cyp * inv_r_force;
        float az = -steering * czp * inv_r_force;

        /* Gemini surgery step 2: tangent push DISABLED during diagnostic */
        (void)c1; (void)s1; (void)c3; (void)s3;  /* suppress unused warnings */
        (void)r_safe; (void)inv_r;

        /* Orbital-plane damping — damp velocity parallel to L̂ with flat γ.
         * Uses angular momentum direction (not radial) so in-plane motion
         * is preserved while out-of-plane velocity is removed. */
        float Lx=py*vz-pz*vy, Ly=pz*vx-px*vz, Lz=px*vy-py*vx;
        float iL=1.0f/sqrtf(Lx*Lx+Ly*Ly+Lz*Lz+1e-8f);
        float lx=Lx*iL, ly=Ly*iL, lz=Lz*iL;
        if (vx*vx+vy*vy+vz*vz > 0.001f) {
            float v_parallel = vx*lx+vy*ly+vz*lz;
            const float COHERENCE_GAMMA = 0.02f;
            vx -= COHERENCE_GAMMA * v_parallel * lx;
            vy -= COHERENCE_GAMMA * v_parallel * ly;
            vz -= COHERENCE_GAMMA * v_parallel * lz;
        }

        vx+=ax*dt; vy+=ay*dt; vz+=az*dt;
        if (isfinite(vx)&&isfinite(vy)&&isfinite(vz)) {
            px+=vx*dt; py+=vy*dt; pz+=vz*dt;
        }
        ps.pos_x[i]=px; ps.pos_y[i]=py; ps.pos_z[i]=pz;
        ps.vel_x[i]=vx; ps.vel_y[i]=vy; ps.vel_z[i]=vz;
        ps.pump_history[i] = ps.pump_history[i]*0.98f + ps.pump_scale[i]*0.02f;
        float th = ps.theta[i]+ps.omega_nat[i]*dt;
        if (th>=6.28318f) th-=6.28318f;
        ps.theta[i]=th;
    }
}

static void free_particles(ParticleState& ps) {
    free(ps.pos_x); free(ps.pos_y); free(ps.pos_z);
    free(ps.vel_x); free(ps.vel_y); free(ps.vel_z);
    free(ps.pump_scale); free(ps.pump_residual); free(ps.pump_history);
    free(ps.pump_state); free(ps.theta); free(ps.omega_nat);
    free(ps.flags); free(ps.topo_state);
}

/* ========================================================================
 * VULKAN VERTEX BUFFER — host-visible, CPU-writable
 * ======================================================================== */

struct VertexBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped;
    size_t size;
};

static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0;
}

static void createVertexBuffer(VkDevice device, VkPhysicalDevice physDev,
                                VertexBuffer& vb, size_t size) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufInfo, nullptr, &vb.buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, vb.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physDev, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &allocInfo, nullptr, &vb.memory);
    vkBindBufferMemory(device, vb.buffer, vb.memory, 0);
    vkMapMemory(device, vb.memory, 0, size, 0, &vb.mapped);
    vb.size = size;
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(int argc, char** argv) {
    int num_particles = DEFAULT_PARTICLES;
    unsigned int seed = 42;
    bool use_gpu_physics = false;
    bool project_only = false;
    ScatterMode scatter_mode = SCATTER_MODE_BASELINE;
    InitSortMode init_sort = INIT_SORT_NONE;
    RigidBodyMode rigid_body_mode = RIGID_BODY_OFF;
    int   rigid_count = 1;      /* number of cubes; only meaningful when --rigid-body cube1000 */
    float spin_rate   = 0.0f;   /* cube 1 angular velocity around hinge axis (rad/frame); cube2-hinge only */
    int   max_frames  = 0;      /* exit after this many frames; 0 = run until window closed */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-physics") == 0)
            use_gpu_physics = true;
        else if (strcmp(argv[i], "--project-only") == 0)
            { use_gpu_physics = true; project_only = true; }
        else if (strcmp(argv[i], "--scatter-mode") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if      (strcmp(m, "baseline")  == 0) scatter_mode = SCATTER_MODE_BASELINE;
            else if (strcmp(m, "uniform")   == 0) scatter_mode = SCATTER_MODE_UNIFORM;
            else if (strcmp(m, "squaragon") == 0) scatter_mode = SCATTER_MODE_SQUARAGON;
            else {
                fprintf(stderr, "unknown --scatter-mode '%s' "
                                "(expected baseline, uniform, or squaragon)\n", m);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--init-sort") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if      (strcmp(m, "none")    == 0) init_sort = INIT_SORT_NONE;
            else if (strcmp(m, "cell")    == 0) init_sort = INIT_SORT_CELL;
            else if (strcmp(m, "viviani") == 0) init_sort = INIT_SORT_VIVIANI;
            else {
                fprintf(stderr, "unknown --init-sort '%s' "
                                "(expected none, cell, or viviani)\n", m);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--rigid-body") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if      (strcmp(m, "off")              == 0) rigid_body_mode = RIGID_BODY_OFF;
            else if (strcmp(m, "cube1000")         == 0) rigid_body_mode = RIGID_BODY_CUBE1000;
            else if (strcmp(m, "cube2-ballsocket") == 0) rigid_body_mode = RIGID_BODY_CUBE2_BALLSOCKET;
            else if (strcmp(m, "cube2-hinge")      == 0) rigid_body_mode = RIGID_BODY_CUBE2_HINGE;
            else if (strcmp(m, "cube2-collide")    == 0) rigid_body_mode = RIGID_BODY_CUBE2_COLLIDE;
            else {
                fprintf(stderr, "unknown --rigid-body '%s' "
                                "(expected off, cube1000, cube2-ballsocket, cube2-hinge, or cube2-collide)\n", m);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--rigid-count") == 0 && i+1 < argc) {
            rigid_count = atoi(argv[++i]);
            if (rigid_count < 1) {
                fprintf(stderr, "--rigid-count must be >= 1 (got %d)\n", rigid_count);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--spin-rate") == 0 && i+1 < argc) {
            spin_rate = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc) {
            max_frames = atoi(argv[++i]);
            if (max_frames < 0) max_frames = 0;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole_v21_visual [-n particles] [--rng-seed S] "
                   "[--gpu-physics] [--project-only] "
                   "[--scatter-mode baseline|uniform|squaragon] "
                   "[--init-sort none|cell|viviani] "
                   "[--rigid-body off|cube1000|cube2-ballsocket|cube2-hinge|cube2-collide] "
                   "[--rigid-count N] [--spin-rate R] [--frames F]\n");
            return 0;
        }
    }

    printf("================================================================\n");
    printf("  BLACKHOLE V21 VISUAL\n");
    printf("  CPU Physics + Vulkan Rendering\n");
    printf("  The framebuffer IS the harmonic accumulator.\n");
    printf("================================================================\n\n");

    /* Split N into galaxy + optional rigid-body lattice. `-n` is always the
     * galaxy count; the lattice (if enabled) is appended to the end of all
     * SoA arrays so its indices are known and its presence does not perturb
     * the Viviani sort ordering on the first n_galaxy particles. */
    const int n_galaxy = num_particles;
    int n_rigid = 0;
    if      (rigid_body_mode == RIGID_BODY_CUBE1000)         n_rigid = rigid_count * 1000;
    else if (rigid_body_mode == RIGID_BODY_CUBE2_BALLSOCKET) n_rigid = 2000;
    else if (rigid_body_mode == RIGID_BODY_CUBE2_HINGE)      n_rigid = 2000;
    else if (rigid_body_mode == RIGID_BODY_CUBE2_COLLIDE)    n_rigid = 2000;
    num_particles = n_galaxy + n_rigid;   /* total allocation from here on */

    /* Init particles — only [0, n_galaxy) gets generated and sorted;
     * [n_galaxy, n_galaxy + n_rigid) remains zeroed for init_rigid_body_cube. */
    ParticleState particles;
    init_particles(particles, num_particles, seed, init_sort, n_galaxy);

    /* Fill the trailing lattice slots (if enabled) before the GPU upload. */
    ConstraintLattice lattice = {};
    if (rigid_body_mode == RIGID_BODY_CUBE1000) {
        lattice = init_rigid_body_cube(particles, n_galaxy, rigid_count);
    } else if (rigid_body_mode == RIGID_BODY_CUBE2_BALLSOCKET) {
        lattice = init_rigid_body_cube2_ballsocket(particles, n_galaxy);
    } else if (rigid_body_mode == RIGID_BODY_CUBE2_HINGE) {
        lattice = init_rigid_body_cube2_hinge(particles, n_galaxy, spin_rate);
    } else if (rigid_body_mode == RIGID_BODY_CUBE2_COLLIDE) {
        lattice = init_rigid_body_cube2_collide(particles, n_galaxy);
    }

    /* Init validation oracle */
    v21_oracle_t oracle;
    v21_oracle_init(&oracle);

    /* Allocate vertex pack buffer */
    v21_packed_vertex_t* vertex_data = (v21_packed_vertex_t*)malloc(
        num_particles * sizeof(v21_packed_vertex_t));

    /* Init Vulkan */
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Blackhole V21", nullptr, nullptr);

    VulkanContext vkCtx;
    vkCtx.window = window;
    vkCtx.particleCount = num_particles;

    /* Mouse-based camera: drag to rotate, scroll to zoom */
    glfwSetWindowUserPointer(window, &vkCtx);
    glfwSetMouseButtonCallback(window, mouse_button_cb);
    glfwSetCursorPosCallback(window, cursor_pos_cb);
    glfwSetScrollCallback(window, scroll_cb);

    vk::createInstance(vkCtx);
    vk::setupDebugMessenger(vkCtx);
    glfwCreateWindowSurface(vkCtx.instance, window, nullptr, &vkCtx.surface);
    vk::pickPhysicalDevice(vkCtx);
    vk::createLogicalDevice(vkCtx);
    vk::createSwapchain(vkCtx);
    vk::createImageViews(vkCtx);
    vk::createRenderPass(vkCtx);
    vk::createDescriptorSetLayout(vkCtx);
    vk::createGraphicsPipeline(vkCtx);
    vk::createDepthResources(vkCtx);
    vk::createFramebuffers(vkCtx);
    vk::createCommandPool(vkCtx);
    vk::createUniformBuffers(vkCtx);
    vk::createDescriptorPool(vkCtx);
    vk::createDescriptorSets(vkCtx);
    vk::createCommandBuffers(vkCtx);
    vk::createSyncObjects(vkCtx);

    /* Create host-visible vertex buffer */
    VertexBuffer vertexBuf;
    createVertexBuffer(vkCtx.device, vkCtx.physicalDevice, vertexBuf,
                       num_particles * sizeof(v21_packed_vertex_t));

    /* Store in context so pipeline can use it */
    vkCtx.particleBuffer = vertexBuf.buffer;
    vkCtx.particleBufferMemory = vertexBuf.memory;

    vkCtx.currentFrame = 0;

    /* GPU physics compute (optional — use --gpu-physics flag) */
    PhysicsCompute gpuPhys = {};
    if (use_gpu_physics) {
        initPhysicsCompute(gpuPhys, vkCtx,
            particles.pos_x, particles.pos_y, particles.pos_z,
            particles.vel_x, particles.vel_y, particles.vel_z,
            particles.pump_scale, particles.pump_residual,
            particles.pump_history, particles.pump_state,
            particles.theta, particles.omega_nat,
            particles.flags, particles.topo_state,
            particles.N, scatter_mode);

        /* Rigid-body constraint solver (Pass 4) — only when --rigid-body != off.
         * Uses the lattice built by init_rigid_body_cube* above. Hardcoded to
         * 4 PBD iterations per frame per the constraint_experiment plan. */
        if (rigid_body_mode != RIGID_BODY_OFF) {
            initConstraintCompute(gpuPhys, vkCtx,
                                  lattice.pairs.data(),
                                  lattice.rest.data(),
                                  lattice.inv_m.data(),
                                  lattice.bucket_offsets,
                                  lattice.bucket_counts,
                                  lattice.base_index,
                                  lattice.rigid_count,
                                  /*iterations=*/4);
        }

        /* Phase 2.2 collision pipeline — only when the scenario populated
         * lattice.rigid_body_id (currently only cube2-collide). */
        if (!lattice.rigid_body_id.empty()) {
            initCollisionCompute(gpuPhys, vkCtx,
                                 lattice.rigid_body_id.data(),
                                 lattice.base_index,
                                 lattice.rigid_count);

            /* Mark lattice particles as out-of-active-region so siphon skips
             * them. Their position integration happens in collision_apply.comp
             * instead (inertial, no Viviani field, no damping). This is
             * necessary because the Viviani field's rotational steering +
             * anisotropic radial damping would prevent inward radial motion,
             * making it impossible for the cubes to approach each other. */
            std::vector<uint32_t> zero_active(lattice.rigid_count, 0u);
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            size_t sz = lattice.rigid_count * sizeof(uint32_t);
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = sz;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            vkCreateBuffer(vkCtx.device, &bi, nullptr, &stagingBuf);
            VkMemoryRequirements mr;
            vkGetBufferMemoryRequirements(vkCtx.device, stagingBuf, &mr);
            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = findMemoryType(vkCtx.physicalDevice, mr.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(vkCtx.device, &ai, nullptr, &stagingMem);
            vkBindBufferMemory(vkCtx.device, stagingBuf, stagingMem, 0);
            void* mapped;
            vkMapMemory(vkCtx.device, stagingMem, 0, sz, 0, &mapped);
            memset(mapped, 0, sz);
            vkUnmapMemory(vkCtx.device, stagingMem);

            VkCommandBuffer ucmd;
            VkCommandBufferAllocateInfo cba = {};
            cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cba.commandPool = vkCtx.commandPool;
            cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cba.commandBufferCount = 1;
            vkAllocateCommandBuffers(vkCtx.device, &cba, &ucmd);
            VkCommandBufferBeginInfo cbegin = {};
            cbegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            cbegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(ucmd, &cbegin);
            VkBufferCopy region = {0, lattice.base_index * sizeof(uint32_t), sz};
            vkCmdCopyBuffer(ucmd, stagingBuf, gpuPhys.soa_buffers[15], 1, &region);
            vkEndCommandBuffer(ucmd);
            VkSubmitInfo usi = {};
            usi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            usi.commandBufferCount = 1;
            usi.pCommandBuffers = &ucmd;
            vkQueueSubmit(vkCtx.graphicsQueue, 1, &usi, VK_NULL_HANDLE);
            vkQueueWaitIdle(vkCtx.graphicsQueue);
            vkFreeCommandBuffers(vkCtx.device, vkCtx.commandPool, 1, &ucmd);
            vkDestroyBuffer(vkCtx.device, stagingBuf, nullptr);
            vkFreeMemory(vkCtx.device, stagingMem, nullptr);

            printf("[vk-compute] Lattice particles [%u, %u) marked in_active_region=0 "
                   "(siphon skip; position integration in collision_apply)\n",
                   lattice.base_index, lattice.base_index + lattice.rigid_count);
        }

        initDensityRender(gpuPhys, vkCtx);
    }

    printf("[v21-visual] %d particles, Vulkan ready, physics=%s\n",
           num_particles, use_gpu_physics ? "GPU" : "CPU");

    /* Readback arrays for oracle (when using GPU physics).
     * Capped at ORACLE_SUBSET_SIZE regardless of total particle count. */
    int oracle_count = num_particles < ORACLE_SUBSET_SIZE
                     ? num_particles : ORACLE_SUBSET_SIZE;
    float *rb_px = NULL, *rb_py = NULL, *rb_pz = NULL;
    float *rb_vx = NULL, *rb_vy = NULL, *rb_vz = NULL;
    float *rb_theta = NULL, *rb_scale = NULL;
    uint8_t *rb_flags = NULL;
    int *rb_pump_state = NULL;
    if (use_gpu_physics) {
        size_t bf = oracle_count * sizeof(float);
        rb_px    = (float*)malloc(bf);
        rb_py    = (float*)malloc(bf);
        rb_pz    = (float*)malloc(bf);
        rb_vx    = (float*)malloc(bf);
        rb_vy    = (float*)malloc(bf);
        rb_vz    = (float*)malloc(bf);
        rb_theta = (float*)malloc(bf);
        rb_scale = (float*)malloc(bf);
        rb_flags = (uint8_t*)malloc(oracle_count * sizeof(uint8_t));
        rb_pump_state = (int*)malloc(oracle_count * sizeof(int));
    }

    /* Pack initial frame so there's something to render immediately */
    v21_pack_vertices(vertex_data,
        particles.pos_x, particles.pos_y, particles.pos_z,
        particles.vel_x, particles.vel_y, particles.vel_z,
        particles.pump_scale, particles.pump_residual,
        particles.flags, particles.topo_state,
        particles.N);
    memcpy(vertexBuf.mapped, vertex_data,
           particles.N * sizeof(v21_packed_vertex_t));

    /* Main loop */
    float dt = DEFAULT_DT;
    float sim_time = 0.0f;
    int frame = 0;
    auto t0 = std::chrono::steady_clock::now();

    /* GPU timestamp accumulators for profiling */
    double gpu_scatter_sum = 0.0, gpu_stencil_sum = 0.0, gpu_gather_sum = 0.0;
    double gpu_constraint_sum = 0.0;
    double gpu_collision_sum = 0.0;
    double gpu_siphon_sum = 0.0;
    double gpu_project_sum = 0.0, gpu_tonemap_sum = 0.0;
    int gpu_samples = 0;

    while (!glfwWindowShouldClose(window)) {
        if (max_frames > 0 && frame >= max_frames) break;
        glfwPollEvents();

        if (!use_gpu_physics) {
            /* CPU physics path */
            physics_step(particles, dt * 2.0f);
        }
        /* GPU physics is dispatched inside the command buffer below */
        sim_time += dt;

        /* Oracle validation.
         * CPU physics: every frame on the full particle set.
         * GPU physics: every 500 frames, ~100K-particle subset readback via fence. */
        if (!use_gpu_physics) {
            v21_oracle_check(&oracle,
                particles.pos_x, particles.pos_y, particles.pos_z,
                particles.vel_x, particles.vel_y, particles.vel_z,
                particles.theta, particles.pump_scale,
                particles.flags, particles.topo_state,
                particles.N, frame);
        } else if (frame > 0 && frame % 500 == 0) {
            auto rb_start = std::chrono::steady_clock::now();
            readbackForOracle(gpuPhys, vkCtx,
                rb_px, rb_py, rb_pz,
                rb_vx, rb_vy, rb_vz,
                rb_theta, rb_scale, rb_flags,
                oracle_count);
            auto rb_end = std::chrono::steady_clock::now();
            double rb_ms = std::chrono::duration<double, std::milli>(rb_end - rb_start).count();

            int prev_passes = oracle.total_passes;
            int prev_fails = oracle.total_fails;
            /* topo_state is CPU-side only (GPU siphon doesn't mutate it),
             * so we pass the original array for the same prefix. */
            v21_oracle_check(&oracle,
                rb_px, rb_py, rb_pz,
                rb_vx, rb_vy, rb_vz,
                rb_theta, rb_scale,
                rb_flags, particles.topo_state,
                oracle_count, frame);
            const char* result = (oracle.total_fails > prev_fails) ? "FAIL" :
                                 (oracle.total_passes > prev_passes) ? "pass" : "skip";
            printf("[oracle] frame=%d readback=%.2fms count=%d -> %s  p0=(%.3f,%.3f,%.3f) v0=(%.3f,%.3f,%.3f) flags0=0x%02x\n",
                   frame, rb_ms, oracle_count, result,
                   rb_px[0], rb_py[0], rb_pz[0],
                   rb_vx[0], rb_vy[0], rb_vz[0],
                   (unsigned)rb_flags[0]);

            /* Lattice + joint stability probe — fires only when the full
             * rigid-body particle range [n_galaxy, n_galaxy + n_rigid) fits
             * inside the oracle readback window (ORACLE_SUBSET_SIZE = 100000).
             * For -n 99000 --rigid-body cube1000: covers cube 0 at [99000,
             *   100000). Shows cube-internal distances to neighbors.
             * For -n 98000 --rigid-body cube2-ballsocket: covers both cubes
             *   at [98000, 100000). Shows cube-internal distances AND the
             *   joint anchor distance.
             * At -n 20M the lattice is at [20M, 20M+n_rigid) which is far
             * outside the first 100K oracle window, so the probe silently
             * no-ops — measurement runs are unaffected. */
            if ((rigid_body_mode == RIGID_BODY_CUBE1000 ||
                 rigid_body_mode == RIGID_BODY_CUBE2_BALLSOCKET ||
                 rigid_body_mode == RIGID_BODY_CUBE2_HINGE ||
                 rigid_body_mode == RIGID_BODY_CUBE2_COLLIDE) &&
                (n_galaxy + n_rigid) <= oracle_count) {
                int b = n_galaxy;
                auto dist_idx = [&](int i, int j) -> float {
                    float dx = rb_px[i] - rb_px[j];
                    float dy = rb_py[i] - rb_py[j];
                    float dz = rb_pz[i] - rb_pz[j];
                    return sqrtf(dx*dx + dy*dy + dz*dz);
                };
                /* Cube 0 internal: particle at b+0 is its (ix=0,iy=0,iz=0)
                 * corner. Neighbors are at +1 (X axis), +10 (Y axis), +100
                 * (Z axis) — matches the idx_of formula for LX=LY=LZ=10. */
                float d0_x = dist_idx(b + 0, b + 1);
                float d0_y = dist_idx(b + 0, b + 10);
                float d0_z = dist_idx(b + 0, b + 100);
                printf("[rigid] frame=%d  cube0[0]=(%.3f,%.3f,%.3f)  "
                       "d(0,1)=%.4f d(0,10)=%.4f d(0,100)=%.4f (rest=0.5)\n",
                       frame, rb_px[b], rb_py[b], rb_pz[b], d0_x, d0_y, d0_z);

                if (rigid_body_mode == RIGID_BODY_CUBE2_BALLSOCKET) {
                    /* Joint anchors: cube 0 (ix=5, iy=9, iz=5) → offset
                     * 5 + 9*10 + 5*100 = 595. Cube 1 (ix=5, iy=0, iz=5) →
                     * offset 5 + 0 + 500 = 505. Cube 1's base is b+1000. */
                    int anchor0 = b + 595;
                    int anchor1 = b + 1000 + 505;
                    float d_joint = dist_idx(anchor0, anchor1);
                    /* Cube 1 internal check: particle at b+1000 is its
                     * (0,0,0) corner. */
                    float d1_x = dist_idx(b + 1000, b + 1001);
                    float d1_y = dist_idx(b + 1000, b + 1010);
                    float d1_z = dist_idx(b + 1000, b + 1100);
                    printf("[rigid] frame=%d  cube1[0]=(%.3f,%.3f,%.3f)  "
                           "d(0,1)=%.4f d(0,10)=%.4f d(0,100)=%.4f  "
                           "d_joint=%.4f (rest=1.0)\n",
                           frame, rb_px[b + 1000], rb_py[b + 1000], rb_pz[b + 1000],
                           d1_x, d1_y, d1_z, d_joint);
                } else if (rigid_body_mode == RIGID_BODY_CUBE2_HINGE) {
                    /* Hinge anchor A: cube 0 (ix=0, iy=9, iz=5) → offset
                     *   0 + 9*10 + 5*100 = 590. Cube 1 (ix=0, iy=0, iz=5)
                     *   → offset 0 + 0 + 500 = 500.
                     * Hinge anchor B: cube 0 (ix=9, iy=9, iz=5) → offset
                     *   9 + 90 + 500 = 599. Cube 1 (ix=9, iy=0, iz=5) →
                     *   offset 9 + 0 + 500 = 509. Cube 1 base = b+1000. */
                    int A0 = b + 590;
                    int A1 = b + 1000 + 500;
                    int B0 = b + 599;
                    int B1 = b + 1000 + 509;
                    float d_hinge_A = dist_idx(A0, A1);
                    float d_hinge_B = dist_idx(B0, B1);
                    /* Cube 1 internal check: particle at b+1000 is its
                     * (0,0,0) corner. */
                    float d1_x = dist_idx(b + 1000, b + 1001);
                    float d1_y = dist_idx(b + 1000, b + 1010);
                    float d1_z = dist_idx(b + 1000, b + 1100);
                    printf("[rigid] frame=%d  cube1[0]=(%.3f,%.3f,%.3f)  "
                           "d(0,1)=%.4f d(0,10)=%.4f d(0,100)=%.4f  "
                           "d_hinge_A=%.4f d_hinge_B=%.4f (rest=1.0)\n",
                           frame, rb_px[b + 1000], rb_py[b + 1000], rb_pz[b + 1000],
                           d1_x, d1_y, d1_z, d_hinge_A, d_hinge_B);
                } else if (rigid_body_mode == RIGID_BODY_CUBE2_COLLIDE) {
                    /* Cube 1 internal check + inter-cube center distance.
                     * Cube center particle ≈ (5, 5, 5) in local indexing =
                     * offset 5 + 5*10 + 5*100 = 555. */
                    float d1_x = dist_idx(b + 1000, b + 1001);
                    float d1_y = dist_idx(b + 1000, b + 1010);
                    float d1_z = dist_idx(b + 1000, b + 1100);
                    int center0 = b + 555;
                    int center1 = b + 1000 + 555;
                    float d_centers = dist_idx(center0, center1);
                    printf("[rigid] frame=%d  cube0_ctr=(%.3f,%.3f,%.3f)  "
                           "cube1_ctr=(%.3f,%.3f,%.3f)  d_centers=%.3f  "
                           "d0(0,1)=%.4f d1(0,1)=%.4f (rest=0.5)\n",
                           frame,
                           rb_px[center0], rb_py[center0], rb_pz[center0],
                           rb_px[center1], rb_py[center1], rb_pz[center1],
                           d_centers, d0_x, d1_x);
                }
            }

            /* Diagnostic: pump-state histogram + pos/vel/scale distributions */
            readbackPumpStateSample(gpuPhys, vkCtx, rb_pump_state, oracle_count);
            int state_hist[8] = {0};
            for (int i = 0; i < oracle_count; i++) {
                int s = rb_pump_state[i];
                if (s >= 0 && s < 8) state_hist[s]++;
            }

            double r_min = 1e30, r_max = 0.0, r_sum = 0.0;
            double v_min = 1e30, v_max = 0.0, v_sum = 0.0;
            double scale_sum = 0.0, scale_max = 0.0;
            int ejected = 0;
            for (int i = 0; i < oracle_count; i++) {
                double r = sqrt((double)rb_px[i]*rb_px[i]
                              + (double)rb_py[i]*rb_py[i]
                              + (double)rb_pz[i]*rb_pz[i]);
                double v = sqrt((double)rb_vx[i]*rb_vx[i]
                              + (double)rb_vy[i]*rb_vy[i]
                              + (double)rb_vz[i]*rb_vz[i]);
                if (r < r_min) r_min = r;
                if (r > r_max) r_max = r;
                r_sum += r;
                if (v < v_min) v_min = v;
                if (v > v_max) v_max = v;
                v_sum += v;
                scale_sum += rb_scale[i];
                if (rb_scale[i] > scale_max) scale_max = rb_scale[i];
                if (rb_flags[i] & 0x02) ejected++;  /* PFLAG_EJECTED */
            }
            double inv_n = 1.0 / (double)oracle_count;
            printf("[diag ] frame=%d  states: IDLE=%d PRIMED=%d UC=%d UH=%d EXP=%d DWN=%d VO=%d REC=%d\n",
                   frame,
                   state_hist[0], state_hist[1], state_hist[2], state_hist[3],
                   state_hist[4], state_hist[5], state_hist[6], state_hist[7]);
            printf("[diag ] frame=%d  |p|: min=%.2f mean=%.2f max=%.2f  |v|: min=%.3f mean=%.3f max=%.3f\n",
                   frame,
                   r_min, r_sum*inv_n, r_max,
                   v_min, v_sum*inv_n, v_max);
            printf("[diag ] frame=%d  scale: mean=%.3f max=%.3f  ejected=%d/%d (%.1f%%)\n",
                   frame, scale_sum*inv_n, scale_max,
                   ejected, oracle_count, 100.0 * ejected * inv_n);
        }

        if (!use_gpu_physics) {
            /* CPU path: pack SoA → AoS and upload */
            v21_pack_vertices(
                vertex_data,
                particles.pos_x, particles.pos_y, particles.pos_z,
                particles.vel_x, particles.vel_y, particles.vel_z,
                particles.pump_scale, particles.pump_residual,
                particles.flags, particles.topo_state,
                particles.N);
            memcpy(vertexBuf.mapped, vertex_data,
                   particles.N * sizeof(v21_packed_vertex_t));
        }
        /* GPU physics: readback ONLY for oracle validation (every 100 frames).
         * Rendering uses the initial vertex data — the GPU compute updates
         * the SSBOs but we don't read them back for display every frame.
         * The proper fix is a GPU repack shader; for now, the visual is
         * static but the physics is validated. */

        /* Camera is static — no auto-rotate (use mouse/IDE controls if any) */
        vkCtx.particleCount = particles.N;

        /* Draw: acquire → record → submit → present */
        {
            vkWaitForFences(vkCtx.device, 1, &vkCtx.inFlightFences[vkCtx.currentFrame], VK_TRUE, UINT64_MAX);

            uint32_t imageIndex;
            VkResult res = vkAcquireNextImageKHR(vkCtx.device, vkCtx.swapchain, UINT64_MAX,
                vkCtx.imageAvailableSemaphores[vkCtx.currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (res == VK_ERROR_OUT_OF_DATE_KHR) { vk::recreateSwapchain(vkCtx); continue; }

            vkResetFences(vkCtx.device, 1, &vkCtx.inFlightFences[vkCtx.currentFrame]);

            VkCommandBuffer cmd = vkCtx.commandBuffers[vkCtx.currentFrame];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmd, &beginInfo);

            /* Update camera uniforms (need viewProj before recording) */
            vk::updateUniformBuffer(vkCtx, vkCtx.currentFrame);

            if (use_gpu_physics) {
                /* GPU path: physics dispatch → compute projection → tone-map */
                if (!project_only)
                    dispatchPhysicsCompute(gpuPhys, cmd, frame, sim_time, dt * 2.0f);

                /* Get viewProj from the mapped UBO */
                GlobalUBO* ubo = (GlobalUBO*)vkCtx.uniformBuffersMapped[vkCtx.currentFrame];
                recordDensityRender(gpuPhys, cmd, vkCtx, imageIndex, ubo->viewProj);
            } else {
                /* CPU path: rasterizer */
                VkRenderPassBeginInfo rpInfo = {};
                rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                rpInfo.renderPass = vkCtx.renderPass;
                rpInfo.framebuffer = vkCtx.framebuffers[imageIndex];
                rpInfo.renderArea.offset = {0, 0};
                rpInfo.renderArea.extent = vkCtx.swapchainExtent;
                VkClearValue clearValues[2] = {};
                clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
                clearValues[1].depthStencil = {1.0f, 0};
                rpInfo.clearValueCount = 2;
                rpInfo.pClearValues = clearValues;

                vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

                VkViewport viewport = {};
                viewport.width = (float)vkCtx.swapchainExtent.width;
                viewport.height = (float)vkCtx.swapchainExtent.height;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                VkRect2D scissor = {{0,0}, vkCtx.swapchainExtent};
                vkCmdSetScissor(cmd, 0, 1, &scissor);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkCtx.graphicsPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vkCtx.pipelineLayout, 0, 1, &vkCtx.descriptorSets[vkCtx.currentFrame], 0, nullptr);
                VkBuffer vbufs[] = {vertexBuf.buffer};
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
                vkCmdDraw(cmd, 1, particles.N, 0, 0);

                vkCmdEndRenderPass(cmd);
            }

            vkEndCommandBuffer(cmd);

            /* Submit */
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            VkSemaphore waitSems[] = {vkCtx.imageAvailableSemaphores[vkCtx.currentFrame]};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSems;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VkSemaphore sigSems[] = {vkCtx.renderFinishedSemaphores[vkCtx.currentFrame]};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = sigSems;
            vkQueueSubmit(vkCtx.graphicsQueue, 1, &submitInfo, vkCtx.inFlightFences[vkCtx.currentFrame]);

            /* Present */
            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = sigSems;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &vkCtx.swapchain;
            presentInfo.pImageIndices = &imageIndex;
            vkQueuePresentKHR(vkCtx.presentQueue, &presentInfo);

            vkCtx.currentFrame = (vkCtx.currentFrame + 1) % 2;
        }

        frame++;

        /* Sample GPU timestamps (non-blocking; silently skipped if not ready) */
        if (use_gpu_physics) {
            double sc_ms = 0, st_ms = 0, g_ms = 0, c_ms = 0;
            double col_ms = 0, s_ms = 0, p_ms = 0, t_ms = 0;
            if (readTimestamps(gpuPhys, vkCtx.device,
                               &sc_ms, &st_ms, &g_ms, &c_ms,
                               &col_ms, &s_ms, &p_ms, &t_ms)) {
                gpu_scatter_sum    += sc_ms;
                gpu_stencil_sum    += st_ms;
                gpu_gather_sum     += g_ms;
                gpu_constraint_sum += c_ms;
                gpu_collision_sum  += col_ms;
                gpu_siphon_sum     += s_ms;
                gpu_project_sum    += p_ms;
                gpu_tonemap_sum    += t_ms;
                gpu_samples++;
            }
        }

        /* FPS counter — print every 100 frames */
        if (frame % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            printf("[perf] frame=%d  %.1f fps  (%.1f ms/frame)\n",
                   frame, frame / elapsed, elapsed / frame * 1000.0);
        }

        /* GPU breakdown — print every 500 frames */
        if (use_gpu_physics && frame % 500 == 0 && gpu_samples > 0) {
            double inv = 1.0 / (double)gpu_samples;
            double sc  = gpu_scatter_sum    * inv;
            double st  = gpu_stencil_sum    * inv;
            double g   = gpu_gather_sum     * inv;
            double c   = gpu_constraint_sum * inv;
            double col = gpu_collision_sum  * inv;
            double s   = gpu_siphon_sum     * inv;
            double p   = gpu_project_sum    * inv;
            double t   = gpu_tonemap_sum    * inv;
            double total = sc + st + g + c + col + s + p + t;
            printf("[gpu] scatter=%.3f stencil=%.3f gather=%.3f constraint=%.3f "
                   "collision=%.3f siphon=%.3f project=%.3f tonemap=%.3f  "
                   "total=%.3f ms  (%d samples)\n",
                   sc, st, g, c, col, s, p, t, total, gpu_samples);
            gpu_scatter_sum = gpu_stencil_sum = gpu_gather_sum = 0.0;
            gpu_constraint_sum = gpu_collision_sum = 0.0;
            gpu_siphon_sum = 0.0;
            gpu_project_sum = gpu_tonemap_sum = 0.0;
            gpu_samples = 0;
        }
    }

    /* Cleanup */
    vkDeviceWaitIdle(vkCtx.device);
    if (use_gpu_physics) {
        cleanupPhysicsCompute(gpuPhys, vkCtx.device);
        free(rb_px); free(rb_py); free(rb_pz);
        free(rb_vx); free(rb_vy); free(rb_vz);
        free(rb_theta); free(rb_scale); free(rb_flags); free(rb_pump_state);
    }
    v21_oracle_summary(&oracle);

    /* Destroy our vertex buffer before vk::cleanup touches the context */
    if (vertexBuf.buffer != VK_NULL_HANDLE) {
        vkUnmapMemory(vkCtx.device, vertexBuf.memory);
        vkDestroyBuffer(vkCtx.device, vertexBuf.buffer, nullptr);
        vkFreeMemory(vkCtx.device, vertexBuf.memory, nullptr);
    }
    /* Null out handles so V20's cleanup doesn't double-free */
    vkCtx.particleBuffer = VK_NULL_HANDLE;
    vkCtx.particleBufferMemory = VK_NULL_HANDLE;

    vk::cleanup(vkCtx);
    glfwDestroyWindow(window);
    glfwTerminate();

    free(vertex_data);
    free_particles(particles);
    printf("[shutdown] Complete.\n");
    return 0;
}
