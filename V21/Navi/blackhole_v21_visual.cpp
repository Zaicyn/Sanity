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
#include "core/v21_physics_diag.h"
}
#include "vk_compute.h"

/* ========================================================================
 * SIMULATION PARAMETERS
 * ======================================================================== */

#define DEFAULT_PARTICLES    10000
#define DEFAULT_DT           (1.0f / 60.0f)
#define BH_MASS              100.0f
#define ISCO_R               1.0f    /* soft floor, no GR ISCO — allocator has no such concept */
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
    INIT_SORT_VIVIANI = 0   /* sorted by Viviani curve parameter (atan2 in xz) */
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
    RIGID_BODY_CUBE2_COLLIDE    = 4,  /* 2 antipodal cubes drifting inward, dynamic contact (Phase 2.2) */
    RIGID_BODY_CUBE3_HINGE_COLLIDE = 5  /* 2 hinged cubes + 1 free projectile (Phase 2.2 phase-coherent test) */
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

    /* Per-lattice-particle override for in_active_region. When non-empty
     * (length = rigid_count), uploaded over the lattice range after init.
     * 1 = siphon governs this particle (Viviani field active),
     * 0 = siphon skips it (inertial, governed by collision_apply only). */
    std::vector<uint32_t> active_override;
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
                           InitSortMode sort_mode = INIT_SORT_VIVIANI,
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
        if (r < 1.0f) { float s=1.0f/r; x*=s; y*=s; z*=s; }

        Particle p = {};
        p.px = x; p.py = y; p.pz = z;

        float r_xz = sqrtf(x*x+z*z);
        if (r_xz > 0.1f) {
            float v = sqrtf(BH_MASS / fmaxf(r_xz, 1.0f));
            p.vx = -v*(z/r_xz);
            p.vz =  v*(x/r_xz);
        }
        /* Theta seeded from orbital position + small perturbation */
        p.theta = atan2f(z, x) + ((float)rand()/RAND_MAX) * 0.2f;

        /* Keplerian omega: inner particles faster, outer slower.
         * omega ∝ r^(-3/2) gives real differential rotation. */
        {
            float r_ref = 50.0f;
            float r_xz_init = sqrtf(x*x + z*z);
            float r_clamped = fmaxf(r_xz_init, 1.0f);
            p.omega_nat = 0.1f * powf(r_ref / r_clamped, 1.5f);
        }
        int axis = rand()%4; int sign = (rand()%2)?1:-1;
        p.topo_state = (uint8_t)(((sign>0)?1:2) << (axis*2));
        tmp[i] = p;

        /* Viviani sort key: θ = atan2(pz, px) as primary axis, radius as
         * tiebreak. Particles nearby on the disk are nearby in memory →
         * coalesced GPU reads. */
        float theta_v = atan2f(z, x) + 3.14159265f;   /* [0, 2π) */
        float r2d = sqrtf(x*x + z*z);
        uint32_t theta_bucket = (uint32_t)(theta_v * (1024.0f / 6.28318f));
        uint32_t r_bucket     = (uint32_t)fminf(16383.0f, r2d * (16384.0f / DISK_OUTER_R));
        sort_key[i] = ((uint64_t)theta_bucket << 32) | r_bucket;
    }

    /* Build permutation indices and sort by key. */
    std::vector<uint32_t> perm(n_generate);
    for (int i = 0; i < n_generate; i++) perm[i] = i;
    std::stable_sort(perm.begin(), perm.end(),
        [&sort_key](uint32_t a, uint32_t b) { return sort_key[a] < sort_key[b]; });

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

    const char* sort_name = "viviani";
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

    /* Both cubes are inertial (siphon-skipped). */
    L.active_override.assign((size_t)N_LATTICE_TOT, 0u);

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

/* Phase 2.2 phase-coherent test: two hinged cubes (orbiting, spinning) + one
 * free cube (inertial projectile) that collides with the spinning cube.
 *
 * Tests whether the hinge beat pattern (ω^1.16 response, orbital-phase-locked
 * period from Phase 2.3.1 Stage 6b) survives an external collision event.
 *
 * Geometry:
 *   Cubes 0,1: y-stacked at (50, ±2.75, 0), hinged along x-axis (same as
 *     cube2-hinge). Cube 1 has spin_rate around x-axis. Both orbit in the
 *     Viviani field (in_active_region=1).
 *   Cube 2: free projectile at (50, +2.75, Z_START) with zero tangent and
 *     inward z-drift toward cube 1's orbital path. Siphon-skipped
 *     (in_active_region=0), coasts inertially via collision_apply.
 *
 * The hinged pair orbits in xz, cube 1 spinning around the hinge. Cube 2
 * is stationary in x/y but placed at +z of the pair's starting location,
 * so as the pair orbits (moving +z initially), cube 1 collides with cube 2
 * after ~2000 frames.
 *
 * 3 rigid bodies → rigid_body_id: 0=field, 1=cube0, 2=cube1, 3=cube2.
 * active_override: cubes 0,1 = 1 (Viviani), cube 2 = 0 (inertial). */
static ConstraintLattice init_rigid_body_cube3_hinge_collide(ParticleState& ps,
                                                              int n_galaxy,
                                                              float spin_rate) {
    const int   LX = 10, LY = 10, LZ = 10;
    const int   N_LATTICE     = LX * LY * LZ;          /* 1000 per cube */
    const int   N_LATTICE_TOT = 3 * N_LATTICE;         /* 3000 */
    const float SPACING       = 0.5f;
    const float R_ORBIT       = 50.0f;
    const float Y_OFFSET      = 2.75f;
    const float Z_PROJECTILE  = 7.0f;  /* cube 2 placed at z=+7, hit by pair moving +z */

    if (n_galaxy + N_LATTICE_TOT > ps.N) {
        fprintf(stderr, "[rigid] ERROR: ps.N=%d insufficient for n_galaxy=%d + "
                        "3 cubes (cube3-hinge-collide)\n", ps.N, n_galaxy);
        return ConstraintLattice{};
    }

    const float v_orbit = sqrtf(BH_MASS / fmaxf(R_ORBIT, ISCO_R));
    const uint32_t base = (uint32_t)n_galaxy;

    /* 7 super-buckets: 0..5 lattice (all 3 cubes), 6 hinge (cubes 0,1 only). */
    std::vector<std::pair<uint32_t, uint32_t>> super[7];
    for (int k = 0; k < 6; k++) super[k].reserve(1350);
    super[6].reserve(2);

    /* Cubes 0 and 1: y-stacked at (R_ORBIT, ±Y_OFFSET, 0), same as cube2-hinge. */
    const float cube01_centers_y[2] = { -Y_OFFSET, +Y_OFFSET };
    uint32_t anchor_A[2] = { 0, 0 };
    uint32_t anchor_B[2] = { 0, 0 };

    for (int c = 0; c < 2; c++) {
        const float cx = R_ORBIT;
        const float cy = cube01_centers_y[c];
        const float cz = 0.0f;
        const uint32_t cube_base = base + (uint32_t)(c * N_LATTICE);
        auto idx_of = [&](int ix, int iy, int iz) -> uint32_t {
            return cube_base + (uint32_t)(ix + iy * LX + iz * LX * LY);
        };

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

            ps.vel_x[p] = 0.0f;
            ps.vel_y[p] = 0.0f + vy_extra;
            ps.vel_z[p] = v_orbit + vz_extra;
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;
        }

        /* Build lattice buckets for this cube. */
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

        /* Hinge anchors (same as cube2-hinge). */
        int anchor_iy = (c == 0) ? (LY - 1) : 0;
        anchor_A[c] = idx_of(0,      anchor_iy, LZ / 2);
        anchor_B[c] = idx_of(LX - 1, anchor_iy, LZ / 2);
    }

    /* Cube 2: free projectile at (R_ORBIT, +Y_OFFSET, Z_PROJECTILE).
     * Same y-level as cube 1 so they collide face-to-face in z.
     * Zero velocity — it just waits for the orbiting pair to reach it. */
    {
        const float cx = R_ORBIT;
        const float cy = +Y_OFFSET;
        const float cz = Z_PROJECTILE;
        const uint32_t cube_base = base + 2u * N_LATTICE;
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
            ps.vel_x[p] = 0.0f;
            ps.vel_y[p] = 0.0f;
            ps.vel_z[p] = 0.0f;
            ps.theta[p] = 0.0f;
            ps.omega_nat[p] = 0.1f;
            ps.flags[p] = 0x01;
            ps.pump_scale[p] = 1.0f;
            ps.topo_state[p] = 0x01;
        }

        /* Build lattice buckets for cube 2. */
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

    /* Hinge edges in bucket 6 (cubes 0,1 only). */
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
        L.rest.push_back(1.0f);  /* y-gap between cube faces */
    }

    /* rigid_body_id: 0=field, 1=cube0, 2=cube1, 3=cube2. */
    L.rigid_body_id.assign((size_t)ps.N, 0u);
    for (int k = 0; k < N_LATTICE; k++) {
        L.rigid_body_id[(size_t)base + (size_t)k] = 1u;
        L.rigid_body_id[(size_t)base + N_LATTICE + (size_t)k] = 2u;
        L.rigid_body_id[(size_t)base + 2 * N_LATTICE + (size_t)k] = 3u;
    }

    /* active_override: cubes 0,1 = 1 (Viviani field), cube 2 = 0 (inertial). */
    L.active_override.resize(N_LATTICE_TOT);
    for (int k = 0; k < 2 * N_LATTICE; k++) L.active_override[k] = 1u;
    for (int k = 2 * N_LATTICE; k < N_LATTICE_TOT; k++) L.active_override[k] = 0u;

    printf("[rigid] cube3-hinge-collide: %d lattice particles (2 hinged + 1 free), "
           "%u constraints (8100 lattice + %u hinge), "
           "base=%u, spin=%.4f, z_projectile=%.1f, v_orbit=%.4f\n",
           N_LATTICE_TOT, running, L.bucket_counts[6],
           base, spin_rate, Z_PROJECTILE, v_orbit);
    return L;
}

/* ========================================================================
 * VIVIANI FIELD (same as blackhole_v21.c)
 * ======================================================================== */

/* CPU physics — Squaragon ground truth + topological force channels.
 *
 * Layer 1: Viviani tangent + centripetal (the base field)
 * Layer 2: Three-channel topological forces:
 *   Strong (3θ): local bunching, within-branch
 *   EM (1θ):     long-range carrier, 1/r² along topology
 *   Weak (9θ):   branch-crossing, flow-gated
 * Layer 3: Gravity = residual from harmonic imbalance
 *
 * Topological distance is measured along the torus (ring + bin + branch),
 * not through Euclidean space. Forces flow along the fabric. */

/* Flow mode LUT from Squaragon V2 (inlined to avoid header dependency) */
static const uint8_t SQ2_FLOW_MODE_LUT[32] = {
    0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 0, 0, 2, 1, 1, 1,
    0, 1, 1, 1, 2, 0, 0, 2, 1, 2, 0, 0, 2, 1, 1, 1
};
#define SQ2_FLOW_MODE_ACTIVE 1
#define SQ2_FLOW_MODE_FLOW   2

/* Accumulator grid for topological force: bin particles by torus position,
 * accumulate phase and count per cell. Then each particle reads its
 * neighborhood from the grid instead of doing N-body. */
#define TOPO_GRID_BINS 8
#define TOPO_GRID_GENS 32
#define TOPO_GRID_SIZE (TOPO_GRID_BINS * TOPO_GRID_GENS)

static void physics_step(ParticleState& ps, float dt) {
    const float TWO_PI_F = 6.28318530f;

    /* === PASS 0: Build topo grid — accumulate phase per (bin, gen) cell ===
     * Each particle maps to a torus cell based on its orbital angle (bin)
     * and fiber phase (gen). We accumulate sin/cos of theta for
     * mean-field coupling, plus count for density. */
    float grid_sin[TOPO_GRID_SIZE] = {0};
    float grid_cos[TOPO_GRID_SIZE] = {0};
    int   grid_cnt[TOPO_GRID_SIZE] = {0};

    for (int i = 0; i < ps.N; i++) {
        if (!(ps.flags[i] & 0x01)) continue;

        /* Derive topo coordinates from particle state */
        float phi = atan2f(ps.pos_z[i], ps.pos_x[i]);
        if (phi < 0.0f) phi += TWO_PI_F;
        int bin = (int)(phi * (float)TOPO_GRID_BINS / TWO_PI_F) & 7;

        float th = ps.theta[i];
        if (th < 0.0f) th += TWO_PI_F;
        int gen = (int)(th * 32.0f / TWO_PI_F) & 31;

        int cell = bin * TOPO_GRID_GENS + gen;
        grid_sin[cell] += sinf(th);
        grid_cos[cell] += cosf(th);
        grid_cnt[cell]++;
    }

    /* === PASS 1: Per-particle physics === */
    for (int i = 0; i < ps.N; i++) {
        if (!(ps.flags[i] & 0x01)) continue;
        if (ps.flags[i] & 0x02) continue;  /* ejected */

        float px = ps.pos_x[i], py = ps.pos_y[i], pz = ps.pos_z[i];
        float vx = ps.vel_x[i], vy = ps.vel_y[i], vz = ps.vel_z[i];

        float r3d_sq = px*px + py*py + pz*pz;
        float r3d = sqrtf(r3d_sq);
        float inv_r3d = 1.0f / (r3d + 1e-8f);

        /* === VIVIANI FIELD FORCE (Squaragon Layer 1) ===
         * Per-particle fiber phase → tangent + gravity */
        float theta_fiber = ps.theta[i];

        float s1 = sinf(theta_fiber), c1 = cosf(theta_fiber);
        float s3 = sinf(3.0f * theta_fiber), c3 = cosf(3.0f * theta_fiber);

        /* Viviani tangent vector */
        float fx = c1 - 1.5f * c3;
        float fy = s1 - 1.5f * s3;
        float fz = -s1 * c3 - 3.0f * c1 * s3;

        float inv_f = 1.0f / sqrtf(fx*fx + fy*fy + fz*fz + 1e-8f);
        fx *= inv_f; fy *= inv_f; fz *= inv_f;

        float r_safe = fmaxf(r3d, 1.0f);
        float inv_r = (r3d >= 1.0f) ? inv_r3d : (1.0f / r_safe);
        float weight = BH_MASS / (r_safe * r_safe);

        /* Radial unit vector */
        float rx_u = px * inv_r, ry_u = py * inv_r, rz_u = pz * inv_r;

        /* Tangential: field direction projected perpendicular to radial */
        float f_dot_r = fx*rx_u + fy*ry_u + fz*rz_u;
        float tx = fx - f_dot_r * rx_u;
        float ty = fy - f_dot_r * ry_u;
        float tz = fz - f_dot_r * rz_u;

        float ax = 0, ay = 0, az = 0;

        /* Centripetal (gravity) */
        ax += -rx_u * weight;
        ay += -ry_u * weight;
        az += -rz_u * weight;

        /* Tangential (rotation from fiber phase) */
        ax += tx * weight * 0.5f;
        ay += ty * weight * 0.5f;
        az += tz * weight * 0.5f;

        /* === TOPOLOGICAL FORCES (Squaragon Layer 2) ===
         * Derive topo coordinates, read neighbor cells, apply three channels */
        float phi_i = atan2f(pz, px);
        if (phi_i < 0.0f) phi_i += TWO_PI_F;
        int my_bin = (int)(phi_i * (float)TOPO_GRID_BINS / TWO_PI_F) & 7;
        int my_gen = (int)(theta_fiber * 32.0f / TWO_PI_F) & 31;
        int my_mode = SQ2_FLOW_MODE_LUT[my_gen];

        /* Scan local topo neighborhood */
        float topo_ax = 0, topo_ay = 0, topo_az = 0;

        for (int db = -1; db <= 1; db++) {
            int nb = (my_bin + db + 8) & 7;
            for (int dg = -2; dg <= 2; dg++) {
                if (db == 0 && dg == 0) continue;
                int ng = (my_gen + dg + 32) & 31;
                int cell = nb * TOPO_GRID_GENS + ng;

                if (grid_cnt[cell] == 0) continue;

                /* Mean phase of this neighbor cell */
                float mean_sin = grid_sin[cell] / (float)grid_cnt[cell];
                float mean_cos = grid_cos[cell] / (float)grid_cnt[cell];
                float mean_theta = atan2f(mean_sin, mean_cos);

                /* Phase difference drives harmonic interaction */
                float dtheta = mean_theta - theta_fiber;

                /* Topological distance */
                int rdist = abs(dg);
                if (rdist > 16) rdist = 32 - rdist;
                int bdist = abs(db);
                if (bdist > 4) bdist = 8 - bdist;
                float r_topo = (float)rdist + 0.5f * (float)bdist;
                if (r_topo < 0.01f) continue;

                int neighbor_mode = SQ2_FLOW_MODE_LUT[ng];
                int active_mode = (my_mode < neighbor_mode) ? my_mode : neighbor_mode;

                /* --- EM (always active): 1θ carrier, 1/r² along topology --- */
                float k_em = 1.0f / (r_topo * r_topo + 1.0f);
                float f_em = (1.0f / 3.0f) * k_em * sinf(dtheta);

                /* --- Strong (ACTIVE+): 3θ bunching, local only --- */
                float f_strong = 0.0f;
                if (active_mode >= SQ2_FLOW_MODE_ACTIVE && rdist <= 1 && bdist <= 1) {
                    float k_strong = 1.0f / (1.0f + (float)bdist);
                    f_strong = (1.0f / 3.0f) * k_strong * sinf(3.0f * dtheta) / 3.0f;
                }

                /* --- Weak (FLOW only): 9θ, cross-bin trigger --- */
                float f_weak = 0.0f;
                if (active_mode >= SQ2_FLOW_MODE_FLOW && bdist >= 1 && rdist <= 2) {
                    f_weak = (1.0f / 9.0f) * 0.2f * sinf(9.0f * dtheta) / 9.0f;
                }

                float f_total = f_em + f_strong + f_weak;

                /* Density of neighbor cell */
                float density_scale = (float)grid_cnt[cell] / fmaxf((float)ps.N / (float)TOPO_GRID_SIZE, 1.0f);

                /* Project along tangential direction (ring flow) and radial */
                float r_xz = sqrtf(px*px + pz*pz);
                if (r_xz > 0.1f) {
                    float tang_x = -pz / r_xz;
                    float tang_z =  px / r_xz;

                    /* Ring force → tangential */
                    float ring_sign = (dg > 0) ? 1.0f : -1.0f;
                    topo_ax += f_total * tang_x * ring_sign * density_scale;
                    topo_az += f_total * tang_z * ring_sign * density_scale;

                    /* Bin force → radial modulation */
                    if (db != 0) {
                        float bin_sign = (db > 0) ? 1.0f : -1.0f;
                        topo_ax += f_total * rx_u * bin_sign * density_scale * 0.3f;
                        topo_az += f_total * rz_u * bin_sign * density_scale * 0.3f;
                    }
                }

                /* Vertical component from cross-bin interaction */
                topo_ay += f_total * density_scale * 0.1f * (float)db;
            }
        }

        /* Scale topological forces relative to the base Viviani field.
         * These are perturbations that create structure, not the primary driver. */
        float topo_scale = weight * 0.1f;
        ax += topo_ax * topo_scale;
        ay += topo_ay * topo_scale;
        az += topo_az * topo_scale;

        /* === ORBITAL-PLANE DAMPING === */
        float Lx = py*vz - pz*vy, Ly = pz*vx - px*vz, Lz = px*vy - py*vx;
        float inv_L = 1.0f / sqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
        float lx = Lx*inv_L, ly = Ly*inv_L, lz = Lz*inv_L;

        float v_sq = vx*vx + vy*vy + vz*vz;
        if (v_sq > 0.001f) {
            float v_normal = vx*lx + vy*ly + vz*lz;
            float r3 = r3d * r3d * r3d;
            float damping = 0.02f;
            vx -= damping * v_normal * lx;
            vy -= damping * v_normal * ly;
            vz -= damping * v_normal * lz;
        }

        /* === INTEGRATE === */
        vx += ax * dt;
        vy += ay * dt;
        vz += az * dt;

        if (isfinite(vx) && isfinite(vy) && isfinite(vz)) {
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;
        }

        ps.pos_x[i] = px; ps.pos_y[i] = py; ps.pos_z[i] = pz;
        ps.vel_x[i] = vx; ps.vel_y[i] = vy; ps.vel_z[i] = vz;

        /* Pump history update */
        float history = ps.pump_history[i];
        ps.pump_history[i] = history * 0.98f + ps.pump_scale[i] * 0.02f;

        /* Kuramoto phase advance */
        float th = ps.theta[i] + ps.omega_nat[i] * dt;
        if (th >= TWO_PI_F) th -= TWO_PI_F;
        if (th < 0.0f) th += TWO_PI_F;
        ps.theta[i] = th;
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

/* ========================================================================
 * HEADLESS VULKAN — compute-only, no window/surface/swapchain
 * ======================================================================== */

static bool initHeadlessVulkan(VulkanContext& ctx) {
    /* Instance — no surface extensions needed */
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Blackhole V21 Headless";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&createInfo, nullptr, &ctx.instance) != VK_SUCCESS) {
        fprintf(stderr, "[headless] Failed to create Vulkan instance\n");
        return false;
    }

    /* Pick physical device — first one with a compute queue */
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    if (devCount == 0) {
        fprintf(stderr, "[headless] No Vulkan devices found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devices.data());

    uint32_t computeFamily = UINT32_MAX;
    for (auto& dev : devices) {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfProps.data());
        for (uint32_t i = 0; i < qfCount; i++) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx.physicalDevice = dev;
                computeFamily = i;
                break;
            }
        }
        if (computeFamily != UINT32_MAX) break;
    }
    if (computeFamily == UINT32_MAX) {
        fprintf(stderr, "[headless] No compute-capable queue found\n");
        return false;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
    printf("[headless] GPU: %s\n", props.deviceName);

    /* Logical device with one compute queue */
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    if (vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device) != VK_SUCCESS) {
        fprintf(stderr, "[headless] Failed to create logical device\n");
        return false;
    }
    vkGetDeviceQueue(ctx.device, computeFamily, 0, &ctx.graphicsQueue);

    /* Command pool */
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool) != VK_SUCCESS) {
        fprintf(stderr, "[headless] Failed to create command pool\n");
        return false;
    }

    ctx.queueFamilies.graphicsFamily = computeFamily;
    return true;
}

static void cleanupHeadlessVulkan(VulkanContext& ctx) {
    vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
}

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
    RigidBodyMode rigid_body_mode = RIGID_BODY_OFF;
    int   rigid_count = 1;      /* number of cubes; only meaningful when --rigid-body cube1000 */
    float spin_rate   = 0.0f;   /* cube 1 angular velocity around hinge axis (rad/frame); cube2-hinge only */
    int   max_frames  = 0;      /* exit after this many frames; 0 = run until window closed */
    bool  headless    = false;  /* CPU physics only, no window, no Vulkan */
    bool  benchmark   = false;  /* headless + timing breakdown */
    bool  use_forward = false;  /* use forward-pass siphon (V8-style double buffer) */
    bool  fourier_render = false; /* sparse Fourier renderer instead of per-particle projection */

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
        else if (strcmp(argv[i], "--sort") == 0 && i+1 < argc) {
            /* Legacy flag — viviani is now the only sort. Accept and ignore. */
            ++i;
        }
        else if (strcmp(argv[i], "--init-sort") == 0 && i+1 < argc) {
            /* Legacy flag — viviani is now the only sort. Accept and ignore. */
            ++i;
        }
        else if (strcmp(argv[i], "--rigid-body") == 0 && i+1 < argc) {
            const char* m = argv[++i];
            if      (strcmp(m, "off")              == 0) rigid_body_mode = RIGID_BODY_OFF;
            else if (strcmp(m, "cube1000")         == 0) rigid_body_mode = RIGID_BODY_CUBE1000;
            else if (strcmp(m, "cube2-ballsocket") == 0) rigid_body_mode = RIGID_BODY_CUBE2_BALLSOCKET;
            else if (strcmp(m, "cube2-hinge")      == 0) rigid_body_mode = RIGID_BODY_CUBE2_HINGE;
            else if (strcmp(m, "cube2-collide")    == 0) rigid_body_mode = RIGID_BODY_CUBE2_COLLIDE;
            else if (strcmp(m, "cube3-hinge-collide") == 0) rigid_body_mode = RIGID_BODY_CUBE3_HINGE_COLLIDE;
            else {
                fprintf(stderr, "unknown --rigid-body '%s'\n", m);
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
        else if (strcmp(argv[i], "--headless") == 0) {
            headless = true;
            if (max_frames == 0) max_frames = 5000;
        }
        else if (strcmp(argv[i], "--forward") == 0) {
            use_forward = true;
        }
        else if (strcmp(argv[i], "--fourier-render") == 0) {
            fourier_render = true;
        }
        else if (strcmp(argv[i], "--benchmark") == 0) {
            headless = true;
            benchmark = true;
            use_gpu_physics = true;
            if (max_frames == 0) max_frames = 1000;
            if (num_particles == DEFAULT_PARTICLES) num_particles = 20000000;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole_v21_visual [-n particles] [--rng-seed S] "
                   "[--gpu-physics] [--project-only] [--headless] "
                   "[--scatter-mode baseline|uniform|squaragon] "
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
    else if (rigid_body_mode == RIGID_BODY_CUBE3_HINGE_COLLIDE) n_rigid = 3000;
    num_particles = n_galaxy + n_rigid;   /* total allocation from here on */

    /* Init particles — only [0, n_galaxy) gets generated and sorted;
     * [n_galaxy, n_galaxy + n_rigid) remains zeroed for init_rigid_body_cube. */
    ParticleState particles;
    init_particles(particles, num_particles, seed, INIT_SORT_VIVIANI, n_galaxy);

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
    } else if (rigid_body_mode == RIGID_BODY_CUBE3_HINGE_COLLIDE) {
        lattice = init_rigid_body_cube3_hinge_collide(particles, n_galaxy, spin_rate);
    }

    /* Init validation oracle */
    v21_oracle_t oracle;
    v21_oracle_init(&oracle);

    /* Init physics diagnostics */
    v21_physics_diag_t phys_diag;
    v21_physics_diag_init(&phys_diag);

    /* ---- Headless mode: GPU or CPU physics + oracle, no window ---- */
    if (headless) {
        float dt = DEFAULT_DT;
        float sim_time = 0.0f;

        /* Try GPU headless; fall back to CPU if no Vulkan */
        VulkanContext headlessCtx = {};
        PhysicsCompute gpuPhys = {};
        bool gpu_ok = false;

        if (use_gpu_physics) {
            gpu_ok = initHeadlessVulkan(headlessCtx);
            if (gpu_ok) {
                headlessCtx.particleCount = num_particles;
                gpuPhys.headlessMode = false;  /* TODO: headless siphon variant segfaults — investigate */
                try {
                initPhysicsCompute(gpuPhys, headlessCtx,
                    particles.pos_x, particles.pos_y, particles.pos_z,
                    particles.vel_x, particles.vel_y, particles.vel_z,
                    particles.pump_scale, particles.pump_residual,
                    particles.pump_history, particles.pump_state,
                    particles.theta, particles.omega_nat,
                    particles.flags, particles.topo_state,
                    num_particles, scatter_mode);
                initGradedCompute(gpuPhys, headlessCtx);
                /* Counting sort skipped in headless — saves ~240 MB VRAM
                 * (sorted pos buffers) that push 20M over 6 GB on RTX 2060.
                 * countingSortEnabled remains false so dispatch skips it. */
                } catch (const std::exception& e) {
                    fprintf(stderr, "[headless] GPU init exception: %s\n", e.what());
                    gpu_ok = false;
                }

                /* One-shot Cartesian → graded conversion */
                VkCommandBufferAllocateInfo cba = {};
                cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                cba.commandPool = headlessCtx.commandPool;
                cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cba.commandBufferCount = 1;
                VkCommandBuffer initCmd;
                vkAllocateCommandBuffers(headlessCtx.device, &cba, &initCmd);
                VkCommandBufferBeginInfo bi = {};
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(initCmd, &bi);
                dispatchCartesianToGraded(gpuPhys, initCmd);
                vkEndCommandBuffer(initCmd);
                VkSubmitInfo si = {};
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &initCmd;
                vkQueueSubmit(headlessCtx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(headlessCtx.graphicsQueue);
                vkFreeCommandBuffers(headlessCtx.device, headlessCtx.commandPool, 1, &initCmd);

                /* Forward siphon init: after Cartesian→graded so buffers have data */
                if (use_forward)
                    initForwardSiphonCompute(gpuPhys, headlessCtx);

                printf("[headless] GPU physics initialized (%d particles)\n", num_particles);
            } else {
                printf("[headless] GPU init failed, falling back to CPU\n");
            }
        }

        printf("[headless] Running %d frames, %d particles, %s\n",
               max_frames, num_particles, gpu_ok ? "GPU" : "CPU");
        fflush(stdout);

        /* Oracle readback buffers (GPU path) */
        int oracle_count = (num_particles < ORACLE_SUBSET_SIZE) ? num_particles : ORACLE_SUBSET_SIZE;
        float *rb_px=NULL, *rb_py=NULL, *rb_pz=NULL;
        float *rb_vx=NULL, *rb_vy=NULL, *rb_vz=NULL;
        float *rb_theta_rb=NULL, *rb_scale=NULL;
        uint8_t *rb_flags=NULL;
        float *rb_r=NULL, *rb_vel_r=NULL, *rb_phi=NULL, *rb_omega_orb=NULL;
        if (gpu_ok) {
            size_t bf = oracle_count * sizeof(float);
            rb_px = (float*)malloc(bf); rb_py = (float*)malloc(bf); rb_pz = (float*)malloc(bf);
            rb_vx = (float*)malloc(bf); rb_vy = (float*)malloc(bf); rb_vz = (float*)malloc(bf);
            rb_theta_rb = (float*)malloc(bf); rb_scale = (float*)malloc(bf);
            rb_flags = (uint8_t*)malloc(oracle_count);
            rb_r = (float*)malloc(bf); rb_vel_r = (float*)malloc(bf);
            rb_phi = (float*)malloc(bf); rb_omega_orb = (float*)malloc(bf);
        }

        /* Benchmark accumulators */
        double bm_scatter = 0, bm_stencil = 0, bm_gather = 0;
        double bm_constraint = 0, bm_collision = 0, bm_siphon = 0;
        double bm_project = 0, bm_tonemap = 0;
        int bm_samples = 0;

        auto t0 = std::chrono::steady_clock::now();

        for (int frame = 0; frame < max_frames; frame++) {
            if (gpu_ok) {
                /* GPU dispatch */
                VkCommandBufferAllocateInfo cba = {};
                cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                cba.commandPool = headlessCtx.commandPool;
                cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cba.commandBufferCount = 1;
                VkCommandBuffer cmd;
                if (vkAllocateCommandBuffers(headlessCtx.device, &cba, &cmd) != VK_SUCCESS) {
                    fprintf(stderr, "[headless] Failed to allocate command buffer frame %d\n", frame);
                    break;
                }
                VkCommandBufferBeginInfo bi = {};
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &bi);
                if (use_forward && gpuPhys.forwardSiphonEnabled) {
                    dispatchCylDensity(gpuPhys, cmd);
                    dispatchForwardSiphon(gpuPhys, cmd, sim_time, dt * 2.0f);
                    /* Projection: update Cartesian set 0 for oracle readback + next frame's scatter */
                    VkMemoryBarrier fb = {};
                    fb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    fb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &fb, 0, nullptr, 0, nullptr);
                    dispatchForwardProject(gpuPhys, cmd);
                } else {
                    dispatchPhysicsCompute(gpuPhys, cmd, frame, sim_time, dt * 2.0f);
                }
                vkEndCommandBuffer(cmd);
                VkSubmitInfo si = {};
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                vkQueueSubmit(headlessCtx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(headlessCtx.graphicsQueue);
                vkFreeCommandBuffers(headlessCtx.device, headlessCtx.commandPool, 1, &cmd);

                /* Read timestamps directly — we just waited for idle,
                 * so current frame's queries are guaranteed available. */
                if (benchmark) {
                    /* Read timestamps 0-6 directly (headless: no project/tonemap).
                     * dispatchPhysicsCompute wrote to slot gpuPhys.queryFrame. */
                    uint32_t slot = gpuPhys.queryFrame;
                    uint32_t base = slot * 9;
                    uint64_t ts[7];
                    VkResult r = vkGetQueryPoolResults(headlessCtx.device,
                        gpuPhys.queryPool, base, 7,
                        sizeof(ts), ts, sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                    if (r == VK_SUCCESS) {
                        double ns = (double)gpuPhys.timestampPeriodNs;
                        bm_scatter    += (double)(ts[1] - ts[0]) * ns / 1e6;
                        bm_stencil    += (double)(ts[2] - ts[1]) * ns / 1e6;
                        bm_gather     += (double)(ts[3] - ts[2]) * ns / 1e6;
                        bm_constraint += (double)(ts[4] - ts[3]) * ns / 1e6;
                        bm_collision  += (double)(ts[5] - ts[4]) * ns / 1e6;
                        bm_siphon     += (double)(ts[6] - ts[5]) * ns / 1e6;
                        bm_samples++;
                    }
                }
                gpuPhys.queryValid[gpuPhys.queryFrame] = true;
                gpuPhys.queryFrame ^= 1;
            } else {
                physics_step(particles, dt * 2.0f);
            }
            sim_time += dt;

            /* Oracle — CPU direct or GPU readback every 500 frames */
            if (!gpu_ok) {
                v21_oracle_check(&oracle,
                    particles.pos_x, particles.pos_y, particles.pos_z,
                    particles.vel_x, particles.vel_y, particles.vel_z,
                    particles.theta, particles.pump_scale,
                    particles.flags, particles.topo_state,
                    particles.N, frame);
            } else if (!benchmark && frame > 0 && frame % 500 == 0) {
                /* Skip oracle in benchmark mode — headless siphon doesn't write
                 * Cartesian projection, so readback would return stale data. */
                readbackForOracle(gpuPhys, headlessCtx,
                    rb_px, rb_py, rb_pz, rb_vx, rb_vy, rb_vz,
                    rb_theta_rb, rb_scale, rb_flags, oracle_count,
                    rb_r, rb_vel_r, rb_phi, rb_omega_orb);
                v21_oracle_check(&oracle,
                    rb_px, rb_py, rb_pz, rb_vx, rb_vy, rb_vz,
                    rb_theta_rb, rb_scale, rb_flags,
                    particles.topo_state,
                    oracle_count, frame);

                /* Inline diagnostics: velocity dispersion + radial stats */
                {
                    double sum_vr2 = 0, sum_vmag = 0, max_vr = 0;
                    double sum_r = 0;
                    int n_active = 0;
                    for (int k = 0; k < oracle_count; k++) {
                        if (!(rb_flags[k] & 0x01)) continue;
                        float px = rb_px[k], pz = rb_pz[k];
                        float r = sqrtf(px*px + pz*pz);
                        float vx = rb_vx[k], vy = rb_vy[k], vz = rb_vz[k];
                        float vmag = sqrtf(vx*vx + vy*vy + vz*vz);
                        // Radial velocity = v · r_hat
                        float vr = (r > 0.01f) ? (vx*px + vz*pz) / r : 0.0f;
                        sum_vr2 += (double)vr * vr;
                        sum_vmag += vmag;
                        if (fabsf(vr) > max_vr) max_vr = fabsf(vr);
                        sum_r += r;
                        n_active++;
                    }
                    if (n_active > 0) {
                        double sigma_r = sqrt(sum_vr2 / n_active);
                        printf("[fwd-diag] frame=%d  N=%d  sigma_r=%.6f  max_vr=%.6f  "
                               "mean_v=%.4f  mean_r=%.1f\n",
                               frame, n_active, sigma_r, max_vr,
                               sum_vmag / n_active, sum_r / n_active);
                    }
                }
            }

            if (frame > 0 && frame % 1000 == 0) {
                auto t1 = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();
                printf("[headless] frame=%d  %.1f fps  (%.1f ms/frame)\n",
                       frame, frame / elapsed, elapsed / frame * 1000.0);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("[headless] Done: %d frames in %.1f s (%.1f fps)\n",
               max_frames, elapsed, max_frames / elapsed);

        if (benchmark && bm_samples > 0) {
            double inv = 1.0 / bm_samples;
            double total = (bm_scatter + bm_stencil + bm_gather + bm_constraint +
                           bm_collision + bm_siphon) * inv;
            size_t graded_bytes = (size_t)num_particles * 10 * sizeof(float);  /* set 2: 10 arrays */
            size_t set0_rw_bytes = (size_t)num_particles * (5 * 4 + 6 * 4);   /* 5 pump read/write + 6 Cartesian write */
            size_t density_bytes = (size_t)(V21_CYL_CELLS) * 4 * sizeof(float); /* cyl density + 3 pressure */
            double siphon_bw = (double)(graded_bytes + set0_rw_bytes + density_bytes) / (bm_siphon * inv * 1e-3) / 1e9;

            printf("\n");
            printf("╔══════════════════════════════════════════════════════╗\n");
            printf("║  BENCHMARK: %d particles, %d frames (%d samples)    \n",
                   num_particles, max_frames, bm_samples);
            printf("╠══════════════════════════════════════════════════════╣\n");
            printf("║  Pass          Avg (ms)    %%total                  \n");
            printf("║  scatter       %7.3f     %5.1f%%                   \n", bm_scatter*inv, bm_scatter*inv/total*100);
            printf("║  stencil       %7.3f     %5.1f%%                   \n", bm_stencil*inv, bm_stencil*inv/total*100);
            printf("║  gather        %7.3f     %5.1f%%                   \n", bm_gather*inv, bm_gather*inv/total*100);
            printf("║  constraint    %7.3f     %5.1f%%                   \n", bm_constraint*inv, bm_constraint*inv/total*100);
            printf("║  collision     %7.3f     %5.1f%%                   \n", bm_collision*inv, bm_collision*inv/total*100);
            printf("║  siphon        %7.3f     %5.1f%%                   \n", bm_siphon*inv, bm_siphon*inv/total*100);
            printf("║  ─────────────────────────────────────              \n");
            printf("║  TOTAL         %7.3f ms/frame                      \n", total);
            printf("║  FPS           %7.1f (GPU-only, no present)        \n", 1000.0 / total);
            printf("╠══════════════════════════════════════════════════════╣\n");
            printf("║  Memory layout: graded (set 2, cylindrical)        \n");
            printf("║  Set 2 (graded):  %6.1f MB  (%d B/particle × 10)  \n",
                   (float)graded_bytes / (1024*1024), (int)(10 * sizeof(float)));
            printf("║  Set 0 (pump+proj): %4.1f MB  (5 R/W + 6 W-only)  \n",
                   (float)set0_rw_bytes / (1024*1024));
            printf("║  Density grid:   %6.1f MB  (cyl %d×%d×%d)         \n",
                   (float)density_bytes / (1024*1024), V21_CYL_NR, V21_CYL_NPHI, V21_CYL_NY);
            printf("║  Siphon BW est:  %5.1f GB/s                       \n", siphon_bw);
            printf("╚══════════════════════════════════════════════════════╝\n");
        }

        v21_oracle_summary(&oracle);
        v21_physics_diag_summary(&phys_diag);

        if (gpu_ok) {
            vkDeviceWaitIdle(headlessCtx.device);
            cleanupPhysicsCompute(gpuPhys, headlessCtx.device);
            cleanupHeadlessVulkan(headlessCtx);
            free(rb_px); free(rb_py); free(rb_pz);
            free(rb_vx); free(rb_vy); free(rb_vz);
            free(rb_theta_rb); free(rb_scale); free(rb_flags);
            free(rb_r); free(rb_vel_r); free(rb_phi); free(rb_omega_orb);
        }
        free_particles(particles);
        return 0;
    }

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

    /* Keyboard: R toggles render mode */
    static bool* s_fourier_render_ptr = &fourier_render;
    glfwSetKeyCallback(window, [](GLFWwindow*, int key, int, int action, int) {
        if (key == GLFW_KEY_R && action == GLFW_PRESS) {
            *s_fourier_render_ptr = !*s_fourier_render_ptr;
            printf("[render] Switched to %s renderer\n",
                   *s_fourier_render_ptr ? "Fourier" : "particle");
        }
    });

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

            /* Override in_active_region for lattice particles based on
             * active_override[]. Particles with 0 are siphon-skipped (inertial,
             * governed by collision_apply). Particles with 1 stay in the
             * Viviani field (siphon governs them). */
            if (!lattice.active_override.empty()) {
                size_t sz = lattice.rigid_count * sizeof(uint32_t);
                VkBuffer stagingBuf;
                VkDeviceMemory stagingMem;
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
                memcpy(mapped, lattice.active_override.data(), sz);
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

                int n_skipped = 0;
                for (uint32_t v : lattice.active_override) if (v == 0) n_skipped++;
                printf("[vk-compute] Lattice in_active_region override: %d skipped, "
                       "%d active (of %u total lattice)\n",
                       n_skipped, (int)lattice.rigid_count - n_skipped,
                       lattice.rigid_count);
            }
        }

        initDensityRender(gpuPhys, vkCtx);
        initFourierRender(gpuPhys, vkCtx);  /* always init — R key toggles at runtime */

        /* Phase 3.1: grade-separated state scaffolding.
         * Allocate graded buffers and run one-shot Cartesian → graded conversion
         * so the graded buffers mirror the Cartesian initial state. */
        if (use_forward) gpuPhys.forwardSiphonEnabled = true;  /* hint to skip packed siphon alloc */
        initGradedCompute(gpuPhys, vkCtx);
        initCountingSortCompute(gpuPhys, vkCtx);
        {
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
            dispatchCartesianToGraded(gpuPhys, ucmd);
            vkEndCommandBuffer(ucmd);
            VkSubmitInfo usi = {};
            usi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            usi.commandBufferCount = 1;
            usi.pCommandBuffers = &ucmd;
            vkQueueSubmit(vkCtx.graphicsQueue, 1, &usi, VK_NULL_HANDLE);
            vkQueueWaitIdle(vkCtx.graphicsQueue);
            vkFreeCommandBuffers(vkCtx.device, vkCtx.commandPool, 1, &ucmd);
            printf("[vk-compute] Cartesian → graded initial conversion complete\n");

            /* Forward siphon init: must happen AFTER Cartesian→graded so
             * the graded buffers have real data to copy into A/B. */
            if (use_forward)
                initForwardSiphonCompute(gpuPhys, vkCtx);

            /* Pack graded state into AoS struct for packed siphon.
             * CPU-side pack from host particle data, uploaded via staging. */
            if (gpuPhys.packedSiphonEnabled) {
                struct PackedParticle { float k0[4]; float k1[4]; float k2[4]; float k3[4]; uint32_t meta; uint32_t _pad[3]; };
                /* 80 bytes: 4×vec4(64) + uint(4) + pad(12) = 80 */
                size_t packed_size = (size_t)num_particles * 80;
                std::vector<uint8_t> packed(packed_size, 0);
                PackedParticle* pp = (PackedParticle*)packed.data();
                for (int k = 0; k < num_particles; k++) {
                    float r_cyl = sqrtf(particles.pos_x[k]*particles.pos_x[k] + particles.pos_z[k]*particles.pos_z[k]);
                    float phi = atan2f(particles.pos_z[k], particles.pos_x[k]);
                    if (phi < 0) phi += 6.28318530718f;
                    float r_safe = fmaxf(r_cyl, 1e-6f);
                    float cp = particles.pos_x[k] / r_safe;
                    float sp = particles.pos_z[k] / r_safe;
                    float vr = particles.vel_x[k] * cp + particles.vel_z[k] * sp;
                    float vt = -particles.vel_x[k] * sp + particles.vel_z[k] * cp;
                    float omega = vt / r_safe;

                    pp[k].k0[0] = r_cyl;
                    pp[k].k0[1] = 0.0f;  /* delta_r */
                    pp[k].k0[2] = particles.pos_y[k];
                    pp[k].k0[3] = vr;
                    pp[k].k1[0] = particles.vel_y[k];
                    pp[k].k1[1] = phi;
                    pp[k].k1[2] = omega;
                    pp[k].k1[3] = particles.theta[k];
                    pp[k].k2[0] = particles.pump_scale[k];
                    pp[k].k2[1] = particles.pump_residual[k];
                    pp[k].k2[2] = 0.0f;  /* pump_work */
                    pp[k].k2[3] = particles.pump_history[k];
                    pp[k].k3[0] = particles.omega_nat[k];
                    pp[k].k3[1] = pp[k].k3[2] = pp[k].k3[3] = 0.0f;

                    uint32_t ps = (uint32_t)particles.pump_state[k] & 0x7u;
                    uint32_t fl = (uint32_t)particles.flags[k] & 0xFu;
                    pp[k].meta = ps | (0u << 3) | (fl << 4) | (1u << 8);
                }
                uploadToSSBO(vkCtx, gpuPhys.packedParticleBuffer, packed.data(), packed_size);
                printf("[vk-compute] Packed particle buffer uploaded (%.1f MB)\n",
                       (float)packed_size / (1024.0f * 1024.0f));
            }

            /* Round-trip validation: graded → Cartesian, compare with original.
             * Save original pos/vel, run graded_to_cartesian, readback, diff. */
            {
                int Nv = std::min(gpuPhys.N, 1000);  /* validate first 1000 */
                size_t bf = Nv * sizeof(float);

                /* Save original Cartesian (already on host from init) */
                std::vector<float> orig_px(particles.pos_x, particles.pos_x + Nv);
                std::vector<float> orig_py(particles.pos_y, particles.pos_y + Nv);
                std::vector<float> orig_pz(particles.pos_z, particles.pos_z + Nv);
                std::vector<float> orig_vx(particles.vel_x, particles.vel_x + Nv);
                std::vector<float> orig_vy(particles.vel_y, particles.vel_y + Nv);
                std::vector<float> orig_vz(particles.vel_z, particles.vel_z + Nv);

                /* Run graded → Cartesian (overwrites set 0 pos/vel) */
                VkCommandBuffer vcmd;
                VkCommandBufferAllocateInfo vcba = {};
                vcba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                vcba.commandPool = vkCtx.commandPool;
                vcba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                vcba.commandBufferCount = 1;
                vkAllocateCommandBuffers(vkCtx.device, &vcba, &vcmd);
                VkCommandBufferBeginInfo vbegin = {};
                vbegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                vbegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(vcmd, &vbegin);
                dispatchGradedToCartesian(gpuPhys, vcmd);
                vkEndCommandBuffer(vcmd);
                VkSubmitInfo vsi = {};
                vsi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                vsi.commandBufferCount = 1;
                vsi.pCommandBuffers = &vcmd;
                vkQueueSubmit(vkCtx.graphicsQueue, 1, &vsi, VK_NULL_HANDLE);
                vkQueueWaitIdle(vkCtx.graphicsQueue);
                vkFreeCommandBuffers(vkCtx.device, vkCtx.commandPool, 1, &vcmd);

                /* Readback reconstructed Cartesian */
                std::vector<float> rt_px(Nv), rt_py(Nv), rt_pz(Nv);
                std::vector<float> rt_vx(Nv), rt_vy(Nv), rt_vz(Nv);
                std::vector<float> rt_theta(Nv), rt_scale(Nv);
                std::vector<uint8_t> rt_flags(Nv);
                readbackForOracle(gpuPhys, vkCtx,
                    rt_px.data(), rt_py.data(), rt_pz.data(),
                    rt_vx.data(), rt_vy.data(), rt_vz.data(),
                    rt_theta.data(), rt_scale.data(), rt_flags.data(), Nv);

                /* Compare */
                double max_pos_err = 0, max_vel_err = 0;
                double sum_pos_err = 0, sum_vel_err = 0;
                int n_checked = 0;
                for (int k = 0; k < Nv; k++) {
                    if (!(particles.flags[k] & 0x01)) continue;
                    float r = sqrtf(orig_px[k]*orig_px[k] + orig_pz[k]*orig_pz[k]);
                    if (r < 0.001f) continue;  /* skip origin singularity */

                    double dpx = fabs(rt_px[k] - orig_px[k]);
                    double dpy = fabs(rt_py[k] - orig_py[k]);
                    double dpz = fabs(rt_pz[k] - orig_pz[k]);
                    double dvx = fabs(rt_vx[k] - orig_vx[k]);
                    double dvy = fabs(rt_vy[k] - orig_vy[k]);
                    double dvz = fabs(rt_vz[k] - orig_vz[k]);

                    double pos_err = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
                    double vel_err = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

                    if (pos_err > max_pos_err) max_pos_err = pos_err;
                    if (vel_err > max_vel_err) max_vel_err = vel_err;
                    sum_pos_err += pos_err;
                    sum_vel_err += vel_err;
                    n_checked++;
                }
                double avg_pos = n_checked > 0 ? sum_pos_err / n_checked : 0;
                double avg_vel = n_checked > 0 ? sum_vel_err / n_checked : 0;

                printf("[phase3.1] Round-trip validation (%d particles):\n", n_checked);
                printf("  pos error: max=%.2e  avg=%.2e\n", max_pos_err, avg_pos);
                printf("  vel error: max=%.2e  avg=%.2e\n", max_vel_err, avg_vel);

                if (max_pos_err < 1e-3 && max_vel_err < 1e-3) {
                    printf("  PASS — round-trip within tolerance\n");
                } else {
                    printf("  WARNING — round-trip error exceeds 1e-3\n");
                }

                /* Re-upload original Cartesian so physics isn't corrupted */
                /* (graded_to_cartesian overwrote set 0) */
                /* For now, just re-run cartesian_to_graded to restore consistency */
            }
        }
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
    float *rb_r = NULL, *rb_vel_r = NULL, *rb_phi = NULL, *rb_omega_orb = NULL;
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
        rb_r         = (float*)malloc(bf);
        rb_vel_r     = (float*)malloc(bf);
        rb_phi       = (float*)malloc(bf);
        rb_omega_orb = (float*)malloc(bf);
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
                oracle_count,
                rb_r, rb_vel_r, rb_phi, rb_omega_orb);
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
                 rigid_body_mode == RIGID_BODY_CUBE2_COLLIDE ||
                 rigid_body_mode == RIGID_BODY_CUBE3_HINGE_COLLIDE) &&
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
                    /* Cube 1 internal check + inter-cube center distance. */
                    float d1_x = dist_idx(b + 1000, b + 1001);
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
                } else if (rigid_body_mode == RIGID_BODY_CUBE3_HINGE_COLLIDE) {
                    /* Hinge distances + cube 2 position + cube1-cube2 gap. */
                    int A0 = b + 590;       /* cube 0, ix=0, iy=9, iz=5 */
                    int A1 = b + 1000 + 500; /* cube 1, ix=0, iy=0, iz=5 */
                    int B0 = b + 599;       /* cube 0, ix=9, iy=9, iz=5 */
                    int B1 = b + 1000 + 509; /* cube 1, ix=9, iy=0, iz=5 */
                    float d_hinge_A = dist_idx(A0, A1);
                    float d_hinge_B = dist_idx(B0, B1);
                    int center1 = b + 1000 + 555;
                    int center2 = b + 2000 + 555;
                    float d_12 = dist_idx(center1, center2);
                    printf("[rigid] frame=%d  d_hinge_A=%.4f d_hinge_B=%.4f (rest=1.0)  "
                           "cube1_ctr=(%.2f,%.2f,%.2f) cube2_ctr=(%.2f,%.2f,%.2f) "
                           "d_12=%.3f  d0(0,1)=%.4f\n",
                           frame, d_hinge_A, d_hinge_B,
                           rb_px[center1], rb_py[center1], rb_pz[center1],
                           rb_px[center2], rb_py[center2], rb_pz[center2],
                           d_12, d0_x);
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

            /* Ejection azimuthal probe — are ejected particles bunched at m=3? */
            if (ejected > 10) {
                double ej_fc[5] = {}, ej_fs[5] = {};
                int ej_n = 0;
                for (int p = 0; p < oracle_count; p++) {
                    if (!(rb_flags[p] & 0x02)) continue;
                    double phi_ej = atan2((double)rb_pz[p], (double)rb_px[p]);
                    for (int m = 0; m < 5; m++) {
                        ej_fc[m] += cos(m * phi_ej);
                        ej_fs[m] += sin(m * phi_ej);
                    }
                    ej_n++;
                }
                if (ej_n > 0) {
                    double ej_inv = 1.0 / ej_n;
                    printf("[eject] frame=%d  N=%d  azimuthal modes: ", frame, ej_n);
                    for (int m = 0; m < 5; m++) {
                        double amp = sqrt(ej_fc[m]*ej_fc[m] + ej_fs[m]*ej_fs[m]) * ej_inv;
                        printf("m%d=%.4f ", m, amp);
                    }
                    printf("\n");
                }
            }

            /* Harmonic structure probe — azimuthal Fourier decomposition.
             * Decomposes particle density into m=0,1,2,3,4 modes per radial shell.
             * Measures whether secondary structures (arms, streams) exist. */
            {
                const int N_SHELLS = 4;
                const double shell_edges[N_SHELLS + 1] = {0.0, 15.0, 50.0, 120.0, 250.0};
                const char* shell_names[N_SHELLS] = {"core", "inner", "mid", "outer"};
                const int N_MODES = 5;  /* m = 0,1,2,3,4 */

                /* Fourier accumulators: cos and sin components per shell per mode */
                double fc[N_SHELLS][N_MODES] = {};  /* cos(m*phi) */
                double fs[N_SHELLS][N_MODES] = {};  /* sin(m*phi) */
                int shell_count[N_SHELLS] = {};

                /* Also measure Viviani phase coherence per shell:
                 * R_theta = |<exp(i*m*theta)>| — Kuramoto-like order parameter */
                double tc[N_SHELLS][N_MODES] = {};
                double ts[N_SHELLS][N_MODES] = {};

                for (int p = 0; p < oracle_count; p++) {
                    if (rb_flags[p] & 0x02) continue;  /* skip ejected */
                    double px = rb_px[p], py = rb_py[p], pz = rb_pz[p];
                    double r_cyl = sqrt(px*px + pz*pz);
                    double phi_p = atan2(pz, px);
                    double theta_p = rb_theta[p];

                    /* Find shell */
                    int sh = -1;
                    for (int s = 0; s < N_SHELLS; s++) {
                        if (r_cyl >= shell_edges[s] && r_cyl < shell_edges[s+1]) { sh = s; break; }
                    }
                    if (sh < 0) continue;
                    shell_count[sh]++;

                    for (int m = 0; m < N_MODES; m++) {
                        double angle = m * phi_p;
                        fc[sh][m] += cos(angle);
                        fs[sh][m] += sin(angle);
                        double tangle = m * theta_p;
                        tc[sh][m] += cos(tangle);
                        ts[sh][m] += sin(tangle);
                    }
                }

                /* Print azimuthal mode amplitudes (normalized: 0 = uniform, 1 = perfect arm) */
                printf("[harmonic] frame=%d  azimuthal density modes |A_m| / N_shell:\n", frame);
                for (int s = 0; s < N_SHELLS; s++) {
                    if (shell_count[s] < 10) continue;
                    double inv = 1.0 / shell_count[s];
                    printf("[harmonic]   %5s (N=%6d, r=[%3.0f,%3.0f)):  ",
                           shell_names[s], shell_count[s],
                           shell_edges[s], shell_edges[s+1]);
                    for (int m = 0; m < N_MODES; m++) {
                        double amp = sqrt(fc[s][m]*fc[s][m] + fs[s][m]*fs[s][m]) * inv;
                        printf("m%d=%.4f ", m, amp);
                    }
                    printf("\n");
                }

                /* Print Viviani phase coherence (Kuramoto R per mode) */
                printf("[harmonic] frame=%d  Viviani phase coherence R_theta:\n", frame);
                for (int s = 0; s < N_SHELLS; s++) {
                    if (shell_count[s] < 10) continue;
                    double inv = 1.0 / shell_count[s];
                    printf("[harmonic]   %5s:  ", shell_names[s]);
                    for (int m = 0; m < N_MODES; m++) {
                        double R = sqrt(tc[s][m]*tc[s][m] + ts[s][m]*ts[s][m]) * inv;
                        printf("m%d=%.4f ", m, R);
                    }
                    printf("\n");
                }

                /* Compact summary line — track m=3 growth curve */
                double m3_inner = 0.0, m3_mid = 0.0, m3_core = 0.0;
                double R3_inner = 0.0;
                for (int s = 0; s < N_SHELLS; s++) {
                    if (shell_count[s] < 10) continue;
                    double inv = 1.0 / shell_count[s];
                    double a3 = sqrt(fc[s][3]*fc[s][3] + fs[s][3]*fs[s][3]) * inv;
                    double r3 = sqrt(tc[s][3]*tc[s][3] + ts[s][3]*ts[s][3]) * inv;
                    if (s == 0) m3_core = a3;
                    if (s == 1) { m3_inner = a3; R3_inner = r3; }
                    if (s == 2) m3_mid = a3;
                }
                printf("[m3-track] frame=%d  A3: core=%.5f inner=%.5f mid=%.5f  R3_inner=%.5f\n",
                       frame, m3_core, m3_inner, m3_mid, R3_inner);
            }

            /* Physics diagnostics — rotation curve, Toomre Q, energy budget,
             * pattern speed, pitch angle, Reynolds stress */
            v21_physics_diag_compute(&phys_diag,
                rb_r, rb_vel_r, rb_phi, rb_omega_orb,
                rb_px, rb_py, rb_pz,
                rb_vx, rb_vy, rb_vz,
                rb_flags,
                oracle_count, frame);
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
                if (!project_only) {
                    if (use_forward && gpuPhys.forwardSiphonEnabled) {
                        /* Density grid from last frame's Cartesian positions */
                        dispatchCylDensity(gpuPhys, cmd);
                        /* Forward siphon reads density grid for neighbor bonding */
                        dispatchForwardSiphon(gpuPhys, cmd, sim_time, dt * 2.0f);
                        /* Barrier: siphon writes must be visible to projection */
                        VkMemoryBarrier b = {};
                        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkCmdPipelineBarrier(cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &b, 0, nullptr, 0, nullptr);
                        dispatchForwardProject(gpuPhys, cmd);
                        /* Barrier: projection writes must be visible to density render */
                        VkMemoryBarrier b2 = {};
                        b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                        b2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                        vkCmdPipelineBarrier(cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 1, &b2, 0, nullptr, 0, nullptr);
                    } else {
                        dispatchPhysicsCompute(gpuPhys, cmd, frame, sim_time, dt * 2.0f);

                        /* Barrier: siphon writes must complete before projection reads */
                        VkMemoryBarrier siphonBarrier = {};
                        siphonBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                        siphonBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        siphonBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkCmdPipelineBarrier(cmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &siphonBarrier, 0, nullptr, 0, nullptr);
                    }
                }

                /* Get viewProj from the mapped UBO */
                GlobalUBO* ubo = (GlobalUBO*)vkCtx.uniformBuffersMapped[vkCtx.currentFrame];
                if (fourier_render && gpuPhys.fourierRenderEnabled)
                    recordFourierRender(gpuPhys, cmd, vkCtx, imageIndex, ubo->viewProj, frame);
                else
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
        free(rb_r); free(rb_vel_r); free(rb_phi); free(rb_omega_orb);
    }
    v21_oracle_summary(&oracle);
    v21_physics_diag_summary(&phys_diag);

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
