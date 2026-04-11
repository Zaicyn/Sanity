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

static void init_particles(ParticleState& ps, int N, unsigned int seed) {
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

    srand(seed);
    for (int i = 0; i < N; i++) {
        float x = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float y = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R * 0.3f;
        float z = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float r = sqrtf(x*x+y*y+z*z);
        if (r < 6.0f) { float s=6.0f/r; x*=s; y*=s; z*=s; }
        ps.pos_x[i]=x; ps.pos_y[i]=y; ps.pos_z[i]=z;

        float r_xz = sqrtf(x*x+z*z);
        if (r_xz > 0.1f) {
            float v = sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            ps.vel_x[i] = -v*(z/r_xz);
            ps.vel_z[i] =  v*(x/r_xz);
        }
        ps.pump_scale[i] = 1.0f;
        ps.flags[i] = 0x01;
        ps.theta[i] = (float)rand()/RAND_MAX * 6.28318f;
        /* Per-particle phase rate — r-dependent so inner shells advance
         * faster than outer shells. This is GPT's phase-field solution
         * to the rigid-rotation problem: instead of all particles sharing
         * a global cos(theta_pos) target, each particle carries its own
         * phase that advances at a rate proportional to 1/r. Inner
         * particles cycle the Viviani target 20× faster than outer
         * particles, creating the shear that differentiates shells into
         * a layered galaxy rather than a rigid sheet. */
        float r3d_init = sqrtf(x*x + y*y + z*z);
        float r_eff = fmaxf(r3d_init, ISCO_R);
        ps.omega_nat[i] = 0.377f / r_eff;  /* ≈ 1 cycle / 100 frames at ISCO */
        int axis = rand()%4; int sign = (rand()%2)?1:-1;
        ps.topo_state[i] = (uint8_t)(((sign>0)?1:2) << (axis*2));
    }
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

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-physics") == 0)
            use_gpu_physics = true;
        else if (strcmp(argv[i], "--project-only") == 0)
            { use_gpu_physics = true; project_only = true; }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole_v21_visual [-n particles] [--rng-seed S] [--gpu-physics] [--project-only]\n");
            return 0;
        }
    }

    printf("================================================================\n");
    printf("  BLACKHOLE V21 VISUAL\n");
    printf("  CPU Physics + Vulkan Rendering\n");
    printf("  The framebuffer IS the harmonic accumulator.\n");
    printf("================================================================\n\n");

    /* Init particles */
    ParticleState particles;
    init_particles(particles, num_particles, seed);

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
            particles.N);
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
    double gpu_scatter_sum = 0.0, gpu_siphon_sum = 0.0;
    double gpu_project_sum = 0.0, gpu_tonemap_sum = 0.0;
    int gpu_samples = 0;

    while (!glfwWindowShouldClose(window)) {
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
            double sc_ms = 0, s_ms = 0, p_ms = 0, t_ms = 0;
            if (readTimestamps(gpuPhys, vkCtx.device, &sc_ms, &s_ms, &p_ms, &t_ms)) {
                gpu_scatter_sum += sc_ms;
                gpu_siphon_sum  += s_ms;
                gpu_project_sum += p_ms;
                gpu_tonemap_sum += t_ms;
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
            double sc = gpu_scatter_sum * inv;
            double s  = gpu_siphon_sum  * inv;
            double p  = gpu_project_sum * inv;
            double t  = gpu_tonemap_sum * inv;
            double total = sc + s + p + t;
            printf("[gpu] scatter=%.3f ms (%4.1f%%)  siphon=%.3f ms (%4.1f%%)  "
                   "project=%.3f ms (%4.1f%%)  tonemap=%.3f ms (%4.1f%%)  "
                   "total=%.3f ms  (%d samples)\n",
                   sc, 100.0 * sc / total,
                   s,  100.0 * s  / total,
                   p,  100.0 * p  / total,
                   t,  100.0 * t  / total,
                   total, gpu_samples);
            gpu_scatter_sum = gpu_siphon_sum = 0.0;
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
