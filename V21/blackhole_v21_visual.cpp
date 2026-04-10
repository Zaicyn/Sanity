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
#define BH_MASS              100.0f
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
        ps.omega_nat[i] = 0.1f + ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
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

        /* Viviani field */
        float theta_pos = atan2f(pz, px);
        float c1=cosf(theta_pos), s1=sinf(theta_pos);
        float c3=cosf(3*theta_pos), s3=sinf(3*theta_pos);
        float fx=c1-1.5f*c3, fy=s1-1.5f*s3, fz=-s1*c3-3*c1*s3;
        float inv_f=1.0f/sqrtf(fx*fx+fy*fy+fz*fz+1e-8f);
        fx*=inv_f; fy*=inv_f; fz*=inv_f;
        float weight = 1.0f/(1.0f+r_safe*r_safe/100.0f);
        float rx=px*inv_r, ry=py*inv_r, rz=pz*inv_r;
        float fdr = fx*rx+fy*ry+fz*rz;
        float tx=fx-fdr*rx, ty=fy-fdr*ry, tz=fz-fdr*rz;
        float ax=-rx*weight + tx*weight*0.5f;
        float ay=-ry*weight + ty*weight*0.5f;
        float az=-rz*weight + tz*weight*0.5f;

        /* Orbital damping */
        float Lx=py*vz-pz*vy, Ly=pz*vx-px*vz, Lz=px*vy-py*vx;
        float iL=1.0f/sqrtf(Lx*Lx+Ly*Ly+Lz*Lz+1e-8f);
        float lx=Lx*iL, ly=Ly*iL, lz=Lz*iL;
        if (vx*vx+vy*vy+vz*vz > 0.001f) {
            float vn=vx*lx+vy*ly+vz*lz;
            float r3=r3d*r3d*r3d;
            float damp=fminf(2.0f*sqrtf(BH_MASS/fmaxf(r3,1.0f)), 0.5f);
            vx-=damp*vn*lx; vy-=damp*vn*ly; vz-=damp*vn*lz;
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

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-physics") == 0)
            use_gpu_physics = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole_v21_visual [-n particles] [--rng-seed S] [--gpu-physics]\n");
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
    }

    printf("[v21-visual] %d particles, Vulkan ready, physics=%s\n",
           num_particles, use_gpu_physics ? "GPU" : "CPU");

    /* Readback arrays for oracle (when using GPU physics) */
    float *rb_px = NULL, *rb_py = NULL, *rb_pz = NULL;
    float *rb_vx = NULL, *rb_vy = NULL, *rb_vz = NULL;
    if (use_gpu_physics) {
        rb_px = (float*)malloc(num_particles * sizeof(float));
        rb_py = (float*)malloc(num_particles * sizeof(float));
        rb_pz = (float*)malloc(num_particles * sizeof(float));
        rb_vx = (float*)malloc(num_particles * sizeof(float));
        rb_vy = (float*)malloc(num_particles * sizeof(float));
        rb_vz = (float*)malloc(num_particles * sizeof(float));
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

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (!use_gpu_physics) {
            /* CPU physics path */
            physics_step(particles, dt * 2.0f);
        }
        /* GPU physics is dispatched inside the command buffer below */
        sim_time += dt;

        /* Oracle validation (CPU physics: every frame, GPU: readback every 100) */
        if (!use_gpu_physics) {
            v21_oracle_check(&oracle,
                particles.pos_x, particles.pos_y, particles.pos_z,
                particles.vel_x, particles.vel_y, particles.vel_z,
                particles.theta, particles.pump_scale,
                particles.flags, particles.topo_state,
                particles.N, frame);
        } else if (frame % 100 == 0 && frame > 0) {
            /* GPU readback for oracle */
            readbackForOracle(gpuPhys, vkCtx, rb_px, rb_py, rb_pz,
                              rb_vx, rb_vy, rb_vz, particles.N);
            v21_oracle_check(&oracle,
                rb_px, rb_py, rb_pz, rb_vx, rb_vy, rb_vz,
                particles.theta, particles.pump_scale,
                particles.flags, particles.topo_state,
                particles.N, frame);
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

        /* Update camera */
        vkCtx.cameraYaw += 0.002f;
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

            /* GPU physics dispatch (before render pass) */
            if (use_gpu_physics) {
                dispatchPhysicsCompute(gpuPhys, cmd, frame, sim_time, dt * 2.0f);
            }

            /* Render pass */
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

            /* Viewport + scissor */
            VkViewport viewport = {};
            viewport.width = (float)vkCtx.swapchainExtent.width;
            viewport.height = (float)vkCtx.swapchainExtent.height;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{0,0}, vkCtx.swapchainExtent};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            /* Bind particle pipeline + vertex buffer + draw */
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkCtx.graphicsPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vkCtx.pipelineLayout, 0, 1, &vkCtx.descriptorSets[vkCtx.currentFrame], 0, nullptr);
            VkBuffer vbufs[] = {vertexBuf.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdDraw(cmd, 6, particles.N, 0, 0);  /* 6 verts (quad) × N instances */

            vkCmdEndRenderPass(cmd);
            vkEndCommandBuffer(cmd);

            /* Update camera uniforms */
            vk::updateUniformBuffer(vkCtx, vkCtx.currentFrame);

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
    }

    /* Cleanup */
    vkDeviceWaitIdle(vkCtx.device);
    if (use_gpu_physics) {
        cleanupPhysicsCompute(gpuPhys, vkCtx.device);
        free(rb_px); free(rb_py); free(rb_pz);
        free(rb_vx); free(rb_vy); free(rb_vz);
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
