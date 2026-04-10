/*
 * V21 VULKAN COMPUTE — GPU Physics Dispatch
 * ==========================================
 *
 * Loads siphon.spv, creates compute pipeline, dispatches GPU physics.
 * Replaces CPU physics_step() with parallel GPU execution.
 * Includes staging buffer for CPU oracle readback.
 *
 * The GPU's rasterizer does the rendering (additive blend = harmonic sum).
 * The GPU's compute unit does the physics (SPIRV = vendor-neutral).
 * The CPU's oracle validates the invariants.
 */

#ifndef V21_VK_COMPUTE_H
#define V21_VK_COMPUTE_H

#include "../vulkan/vk_types.h"
#include <vulkan/vulkan.h>

/* Push constants for siphon.comp */
struct SiphonPushConstants {
    int   N;
    float time;
    float dt;
    float BH_MASS;
    float FIELD_STRENGTH;
    float FIELD_FALLOFF;
    float TANGENT_SCALE;
    uint32_t seam_bits;
    float bias;
};

/* Push constants for project.comp */
struct ProjectPushConstants {
    int   particle_count;
    float brightness;
    float temp_scale;
};

/* Push constants for tone_map.frag */
struct ToneMapPushConstants {
    float peak_density;
    float exposure;
    float gamma;
    int   colormap;
};

/* Camera UBO for projection */
struct ProjectCameraUBO {
    float view_proj[16];
    float zoom;
    float aspect;
    int   width;
    int   height;
};

/* Number of SoA arrays bound as SSBOs */
#define VK_COMPUTE_NUM_BINDINGS 16

/* Physics compute state */
struct PhysicsCompute {
    /* Per-array SSBOs */
    VkBuffer      soa_buffers[VK_COMPUTE_NUM_BINDINGS];
    VkDeviceMemory soa_memory[VK_COMPUTE_NUM_BINDINGS];
    size_t         soa_sizes[VK_COMPUTE_NUM_BINDINGS];

    /* Siphon compute pipeline (physics) */
    VkDescriptorSetLayout descLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;

    /* Projection compute pipeline (rendering) */
    VkDescriptorSetLayout projDescLayout;
    VkPipelineLayout projPipelineLayout;
    VkPipeline projPipeline;
    VkDescriptorPool projDescPool;
    VkDescriptorSet projDescSet;

    /* Tone-map graphics pipeline */
    VkDescriptorSetLayout toneDescLayout;
    VkPipelineLayout tonePipelineLayout;
    VkPipeline tonePipeline;
    VkDescriptorPool toneDescPool;
    VkDescriptorSet toneDescSet;

    /* Density image (R32_UINT) */
    VkImage densityImage;
    VkDeviceMemory densityMemory;
    VkImageView densityView;
    VkSampler densitySampler;

    /* Camera UBO */
    VkBuffer cameraUBO;
    VkDeviceMemory cameraMemory;
    void* cameraMapped;

    /* Staging buffer for CPU oracle readback */
    VkBuffer staging;
    VkDeviceMemory stagingMemory;
    void* stagingMapped;
    size_t stagingSize;

    /* GPU timestamp profiling — double-buffered.
     * Each slot holds 4 timestamps: siphon_begin, siphon_end,
     * project_end, tonemap_end. Reads previous frame's results
     * non-blocking to avoid stalling the queue. */
    VkQueryPool queryPool;
    float timestampPeriodNs;   /* ns per timestamp tick */
    bool queryValid[2];        /* whether slot N has been written yet */
    uint32_t queryFrame;       /* 0 or 1, current write slot */

    /* Particle count */
    int N;
    bool initialized;
};

/* Initialize compute pipeline + SSBOs, upload initial particle state */
void initPhysicsCompute(PhysicsCompute& phys, VulkanContext& ctx,
                         const float* pos_x, const float* pos_y, const float* pos_z,
                         const float* vel_x, const float* vel_y, const float* vel_z,
                         const float* pump_scale, const float* pump_residual,
                         const float* pump_history, const int* pump_state,
                         const float* theta, const float* omega_nat,
                         const uint8_t* flags, const uint8_t* topo_state,
                         int N);

/* Record compute dispatch commands into command buffer */
void dispatchPhysicsCompute(PhysicsCompute& phys, VkCommandBuffer cmd,
                            int frame, float sim_time, float dt);

/* Initialize density rendering pipeline (projection + tone-map) */
void initDensityRender(PhysicsCompute& phys, VulkanContext& ctx);

/* Record density rendering commands: clear → project → barrier → tone-map */
void recordDensityRender(PhysicsCompute& phys, VkCommandBuffer cmd,
                         VulkanContext& ctx, uint32_t imageIndex,
                         const float* viewProj);

/* Oracle subset size — fixed cap so staging buffer stays small (~3.2 MB) */
#define ORACLE_SUBSET_SIZE 100000

/* Read back particle subset to CPU for oracle validation.
 * Copies the first `count` entries of each of 8 SoA arrays (pos_xyz, vel_xyz,
 * theta, pump_scale) plus flags as uint32 from device-local SSBOs into the
 * host-visible staging buffer, then memcpy's into the output arrays.
 * `count` must be <= ORACLE_SUBSET_SIZE. */
void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       float* out_theta, float* out_pump_scale,
                       uint8_t* out_flags,
                       int count);

/* Read back GPU timestamps from the previous frame (non-blocking).
 * Writes ms values to out_siphon_ms, out_project_ms, out_tonemap_ms.
 * Returns true if results are valid, false if not yet available. */
bool readTimestamps(PhysicsCompute& phys, VkDevice device,
                    double* out_siphon_ms,
                    double* out_project_ms,
                    double* out_tonemap_ms);

/* Cleanup */
void cleanupPhysicsCompute(PhysicsCompute& phys, VkDevice device);

#endif /* V21_VK_COMPUTE_H */
