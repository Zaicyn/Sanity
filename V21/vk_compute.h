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

/* Number of SoA arrays bound as SSBOs */
#define VK_COMPUTE_NUM_BINDINGS 16

/* Physics compute state */
struct PhysicsCompute {
    /* Per-array SSBOs */
    VkBuffer      soa_buffers[VK_COMPUTE_NUM_BINDINGS];
    VkDeviceMemory soa_memory[VK_COMPUTE_NUM_BINDINGS];
    size_t         soa_sizes[VK_COMPUTE_NUM_BINDINGS];

    /* Compute pipeline */
    VkDescriptorSetLayout descLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;

    /* Staging buffer for CPU oracle readback */
    VkBuffer staging;
    VkDeviceMemory stagingMemory;
    void* stagingMapped;
    size_t stagingSize;

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

/* Read back particle subset to CPU for oracle validation */
void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       int count);

/* Read back packed vertices for rendering (SoA → staging → CPU → render buffer) */
void readbackForRender(PhysicsCompute& phys, VulkanContext& ctx,
                       void* vertex_data, int count);

/* Cleanup */
void cleanupPhysicsCompute(PhysicsCompute& phys, VkDevice device);

#endif /* V21_VK_COMPUTE_H */
