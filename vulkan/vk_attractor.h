// vk_attractor.h — Density-Based Galaxy Rendering Pipeline
// =========================================================
//
// Renders the galaxy using compute-based density accumulation:
//   - Compute shader reads actual particle positions from SSBO
//   - Projects particles to screen, accumulates into density buffer
//   - Fragment shader tone-maps density to color with various colormaps
//
// This provides smoother, more astronomical-looking renders than
// individual point sprites, especially for dense regions.
//
#pragma once

#include "vk_types.h"

// Camera UBO for density sampling — matches harmonic_sample.comp binding 2
struct AttractorCameraUBO {
    float view_proj[16];  // 4x4 view-projection matrix
    float zoom;
    float aspect;
    int   width;
    int   height;
};

// Attractor State UBO for pure parametric mode — matches harmonic_attractor.comp binding 0
struct AttractorStateUBO {
    float w;         // accumulated w-component [0, 1]
    float s_theta;   // sqrt(1 - w^2), precomputed
    float phase;     // global pump phase mod 2pi
    float residual;  // bifurcation proximity
};

// Push constants for pure attractor mode
struct AttractorPurePush {
    int   n_samples;     // total theta samples this frame
    float shell_bias;    // >1 = more inner-shell samples
    float brightness;    // pre-tone-map scale
    float time;          // elapsed time
};

// Render mode enum
enum class AttractorMode {
    POSITION_PRIMARY = 0,  // Reads particle xyz from CUDA (harmonic_sample.comp)
    PHASE_PRIMARY    = 1,  // Reads phase state from CUDA (harmonic_phase.comp)
    PURE_ATTRACTOR   = 2   // Pure GPU parametric sampling (harmonic_attractor.comp)
};

// Push constants for compute shader — matches harmonic_sample.comp
struct AttractorComputePush {
    int   particle_count;  // number of active particles
    float brightness;      // pre-tone-map scale
    float temp_scale;      // temperature contribution to brightness
    float time;            // elapsed time for phase evolution (phase-primary only)
};

// Push constants for tone mapping
struct AttractorTonePush {
    float peak_density;   // normalisation: max expected count per pixel
    float exposure;       // linear pre-scale before log compression
    float gamma;          // output gamma (2.2 for sRGB)
    int   colormap;       // 0=thermal, 1=plasma, 2=blue-white, 3=hopf
};

// Attractor pipeline extension to VulkanContext
struct AttractorPipeline {
    // Density buffer (R32_UINT storage image)
    VkImage       densityImage = VK_NULL_HANDLE;
    VkDeviceMemory densityMemory = VK_NULL_HANDLE;
    VkImageView   densityView = VK_NULL_HANDLE;

    // Sampler for tone mapping pass
    VkSampler     densitySampler = VK_NULL_HANDLE;

    // Camera UBO
    VkBuffer      cameraUBO = VK_NULL_HANDLE;
    VkDeviceMemory cameraUBOMemory = VK_NULL_HANDLE;
    void*         cameraUBOMapped = nullptr;

    // Compute pipeline (density accumulation)
    VkDescriptorSetLayout computeDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       computePipeline = VK_NULL_HANDLE;
    VkDescriptorSet  computeDescSet = VK_NULL_HANDLE;

    // Graphics pipeline (tone mapping)
    VkDescriptorSetLayout graphicsDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       graphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSet  graphicsDescSet = VK_NULL_HANDLE;

    // Dedicated descriptor pool
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    // Phase-primary compute pipeline (harmonic_phase.comp)
    // Uses same descriptor layout but different shader
    VkPipeline       phasePipeline = VK_NULL_HANDLE;

    // Pure attractor pipeline (harmonic_attractor.comp)
    // Uses different descriptor layout (UBO at binding 0 instead of SSBO)
    VkDescriptorSetLayout pureDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout purePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       purePipeline = VK_NULL_HANDLE;
    VkDescriptorSet  pureDescSet = VK_NULL_HANDLE;

    // Attractor state UBO (for pure mode)
    VkBuffer      stateUBO = VK_NULL_HANDLE;
    VkDeviceMemory stateUBOMemory = VK_NULL_HANDLE;
    void*         stateUBOMapped = nullptr;

    // Render state
    bool enabled = false;
    AttractorMode mode = AttractorMode::POSITION_PRIMARY;  // Current render mode
    int  particleCount = 0;       // Current particle count (updated each frame)
    int  colormap = 3;            // hopf colormap
    float brightness = 1.0f;
    float temp_scale = 2.0f;      // Temperature contribution
    float exposure = 2.0f;
    float peak_density = 50000.0f;
    float time = 0.0f;            // Elapsed time for phase evolution
};

namespace vk {
    // Initialize attractor pipeline (needs particle buffer from ctx)
    void createAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);

    // Record attractor rendering commands
    void recordAttractorCommands(
        VulkanContext& ctx,
        AttractorPipeline& attractor,
        VkCommandBuffer cmd,
        uint32_t imageIndex
    );

    // Update attractor camera (call when camera changes)
    void updateAttractorCamera(
        AttractorPipeline& attractor,
        const float* viewProj,
        float zoom, float aspect,
        int width, int height
    );

    // Update attractor state (for pure mode - call each frame)
    void updateAttractorState(
        AttractorPipeline& attractor,
        float w, float phase, float residual
    );

    // Cleanup
    void destroyAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);
}
