// V20 Vulkan Types and Common Includes
// =====================================
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <array>
#include <iostream>
#include <cmath>

// Forward declarations
struct VulkanContext;

// Queue family indices
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swapchain support details
struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Particle vertex for instanced rendering
struct ParticleVertex {
    float position[3];    // xyz world position
    float pump_scale;     // for size/color mapping (1.0 = 12D, ~1.33 = 16D)
    float pump_residual;  // dissolution effect (0-1, >0.95 = dissolving)
    float temp;           // temperature for blackbody coloring
    float velocity[3];    // xyz velocity (for motion blur)
    float elongation;     // streak length

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding = {};
        binding.binding = 0;
        binding.stride = sizeof(ParticleVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;  // Per-instance data
        return binding;
    }

    static std::array<VkVertexInputAttributeDescription, 6> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 6> attrs = {};

        // position (vec3)
        attrs[0].binding = 0;
        attrs[0].location = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(ParticleVertex, position);

        // pump_scale (float)
        attrs[1].binding = 0;
        attrs[1].location = 1;
        attrs[1].format = VK_FORMAT_R32_SFLOAT;
        attrs[1].offset = offsetof(ParticleVertex, pump_scale);

        // pump_residual (float)
        attrs[2].binding = 0;
        attrs[2].location = 2;
        attrs[2].format = VK_FORMAT_R32_SFLOAT;
        attrs[2].offset = offsetof(ParticleVertex, pump_residual);

        // temp (float)
        attrs[3].binding = 0;
        attrs[3].location = 3;
        attrs[3].format = VK_FORMAT_R32_SFLOAT;
        attrs[3].offset = offsetof(ParticleVertex, temp);

        // velocity (vec3)
        attrs[4].binding = 0;
        attrs[4].location = 4;
        attrs[4].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[4].offset = offsetof(ParticleVertex, velocity);

        // elongation (float)
        attrs[5].binding = 0;
        attrs[5].location = 5;
        attrs[5].format = VK_FORMAT_R32_SFLOAT;
        attrs[5].offset = offsetof(ParticleVertex, elongation);

        return attrs;
    }
};

// Global uniforms (camera, time, pump metrics)
struct GlobalUBO {
    float viewProj[16];   // 4x4 view-projection matrix
    float cameraPos[3];   // Camera world position
    float time;           // Elapsed time
    float avgScale;       // Average pump scale (from PumpMetrics)
    float avgResidual;    // Average residual
    float heartbeat;      // Oscillating [-1, 1] for pulsing effects
    float pump_phase;     // Actual pump phase from CUDA (for volume shader)
};

// LOD configuration for hybrid rendering
struct LODConfig {
    float nearThreshold;   // Distance below which particles render as points (default: 150)
    float farThreshold;    // Distance above which particles go to volume only (default: 600)
    float blendRange;      // Smooth transition range (default: 100)
    float volumeScale;     // World-space size of density volume (default: 300)
};

// Volume rendering uniforms (matches layout in volume_shells.frag)
struct VolumeUBO {
    float invViewProj[16];  // 4x4 inverse view-projection matrix
    float volumeMin[3];     // World-space AABB min (e.g., -200, -20, -200)
    float volumeScale;      // Total extent (e.g., 400)
    float volumeMax[3];     // World-space AABB max (e.g., 200, 20, 200)
    float seam_angle;       // Locked seam orientation in radians (from CUDA)
    float shellBrightness;  // Shell opacity multiplier (V key cycles 1.0→0.5→0.25→0)
    float padding[3];       // Align to 16-byte boundary
};

// Density grid for volumetric far-field rendering (128³ = 2M voxels)
// Each voxel stores: accumulated pump_scale, temperature, count, dominant state
#define DENSITY_GRID_SIZE 128
#define DENSITY_GRID_VOXELS (DENSITY_GRID_SIZE * DENSITY_GRID_SIZE * DENSITY_GRID_SIZE)

struct DensityVoxel {
    float pump_scale_sum;  // Accumulated pump scale
    float temp_sum;        // Accumulated temperature
    float count;           // Number of particles in voxel
    float coherence;       // Lock ratio (0=chaotic, 1=coherent)
};

// Main Vulkan context
struct VulkanContext {
    // Window
    GLFWwindow* window = nullptr;
    uint32_t windowWidth = 1280;
    uint32_t windowHeight = 720;

    // Instance
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    // Surface
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // Device
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    QueueFamilyIndices queueFamilies;

    // Swapchain
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;

    // Render pass
    VkRenderPass renderPass = VK_NULL_HANDLE;

    // Framebuffers
    std::vector<VkFramebuffer> framebuffers;

    // Command pool and buffers
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // Sync objects
    // Per-frame sync (indexed by currentFrame for frame pacing)
    std::vector<VkSemaphore> imageAvailableSemaphores;  // Signaled when image acquired
    std::vector<VkFence> inFlightFences;                // Signaled when GPU finishes frame

    // Per-swapchain-image sync (indexed by imageIndex to avoid reuse conflicts)
    std::vector<VkSemaphore> renderFinishedSemaphores;  // Signaled when render done, used for present

    uint32_t currentFrame = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;  // More frames in flight for better GPU utilization

    // Depth buffer
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;

    // Pipeline
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // Descriptor sets
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    // Uniform buffers
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // Particle buffer (will be shared with CUDA)
    VkBuffer particleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory particleBufferMemory = VK_NULL_HANDLE;
    uint32_t particleCount = 0;
    uint32_t nearParticleCount = 0;  // Count of particles in NEAR LOD (rendered as points)

    // Indirect draw support for hybrid LOD (stream compaction)
    // CUDA writes visible particles here and updates the draw command count
    VkBuffer compactedParticleBuffer = VK_NULL_HANDLE;      // Compacted visible particles
    VkDeviceMemory compactedParticleBufferMemory = VK_NULL_HANDLE;
    VkBuffer indirectDrawBuffer = VK_NULL_HANDLE;           // VkDrawIndirectCommand written by CUDA
    VkDeviceMemory indirectDrawBufferMemory = VK_NULL_HANDLE;
    bool useIndirectDraw = false;  // Enable indirect draw for hybrid LOD

    // Density volume for far-field rendering (128³ 3D texture)
    VkImage densityVolume = VK_NULL_HANDLE;
    VkDeviceMemory densityVolumeMemory = VK_NULL_HANDLE;
    VkImageView densityVolumeView = VK_NULL_HANDLE;
    VkSampler densityVolumeSampler = VK_NULL_HANDLE;

    // Volume rendering pipeline (fullscreen quad + raymarching)
    VkPipeline volumePipeline = VK_NULL_HANDLE;
    VkPipelineLayout volumePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout volumeDescriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> volumeDescriptorSets;

    // Volume uniform buffers
    std::vector<VkBuffer> volumeUniformBuffers;
    std::vector<VkDeviceMemory> volumeUniformBuffersMemory;
    std::vector<void*> volumeUniformBuffersMapped;

    // Volume rendering mode: 0=off, 1=texture-based, 2=analytic, 3=shells (default)
    int volumeRenderMode = 3;
    bool enableVolumePass = true;  // Toggle volume rendering on/off

    // Shell brightness control (0.0-1.0)
    // Controls volumetric shell opacity independently of particle brightness
    // V key cycles: 100% → 50% → 25% → OFF → 100%
    float shellBrightness = 1.0f;

    // LOD configuration
    LODConfig lodConfig = {150.0f, 600.0f, 100.0f, 300.0f};

    // State
    bool framebufferResized = false;

    // Attractor-first rendering mode (always on, default mode was broken)
    bool useAttractorMode = true;

    // Camera state (mouse-controlled orbit)
    float cameraYaw = 0.0f;      // Horizontal angle (radians)
    float cameraPitch = 0.3f;    // Vertical angle (radians) - start slightly above
    float cameraRadius = 800.0f; // Distance from origin (further back for 3.5M particles)
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    bool mousePressed = false;
};

// Validation layer support
#ifdef NDEBUG
    constexpr bool enableValidationLayers = false;
#else
    constexpr bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
};

// Function declarations
namespace vk {
    // Instance (vk_instance.cpp)
    void createInstance(VulkanContext& ctx);
    void setupDebugMessenger(VulkanContext& ctx);
    void destroyInstance(VulkanContext& ctx);

    // Device (vk_device.cpp)
    void pickPhysicalDevice(VulkanContext& ctx);
    void createLogicalDevice(VulkanContext& ctx);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

    // Swapchain (vk_swapchain.cpp)
    void createSwapchain(VulkanContext& ctx);
    void createImageViews(VulkanContext& ctx);
    void createDepthResources(VulkanContext& ctx);
    void createFramebuffers(VulkanContext& ctx);
    void recreateSwapchain(VulkanContext& ctx);
    void cleanupSwapchain(VulkanContext& ctx);

    // Pipeline (vk_pipeline.cpp)
    void createRenderPass(VulkanContext& ctx);
    void createDescriptorSetLayout(VulkanContext& ctx);
    void createGraphicsPipeline(VulkanContext& ctx);
    void createVolumeDescriptorSetLayout(VulkanContext& ctx);
    void createVolumePipeline(VulkanContext& ctx);

    // Buffer (vk_buffer.cpp)
    void createCommandPool(VulkanContext& ctx);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);
    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx);
    void createVolumeUniformBuffers(VulkanContext& ctx);
    void createVolumeDescriptorSets(VulkanContext& ctx);
    uint32_t findMemoryType(VulkanContext& ctx, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory);

    // Rendering
    void recordCommandBuffer(VulkanContext& ctx, VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void drawFrame(VulkanContext& ctx);
    void updateUniformBuffer(VulkanContext& ctx, uint32_t currentImage);

    // Cleanup
    void cleanup(VulkanContext& ctx);
}
