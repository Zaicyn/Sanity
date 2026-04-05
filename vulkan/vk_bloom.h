// V20 Vulkan Bloom Post-Processing
// ==================================
#pragma once

#include "vk_types.h"

namespace vk {

// Bloom resources
struct BloomResources {
    // HDR render target (particles render here first)
    VkImage hdrImage = VK_NULL_HANDLE;
    VkDeviceMemory hdrImageMemory = VK_NULL_HANDLE;
    VkImageView hdrImageView = VK_NULL_HANDLE;
    VkFramebuffer hdrFramebuffer = VK_NULL_HANDLE;

    // Bright pass extraction
    VkImage brightImage = VK_NULL_HANDLE;
    VkDeviceMemory brightImageMemory = VK_NULL_HANDLE;
    VkImageView brightImageView = VK_NULL_HANDLE;
    VkFramebuffer brightFramebuffer = VK_NULL_HANDLE;

    // Blur ping-pong buffers (half resolution)
    VkImage blurImages[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory blurImageMemory[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView blurImageViews[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFramebuffer blurFramebuffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Sampler for texture reads
    VkSampler sampler = VK_NULL_HANDLE;

    // Render passes
    VkRenderPass hdrRenderPass = VK_NULL_HANDLE;
    VkRenderPass postRenderPass = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline extractPipeline = VK_NULL_HANDLE;
    VkPipeline blurPipeline = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;

    // Descriptor sets
    VkDescriptorSetLayout postDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool postDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet extractDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet blurDescriptorSets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet compositeDescriptorSet = VK_NULL_HANDLE;

    // Dimensions
    uint32_t width = 0;
    uint32_t height = 0;

    // Bloom parameters
    float bloomStrength = 0.8f;
    float exposure = 1.2f;
    float gamma = 2.2f;
    int blurPasses = 5;
};

// Functions
void createBloomResources(VulkanContext& ctx, BloomResources& bloom);
void cleanupBloomResources(VulkanContext& ctx, BloomResources& bloom);
void recreateBloomResources(VulkanContext& ctx, BloomResources& bloom);
void recordBloomCommands(VulkanContext& ctx, BloomResources& bloom,
                         VkCommandBuffer cmd, uint32_t imageIndex);

} // namespace vk
