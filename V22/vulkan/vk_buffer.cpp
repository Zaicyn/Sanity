// V20 Vulkan Buffer Management
// =============================
#include "vk_types.h"
#include "vk_attractor.h"

// External attractor pipeline (defined in blackhole_v20.cu for VULKAN_INTEROP build)
extern AttractorPipeline g_attractor;

namespace vk {

uint32_t findMemoryType(VulkanContext& ctx, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

void createBuffer(VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }

    vkBindBufferMemory(ctx.device, buffer, memory, 0);
}

void createCommandPool(VulkanContext& ctx) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamilies.graphicsFamily.value();

    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void createCommandBuffers(VulkanContext& ctx) {
    ctx.commandBuffers.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(ctx.commandBuffers.size());

    if (vkAllocateCommandBuffers(ctx.device, &allocInfo, ctx.commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void createSyncObjects(VulkanContext& ctx) {
    // Per-frame sync objects (for frame pacing)
    ctx.imageAvailableSemaphores.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    ctx.inFlightFences.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);

    // Per-swapchain-image semaphores (indexed by imageIndex, not currentFrame)
    // This prevents semaphore reuse conflicts when imageIndex != currentFrame
    uint32_t imageCount = static_cast<uint32_t>(ctx.swapchainImages.size());
    ctx.renderFinishedSemaphores.resize(imageCount);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first frame doesn't wait

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &ctx.imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create per-frame sync objects");
        }
    }

    // Create per-swapchain-image semaphores
    for (size_t i = 0; i < imageCount; i++) {
        if (vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &ctx.renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create per-image semaphore");
        }
    }
}

void createUniformBuffers(VulkanContext& ctx) {
    VkDeviceSize bufferSize = sizeof(GlobalUBO);

    ctx.uniformBuffers.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    ctx.uniformBuffersMemory.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    ctx.uniformBuffersMapped.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(ctx, bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ctx.uniformBuffers[i], ctx.uniformBuffersMemory[i]);

        vkMapMemory(ctx.device, ctx.uniformBuffersMemory[i], 0, bufferSize, 0, &ctx.uniformBuffersMapped[i]);
    }
}

void createDescriptorPool(VulkanContext& ctx) {
    // Need descriptors for:
    // - Particle pass: 1 UBO per frame (GlobalUBO) = MAX_FRAMES_IN_FLIGHT
    // - Volume pass: 2 UBOs per frame (GlobalUBO + VolumeUBO) = 2 * MAX_FRAMES_IN_FLIGHT
    // Total UBO descriptors: 3 * MAX_FRAMES_IN_FLIGHT
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(3 * VulkanContext::MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    // 2 descriptor sets per frame: particle + volume
    poolInfo.maxSets = static_cast<uint32_t>(2 * VulkanContext::MAX_FRAMES_IN_FLIGHT);

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &ctx.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void createDescriptorSets(VulkanContext& ctx) {
    std::vector<VkDescriptorSetLayout> layouts(VulkanContext::MAX_FRAMES_IN_FLIGHT, ctx.descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx.descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    ctx.descriptorSets.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, ctx.descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = ctx.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(GlobalUBO);

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = ctx.descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);
    }
}

void createVolumeUniformBuffers(VulkanContext& ctx) {
    VkDeviceSize bufferSize = sizeof(VolumeUBO);

    ctx.volumeUniformBuffers.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    ctx.volumeUniformBuffersMemory.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    ctx.volumeUniformBuffersMapped.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(ctx, bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ctx.volumeUniformBuffers[i], ctx.volumeUniformBuffersMemory[i]);

        vkMapMemory(ctx.device, ctx.volumeUniformBuffersMemory[i], 0, bufferSize, 0, &ctx.volumeUniformBuffersMapped[i]);
    }

    std::cout << "[V20] Volume uniform buffers created" << std::endl;
}

void createVolumeDescriptorSets(VulkanContext& ctx) {
    if (ctx.volumeDescriptorSetLayout == VK_NULL_HANDLE) {
        std::cerr << "[V20] Warning: Volume descriptor set layout not created, skipping descriptor sets" << std::endl;
        return;
    }

    std::vector<VkDescriptorSetLayout> layouts(VulkanContext::MAX_FRAMES_IN_FLIGHT, ctx.volumeDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx.descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    ctx.volumeDescriptorSets.resize(VulkanContext::MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, ctx.volumeDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate volume descriptor sets");
    }

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        // Binding 0: GlobalUBO
        VkDescriptorBufferInfo globalBufferInfo = {};
        globalBufferInfo.buffer = ctx.uniformBuffers[i];
        globalBufferInfo.offset = 0;
        globalBufferInfo.range = sizeof(GlobalUBO);

        // Binding 1: VolumeUBO
        VkDescriptorBufferInfo volumeBufferInfo = {};
        volumeBufferInfo.buffer = ctx.volumeUniformBuffers[i];
        volumeBufferInfo.offset = 0;
        volumeBufferInfo.range = sizeof(VolumeUBO);

        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = ctx.volumeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &globalBufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = ctx.volumeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &volumeBufferInfo;

        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(descriptorWrites.size()),
                               descriptorWrites.data(), 0, nullptr);
    }

    std::cout << "[V20] Volume descriptor sets created" << std::endl;
}

void recordCommandBuffer(VulkanContext& ctx, VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    // === ATTRACTOR MODE: Compute-based density accumulation + tone mapping ===
    // Two sub-modes: position-primary (harmonic_sample.comp) or phase-primary (harmonic_phase.comp)
    // Toggle with 'P' key
    if (g_attractor.enabled) {
        // Record attractor compute pass (before render pass)
        vk::recordAttractorCommands(ctx, g_attractor, commandBuffer, imageIndex);

        // Begin render pass for tone mapping
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = ctx.renderPass;
        renderPassInfo.framebuffer = ctx.framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = ctx.swapchainExtent;

        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};  // Black background for attractor
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Set dynamic viewport and scissor
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(ctx.swapchainExtent.width);
        viewport.height = static_cast<float>(ctx.swapchainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = ctx.swapchainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Tone mapping pass (fullscreen triangle)
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_attractor.graphicsPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            g_attractor.graphicsPipelineLayout, 0, 1, &g_attractor.graphicsDescSet, 0, nullptr);

        AttractorTonePush tonePush = { g_attractor.peak_density, g_attractor.exposure, 2.2f, g_attractor.colormap };
        vkCmdPushConstants(commandBuffer, g_attractor.graphicsPipelineLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(AttractorTonePush), &tonePush);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
    }
}

void updateUniformBuffer(VulkanContext& ctx, uint32_t currentImage) {
    static float time = 0.0f;
    time += 0.016f;  // ~60 FPS increment

    GlobalUBO ubo = {};

    // Camera position from mouse-controlled orbit
    float camX = ctx.cameraRadius * cosf(ctx.cameraPitch) * cosf(ctx.cameraYaw);
    float camY = ctx.cameraRadius * sinf(ctx.cameraPitch);
    float camZ = ctx.cameraRadius * cosf(ctx.cameraPitch) * sinf(ctx.cameraYaw);

    ubo.cameraPos[0] = camX;
    ubo.cameraPos[1] = camY;
    ubo.cameraPos[2] = camZ;
    ubo.time = time;

    // Simple view-projection matrix (look at origin)
    // For a real implementation, use a proper matrix library
    float aspect = (float)ctx.swapchainExtent.width / (float)ctx.swapchainExtent.height;
    float fov = 45.0f * 3.14159f / 180.0f;
    float near = 0.1f;
    float far = 50000.0f;

    // Perspective projection (column-major for Vulkan/GLSL)
    float tanHalfFov = tanf(fov / 2.0f);
    float proj[16] = {0};
    proj[0] = 1.0f / (aspect * tanHalfFov);  // [0][0]
    proj[5] = -1.0f / tanHalfFov;            // [1][1] - negative for Vulkan Y-flip
    proj[10] = far / (near - far);           // [2][2]
    proj[11] = -1.0f;                        // [2][3]
    proj[14] = (near * far) / (near - far);  // [3][2]

    // Look-at view matrix (column-major)
    float lookX = 0.0f, lookY = 0.0f, lookZ = 0.0f;

    // Forward vector (from camera to target)
    float fwdX = lookX - camX, fwdY = lookY - camY, fwdZ = lookZ - camZ;
    float fwdLen = sqrtf(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
    fwdX /= fwdLen; fwdY /= fwdLen; fwdZ /= fwdLen;

    // Right vector (cross product of forward and world up)
    float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
    float rightX = fwdY * upZ - fwdZ * upY;
    float rightY = fwdZ * upX - fwdX * upZ;
    float rightZ = fwdX * upY - fwdY * upX;
    float rightLen = sqrtf(rightX*rightX + rightY*rightY + rightZ*rightZ);
    rightX /= rightLen; rightY /= rightLen; rightZ /= rightLen;

    // Recompute up vector
    float newUpX = rightY * fwdZ - rightZ * fwdY;
    float newUpY = rightZ * fwdX - rightX * fwdZ;
    float newUpZ = rightX * fwdY - rightY * fwdX;

    // View matrix (column-major for GLSL)
    // Column 0: right vector
    // Column 1: up vector
    // Column 2: -forward vector (looking down -Z)
    // Column 3: translation (dot products)
    float dotRight = -(rightX*camX + rightY*camY + rightZ*camZ);
    float dotUp = -(newUpX*camX + newUpY*camY + newUpZ*camZ);
    float dotFwd = (fwdX*camX + fwdY*camY + fwdZ*camZ);

    float view[16] = {
        rightX,   newUpX,   -fwdX,   0.0f,   // Column 0
        rightY,   newUpY,   -fwdY,   0.0f,   // Column 1
        rightZ,   newUpZ,   -fwdZ,   0.0f,   // Column 2
        dotRight, dotUp,    dotFwd,  1.0f    // Column 3
    };

    // Multiply proj * view -> viewProj (column-major, for gl_Position = MVP * pos)
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += proj[k * 4 + row] * view[col * 4 + k];
            }
            ubo.viewProj[col * 4 + row] = sum;
        }
    }

    // Pump metrics (placeholder values - CUDA will update these via context)
    ubo.avgScale = 1.15f;
    ubo.avgResidual = 0.1f;
    ubo.heartbeat = sinf(time * 2.0f * 3.14159f);  // 1 Hz pulse
    ubo.pump_phase = time * 0.125f * 2.0f * 3.14159f;  // ω_pump = 0.125, 8-frame period

    memcpy(ctx.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));

    // NOTE: Attractor camera is updated in blackhole_v20.cu (line ~9310)
    // Do NOT update it here - the two camera calculations use different trig
    // functions (sin/cos swapped), causing a ghost/double-image effect.

    // Update VolumeUBO if volume rendering is enabled
    if (ctx.enableVolumePass && !ctx.volumeUniformBuffersMapped.empty()) {
        VolumeUBO volUbo = {};

        // Compute inverse view-projection for ray unprojection
        // viewProj is column-major, so we need to invert it
        // For simplicity, we compute the inverse matrices separately and multiply
        // inv(proj * view) = inv(view) * inv(proj)

        // Inverse projection (for our simple perspective matrix)
        float invProj[16] = {0};
        invProj[0] = aspect * tanHalfFov;
        invProj[5] = -tanHalfFov;  // Negative to undo Vulkan Y-flip
        invProj[11] = (near - far) / (near * far);
        invProj[14] = -1.0f;
        invProj[15] = 1.0f / near;

        // Inverse view matrix (transpose of rotation, negated translation)
        float invView[16] = {
            rightX,   rightY,   rightZ,   0.0f,   // Row 0 of view rotation = Column 0 of inverse
            newUpX,   newUpY,   newUpZ,   0.0f,   // Row 1
            -fwdX,    -fwdY,    -fwdZ,    0.0f,   // Row 2
            camX,     camY,     camZ,     1.0f    // Camera position
        };

        // Multiply inv(view) * inv(proj) -> invViewProj
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += invView[k * 4 + row] * invProj[col * 4 + k];
                }
                volUbo.invViewProj[col * 4 + row] = sum;
            }
        }

        // Volume bounds for the disk (centered at origin, flattened Y)
        volUbo.volumeMin[0] = -200.0f;
        volUbo.volumeMin[1] = -20.0f;
        volUbo.volumeMin[2] = -200.0f;
        volUbo.volumeScale = 400.0f;
        volUbo.volumeMax[0] = 200.0f;
        volUbo.volumeMax[1] = 20.0f;
        volUbo.volumeMax[2] = 200.0f;
        volUbo.seam_angle = 0.0f;  // Seam orientation - could be locked from CUDA
        volUbo.shellBrightness = ctx.shellBrightness;  // V key cycles brightness
        volUbo.padding[0] = 0.0f;
        volUbo.padding[1] = 0.0f;
        volUbo.padding[2] = 0.0f;

        memcpy(ctx.volumeUniformBuffersMapped[currentImage], &volUbo, sizeof(volUbo));
    }
}

void drawFrame(VulkanContext& ctx) {
    vkWaitForFences(ctx.device, 1, &ctx.inFlightFences[ctx.currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
        ctx.imageAvailableSemaphores[ctx.currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain(ctx);
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    vkResetFences(ctx.device, 1, &ctx.inFlightFences[ctx.currentFrame]);

    vkResetCommandBuffer(ctx.commandBuffers[ctx.currentFrame], 0);
    recordCommandBuffer(ctx, ctx.commandBuffers[ctx.currentFrame], imageIndex);

    updateUniformBuffer(ctx, ctx.currentFrame);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {ctx.imageAvailableSemaphores[ctx.currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx.commandBuffers[ctx.currentFrame];

    // Use per-swapchain-image semaphore indexed by imageIndex (not currentFrame)
    // This avoids signaling a semaphore that may still be in use by presentation
    VkSemaphore signalSemaphores[] = {ctx.renderFinishedSemaphores[imageIndex]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, ctx.inFlightFences[ctx.currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = {ctx.swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(ctx.presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || ctx.framebufferResized) {
        recreateSwapchain(ctx);
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swapchain image");
    }

    ctx.currentFrame = (ctx.currentFrame + 1) % VulkanContext::MAX_FRAMES_IN_FLIGHT;
}

void cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    cleanupSwapchain(ctx);

    vkDestroyDescriptorPool(ctx.device, ctx.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptorSetLayout, nullptr);

    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(ctx.device, ctx.uniformBuffers[i], nullptr);
        vkFreeMemory(ctx.device, ctx.uniformBuffersMemory[i], nullptr);
    }

    if (ctx.particleBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, ctx.particleBuffer, nullptr);
        vkFreeMemory(ctx.device, ctx.particleBufferMemory, nullptr);
    }

    // Destroy volume rendering resources
    for (size_t i = 0; i < ctx.volumeUniformBuffers.size(); i++) {
        vkDestroyBuffer(ctx.device, ctx.volumeUniformBuffers[i], nullptr);
        vkFreeMemory(ctx.device, ctx.volumeUniformBuffersMemory[i], nullptr);
    }

    if (ctx.volumePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, ctx.volumePipeline, nullptr);
    }
    if (ctx.volumePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, ctx.volumePipelineLayout, nullptr);
    }
    if (ctx.volumeDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, ctx.volumeDescriptorSetLayout, nullptr);
    }

    vkDestroyPipeline(ctx.device, ctx.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, ctx.pipelineLayout, nullptr);
    vkDestroyRenderPass(ctx.device, ctx.renderPass, nullptr);

    // Destroy per-frame sync objects
    for (size_t i = 0; i < VulkanContext::MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(ctx.device, ctx.imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(ctx.device, ctx.inFlightFences[i], nullptr);
    }

    // Destroy per-swapchain-image semaphores
    for (size_t i = 0; i < ctx.renderFinishedSemaphores.size(); i++) {
        vkDestroySemaphore(ctx.device, ctx.renderFinishedSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);

    if (ctx.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr);
    }

    destroyInstance(ctx);

    if (ctx.window != nullptr) {
        glfwDestroyWindow(ctx.window);
        glfwTerminate();
    }
}

} // namespace vk
