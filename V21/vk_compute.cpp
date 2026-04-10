/*
 * V21 VULKAN COMPUTE — GPU Physics Implementation
 * ================================================
 *
 * Follows the pattern from vk_attractor.cpp:
 *   Load .spv → create pipeline → bind SSBOs → dispatch → barrier
 */

#include "vk_compute.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <stdexcept>

/* ========================================================================
 * HELPERS (reuse from vk_attractor.cpp pattern)
 * ======================================================================== */

static std::vector<char> readShaderFile(const std::string& filename) {
    std::vector<std::string> paths = {
        "shaders/compute/", "kernels/", "../kernels/",
        "../../V21/kernels/", "shaders/"
    };
    for (auto& p : paths) {
        std::ifstream file(p + filename, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t sz = (size_t)file.tellg();
            std::vector<char> buffer(sz);
            file.seekg(0);
            file.read(buffer.data(), sz);
            return buffer;
        }
    }
    throw std::runtime_error("Shader not found: " + filename);
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
    return mod;
}

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return 0;
}

static void createSSBO(VkDevice dev, VkPhysicalDevice pd,
                       VkBuffer& buf, VkDeviceMemory& mem, size_t size) {
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &bi, nullptr, &buf);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);

    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(dev, &ai, nullptr, &mem);
    vkBindBufferMemory(dev, buf, mem, 0);
}

static void uploadToSSBO(VulkanContext& ctx, VkBuffer dst, const void* src, size_t size) {
    /* Create temp staging buffer */
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(ctx.device, &bi, nullptr, &staging);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx.device, staging, &mr);
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemType(ctx.physicalDevice, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMem);
    vkBindBufferMemory(ctx.device, staging, stagingMem, 0);

    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, size, 0, &mapped);
    memcpy(mapped, src, size);
    vkUnmapMemory(ctx.device, stagingMem);

    /* Copy staging → device */
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba = {};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = ctx.commandPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cba, &cmd);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkBufferCopy copy = {0, 0, size};
    vkCmdCopyBuffer(cmd, staging, dst, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue);

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
}

/* ========================================================================
 * INIT — create SSBOs, pipeline, descriptors
 * ======================================================================== */

void initPhysicsCompute(PhysicsCompute& phys, VulkanContext& ctx,
                         const float* pos_x, const float* pos_y, const float* pos_z,
                         const float* vel_x, const float* vel_y, const float* vel_z,
                         const float* pump_scale, const float* pump_residual,
                         const float* pump_history, const int* pump_state,
                         const float* theta, const float* omega_nat,
                         const uint8_t* flags, const uint8_t* topo_state,
                         int N) {
    phys.N = N;
    size_t float_sz = N * sizeof(float);
    size_t int_sz = N * sizeof(int);
    size_t uint_sz = N * sizeof(uint32_t);

    /* Allocate SSBOs — all float-sized except flags/active_region which are uint-padded */
    size_t sizes[VK_COMPUTE_NUM_BINDINGS] = {
        float_sz, float_sz, float_sz,  /* pos x,y,z */
        float_sz, float_sz, float_sz,  /* vel x,y,z */
        int_sz,                         /* pump_state */
        float_sz,                       /* pump_scale */
        int_sz,                         /* pump_coherent */
        float_sz,                       /* pump_residual */
        float_sz,                       /* pump_work */
        float_sz,                       /* pump_history */
        float_sz,                       /* theta */
        float_sz,                       /* omega_nat */
        uint_sz,                        /* flags (uint32 padded) */
        uint_sz                         /* in_active_region (uint32 padded) */
    };

    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        phys.soa_sizes[i] = sizes[i];
        createSSBO(ctx.device, ctx.physicalDevice, phys.soa_buffers[i], phys.soa_memory[i], sizes[i]);
    }

    /* Upload initial state */
    uploadToSSBO(ctx, phys.soa_buffers[0], pos_x, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[1], pos_y, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[2], pos_z, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[3], vel_x, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[4], vel_y, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[5], vel_z, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[7], pump_scale, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[9], pump_residual, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[11], pump_history, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[12], theta, float_sz);
    uploadToSSBO(ctx, phys.soa_buffers[13], omega_nat, float_sz);

    /* Pad uint8 flags to uint32 for SSBO */
    std::vector<uint32_t> flags32(N), active32(N, 1);  /* all active initially */
    for (int i = 0; i < N; i++) flags32[i] = flags[i];
    uploadToSSBO(ctx, phys.soa_buffers[14], flags32.data(), uint_sz);
    uploadToSSBO(ctx, phys.soa_buffers[15], active32.data(), uint_sz);

    printf("[vk-compute] Uploaded %d particles to %d SSBOs (%.1f MB total)\n",
           N, VK_COMPUTE_NUM_BINDINGS,
           (float)(16 * float_sz) / (1024 * 1024));

    /* Create descriptor set layout — 16 SSBO bindings */
    VkDescriptorSetLayoutBinding bindings[VK_COMPUTE_NUM_BINDINGS] = {};
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = VK_COMPUTE_NUM_BINDINGS;
    layoutInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &phys.descLayout);

    /* Pipeline layout with push constants */
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(SiphonPushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &phys.descLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.pipelineLayout);

    /* Load siphon.spv and create compute pipeline */
    auto code = readShaderFile("siphon.spv");
    VkShaderModule shaderMod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = shaderMod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.pipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &phys.pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create siphon compute pipeline");

    vkDestroyShaderModule(ctx.device, shaderMod, nullptr);
    printf("[vk-compute] Siphon compute pipeline created\n");

    /* Descriptor pool + set */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = VK_COMPUTE_NUM_BINDINGS;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.descPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.descPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.descLayout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.descSet);

    /* Write SSBO descriptors */
    VkDescriptorBufferInfo bufInfos[VK_COMPUTE_NUM_BINDINGS];
    VkWriteDescriptorSet writes[VK_COMPUTE_NUM_BINDINGS];
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        bufInfos[i] = {phys.soa_buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, VK_COMPUTE_NUM_BINDINGS, writes, 0, nullptr);

    /* Staging buffer for readback */
    size_t readback_sz = N * 6 * sizeof(float);  /* pos + vel */
    VkBufferCreateInfo sbi = {};
    sbi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbi.size = readback_sz;
    sbi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(ctx.device, &sbi, nullptr, &phys.staging);

    VkMemoryRequirements smr;
    vkGetBufferMemoryRequirements(ctx.device, phys.staging, &smr);
    VkMemoryAllocateInfo sai = {};
    sai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    sai.allocationSize = smr.size;
    sai.memoryTypeIndex = findMemType(ctx.physicalDevice, smr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(ctx.device, &sai, nullptr, &phys.stagingMemory);
    vkBindBufferMemory(ctx.device, phys.staging, phys.stagingMemory, 0);
    vkMapMemory(ctx.device, phys.stagingMemory, 0, readback_sz, 0, &phys.stagingMapped);
    phys.stagingSize = readback_sz;

    phys.initialized = true;
    printf("[vk-compute] Physics compute ready (%d particles, %d SSBOs)\n", N, VK_COMPUTE_NUM_BINDINGS);
}

/* ========================================================================
 * DISPATCH — record compute commands
 * ======================================================================== */

void dispatchPhysicsCompute(PhysicsCompute& phys, VkCommandBuffer cmd,
                            int frame, float sim_time, float dt) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.pipelineLayout, 0, 1, &phys.descSet, 0, nullptr);

    SiphonPushConstants push = {};
    push.N = phys.N;
    push.time = sim_time;
    push.dt = dt;
    push.BH_MASS = 100.0f;
    push.FIELD_STRENGTH = 1.0f;
    push.FIELD_FALLOFF = 100.0f;
    push.TANGENT_SCALE = 0.5f;
    push.seam_bits = 0x03;  /* SEAM_FULL */
    push.bias = 0.75f;

    vkCmdPushConstants(cmd, phys.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Memory barrier: compute writes → vertex read / transfer read */
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

/* ========================================================================
 * READBACK — copy subset to CPU for oracle
 * ======================================================================== */

void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       int count) {
    size_t sz = count * sizeof(float);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba = {};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = ctx.commandPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cba, &cmd);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    /* Copy pos_x, pos_y, pos_z, vel_x, vel_y, vel_z to staging */
    VkBufferCopy region = {0, 0, sz};
    size_t offset = 0;
    for (int b = 0; b < 6; b++) {
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size = sz;
        vkCmdCopyBuffer(cmd, phys.soa_buffers[b], phys.staging, 1, &region);
        offset += sz;
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue);

    /* Copy from staging to CPU arrays */
    char* mapped = (char*)phys.stagingMapped;
    memcpy(out_pos_x, mapped + 0 * sz, sz);
    memcpy(out_pos_y, mapped + 1 * sz, sz);
    memcpy(out_pos_z, mapped + 2 * sz, sz);
    memcpy(out_vel_x, mapped + 3 * sz, sz);
    memcpy(out_vel_y, mapped + 4 * sz, sz);
    memcpy(out_vel_z, mapped + 5 * sz, sz);

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
}

/* ========================================================================
 * CLEANUP
 * ======================================================================== */

void cleanupPhysicsCompute(PhysicsCompute& phys, VkDevice device) {
    if (!phys.initialized) return;

    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        vkDestroyBuffer(device, phys.soa_buffers[i], nullptr);
        vkFreeMemory(device, phys.soa_memory[i], nullptr);
    }
    vkDestroyBuffer(device, phys.staging, nullptr);
    vkFreeMemory(device, phys.stagingMemory, nullptr);
    vkDestroyPipeline(device, phys.pipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.descLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.descPool, nullptr);

    phys.initialized = false;
    printf("[vk-compute] Physics compute cleaned up\n");
}
