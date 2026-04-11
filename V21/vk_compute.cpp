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
    /* V21 kernels first — "shaders/compute/" is a legacy V20 path that
     * may contain stale artifacts and must NOT take precedence. A stale
     * build/shaders/compute/siphon.spv previously caused shader edits to
     * be silently ignored at runtime; rooted out during the packing
     * experiment, see memory: feedback_shader_search_order.md */
    std::vector<std::string> paths = {
        "kernels/", "../kernels/", "../../V21/kernels/",
        "shaders/compute/", "shaders/",
        "../../vulkan/shaders/",   /* V20 tone-map SPVs */
        "../vulkan/shaders/"
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

    /* Device-local: GPU reads at full VRAM bandwidth, not PCIe. */
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

    /* Allocate device-local SSBOs */
    size_t total_sz = 0;
    size_t offsets[VK_COMPUTE_NUM_BINDINGS];
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        phys.soa_sizes[i] = sizes[i];
        offsets[i] = total_sz;
        total_sz += sizes[i];
        createSSBO(ctx.device, ctx.physicalDevice, phys.soa_buffers[i],
                   phys.soa_memory[i], sizes[i]);
    }

    /* Single large host-visible staging buffer — batched upload */
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    {
        VkBufferCreateInfo ubi = {};
        ubi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ubi.size = total_sz;
        ubi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(ctx.device, &ubi, nullptr, &stagingBuf);

        VkMemoryRequirements umr;
        vkGetBufferMemoryRequirements(ctx.device, stagingBuf, &umr);
        VkMemoryAllocateInfo uai = {};
        uai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        uai.allocationSize = umr.size;
        uai.memoryTypeIndex = findMemType(ctx.physicalDevice, umr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(ctx.device, &uai, nullptr, &stagingMem);
        vkBindBufferMemory(ctx.device, stagingBuf, stagingMem, 0);
    }

    /* Map and fill staging in one shot */
    char* staging_base;
    vkMapMemory(ctx.device, stagingMem, 0, total_sz, 0, (void**)&staging_base);
    memset(staging_base, 0, total_sz);  /* Zero-init all */

    memcpy(staging_base + offsets[0], pos_x, float_sz);
    memcpy(staging_base + offsets[1], pos_y, float_sz);
    memcpy(staging_base + offsets[2], pos_z, float_sz);
    memcpy(staging_base + offsets[3], vel_x, float_sz);
    memcpy(staging_base + offsets[4], vel_y, float_sz);
    memcpy(staging_base + offsets[5], vel_z, float_sz);
    memcpy(staging_base + offsets[7], pump_scale, float_sz);
    memcpy(staging_base + offsets[9], pump_residual, float_sz);
    memcpy(staging_base + offsets[11], pump_history, float_sz);
    memcpy(staging_base + offsets[12], theta, float_sz);
    memcpy(staging_base + offsets[13], omega_nat, float_sz);

    /* Pad uint8 flags to uint32 for SSBO */
    uint32_t* flags_s = (uint32_t*)(staging_base + offsets[14]);
    uint32_t* active_s = (uint32_t*)(staging_base + offsets[15]);
    for (int i = 0; i < N; i++) {
        flags_s[i] = flags[i];
        active_s[i] = 1;  /* All active initially */
    }
    vkUnmapMemory(ctx.device, stagingMem);

    /* Single command buffer: 16 copies + one wait */
    VkCommandBuffer ucmd;
    VkCommandBufferAllocateInfo cba = {};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = ctx.commandPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cba, &ucmd);

    VkCommandBufferBeginInfo cbegin = {};
    cbegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ucmd, &cbegin);

    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        VkBufferCopy copy = {offsets[i], 0, sizes[i]};
        vkCmdCopyBuffer(ucmd, stagingBuf, phys.soa_buffers[i], 1, &copy);
    }
    vkEndCommandBuffer(ucmd);

    VkSubmitInfo usi = {};
    usi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    usi.commandBufferCount = 1;
    usi.pCommandBuffers = &ucmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &usi, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue);

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &ucmd);
    vkDestroyBuffer(ctx.device, stagingBuf, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    printf("[vk-compute] Uploaded %d particles to %d device-local SSBOs (%.1f MB total)\n",
           N, VK_COMPUTE_NUM_BINDINGS, (float)total_sz / (1024 * 1024));

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

    /* Staging buffer for readback — capped at ORACLE_SUBSET_SIZE × 9 floats.
     * Layout: [pos_x][pos_y][pos_z][vel_x][vel_y][vel_z][theta][pump_scale][flags]
     * 9 arrays × 100K × 4 bytes = 3.6 MB. Stays small even at 80M particles. */
    int subset_cap = N < ORACLE_SUBSET_SIZE ? N : ORACLE_SUBSET_SIZE;
    size_t readback_sz = (size_t)subset_cap * 9 * sizeof(float);
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

    /* --- GPU timestamp query pool (double-buffered, 4 timestamps per slot) --- */
    VkPhysicalDeviceProperties pdProps;
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &pdProps);
    phys.timestampPeriodNs = pdProps.limits.timestampPeriod;

    VkQueryPoolCreateInfo qpInfo = {};
    qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = 10;  /* 2 frames × 5 timestamps (begin, scatter_end, siphon_end, project_end, tonemap_end) */
    if (vkCreateQueryPool(ctx.device, &qpInfo, nullptr, &phys.queryPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create timestamp query pool");

    phys.queryValid[0] = false;
    phys.queryValid[1] = false;
    phys.queryFrame = 0;

    phys.initialized = true;
    printf("[vk-compute] Physics compute ready (%d particles, %d SSBOs, timestamp period %.2f ns)\n",
           N, VK_COMPUTE_NUM_BINDINGS, phys.timestampPeriodNs);

    /* Pass 1 scatter pipeline (particles → cell grid) */
    initScatterCompute(phys, ctx);
}

/* ========================================================================
 * SCATTER (Pass 1) — particles → cell grid
 * ======================================================================== */

void initScatterCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Allocate grid density buffer (uint[V21_GRID_CELLS]) --- */
    size_t grid_bytes = (size_t)V21_GRID_CELLS * sizeof(uint32_t);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.gridDensityBuffer, phys.gridDensityMemory, grid_bytes);

    /* --- Allocate particle_cell buffer (uint[N]) --- */
    size_t pcell_bytes = (size_t)phys.N * sizeof(uint32_t);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.particleCellBuffer, phys.particleCellMemory, pcell_bytes);

    printf("[vk-compute] Scatter grid: %d cells (%.1f MB) + particle_cell (%.1f MB)\n",
           V21_GRID_CELLS,
           (double)grid_bytes / (1024.0 * 1024.0),
           (double)pcell_bytes / (1024.0 * 1024.0));

    /* --- Descriptor set layout for set 1 (grid buffers only) --- */
    VkDescriptorSetLayoutBinding set1Bindings[2] = {};
    for (int i = 0; i < 2; i++) {
        set1Bindings[i].binding = i;
        set1Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set1Bindings[i].descriptorCount = 1;
        set1Bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set1Info = {};
    set1Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set1Info.bindingCount = 2;
    set1Info.pBindings = set1Bindings;
    vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &phys.scatterSet1Layout);

    /* --- Pipeline layout: set 0 = siphon's existing layout, set 1 = grid --- */
    VkDescriptorSetLayout layouts[2] = { phys.descLayout, phys.scatterSet1Layout };
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(ScatterPushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 2;
    plInfo.pSetLayouts = layouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.scatterPipelineLayout);

    /* --- Load scatter.spv and create compute pipeline --- */
    auto code = readShaderFile("scatter.spv");
    VkShaderModule mod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.scatterPipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                  nullptr, &phys.scatterPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create scatter compute pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);

    /* --- Descriptor pool + set for set 1 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.scatterDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.scatterDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.scatterSet1Layout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.scatterSet1);

    /* --- Write descriptors for set 1 --- */
    VkDescriptorBufferInfo bufInfos[2] = {
        { phys.gridDensityBuffer,  0, VK_WHOLE_SIZE },
        { phys.particleCellBuffer, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.scatterSet1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    printf("[vk-compute] Scatter compute pipeline created\n");
}

/* ========================================================================
 * DISPATCH — record compute commands
 * ======================================================================== */

void dispatchPhysicsCompute(PhysicsCompute& phys, VkCommandBuffer cmd,
                            int frame, float sim_time, float dt) {
    /* Reset this slot's 5 timestamps and write the begin marker */
    uint32_t base = phys.queryFrame * 5;
    vkCmdResetQueryPool(cmd, phys.queryPool, base, 5);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        phys.queryPool, base + 0);

    /* ---- Pass 1: Scatter (particles → cell grid) ---------------------- */
    /* Zero the density buffer before scatter accumulates into it. */
    vkCmdFillBuffer(cmd, phys.gridDensityBuffer, 0,
                    (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t), 0);

    /* Barrier: fill must complete before scatter reads/writes density. */
    {
        VkMemoryBarrier fillBarrier = {};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.scatterPipeline);
    VkDescriptorSet scatterSets[2] = { phys.descSet, phys.scatterSet1 };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.scatterPipelineLayout, 0, 2, scatterSets, 0, nullptr);

    ScatterPushConstants scatterPush = {};
    scatterPush.N              = phys.N;
    scatterPush.grid_dim       = V21_GRID_DIM;
    scatterPush.grid_cell_size = V21_GRID_CELL_SIZE;
    scatterPush.grid_half_size = V21_GRID_HALF_SIZE;
    vkCmdPushConstants(cmd, phys.scatterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(scatterPush), &scatterPush);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Barrier: scatter's writes to grid buffers are not read by siphon, but
     * they will be read by any future Pass 2/3 shaders. For now this is just
     * a safety fence; cost is negligible. */
    {
        VkMemoryBarrier scatterBarrier = {};
        scatterBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        scatterBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        scatterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &scatterBarrier, 0, nullptr, 0, nullptr);
    }

    /* Scatter end — record after the drain-barrier */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 1);

    /* ---- Siphon (physics) -------------------------------------------- */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.pipelineLayout, 0, 1, &phys.descSet, 0, nullptr);

    SiphonPushConstants push = {};
    push.N = phys.N;
    push.time = sim_time;
    push.dt = dt;
    push.BH_MASS = 1.0f;
    push.FIELD_STRENGTH = 0.01f;
    push.FIELD_FALLOFF = 100.0f;
    push.TANGENT_SCALE = 2.0f;
    push.seam_bits = 0x03;  /* SEAM_FULL */
    push.bias = 0.75f;

    vkCmdPushConstants(cmd, phys.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Memory barrier: siphon's SSBO writes must be visible to:
     *   - the subsequent projection compute pass (shader read)
     *   - transfer reads (oracle readback copies)
     * Compute→compute is critical; without it the driver may reorder or
     * even eliminate the siphon dispatch when the next consumer only sees
     * transfer/vertex reads. */
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    /* Siphon end — after the write-visibility barrier drains */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 2);
}

/* ========================================================================
 * DENSITY RENDERING — compute projection + tone-map (no rasterizer)
 * ======================================================================== */

void initDensityRender(PhysicsCompute& phys, VulkanContext& ctx) {
    uint32_t w = ctx.swapchainExtent.width;
    uint32_t h = ctx.swapchainExtent.height;

    /* --- Density image (R32_UINT) --- */
    VkImageCreateInfo imgInfo = {};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R32_UINT;
    imgInfo.extent = {w, h, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(ctx.device, &imgInfo, nullptr, &phys.densityImage);

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(ctx.device, phys.densityImage, &mr);
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemType(ctx.physicalDevice, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(ctx.device, &ai, nullptr, &phys.densityMemory);
    vkBindImageMemory(ctx.device, phys.densityImage, phys.densityMemory, 0);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = phys.densityImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32_UINT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(ctx.device, &viewInfo, nullptr, &phys.densityView);

    VkSamplerCreateInfo sampInfo = {};
    sampInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampInfo.magFilter = VK_FILTER_NEAREST;
    sampInfo.minFilter = VK_FILTER_NEAREST;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(ctx.device, &sampInfo, nullptr, &phys.densitySampler);

    /* --- Camera UBO --- */
    VkBufferCreateInfo uboBuf = {};
    uboBuf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    uboBuf.size = sizeof(ProjectCameraUBO);
    uboBuf.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vkCreateBuffer(ctx.device, &uboBuf, nullptr, &phys.cameraUBO);

    VkMemoryRequirements uboMr;
    vkGetBufferMemoryRequirements(ctx.device, phys.cameraUBO, &uboMr);
    VkMemoryAllocateInfo uboAi = {};
    uboAi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    uboAi.allocationSize = uboMr.size;
    uboAi.memoryTypeIndex = findMemType(ctx.physicalDevice, uboMr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(ctx.device, &uboAi, nullptr, &phys.cameraMemory);
    vkBindBufferMemory(ctx.device, phys.cameraUBO, phys.cameraMemory, 0);
    vkMapMemory(ctx.device, phys.cameraMemory, 0, sizeof(ProjectCameraUBO), 0, &phys.cameraMapped);

    /* --- Projection compute pipeline --- */
    /* Descriptor layout: 16 SoA SSBOs (bindings 0-15) + density image (16) + camera UBO (17) */
    VkDescriptorSetLayoutBinding projBindings[18] = {};
    for (int i = 0; i < 16; i++) {
        projBindings[i].binding = i;
        projBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        projBindings[i].descriptorCount = 1;
        projBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    projBindings[16].binding = 16;
    projBindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    projBindings[16].descriptorCount = 1;
    projBindings[16].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    projBindings[17].binding = 17;
    projBindings[17].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    projBindings[17].descriptorCount = 1;
    projBindings[17].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo projLayoutInfo = {};
    projLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    projLayoutInfo.bindingCount = 18;
    projLayoutInfo.pBindings = projBindings;
    vkCreateDescriptorSetLayout(ctx.device, &projLayoutInfo, nullptr, &phys.projDescLayout);

    VkPushConstantRange projPush = {};
    projPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    projPush.size = sizeof(ProjectPushConstants);

    VkPipelineLayoutCreateInfo projPlInfo = {};
    projPlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    projPlInfo.setLayoutCount = 1;
    projPlInfo.pSetLayouts = &phys.projDescLayout;
    projPlInfo.pushConstantRangeCount = 1;
    projPlInfo.pPushConstantRanges = &projPush;
    vkCreatePipelineLayout(ctx.device, &projPlInfo, nullptr, &phys.projPipelineLayout);

    auto projCode = readShaderFile("project.spv");
    VkShaderModule projMod = createShaderModule(ctx.device, projCode);

    VkComputePipelineCreateInfo projCpInfo = {};
    projCpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    projCpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    projCpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    projCpInfo.stage.module = projMod;
    projCpInfo.stage.pName = "main";
    projCpInfo.layout = phys.projPipelineLayout;
    vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &projCpInfo, nullptr, &phys.projPipeline);
    vkDestroyShaderModule(ctx.device, projMod, nullptr);

    /* Projection descriptor pool + set */
    VkDescriptorPoolSize projPoolSizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo projDpInfo = {};
    projDpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    projDpInfo.maxSets = 1;
    projDpInfo.poolSizeCount = 3;
    projDpInfo.pPoolSizes = projPoolSizes;
    vkCreateDescriptorPool(ctx.device, &projDpInfo, nullptr, &phys.projDescPool);

    VkDescriptorSetAllocateInfo projDsInfo = {};
    projDsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    projDsInfo.descriptorPool = phys.projDescPool;
    projDsInfo.descriptorSetCount = 1;
    projDsInfo.pSetLayouts = &phys.projDescLayout;
    vkAllocateDescriptorSets(ctx.device, &projDsInfo, &phys.projDescSet);

    /* Write projection descriptors */
    VkDescriptorBufferInfo soaBufInfos[16];
    VkWriteDescriptorSet projWrites[18] = {};
    for (int i = 0; i < 16; i++) {
        soaBufInfos[i] = {phys.soa_buffers[i], 0, VK_WHOLE_SIZE};
        projWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        projWrites[i].dstSet = phys.projDescSet;
        projWrites[i].dstBinding = i;
        projWrites[i].descriptorCount = 1;
        projWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        projWrites[i].pBufferInfo = &soaBufInfos[i];
    }

    VkDescriptorImageInfo densImgInfo = {VK_NULL_HANDLE, phys.densityView, VK_IMAGE_LAYOUT_GENERAL};
    projWrites[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    projWrites[16].dstSet = phys.projDescSet;
    projWrites[16].dstBinding = 16;
    projWrites[16].descriptorCount = 1;
    projWrites[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    projWrites[16].pImageInfo = &densImgInfo;

    VkDescriptorBufferInfo camBufInfo = {phys.cameraUBO, 0, sizeof(ProjectCameraUBO)};
    projWrites[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    projWrites[17].dstSet = phys.projDescSet;
    projWrites[17].dstBinding = 17;
    projWrites[17].descriptorCount = 1;
    projWrites[17].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    projWrites[17].pBufferInfo = &camBufInfo;

    vkUpdateDescriptorSets(ctx.device, 18, projWrites, 0, nullptr);

    /* --- Tone-map graphics pipeline --- */
    VkDescriptorSetLayoutBinding toneBindings[2] = {};
    toneBindings[0].binding = 0;
    toneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    toneBindings[0].descriptorCount = 1;
    toneBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    toneBindings[1].binding = 1;
    toneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    toneBindings[1].descriptorCount = 1;
    toneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo toneLayoutInfo = {};
    toneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    toneLayoutInfo.bindingCount = 2;
    toneLayoutInfo.pBindings = toneBindings;
    vkCreateDescriptorSetLayout(ctx.device, &toneLayoutInfo, nullptr, &phys.toneDescLayout);

    VkPushConstantRange tonePush = {};
    tonePush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    tonePush.size = sizeof(ToneMapPushConstants);

    VkPipelineLayoutCreateInfo tonePlInfo = {};
    tonePlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tonePlInfo.setLayoutCount = 1;
    tonePlInfo.pSetLayouts = &phys.toneDescLayout;
    tonePlInfo.pushConstantRangeCount = 1;
    tonePlInfo.pPushConstantRanges = &tonePush;
    vkCreatePipelineLayout(ctx.device, &tonePlInfo, nullptr, &phys.tonePipelineLayout);

    /* Load tone_map shaders from V20's precompiled SPVs */
    auto vertCode = readShaderFile("tone_map.vert.spv");
    auto fragCode = readShaderFile("tone_map.frag.spv");
    VkShaderModule vertMod = createShaderModule(ctx.device, vertCode);
    VkShaderModule fragMod = createShaderModule(ctx.device, fragCode);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    std::vector<VkDynamicState> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = (uint32_t)dynStates.size();
    dynamicState.pDynamicStates = dynStates.data();

    VkGraphicsPipelineCreateInfo gpInfo = {};
    gpInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpInfo.stageCount = 2;
    gpInfo.pStages = stages;
    gpInfo.pVertexInputState = &vertexInput;
    gpInfo.pInputAssemblyState = &inputAssembly;
    gpInfo.pViewportState = &viewportState;
    gpInfo.pRasterizationState = &rasterizer;
    gpInfo.pMultisampleState = &multisampling;
    gpInfo.pDepthStencilState = &depthStencil;
    gpInfo.pColorBlendState = &colorBlend;
    gpInfo.pDynamicState = &dynamicState;
    gpInfo.layout = phys.tonePipelineLayout;
    gpInfo.renderPass = ctx.renderPass;
    gpInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpInfo, nullptr, &phys.tonePipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create tone-map pipeline");

    vkDestroyShaderModule(ctx.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx.device, fragMod, nullptr);

    /* Tone-map descriptor pool + set */
    VkDescriptorPoolSize tonePoolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1}
    };
    VkDescriptorPoolCreateInfo toneDpInfo = {};
    toneDpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    toneDpInfo.maxSets = 1;
    toneDpInfo.poolSizeCount = 2;
    toneDpInfo.pPoolSizes = tonePoolSizes;
    vkCreateDescriptorPool(ctx.device, &toneDpInfo, nullptr, &phys.toneDescPool);

    VkDescriptorSetAllocateInfo toneDsInfo = {};
    toneDsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    toneDsInfo.descriptorPool = phys.toneDescPool;
    toneDsInfo.descriptorSetCount = 1;
    toneDsInfo.pSetLayouts = &phys.toneDescLayout;
    vkAllocateDescriptorSets(ctx.device, &toneDsInfo, &phys.toneDescSet);

    VkDescriptorImageInfo sampledInfo = {VK_NULL_HANDLE, phys.densityView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo samplerInfo = {phys.densitySampler, VK_NULL_HANDLE,
        VK_IMAGE_LAYOUT_UNDEFINED};

    VkWriteDescriptorSet toneWrites[2] = {};
    toneWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    toneWrites[0].dstSet = phys.toneDescSet;
    toneWrites[0].dstBinding = 0;
    toneWrites[0].descriptorCount = 1;
    toneWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    toneWrites[0].pImageInfo = &sampledInfo;

    toneWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    toneWrites[1].dstSet = phys.toneDescSet;
    toneWrites[1].dstBinding = 1;
    toneWrites[1].descriptorCount = 1;
    toneWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    toneWrites[1].pImageInfo = &samplerInfo;

    vkUpdateDescriptorSets(ctx.device, 2, toneWrites, 0, nullptr);

    printf("[vk-compute] Density render pipeline ready (%dx%d)\n", w, h);
}

/* Record density rendering: clear → physics barrier → project → barrier → tone-map */
void recordDensityRender(PhysicsCompute& phys, VkCommandBuffer cmd,
                         VulkanContext& ctx, uint32_t imageIndex,
                         const float* viewProj) {
    uint32_t w = ctx.swapchainExtent.width;
    uint32_t h = ctx.swapchainExtent.height;

    /* Update camera UBO */
    ProjectCameraUBO cam = {};
    memcpy(cam.view_proj, viewProj, 16 * sizeof(float));
    cam.zoom = 1.0f;
    cam.aspect = (float)w / (float)h;
    cam.width = (int)w;
    cam.height = (int)h;
    memcpy(phys.cameraMapped, &cam, sizeof(cam));

    /* Barrier 1: Undefined → General (for clear) */
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.image = phys.densityImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Clear density to zero */
    VkClearColorValue clearColor = {};
    clearColor.uint32[0] = 0;
    VkImageSubresourceRange clearRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, phys.densityImage, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &clearRange);

    /* Barrier 2: Clear → Compute write */
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Projection compute dispatch */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.projPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.projPipelineLayout, 0, 1, &phys.projDescSet, 0, nullptr);

    ProjectPushConstants projPush = {};
    projPush.particle_count = phys.N;
    projPush.brightness = 1.0f;
    projPush.temp_scale = 0.1f;
    vkCmdPushConstants(cmd, phys.projPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(projPush), &projPush);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Barrier 3: Compute write → Fragment read */
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Projection end — after compute work drains */
    {
        uint32_t base = phys.queryFrame * 5;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            phys.queryPool, base + 3);
    }

    /* Tone-map render pass */
    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = ctx.renderPass;
    rpInfo.framebuffer = ctx.framebuffers[imageIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = ctx.swapchainExtent;
    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    rpInfo.clearValueCount = 2;
    rpInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {};
    viewport.width = (float)w;
    viewport.height = (float)h;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, ctx.swapchainExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, phys.tonePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        phys.tonePipelineLayout, 0, 1, &phys.toneDescSet, 0, nullptr);

    ToneMapPushConstants tonePush = {};
    tonePush.peak_density = (float)phys.N * 0.001f;  /* Auto-scale with particle count */
    tonePush.exposure = 5.0f;
    tonePush.gamma = 2.2f;
    tonePush.colormap = 3;  /* hopf */
    vkCmdPushConstants(cmd, phys.tonePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(tonePush), &tonePush);

    vkCmdDraw(cmd, 3, 1, 0, 0);  /* Fullscreen triangle */

    vkCmdEndRenderPass(cmd);

    /* Tone-map end — after color output drains */
    {
        uint32_t base = phys.queryFrame * 5;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            phys.queryPool, base + 4);
    }

    /* Mark this slot as written so the next readback can consume it */
    phys.queryValid[phys.queryFrame] = true;
    phys.queryFrame ^= 1;
}

/* ========================================================================
 * TIMESTAMP READBACK — non-blocking
 * ======================================================================== */

bool readTimestamps(PhysicsCompute& phys, VkDevice device,
                    double* out_scatter_ms,
                    double* out_siphon_ms,
                    double* out_project_ms,
                    double* out_tonemap_ms) {
    /* Read from the slot we just finished writing (phys.queryFrame was flipped
     * at the end of recordDensityRender, so ^1 is the most recent write).
     * Use WITHOUT_WAIT so we don't block — if the GPU hasn't finished yet,
     * VK_NOT_READY is returned and we skip this report. */
    uint32_t slot = phys.queryFrame ^ 1;
    if (!phys.queryValid[slot]) return false;

    uint32_t base = slot * 5;
    uint64_t ts[5];
    VkResult r = vkGetQueryPoolResults(device, phys.queryPool, base, 5,
        sizeof(ts), ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return false;

    double nsPerTick = (double)phys.timestampPeriodNs;
    *out_scatter_ms = (double)(ts[1] - ts[0]) * nsPerTick / 1.0e6;
    *out_siphon_ms  = (double)(ts[2] - ts[1]) * nsPerTick / 1.0e6;
    *out_project_ms = (double)(ts[3] - ts[2]) * nsPerTick / 1.0e6;
    *out_tonemap_ms = (double)(ts[4] - ts[3]) * nsPerTick / 1.0e6;
    return true;
}

/* ========================================================================
 * READBACK — copy subset to CPU for oracle
 * ======================================================================== */

void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       float* out_theta, float* out_pump_scale,
                       uint8_t* out_flags,
                       int count) {
    if (count <= 0) return;
    if (count > ORACLE_SUBSET_SIZE) count = ORACLE_SUBSET_SIZE;

    size_t sz = (size_t)count * sizeof(float);

    /* Source bindings for the 9 values we copy (8 floats + 1 uint32 for flags).
     * Binding layout from siphon.comp:
     *   0..5 = pos_xyz, vel_xyz     6 = pump_state   7 = pump_scale
     *   8 = pump_coherent           9 = pump_residual  10 = pump_work
     *   11 = pump_history          12 = theta        13 = omega_nat
     *   14 = flags (uint32-padded) 15 = in_active_region */
    const int src_bindings[9] = {0, 1, 2, 3, 4, 5, 12, 7, 14};

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

    size_t offset = 0;
    for (int i = 0; i < 9; i++) {
        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size = sz;
        vkCmdCopyBuffer(cmd, phys.soa_buffers[src_bindings[i]],
                        phys.staging, 1, &region);
        offset += sz;
    }

    vkEndCommandBuffer(cmd);

    /* Dedicated fence — only wait for this submission, not the whole queue */
    VkFence fence;
    VkFenceCreateInfo fi = {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fi, nullptr, &fence);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, fence);

    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    /* Copy from staging to CPU arrays */
    char* mapped = (char*)phys.stagingMapped;
    memcpy(out_pos_x,      mapped + 0 * sz, sz);
    memcpy(out_pos_y,      mapped + 1 * sz, sz);
    memcpy(out_pos_z,      mapped + 2 * sz, sz);
    memcpy(out_vel_x,      mapped + 3 * sz, sz);
    memcpy(out_vel_y,      mapped + 4 * sz, sz);
    memcpy(out_vel_z,      mapped + 5 * sz, sz);
    memcpy(out_theta,      mapped + 6 * sz, sz);
    memcpy(out_pump_scale, mapped + 7 * sz, sz);

    /* Flags are stored as uint32 in the SSBO, narrow back to uint8 */
    const uint32_t* flags32 = (const uint32_t*)(mapped + 8 * sz);
    for (int i = 0; i < count; i++) {
        out_flags[i] = (uint8_t)(flags32[i] & 0xFF);
    }

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);
}

/* ========================================================================
 * DIAGNOSTIC — pump_state readback
 * ======================================================================== */

void readbackPumpStateSample(PhysicsCompute& phys, VulkanContext& ctx,
                             int* out_states, int count) {
    if (count <= 0) return;
    if (count > ORACLE_SUBSET_SIZE) count = ORACLE_SUBSET_SIZE;

    size_t sz = (size_t)count * sizeof(int);

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

    /* Copy pump_state (binding 6) into the first `sz` bytes of staging */
    VkBufferCopy region = {0, 0, sz};
    vkCmdCopyBuffer(cmd, phys.soa_buffers[6], phys.staging, 1, &region);

    vkEndCommandBuffer(cmd);

    VkFence fence;
    VkFenceCreateInfo fi = {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fi, nullptr, &fence);

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, fence);
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    memcpy(out_states, phys.stagingMapped, sz);

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

    /* Physics pipeline */
    vkDestroyPipeline(device, phys.pipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.descLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.descPool, nullptr);

    /* Scatter pipeline + grid buffers */
    vkDestroyPipeline(device, phys.scatterPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.scatterPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.scatterSet1Layout, nullptr);
    vkDestroyDescriptorPool(device, phys.scatterDescPool, nullptr);
    vkDestroyBuffer(device, phys.gridDensityBuffer, nullptr);
    vkFreeMemory(device, phys.gridDensityMemory, nullptr);
    vkDestroyBuffer(device, phys.particleCellBuffer, nullptr);
    vkFreeMemory(device, phys.particleCellMemory, nullptr);

    /* Projection pipeline */
    vkDestroyPipeline(device, phys.projPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.projPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.projDescLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.projDescPool, nullptr);

    /* Tone-map pipeline */
    vkDestroyPipeline(device, phys.tonePipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.tonePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.toneDescLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.toneDescPool, nullptr);

    /* Density image */
    vkDestroySampler(device, phys.densitySampler, nullptr);
    vkDestroyImageView(device, phys.densityView, nullptr);
    vkDestroyImage(device, phys.densityImage, nullptr);
    vkFreeMemory(device, phys.densityMemory, nullptr);

    /* Camera UBO */
    vkUnmapMemory(device, phys.cameraMemory);
    vkDestroyBuffer(device, phys.cameraUBO, nullptr);
    vkFreeMemory(device, phys.cameraMemory, nullptr);

    /* Query pool */
    vkDestroyQueryPool(device, phys.queryPool, nullptr);

    phys.initialized = false;
    printf("[vk-compute] Physics compute cleaned up\n");
}
