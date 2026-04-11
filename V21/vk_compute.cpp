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
                         int N, ScatterMode scatterMode) {
    phys.N = N;
    phys.scatterMode = scatterMode;
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
    qpInfo.queryCount = 16;  /* 2 frames × 8 timestamps:
                              *   0 begin         1 scatter_end    2 stencil_end
                              *   3 gather_end    4 constraint_end 5 siphon_end
                              *   6 project_end   7 tonemap_end */
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

    /* Pass 1b reduce pipeline (8 shards → canonical density) */
    initScatterReduceCompute(phys, ctx);

    /* Pass 2 stencil pipeline (density → pressure gradient, 6-neighbor stencil) */
    initStencilCompute(phys, ctx);

    /* Pass 3 gather-measure pipeline (cells → per-particle scratch, measurement-only) */
    initGatherMeasureCompute(phys, ctx);
}

/* ========================================================================
 * SCATTER (Pass 1) — particles → cell grid
 * ======================================================================== */

void initScatterCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Allocate canonical grid density buffer (uint[V21_GRID_CELLS]) --- */
    size_t grid_bytes = (size_t)V21_GRID_CELLS * sizeof(uint32_t);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.gridDensityBuffer, phys.gridDensityMemory, grid_bytes);

    /* --- Allocate privatized shards buffer (uint[SHARD_COUNT * V21_GRID_CELLS]) --- */
    size_t shards_bytes = (size_t)V21_GRID_SHARD_COUNT * V21_GRID_CELLS * sizeof(uint32_t);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.gridDensityShardsBuffer, phys.gridDensityShardsMemory, shards_bytes);

    /* --- Allocate particle_cell buffer (uint[N]) --- */
    size_t pcell_bytes = (size_t)phys.N * sizeof(uint32_t);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.particleCellBuffer, phys.particleCellMemory, pcell_bytes);

    const char* mode_name =
        (phys.scatterMode == SCATTER_MODE_SQUARAGON) ? "squaragon" :
        (phys.scatterMode == SCATTER_MODE_UNIFORM)   ? "uniform"   : "baseline";
    printf("[vk-compute] Scatter grid: %d cells × %d shards (%.1f MB), "
           "canonical (%.1f MB), particle_cell (%.1f MB), mode=%s\n",
           V21_GRID_CELLS, V21_GRID_SHARD_COUNT,
           (double)shards_bytes / (1024.0 * 1024.0),
           (double)grid_bytes / (1024.0 * 1024.0),
           (double)pcell_bytes / (1024.0 * 1024.0),
           mode_name);

    /* --- Descriptor set layout for set 1 (shards + particle_cell) --- */
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

    /* --- Load the right scatter variant SPIRV --- */
    const char* spv_name =
        (phys.scatterMode == SCATTER_MODE_SQUARAGON) ? "scatter_squaragon.spv" :
        (phys.scatterMode == SCATTER_MODE_UNIFORM)   ? "scatter_uniform.spv"   :
                                                        "scatter_baseline.spv";
    auto code = readShaderFile(spv_name);
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

    /* --- Write descriptors for set 1: shards buffer + particle_cell --- */
    VkDescriptorBufferInfo bufInfos[2] = {
        { phys.gridDensityShardsBuffer, 0, VK_WHOLE_SIZE },
        { phys.particleCellBuffer,      0, VK_WHOLE_SIZE },
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

    printf("[vk-compute] Scatter compute pipeline created (%s)\n", mode_name);
}

/* ========================================================================
 * SCATTER REDUCE — sum 8 shards into canonical density
 * ======================================================================== */

void initScatterReduceCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Descriptor set layout: 2 SSBO bindings (shards readonly, canonical writeonly) --- */
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (int i = 0; i < 2; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lInfo = {};
    lInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lInfo.bindingCount = 2;
    lInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &lInfo, nullptr, &phys.scatterReduceSetLayout);

    /* --- Pipeline layout --- */
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(ScatterReducePushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &phys.scatterReduceSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.scatterReducePipelineLayout);

    /* --- Pipeline --- */
    auto code = readShaderFile("scatter_reduce.spv");
    VkShaderModule mod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.scatterReducePipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                  nullptr, &phys.scatterReducePipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create scatter_reduce pipeline");

    vkDestroyShaderModule(ctx.device, mod, nullptr);

    /* --- Descriptor pool + set --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.scatterReduceDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.scatterReduceDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.scatterReduceSetLayout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.scatterReduceSet);

    /* --- Write descriptors --- */
    VkDescriptorBufferInfo bufInfos[2] = {
        { phys.gridDensityShardsBuffer, 0, VK_WHOLE_SIZE },
        { phys.gridDensityBuffer,       0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.scatterReduceSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    printf("[vk-compute] Scatter reduce pipeline created\n");
}

/* ========================================================================
 * STENCIL (Pass 2) — density → pressure gradient via 6-neighbor stencil
 * ======================================================================== */

void initStencilCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Allocate pressure_x/y/z buffers (float[V21_GRID_CELLS]) --- */
    size_t pbytes = (size_t)V21_GRID_CELLS * sizeof(float);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.pressureXBuffer, phys.pressureXMemory, pbytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.pressureYBuffer, phys.pressureYMemory, pbytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.pressureZBuffer, phys.pressureZMemory, pbytes);
    printf("[vk-compute] Pressure grid: 3 × %.1f MB\n",
           (double)pbytes / (1024.0 * 1024.0));

    /* --- Descriptor set layout: density (read) + 3 pressure (write) --- */
    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lInfo = {};
    lInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lInfo.bindingCount = 4;
    lInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &lInfo, nullptr, &phys.stencilSetLayout);

    /* --- Pipeline layout --- */
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(StencilPushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &phys.stencilSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.stencilPipelineLayout);

    /* --- Pipeline --- */
    auto code = readShaderFile("stencil.spv");
    VkShaderModule mod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.stencilPipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                  nullptr, &phys.stencilPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create stencil pipeline");
    vkDestroyShaderModule(ctx.device, mod, nullptr);

    /* --- Descriptor pool + set --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.stencilDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.stencilDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.stencilSetLayout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.stencilSet);

    /* --- Write descriptors --- */
    VkDescriptorBufferInfo bufInfos[4] = {
        { phys.gridDensityBuffer, 0, VK_WHOLE_SIZE },
        { phys.pressureXBuffer,   0, VK_WHOLE_SIZE },
        { phys.pressureYBuffer,   0, VK_WHOLE_SIZE },
        { phys.pressureZBuffer,   0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[4] = {};
    for (int i = 0; i < 4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.stencilSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

    printf("[vk-compute] Stencil pipeline created\n");
}

/* ========================================================================
 * GATHER-MEASURE (Pass 3, measurement-only) — cells → particle scratch
 * ======================================================================== */

void initGatherMeasureCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Allocate gather_scratch[N] --- */
    size_t sbytes = (size_t)phys.N * sizeof(float);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.gatherScratchBuffer, phys.gatherScratchMemory, sbytes);
    printf("[vk-compute] Gather scratch buffer: %.1f MB\n",
           (double)sbytes / (1024.0 * 1024.0));

    /* --- Descriptor set layout for set 1: 5 SSBOs --- */
    VkDescriptorSetLayoutBinding set1Bindings[5] = {};
    for (int i = 0; i < 5; i++) {
        set1Bindings[i].binding = i;
        set1Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set1Bindings[i].descriptorCount = 1;
        set1Bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set1Info = {};
    set1Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set1Info.bindingCount = 5;
    set1Info.pBindings = set1Bindings;
    vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &phys.gatherMeasureSet1Layout);

    /* --- Pipeline layout: set 0 (siphon particles) + set 1 (gather IO) --- */
    VkDescriptorSetLayout layouts[2] = { phys.descLayout, phys.gatherMeasureSet1Layout };
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(GatherMeasurePushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 2;
    plInfo.pSetLayouts = layouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.gatherMeasurePipelineLayout);

    /* --- Pipeline --- */
    auto code = readShaderFile("gather_measure.spv");
    VkShaderModule mod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.gatherMeasurePipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                  nullptr, &phys.gatherMeasurePipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create gather_measure pipeline");
    vkDestroyShaderModule(ctx.device, mod, nullptr);

    /* --- Descriptor pool + set for set 1 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 5;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.gatherMeasureDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.gatherMeasureDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.gatherMeasureSet1Layout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.gatherMeasureSet1);

    /* --- Write descriptors for set 1 --- */
    VkDescriptorBufferInfo bufInfos[5] = {
        { phys.particleCellBuffer,   0, VK_WHOLE_SIZE },
        { phys.pressureXBuffer,      0, VK_WHOLE_SIZE },
        { phys.pressureYBuffer,      0, VK_WHOLE_SIZE },
        { phys.pressureZBuffer,      0, VK_WHOLE_SIZE },
        { phys.gatherScratchBuffer,  0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.gatherMeasureSet1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);

    printf("[vk-compute] Gather-measure pipeline created\n");
}

/* ========================================================================
 * CONSTRAINT SOLVE (Pass 4) — PBD distance constraints for rigid-body MVP
 * ======================================================================== */

void initConstraintCompute(PhysicsCompute& phys, VulkanContext& ctx,
                           const uint32_t* pair_indices,
                           const float*    rest_lengths,
                           const float*    inv_masses,
                           const uint32_t* bucket_offsets,
                           const uint32_t* bucket_counts,
                           uint32_t        rigid_base,
                           uint32_t        rigid_count,
                           uint32_t        iterations) {
    /* Total constraint count = sum of the 6 buckets */
    uint32_t M = 0;
    for (int k = 0; k < 6; k++) M += bucket_counts[k];

    /* --- Allocate device-local SSBOs --- */
    size_t pairs_bytes = (size_t)M * 2 * sizeof(uint32_t);        /* uvec2 × M */
    size_t rest_bytes  = (size_t)M * sizeof(float);
    size_t invm_bytes  = (size_t)rigid_count * sizeof(float);

    createSSBO(ctx.device, ctx.physicalDevice,
               phys.constraintPairsBuffer, phys.constraintPairsMemory, pairs_bytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.restLengthsBuffer,     phys.restLengthsMemory,     rest_bytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.invMassesBuffer,       phys.invMassesMemory,       invm_bytes);

    printf("[vk-compute] Constraint solver: %u particles, %u constraints, "
           "buckets=[%u %u %u %u %u %u], %u iters/frame, %.1f KB total\n",
           rigid_count, M,
           bucket_counts[0], bucket_counts[1], bucket_counts[2],
           bucket_counts[3], bucket_counts[4], bucket_counts[5],
           iterations,
           (double)(pairs_bytes + rest_bytes + invm_bytes) / 1024.0);

    /* --- Stage + upload all 3 buffers in one command --- */
    size_t total_sz = pairs_bytes + rest_bytes + invm_bytes;
    size_t off_pairs = 0;
    size_t off_rest  = pairs_bytes;
    size_t off_invm  = pairs_bytes + rest_bytes;

    VkBuffer       stagingBuf;
    VkDeviceMemory stagingMem;
    {
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = total_sz;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuf);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(ctx.device, stagingBuf, &mr);
        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemType(ctx.physicalDevice, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMem);
        vkBindBufferMemory(ctx.device, stagingBuf, stagingMem, 0);
    }

    char* staging_base;
    vkMapMemory(ctx.device, stagingMem, 0, total_sz, 0, (void**)&staging_base);
    memcpy(staging_base + off_pairs, pair_indices, pairs_bytes);
    memcpy(staging_base + off_rest,  rest_lengths, rest_bytes);
    memcpy(staging_base + off_invm,  inv_masses,   invm_bytes);
    vkUnmapMemory(ctx.device, stagingMem);

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

    VkBufferCopy c0 = {off_pairs, 0, pairs_bytes};
    vkCmdCopyBuffer(ucmd, stagingBuf, phys.constraintPairsBuffer, 1, &c0);
    VkBufferCopy c1 = {off_rest, 0, rest_bytes};
    vkCmdCopyBuffer(ucmd, stagingBuf, phys.restLengthsBuffer, 1, &c1);
    VkBufferCopy c2 = {off_invm, 0, invm_bytes};
    vkCmdCopyBuffer(ucmd, stagingBuf, phys.invMassesBuffer, 1, &c2);
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

    /* --- Descriptor set layout for set 1: 3 SSBOs --- */
    VkDescriptorSetLayoutBinding set1Bindings[3] = {};
    for (int i = 0; i < 3; i++) {
        set1Bindings[i].binding = i;
        set1Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set1Bindings[i].descriptorCount = 1;
        set1Bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set1Info = {};
    set1Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set1Info.bindingCount = 3;
    set1Info.pBindings = set1Bindings;
    vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &phys.constraintSet1Layout);

    /* --- Pipeline layout: set 0 (siphon particles) + set 1 (constraint data) --- */
    VkDescriptorSetLayout layouts[2] = { phys.descLayout, phys.constraintSet1Layout };
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(ConstraintPushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 2;
    plInfo.pSetLayouts = layouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.constraintPipelineLayout);

    /* --- Pipeline --- */
    auto code = readShaderFile("constraint_solve.spv");
    VkShaderModule mod = createShaderModule(ctx.device, code);

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName = "main";
    cpInfo.layout = phys.constraintPipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                  nullptr, &phys.constraintPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create constraint_solve pipeline");
    vkDestroyShaderModule(ctx.device, mod, nullptr);

    /* --- Descriptor pool + set for set 1 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.constraintDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.constraintDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.constraintSet1Layout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.constraintSet1);

    /* --- Write descriptors for set 1 --- */
    VkDescriptorBufferInfo bufInfos[3] = {
        { phys.constraintPairsBuffer, 0, VK_WHOLE_SIZE },
        { phys.restLengthsBuffer,     0, VK_WHOLE_SIZE },
        { phys.invMassesBuffer,       0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.constraintSet1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    /* --- Store solver config --- */
    phys.rigidBaseIndex      = rigid_base;
    phys.rigidCount          = rigid_count;
    phys.constraintIterations = iterations;
    for (int k = 0; k < 6; k++) {
        phys.constraintBucketOffsets[k] = bucket_offsets[k];
        phys.constraintBucketCounts[k]  = bucket_counts[k];
    }
    phys.constraintEnabled = true;

    printf("[vk-compute] Constraint solve pipeline created (base=%u count=%u iters=%u)\n",
           rigid_base, rigid_count, iterations);
}

/* ========================================================================
 * DISPATCH — record compute commands
 * ======================================================================== */

void dispatchPhysicsCompute(PhysicsCompute& phys, VkCommandBuffer cmd,
                            int frame, float sim_time, float dt) {
    /* Reset this slot's 8 timestamps and write the begin marker.
     * Layout:
     *   +0 begin            +1 scatter_end      +2 stencil_end
     *   +3 gather_end       +4 constraint_end   +5 siphon_end
     *   +6 project_end      +7 tonemap_end (written from recordDensityRender) */
    uint32_t base = phys.queryFrame * 8;
    vkCmdResetQueryPool(cmd, phys.queryPool, base, 8);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        phys.queryPool, base + 0);

    /* ---- Pass 1: Scatter (particles → privatized cell grid shards) ---- */
    /* Zero the shards buffer (not the canonical density) before scatter. */
    vkCmdFillBuffer(cmd, phys.gridDensityShardsBuffer, 0,
                    (VkDeviceSize)V21_GRID_SHARD_COUNT * V21_GRID_CELLS * sizeof(uint32_t), 0);

    /* Barrier: fill must complete before scatter reads/writes shards. */
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

    /* Barrier: scatter writes (shards) must be visible to the reduce pass. */
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

    /* ---- Pass 1b: Reduce (8 shards → canonical grid_density) --------- */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.scatterReducePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.scatterReducePipelineLayout, 0, 1, &phys.scatterReduceSet, 0, nullptr);

    ScatterReducePushConstants reducePush = {};
    reducePush.total_cells = V21_GRID_CELLS;
    vkCmdPushConstants(cmd, phys.scatterReducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(reducePush), &reducePush);

    vkCmdDispatch(cmd, (V21_GRID_CELLS + 255) / 256, 1, 1);

    /* Barrier: reduce's writes to canonical density must be visible to any
     * Pass 2/3 consumers (none yet, but keep the fence for correctness). */
    {
        VkMemoryBarrier reduceBarrier = {};
        reduceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        reduceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        reduceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &reduceBarrier, 0, nullptr, 0, nullptr);
    }

    /* Scatter + reduce end — record after the drain-barrier.
     * The scatter_ms timestamp covers BOTH passes (scatter + reduce). */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 1);

    /* ---- Pass 2: Stencil (density → pressure gradient) --------------- */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.stencilPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.stencilPipelineLayout, 0, 1, &phys.stencilSet, 0, nullptr);

    StencilPushConstants stencilPush = {};
    stencilPush.grid_dim       = V21_GRID_DIM;
    stencilPush.total_cells    = V21_GRID_CELLS;
    stencilPush.grid_cell_size = V21_GRID_CELL_SIZE;
    stencilPush.pressure_k     = 0.01f;  /* measurement-only scale factor */
    vkCmdPushConstants(cmd, phys.stencilPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(stencilPush), &stencilPush);

    vkCmdDispatch(cmd, (V21_GRID_CELLS + 255) / 256, 1, 1);

    /* Barrier: stencil's pressure writes must be visible to gather. */
    {
        VkMemoryBarrier stencilBarrier = {};
        stencilBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        stencilBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        stencilBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &stencilBarrier, 0, nullptr, 0, nullptr);
    }

    /* Stencil end */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 2);

    /* ---- Pass 3: Gather-measure (cells → per-particle scratch) ------- */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.gatherMeasurePipeline);
    VkDescriptorSet gatherSets[2] = { phys.descSet, phys.gatherMeasureSet1 };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.gatherMeasurePipelineLayout, 0, 2, gatherSets, 0, nullptr);

    GatherMeasurePushConstants gatherPush = {};
    gatherPush.N           = phys.N;
    gatherPush.total_cells = V21_GRID_CELLS;
    gatherPush.dt          = dt;
    vkCmdPushConstants(cmd, phys.gatherMeasurePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(gatherPush), &gatherPush);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Barrier: gather writes to scratch — no downstream consumer in
     * measurement-only mode, but the fence prevents the driver from
     * elision-reordering the dispatch relative to siphon. */
    {
        VkMemoryBarrier gatherBarrier = {};
        gatherBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        gatherBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        gatherBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &gatherBarrier, 0, nullptr, 0, nullptr);
    }

    /* Gather end */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 3);

    /* ---- Pass 4: Constraint Solve (PBD distance constraints, 6-bucket GS)
     * Dispatches the constraint solver when the rigid-body mode is enabled.
     * Each outer iteration runs 6 vertex-disjoint bucket dispatches with a
     * memory barrier between each bucket so later buckets see earlier writes
     * to pos_x/y/z. The solver touches only positions (PBD style); siphon's
     * next integration step will feel the new positions via gravitational
     * pull. When disabled, constraint_ms ≈ 0 from the two adjacent timestamp
     * writes. */
    if (phys.constraintEnabled) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.constraintPipeline);
        VkDescriptorSet csets[2] = { phys.descSet, phys.constraintSet1 };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.constraintPipelineLayout, 0, 2, csets, 0, nullptr);

        ConstraintPushConstants cpc = {};
        cpc.rigid_base  = phys.rigidBaseIndex;
        cpc.rigid_count = phys.rigidCount;
        cpc.beta        = 0.2f;
        cpc.compliance  = 0.0f;
        cpc.dt          = dt;
        cpc._pad        = 0;

        VkMemoryBarrier colorBarrier = {};
        colorBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        colorBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        for (uint32_t it = 0; it < phys.constraintIterations; it++) {
            for (int bucket = 0; bucket < 6; bucket++) {
                cpc.constraint_offset = phys.constraintBucketOffsets[bucket];
                cpc.constraint_count  = phys.constraintBucketCounts[bucket];
                if (cpc.constraint_count == 0) continue;

                vkCmdPushConstants(cmd, phys.constraintPipelineLayout,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(cpc), &cpc);
                vkCmdDispatch(cmd, (cpc.constraint_count + 63) / 64, 1, 1);

                /* Barrier between buckets (and between iterations): pos writes
                 * must be visible to the next bucket's reads. Skip the final
                 * barrier after the last bucket of the last iteration — the
                 * final visibility barrier to siphon is emitted below. */
                bool is_last = (it + 1 == phys.constraintIterations) && (bucket == 5);
                if (!is_last) {
                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &colorBarrier, 0, nullptr, 0, nullptr);
                }
            }
        }

        /* Final barrier: constraint pos writes → siphon reads */
        VkMemoryBarrier finalBarrier = {};
        finalBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        finalBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        finalBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &finalBarrier, 0, nullptr, 0, nullptr);
    }

    /* Constraint end — written unconditionally so constraint_ms is always
     * defined. When disabled, constraint_ms reads ≈ 0. */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 4);

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
                        phys.queryPool, base + 5);
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
        uint32_t base = phys.queryFrame * 8;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            phys.queryPool, base + 6);
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
        uint32_t base = phys.queryFrame * 8;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            phys.queryPool, base + 7);
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
                    double* out_stencil_ms,
                    double* out_gather_ms,
                    double* out_constraint_ms,
                    double* out_siphon_ms,
                    double* out_project_ms,
                    double* out_tonemap_ms) {
    uint32_t slot = phys.queryFrame ^ 1;
    if (!phys.queryValid[slot]) return false;

    uint32_t base = slot * 8;
    uint64_t ts[8];
    VkResult r = vkGetQueryPoolResults(device, phys.queryPool, base, 8,
        sizeof(ts), ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return false;

    double nsPerTick = (double)phys.timestampPeriodNs;
    *out_scatter_ms    = (double)(ts[1] - ts[0]) * nsPerTick / 1.0e6;
    *out_stencil_ms    = (double)(ts[2] - ts[1]) * nsPerTick / 1.0e6;
    *out_gather_ms     = (double)(ts[3] - ts[2]) * nsPerTick / 1.0e6;
    *out_constraint_ms = (double)(ts[4] - ts[3]) * nsPerTick / 1.0e6;
    *out_siphon_ms     = (double)(ts[5] - ts[4]) * nsPerTick / 1.0e6;
    *out_project_ms    = (double)(ts[6] - ts[5]) * nsPerTick / 1.0e6;
    *out_tonemap_ms    = (double)(ts[7] - ts[6]) * nsPerTick / 1.0e6;
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
    vkDestroyBuffer(device, phys.gridDensityShardsBuffer, nullptr);
    vkFreeMemory(device, phys.gridDensityShardsMemory, nullptr);
    vkDestroyBuffer(device, phys.particleCellBuffer, nullptr);
    vkFreeMemory(device, phys.particleCellMemory, nullptr);

    /* Scatter reduce pipeline */
    vkDestroyPipeline(device, phys.scatterReducePipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.scatterReducePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.scatterReduceSetLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.scatterReduceDescPool, nullptr);

    /* Stencil pipeline + pressure buffers */
    vkDestroyPipeline(device, phys.stencilPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.stencilPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.stencilSetLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.stencilDescPool, nullptr);
    vkDestroyBuffer(device, phys.pressureXBuffer, nullptr);
    vkFreeMemory(device, phys.pressureXMemory, nullptr);
    vkDestroyBuffer(device, phys.pressureYBuffer, nullptr);
    vkFreeMemory(device, phys.pressureYMemory, nullptr);
    vkDestroyBuffer(device, phys.pressureZBuffer, nullptr);
    vkFreeMemory(device, phys.pressureZMemory, nullptr);

    /* Gather-measure pipeline + scratch buffer */
    vkDestroyPipeline(device, phys.gatherMeasurePipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.gatherMeasurePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.gatherMeasureSet1Layout, nullptr);
    vkDestroyDescriptorPool(device, phys.gatherMeasureDescPool, nullptr);
    vkDestroyBuffer(device, phys.gatherScratchBuffer, nullptr);
    vkFreeMemory(device, phys.gatherScratchMemory, nullptr);

    /* Constraint solve pipeline + constraint SSBOs (optional) */
    if (phys.constraintEnabled) {
        vkDestroyPipeline(device, phys.constraintPipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.constraintPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.constraintSet1Layout, nullptr);
        vkDestroyDescriptorPool(device, phys.constraintDescPool, nullptr);
        vkDestroyBuffer(device, phys.constraintPairsBuffer, nullptr);
        vkFreeMemory(device, phys.constraintPairsMemory, nullptr);
        vkDestroyBuffer(device, phys.restLengthsBuffer, nullptr);
        vkFreeMemory(device, phys.restLengthsMemory, nullptr);
        vkDestroyBuffer(device, phys.invMassesBuffer, nullptr);
        vkFreeMemory(device, phys.invMassesMemory, nullptr);
    }

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
