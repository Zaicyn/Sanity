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
            printf("[shader] Loaded %s%s (%zu bytes)\n", p.c_str(), filename.c_str(), sz);
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

uint32_t findMemType(VkPhysicalDevice pd, uint32_t filter, VkMemoryPropertyFlags props) {
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

void uploadToSSBO(VulkanContext& ctx, VkBuffer dst, const void* src, size_t size) {
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

void uploadToSSBO_offset(VulkanContext& ctx, VkBuffer dst, const void* src,
                         size_t offset, size_t size) {
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

    VkBufferCopy copy = {0, offset, size};
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

/* Query free VRAM using VK_EXT_memory_budget.
 * Returns the number of particles that fit in 80% of free device-local VRAM.
 * Each particle costs ~44 bytes (9 floats + 2 uint32s across active SSBOs). */
int queryVramMaxParticles(VkPhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties2 memProps2 = {};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    memProps2.pNext = &budget;

    vkGetPhysicalDeviceMemoryProperties2(pd, &memProps2);

    /* Find the device-local heap with the most free space */
    VkDeviceSize best_free = 0;
    for (uint32_t i = 0; i < memProps2.memoryProperties.memoryHeapCount; i++) {
        if (memProps2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            VkDeviceSize heap_budget = budget.heapBudget[i];
            VkDeviceSize heap_usage  = budget.heapUsage[i];
            VkDeviceSize heap_free   = (heap_budget > heap_usage) ? (heap_budget - heap_usage) : 0;
            if (heap_free > best_free) best_free = heap_free;
        }
    }

    /* 80% of free VRAM, ~44 bytes per particle */
    const size_t BYTES_PER_PARTICLE = 44;
    size_t usable = (size_t)(best_free * 0.8);
    int max_particles = (int)(usable / BYTES_PER_PARTICLE);
    if (max_particles < 1000) max_particles = 1000;  /* floor */

    printf("[vram] Free device-local: %.1f MB, 80%% usable: %.1f MB, max particles: %d\n",
           (double)best_free / (1024.0 * 1024.0),
           (double)usable / (1024.0 * 1024.0),
           max_particles);

    return max_particles;
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
                         int N, ScatterMode scatterMode,
                         int capacity) {
    if (capacity < N) capacity = N;
    phys.N = N;
    phys.capacity = capacity;
    phys.scatterMode = scatterMode;
    /* SSBOs sized to capacity (room for growth), upload only N particles */
    size_t float_sz = capacity * sizeof(float);
    size_t int_sz = capacity * sizeof(int);
    size_t uint_sz = capacity * sizeof(uint32_t);

    /* Allocate SSBOs — full-size for active buffers only.
     * Siphon reads gradient directly from pressure grid (set 1), not per-particle.
     *
     * Active:  0-5 (pos/vel), 12 (theta), 13 (omega_nat), 14 (flags)
     * Stubs:   6-11, 15 (unused legacy) */
    const size_t stub = 4;  /* minimum valid SSBO size */
    size_t sizes[VK_COMPUTE_NUM_BINDINGS] = {
        float_sz, float_sz, float_sz,  /* pos x,y,z */
        float_sz, float_sz, float_sz,  /* vel x,y,z */
        stub,                           /* unused */
        stub,                           /* unused */
        stub,                           /* unused */
        stub,                           /* unused */
        stub,                           /* unused */
        stub,                           /* unused */
        float_sz,                       /* theta */
        float_sz,                       /* omega_nat */
        uint_sz,                        /* flags (uint32 padded) */
        stub                            /* unused */
    };

    /* Unified allocation: one VkDeviceMemory, 16 VkBuffer views at 256-byte aligned offsets.
     * Contiguous layout lets the L2 see one address range instead of 16 scattered allocations. */
    const size_t ALIGN = 256;  /* >= minStorageBufferOffsetAlignment on Turing */
    size_t total_sz = 0;
    size_t offsets[VK_COMPUTE_NUM_BINDINGS];
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        phys.soa_sizes[i] = sizes[i];
        offsets[i] = total_sz;
        phys.soa_offsets[i] = total_sz;
        total_sz += (sizes[i] + ALIGN - 1) & ~(ALIGN - 1);  /* round up to 256 */
    }
    phys.soa_total_size = total_sz;

    /* Create 16 VkBuffer objects (one per binding) */
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = sizes[i];
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(ctx.device, &bi, nullptr, &phys.soa_buffers[i]);
    }

    /* Single device-local memory allocation for all 16 buffers */
    {
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(ctx.device, phys.soa_buffers[0], &mr);

        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = total_sz;
        ai.memoryTypeIndex = findMemType(ctx.physicalDevice, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(ctx.device, &ai, nullptr, &phys.soa_unified_memory);
    }

    /* Bind each buffer at its aligned offset into the unified allocation */
    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        vkBindBufferMemory(ctx.device, phys.soa_buffers[i],
                           phys.soa_unified_memory, offsets[i]);
    }
    printf("[vk-compute] Unified SoA allocation: %.1f MB (16 views, 256-byte aligned)\n",
           (float)total_sz / (1024 * 1024));

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

    /* Upload only N particles (not capacity) — the rest stays zero-initialized.
     * Bindings 6-11, 15 are 4-byte stubs — don't write into them. */
    size_t upload_float = (size_t)N * sizeof(float);
    memcpy(staging_base + offsets[0], pos_x, upload_float);
    memcpy(staging_base + offsets[1], pos_y, upload_float);
    memcpy(staging_base + offsets[2], pos_z, upload_float);
    memcpy(staging_base + offsets[3], vel_x, upload_float);
    memcpy(staging_base + offsets[4], vel_y, upload_float);
    memcpy(staging_base + offsets[5], vel_z, upload_float);
    memcpy(staging_base + offsets[12], theta, upload_float);
    memcpy(staging_base + offsets[13], omega_nat, upload_float);

    /* Flags (binding 14): active particles get 0x01 */
    uint32_t* flags_s = (uint32_t*)(staging_base + offsets[14]);
    for (int i = 0; i < N; i++) {
        flags_s[i] = flags[i];
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

    /* Siphon pipeline layout deferred until after initSiphonSet1,
     * because the siphon binds set 1 (pressure grid) for direct gradient sampling. */

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

    /* Staging buffer for readback — capped at ORACLE_SUBSET_SIZE × 13 floats.
     * Layout: [pos_x][pos_y][pos_z][vel_x][vel_y][vel_z][theta][pump_scale][flags]
     *         [r][vel_r][phi][omega_orb]  (graded state for physics diagnostics)
     * 13 arrays × 100K × 4 bytes = 5.2 MB. Stays small even at 80M particles. */
    int subset_cap = N < ORACLE_SUBSET_SIZE ? N : ORACLE_SUBSET_SIZE;
    size_t readback_sz = (size_t)subset_cap * 13 * sizeof(float);
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
    qpInfo.queryCount = 18;  /* 2 frames × 9 timestamps:
                              *   0 begin         1 scatter_end    2 stencil_end
                              *   3 gather_end    4 constraint_end 5 collision_end
                              *   6 siphon_end    7 project_end    8 tonemap_end */
    if (vkCreateQueryPool(ctx.device, &qpInfo, nullptr, &phys.queryPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create timestamp query pool");

    phys.queryValid[0] = false;
    phys.queryValid[1] = false;
    phys.queryFrame = 0;

    phys.initialized = true;
    printf("[vk-compute] Physics compute ready (%d particles, capacity %d, %d SSBOs, timestamp period %.2f ns)\n",
           N, capacity, VK_COMPUTE_NUM_BINDINGS, phys.timestampPeriodNs);

    /* Pass 1 scatter pipeline (particles → cell grid) */
    initScatterCompute(phys, ctx);

    /* Pass 1b reduce pipeline (8 shards → canonical density) */
    initScatterReduceCompute(phys, ctx);

    /* Pass 2 stencil pipeline (density → pressure gradient, 6-neighbor stencil) */
    initStencilCompute(phys, ctx);

    /* Siphon set 1 — pressure grid descriptors (particle_cell + pressure_x/y/z) */
    initSiphonSet1(phys, ctx);

    /* --- Siphon pipeline: set 0 (particles) + set 1 (pressure grid) ---
     * The siphon reads pressure gradient directly from the grid buffers via
     * particle_cell[i] and computes density magnitude inline (sqrt). */
    {
        VkDescriptorSetLayout siphonSets[2] = {
            phys.descLayout,              /* set 0: 16 SoA particle buffers */
            phys.siphonSet1Layout         /* set 1: particle_cell + pressure xyz */
        };
        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.size = sizeof(SiphonPushConstants);

        VkPipelineLayoutCreateInfo plInfo = {};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 2;
        plInfo.pSetLayouts = siphonSets;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
        vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.pipelineLayout);

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
        printf("[vk-compute] Siphon pipeline created (set 0 + set 1 pressure grid)\n");
    }
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

    /* --- Allocate particle_cell buffer (uint[capacity]) --- */
    size_t pcell_bytes = (size_t)phys.capacity * sizeof(uint32_t);
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

    /* --- Write descriptors ---
     * Binding 0: density from shard 0 of the shards buffer (scatter writes here directly).
     * Limit the view to V21_GRID_CELLS uints so stencil sees shard 0 only. */
    VkDescriptorBufferInfo bufInfos[4] = {
        { phys.gridDensityShardsBuffer, 0, (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t) },
        { phys.pressureXBuffer,         0, VK_WHOLE_SIZE },
        { phys.pressureYBuffer,         0, VK_WHOLE_SIZE },
        { phys.pressureZBuffer,         0, VK_WHOLE_SIZE },
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

    printf("[vk-compute] Stencil pipeline created (reads shard 0 density)\n");
}

/* ========================================================================
 * SIPHON SET 1 — pressure grid descriptors (particle_cell + pressure_x/y/z)
 * ======================================================================== */

void initSiphonSet1(PhysicsCompute& phys, VulkanContext& ctx) {
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
    vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &phys.siphonSet1Layout);

    /* --- Descriptor pool + set for set 1 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 5;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.siphonSet1DescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.siphonSet1DescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.siphonSet1Layout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.siphonSet1);

    /* --- Write descriptors for set 1 --- */
    VkDescriptorBufferInfo bufInfos[5] = {
        { phys.particleCellBuffer,      0, VK_WHOLE_SIZE },
        { phys.pressureXBuffer,         0, VK_WHOLE_SIZE },
        { phys.pressureYBuffer,         0, VK_WHOLE_SIZE },
        { phys.pressureZBuffer,         0, VK_WHOLE_SIZE },
        { phys.gridDensityShardsBuffer, 0, (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t) },
    };
    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.siphonSet1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);

    printf("[vk-compute] Siphon set 1 (pressure grid + density) created — 5 bindings\n");
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
    /* Total constraint count = sum of the 7 buckets (6 lattice + 1 joint) */
    uint32_t M = 0;
    for (int k = 0; k < 7; k++) M += bucket_counts[k];

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
           "buckets=[%u %u %u %u %u %u %u], %u iters/frame, %.1f KB total\n",
           rigid_count, M,
           bucket_counts[0], bucket_counts[1], bucket_counts[2],
           bucket_counts[3], bucket_counts[4], bucket_counts[5],
           bucket_counts[6],
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
    for (int k = 0; k < 7; k++) {
        phys.constraintBucketOffsets[k] = bucket_offsets[k];
        phys.constraintBucketCounts[k]  = bucket_counts[k];
    }
    phys.constraintEnabled = true;

    printf("[vk-compute] Constraint solve pipeline created (base=%u count=%u iters=%u)\n",
           rigid_base, rigid_count, iterations);
}

/* ========================================================================
 * Phase 2.2 — COLLISION PIPELINE (dynamic contact constraints)
 * ========================================================================
 *
 * C1 implements only the apply kernel + buffer infrastructure. The fused
 * broadphase+resolve kernel arrives in C2 and reuses the same descriptor set.
 *
 * Per-pipeline set 1 layout:
 *   binding 0  rigid_body_id[N]                uint32, init-time only
 *   binding 1  vel_delta[N_rigid * 3]          int32,  zeroed/written per frame
 *   binding 2  contact_count[1]                uint32, probe, reset per frame
 *   binding 3  first_contact_frame[1]          uint32, probe, reset per frame
 *
 * The descriptor set is sized for 4 bindings even though C1's apply kernel
 * only needs vel_delta — keeping the layout stable across C1/C2 means C2 only
 * adds new pipelines, no re-layout, no re-bind.
 */
void initCollisionCompute(PhysicsCompute& phys, VulkanContext& ctx,
                          const uint32_t* rigid_body_ids,
                          uint32_t        rigid_base,
                          uint32_t        rigid_count) {
    if (rigid_count == 0) {
        fprintf(stderr, "[vk-compute] initCollisionCompute called with rigid_count=0; skipping\n");
        return;
    }

    /* --- Allocate device-local SSBOs --- */
    size_t rbid_bytes  = (size_t)phys.N * sizeof(uint32_t);
    size_t vdelta_bytes = (size_t)rigid_count * 3 * sizeof(int32_t);
    size_t counter_bytes = sizeof(uint32_t);

    createSSBO(ctx.device, ctx.physicalDevice,
               phys.rigidBodyIdBuffer, phys.rigidBodyIdMemory, rbid_bytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.velDeltaBuffer, phys.velDeltaMemory, vdelta_bytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.contactCountBuffer, phys.contactCountMemory, counter_bytes);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.firstContactFrameBuffer, phys.firstContactFrameMemory, counter_bytes);

    size_t pos_prev_bytes = (size_t)rigid_count * 3 * sizeof(float);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.posPrevBuffer, phys.posPrevMemory, pos_prev_bytes);

    printf("[vk-compute] Collision pipeline: N=%d, rigid_base=%u, rigid_count=%u, "
           "vel_delta=%.1f KB, pos_prev=%.1f KB, rigid_body_id=%.1f KB\n",
           phys.N, rigid_base, rigid_count,
           (double)vdelta_bytes / 1024.0,
           (double)pos_prev_bytes / 1024.0,
           (double)rbid_bytes / 1024.0);

    /* --- Stage + upload rigid_body_id once --- */
    uploadToSSBO(ctx, phys.rigidBodyIdBuffer, rigid_body_ids, rbid_bytes);

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
    vkCreateDescriptorSetLayout(ctx.device, &set1Info, nullptr, &phys.collisionSet1Layout);

    /* --- Pipeline layout for collision_apply: set 0 (siphon particles) + set 1 (collision) --- */
    VkDescriptorSetLayout layouts[2] = { phys.descLayout, phys.collisionSet1Layout };
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(CollisionApplyPushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 2;
    plInfo.pSetLayouts = layouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &phys.collisionApplyPipelineLayout);

    /* --- collision_apply pipeline --- */
    {
        auto code = readShaderFile("collision_apply.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);

        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.collisionApplyPipelineLayout;

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                      nullptr, &phys.collisionApplyPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create collision_apply pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* --- collision_resolve pipeline layout (larger push constants than apply) --- */
    {
        VkDescriptorSetLayout resolveLayouts[2] = { phys.descLayout, phys.collisionSet1Layout };
        VkPushConstantRange resolvePushRange = {};
        resolvePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        resolvePushRange.size = sizeof(CollisionResolvePushConstants);

        VkPipelineLayoutCreateInfo resolvePlInfo = {};
        resolvePlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        resolvePlInfo.setLayoutCount = 2;
        resolvePlInfo.pSetLayouts = resolveLayouts;
        resolvePlInfo.pushConstantRangeCount = 1;
        resolvePlInfo.pPushConstantRanges = &resolvePushRange;
        vkCreatePipelineLayout(ctx.device, &resolvePlInfo, nullptr,
                               &phys.collisionResolvePipelineLayout);
    }

    /* --- collision_resolve pipeline --- */
    {
        auto code = readShaderFile("collision_resolve.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);

        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.collisionResolvePipelineLayout;

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                      nullptr, &phys.collisionResolvePipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create collision_resolve pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* --- collision_sync pipeline (shares apply's pipeline layout) --- */
    phys.collisionSyncPipelineLayout = phys.collisionApplyPipelineLayout;  /* same push constants */
    {
        auto code = readShaderFile("collision_sync.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);

        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.collisionSyncPipelineLayout;

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                      nullptr, &phys.collisionSyncPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create collision_sync pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* --- collision_snapshot pipeline (shares apply's pipeline layout) --- */
    {
        auto code = readShaderFile("collision_snapshot.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);

        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.collisionApplyPipelineLayout;  /* same layout */

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo,
                                      nullptr, &phys.collisionSnapshotPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create collision_snapshot pipeline");
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* --- Descriptor pool + set for set 1 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 5;

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &phys.collisionDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.collisionDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.collisionSet1Layout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.collisionSet1);

    /* --- Write descriptors for set 1 --- */
    VkDescriptorBufferInfo bufInfos[5] = {
        { phys.rigidBodyIdBuffer,       0, VK_WHOLE_SIZE },
        { phys.velDeltaBuffer,          0, VK_WHOLE_SIZE },
        { phys.contactCountBuffer,      0, VK_WHOLE_SIZE },
        { phys.firstContactFrameBuffer, 0, VK_WHOLE_SIZE },
        { phys.posPrevBuffer,           0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.collisionSet1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, nullptr);

    phys.collisionEnabled = true;
    printf("[vk-compute] Collision apply pipeline created (base=%u count=%u)\n",
           rigid_base, rigid_count);
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
    uint32_t base = phys.queryFrame * 9;
    vkCmdResetQueryPool(cmd, phys.queryPool, base, 9);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        phys.queryPool, base + 0);

    /* ---- Pass 1: Scatter (cell assignment + counting sort) ----
     *
     * Step 1: Run scatter.comp to compute particle_cell[N] (cell indices).
     *         The shard density accumulation is no longer used — histogram
     *         replaces it. But scatter.comp still writes particle_cell as
     *         a side effect, which we need.
     * Step 2: Histogram — count particles per cell into gridDensity.
     * Step 3: Copy gridDensity → cellOffset, then prefix-scan cellOffset.
     * Step 4: Reorder — write particles to sorted positions.
     *
     * The gridDensityBuffer (canonical density) is preserved for stencil.
     * The cellOffsetBuffer has exclusive prefix sums for the reorder pass.
     */

    /* Zero shard 0 of the shards buffer — scatter accumulates density here directly. */
    vkCmdFillBuffer(cmd, phys.gridDensityShardsBuffer, 0,
                    (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t), 0);
    if (phys.countingSortEnabled) {
        vkCmdFillBuffer(cmd, phys.writeCounterBuffer, 0,
                        (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t), 0);
        vkCmdFillBuffer(cmd, phys.scanBlockSumsBuffer, 0,
                        1024 * sizeof(uint32_t), 0);
    }

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

    /* Step 1: Scatter — cell assignment (writes particle_cell + shards) */
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

    {
        VkMemoryBarrier bar = {};
        bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &bar, 0, nullptr, 0, nullptr);
    }

    if (phys.countingSortEnabled) {
        /* Step 3a: Copy density → cellOffset (scan will modify cellOffset in-place) */
        VkBufferCopy copyRegion = {0, 0, (VkDeviceSize)V21_GRID_CELLS * sizeof(uint32_t)};
        vkCmdCopyBuffer(cmd, phys.gridDensityBuffer, phys.cellOffsetBuffer, 1, &copyRegion);

        {
            VkMemoryBarrier bar = {};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, nullptr, 0, nullptr);
        }

        /* Step 3b: Prefix scan — 3 dispatches (local scan, block scan, propagate) */
        uint32_t scan_N = V21_GRID_CELLS;
        uint32_t scan_groups = (scan_N + 255) / 256;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.scanPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.scanPipelineLayout, 0, 1, &phys.scanSet, 0, nullptr);

        /* Mode 0: local scan + extract block sums */
        ScanPushConstants scanPC = {};
        scanPC.N = scan_N;
        scanPC.mode = 0;
        vkCmdPushConstants(cmd, phys.scanPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
        vkCmdDispatch(cmd, scan_groups, 1, 1);

        {
            VkMemoryBarrier bar = {};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, nullptr, 0, nullptr);
        }

        /* Mode 1: scan the block sums (single workgroup) */
        scanPC.N = scan_N;
        scanPC.mode = 1;
        vkCmdPushConstants(cmd, phys.scanPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
        vkCmdDispatch(cmd, 1, 1, 1);

        {
            VkMemoryBarrier bar = {};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, nullptr, 0, nullptr);
        }

        /* Mode 2: propagate block prefixes to all elements */
        scanPC.N = scan_N;
        scanPC.mode = 2;
        vkCmdPushConstants(cmd, phys.scanPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
        vkCmdDispatch(cmd, scan_groups, 1, 1);

        {
            VkMemoryBarrier bar = {};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, nullptr, 0, nullptr);
        }

        /* Step 4: Reorder — write particles to sorted Cartesian positions */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.reorderPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.reorderPipelineLayout, 0, 1, &phys.descSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.reorderPipelineLayout, 2, 1, &phys.gradedSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.reorderPipelineLayout, 3, 1, &phys.reorderSet, 0, nullptr);
        ReorderPushConstants reorderPC = {};
        reorderPC.N = (uint32_t)phys.N;
        vkCmdPushConstants(cmd, phys.reorderPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reorderPC), &reorderPC);
        vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

        {
            VkMemoryBarrier bar = {};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, nullptr, 0, nullptr);
        }
    }

    /* Scatter end — covers all scatter passes (cell assign + histogram + scan + reorder) */
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

    /* (Pass 3 removed — siphon computes density inline from pressure grid) */

    /* Gather end */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 3);

    /* ---- Pass 4: Constraint Solve (graded, Grade 1 only) ----
     * No position snapshot or velocity sync needed — grade separation
     * makes constraints and velocity independent by construction. */
    if (phys.constraintEnabled) {
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

        int last_nonempty_bucket = -1;
        for (int b = 0; b < 7; b++) {
            if (phys.constraintBucketCounts[b] > 0) last_nonempty_bucket = b;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          phys.constraintGradedPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.constraintGradedPipelineLayout, 0, 1, &phys.descSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.constraintGradedPipelineLayout, 1, 1, &phys.constraintSet1, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.constraintGradedPipelineLayout, 2, 1, &phys.gradedSet, 0, nullptr);

        for (uint32_t it = 0; it < phys.constraintIterations; it++) {
            for (int bucket = 0; bucket < 7; bucket++) {
                cpc.constraint_offset = phys.constraintBucketOffsets[bucket];
                cpc.constraint_count  = phys.constraintBucketCounts[bucket];
                if (cpc.constraint_count == 0) continue;

                vkCmdPushConstants(cmd, phys.constraintGradedPipelineLayout,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(cpc), &cpc);
                vkCmdDispatch(cmd, (cpc.constraint_count + 63) / 64, 1, 1);

                bool is_last = (it + 1 == phys.constraintIterations) &&
                               (bucket == last_nonempty_bucket);
                if (!is_last) {
                    vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &colorBarrier, 0, nullptr, 0, nullptr);
                }
            }
        }

        VkMemoryBarrier finalBarrier = {};
        finalBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        finalBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        finalBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &finalBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 4);

    /* ---- Phase 2.2: Collision pipeline (dynamic contact constraints) ----
     * C1 only ships the apply kernel + buffer reset. The fused
     * broadphase+resolve kernel arrives in C2 and slots in between the
     * reset and the apply. Until then, vel_delta stays zero each frame and
     * the apply kernel is a no-op (int->float conversion + add of zero).
     *
     * Order:
     *   1. vkCmdFillBuffer: zero vel_delta, contact_count, first_contact_frame
     *   2. transfer->compute barrier
     *   3. collision_apply dispatch (over rigid_count threads)
     *   4. compute->compute barrier so siphon sees the (no-op for C1) vel writes
     *
     * collision_ms metric is deferred to C2 — it would require growing the
     * timestamp pool from 8 to 9 slots per frame, which is more invasive than
     * is justified for the C1 smoke test where there's no real collision work
     * to time. */
    if (phys.collisionEnabled) {
        /* Reset the collision scratch buffers. */
        vkCmdFillBuffer(cmd, phys.velDeltaBuffer, 0,
                        (VkDeviceSize)phys.rigidCount * 3u * sizeof(int32_t), 0u);
        vkCmdFillBuffer(cmd, phys.contactCountBuffer, 0,
                        (VkDeviceSize)sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, phys.firstContactFrameBuffer, 0,
                        (VkDeviceSize)sizeof(uint32_t), 0xFFFFFFFFu);

        /* Transfer -> compute barrier so collision_apply sees the zeros. */
        {
            VkMemoryBarrier b = {};
            b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &b, 0, nullptr, 0, nullptr);
        }

        /* collision_resolve — graded (reads set 2, writes vel_delta set 1) */
        CollisionResolvePushConstants rpc = {};
        rpc.rigid_base    = phys.rigidBaseIndex;
        rpc.rigid_count   = phys.rigidCount;
        rpc.dt            = dt;
        rpc.frame_number  = (uint32_t)frame;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          phys.collisionResolveGradedPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionResolveGradedPipelineLayout, 0, 1, &phys.descSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionResolveGradedPipelineLayout, 1, 1, &phys.collisionSet1, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionResolveGradedPipelineLayout, 2, 1, &phys.gradedSet, 0, nullptr);
        vkCmdPushConstants(cmd, phys.collisionResolveGradedPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), &rpc);
        {
            uint32_t groups = (phys.rigidCount + 15u) / 16u;
            vkCmdDispatch(cmd, groups, groups, 1);
        }

        /* Barrier: resolve writes to vel_delta must be visible to apply. */
        {
            VkMemoryBarrier b = {};
            b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &b, 0, nullptr, 0, nullptr);
        }

        /* collision_apply — graded (decomposes Cartesian vel_delta → Grade 1) */
        CollisionApplyPushConstants apc = {};
        apc.rigid_base  = phys.rigidBaseIndex;
        apc.rigid_count = phys.rigidCount;
        apc.dt          = dt;
        apc._pad        = 0;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          phys.collisionApplyGradedPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionApplyGradedPipelineLayout, 0, 1, &phys.descSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionApplyGradedPipelineLayout, 1, 1, &phys.collisionSet1, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            phys.collisionApplyGradedPipelineLayout, 2, 1, &phys.gradedSet, 0, nullptr);
        vkCmdPushConstants(cmd, phys.collisionApplyGradedPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(apc), &apc);
        vkCmdDispatch(cmd, (phys.rigidCount + 63u) / 64u, 1, 1);

        /* Compute -> compute barrier so siphon sees vel writes. */
        {
            VkMemoryBarrier b = {};
            b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &b, 0, nullptr, 0, nullptr);
        }
    }

    /* Collision end — written unconditionally so collision_ms is always
     * defined. When collisionEnabled is false, collision_ms reads ≈ 0. */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 5);

    /* ---- Cylindrical density grid (before siphon) ---------------------- */
    /* Clear cylindrical density grid */
    vkCmdFillBuffer(cmd, phys.cylDensityBuffer, 0,
                    V21_CYL_CELLS * sizeof(uint32_t), 0);
    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    /* Cylindrical scatter: particles → (r, phi, y) bins */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.cylScatterPipeline);
    VkDescriptorSet cylScSets[2] = { phys.descSet, phys.cylScatterSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cylScatterPipelineLayout, 0, 2, cylScSets, 0, nullptr);
    int cylScN = phys.N;
    vkCmdPushConstants(cmd, phys.cylScatterPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &cylScN);
    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    /* Cylindrical stencil: density → pressure gradients in (r, phi, y) */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.cylStencilPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cylStencilPipelineLayout, 0, 1, &phys.cylStencilSet, 0, nullptr);
    float cylPressureK = 0.01f;
    vkCmdPushConstants(cmd, phys.cylStencilPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &cylPressureK);
    vkCmdDispatch(cmd, (V21_CYL_CELLS + 255) / 256, 1, 1);

    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    /* ---- Siphon (physics) -------------------------------------------- */
    SiphonPushConstants push = {};
    push.N = phys.N;
    push.time = sim_time;
    push.dt = dt;
    push.BH_MASS = 100.0f;
    push.FIELD_STRENGTH = 0.01f;
    push.FIELD_FALLOFF = 100.0f;
    push.TANGENT_SCALE = 2.0f;
    push.seam_bits = 0x03;  /* SEAM_FULL */
    push.bias = 0.75f;

    /* Squaragon V2: use the clean Cartesian siphon pipeline (siphon.comp).
     * This is the ground truth force model — 3 channels + gravity.
     * The graded/packed variants are legacy and don't implement
     * the correct Squaragon physics. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.pipeline);
    VkDescriptorSet siphonSets[2] = { phys.descSet, phys.siphonSet1 };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.pipelineLayout, 0, 2, siphonSets, 0, nullptr);
    vkCmdPushConstants(cmd, phys.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Memory barrier: siphon (or graded_to_cartesian) writes must be visible
     * to the projection pass and transfer reads (oracle readback). */
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    /* Siphon end */
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        phys.queryPool, base + 6);
}

/* ========================================================================
 * GRADE-SEPARATED STATE — Phase 3.1 scaffolding
 * ======================================================================== */

void initGradedCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    int N = phys.N;

    /* --- Allocate 10 graded SSBOs (all float[N]) --- */
    for (int i = 0; i < VK_GRADED_NUM_BINDINGS; i++) {
        createSSBO(ctx.device, ctx.physicalDevice,
                   phys.graded_buffers[i], phys.graded_memory[i],
                   (size_t)N * sizeof(float));
    }

    /* --- Descriptor set layout for set 2 (10 storage buffers) --- */
    VkDescriptorSetLayoutBinding bindings[VK_GRADED_NUM_BINDINGS] = {};
    for (int i = 0; i < VK_GRADED_NUM_BINDINGS; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = VK_GRADED_NUM_BINDINGS;
    layoutInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
                                &phys.gradedSetLayout);

    /* --- cartesian_to_graded pipeline: set 0 (read) + set 2 (write) --- */
    {
        VkDescriptorSetLayout sets[2] = { phys.descLayout, phys.gradedSetLayout };
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(CartesianToGradedPushConstants);
        VkPipelineLayoutCreateInfo plInfo = {};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 3;  /* set 0, (skip set 1), set 2 */
        /* We need a dummy set 1 layout to keep set indices correct.
         * Use an empty layout. */
        VkDescriptorSetLayoutCreateInfo emptyInfo = {};
        emptyInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        emptyInfo.bindingCount = 0;
        VkDescriptorSetLayout emptyLayout;
        vkCreateDescriptorSetLayout(ctx.device, &emptyInfo, nullptr, &emptyLayout);
        VkDescriptorSetLayout allSets[3] = { phys.descLayout, emptyLayout, phys.gradedSetLayout };
        plInfo.pSetLayouts = allSets;
        plInfo.setLayoutCount = 3;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                               &phys.cartToGradedPipelineLayout);

        auto code = readShaderFile("cartesian_to_graded.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.cartToGradedPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                 &phys.cartToGradedPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, emptyLayout, nullptr);
    }

    /* --- graded_to_cartesian pipeline: set 0 (write) + set 2 (read) --- */
    {
        VkDescriptorSetLayoutCreateInfo emptyInfo = {};
        emptyInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        emptyInfo.bindingCount = 0;
        VkDescriptorSetLayout emptyLayout;
        vkCreateDescriptorSetLayout(ctx.device, &emptyInfo, nullptr, &emptyLayout);
        VkDescriptorSetLayout allSets[3] = { phys.descLayout, emptyLayout, phys.gradedSetLayout };
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(GradedToCartesianPushConstants);
        VkPipelineLayoutCreateInfo plInfo = {};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 3;
        plInfo.pSetLayouts = allSets;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                               &phys.gradedToCartPipelineLayout);

        auto code = readShaderFile("graded_to_cartesian.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.gradedToCartPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                 &phys.gradedToCartPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, emptyLayout, nullptr);
    }

    /* --- Descriptor pool + set allocation for set 2 --- */
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = VK_GRADED_NUM_BINDINGS;
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &phys.gradedDescPool);

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = phys.gradedDescPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &phys.gradedSetLayout;
    vkAllocateDescriptorSets(ctx.device, &dsInfo, &phys.gradedSet);

    /* --- Write descriptors for set 2 --- */
    VkDescriptorBufferInfo bufInfos[VK_GRADED_NUM_BINDINGS];
    VkWriteDescriptorSet writes[VK_GRADED_NUM_BINDINGS];
    for (int i = 0; i < VK_GRADED_NUM_BINDINGS; i++) {
        bufInfos[i] = { phys.graded_buffers[i], 0, VK_WHOLE_SIZE };
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = phys.gradedSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, VK_GRADED_NUM_BINDINGS, writes, 0, nullptr);

    /* --- Cylindrical density grid: allocate buffers --- */
    {
        size_t cyl_uint_sz  = V21_CYL_CELLS * sizeof(uint32_t);
        size_t cyl_float_sz = V21_CYL_CELLS * sizeof(float);
        createSSBO(ctx.device, ctx.physicalDevice, phys.cylDensityBuffer,
                   phys.cylDensityMemory, cyl_uint_sz);
        createSSBO(ctx.device, ctx.physicalDevice, phys.cylPressureRBuffer,
                   phys.cylPressureRMemory, cyl_float_sz);
        createSSBO(ctx.device, ctx.physicalDevice, phys.cylPressurePhiBuffer,
                   phys.cylPressurePhiMemory, cyl_float_sz);
        createSSBO(ctx.device, ctx.physicalDevice, phys.cylPressureYBuffer,
                   phys.cylPressureYMemory, cyl_float_sz);
        printf("[vk-compute] Cylindrical grid: %d×%d×%d = %d cells (%.1f MB)\n",
               V21_CYL_NR, V21_CYL_NPHI, V21_CYL_NY, V21_CYL_CELLS,
               (float)(cyl_uint_sz + 3*cyl_float_sz) / (1024*1024));
    }

    /* --- Cylindrical scatter pipeline: set 0 (particles) + set 1 (cyl_density) --- */
    {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &b;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.cylScatterSetLayout);

        VkDescriptorSetLayout sets[2] = { phys.descLayout, phys.cylScatterSetLayout };
        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(int);  /* just N */
        VkPipelineLayoutCreateInfo pl = {};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 2;
        pl.pSetLayouts = sets;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pl, nullptr, &phys.cylScatterPipelineLayout);

        auto code = readShaderFile("cyl_scatter.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.cylScatterPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.cylScatterPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo dpi = {};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &phys.cylScatterDescPool);

        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.cylScatterDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &phys.cylScatterSetLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &phys.cylScatterSet);

        VkDescriptorBufferInfo bi = {phys.cylDensityBuffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = phys.cylScatterSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
        printf("[vk-compute] Cylindrical scatter pipeline created\n");
    }

    /* --- Cylindrical stencil pipeline: set 0 (density + pressure_r/phi/y) --- */
    {
        VkDescriptorSetLayoutBinding binds[4] = {};
        for (int b = 0; b < 4; b++) {
            binds[b].binding = b;
            binds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[b].descriptorCount = 1;
            binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 4;
        li.pBindings = binds;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.cylStencilSetLayout);

        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(float);  /* pressure_k */
        VkPipelineLayoutCreateInfo pl = {};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &phys.cylStencilSetLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pl, nullptr, &phys.cylStencilPipelineLayout);

        auto code = readShaderFile("cyl_stencil.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.cylStencilPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.cylStencilPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo dpi = {};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &phys.cylStencilDescPool);

        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.cylStencilDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &phys.cylStencilSetLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &phys.cylStencilSet);

        VkDescriptorBufferInfo bis[4] = {
            {phys.cylDensityBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureRBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressurePhiBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureYBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet ws[4] = {};
        for (int i = 0; i < 4; i++) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = phys.cylStencilSet;
            ws[i].dstBinding = i;
            ws[i].descriptorCount = 1;
            ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[i].pBufferInfo = &bis[i];
        }
        vkUpdateDescriptorSets(ctx.device, 4, ws, 0, nullptr);
        printf("[vk-compute] Cylindrical stencil pipeline created\n");
    }

    /* --- Siphon density feedback set 1: cyl_density + pressure_r/phi/y --- */
    {
        VkDescriptorSetLayoutBinding densBindings[4] = {};
        for (int b = 0; b < 4; b++) {
            densBindings[b].binding = b;
            densBindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            densBindings[b].descriptorCount = 1;
            densBindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo densLayoutInfo = {};
        densLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        densLayoutInfo.bindingCount = 4;
        densLayoutInfo.pBindings = densBindings;
        vkCreateDescriptorSetLayout(ctx.device, &densLayoutInfo, nullptr,
                                    &phys.siphonDensitySetLayout);

        VkDescriptorPoolSize densPoolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo densPoolInfo = {};
        densPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        densPoolInfo.maxSets = 1;
        densPoolInfo.poolSizeCount = 1;
        densPoolInfo.pPoolSizes = &densPoolSize;
        vkCreateDescriptorPool(ctx.device, &densPoolInfo, nullptr,
                               &phys.siphonDensityDescPool);

        VkDescriptorSetAllocateInfo densAllocInfo = {};
        densAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        densAllocInfo.descriptorPool = phys.siphonDensityDescPool;
        densAllocInfo.descriptorSetCount = 1;
        densAllocInfo.pSetLayouts = &phys.siphonDensitySetLayout;
        vkAllocateDescriptorSets(ctx.device, &densAllocInfo, &phys.siphonDensitySet);

        VkDescriptorBufferInfo densBufInfos[4] = {
            {phys.cylDensityBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureRBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressurePhiBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureYBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet densWrites[4] = {};
        for (int i = 0; i < 4; i++) {
            densWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            densWrites[i].dstSet = phys.siphonDensitySet;
            densWrites[i].dstBinding = i;
            densWrites[i].descriptorCount = 1;
            densWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            densWrites[i].pBufferInfo = &densBufInfos[i];
        }
        vkUpdateDescriptorSets(ctx.device, 4, densWrites, 0, nullptr);
    }

    /* --- Graded siphon pipeline (Phase 3.2): set 0 (pump) + set 1 (density) + set 2 (graded) --- */
    {
        VkDescriptorSetLayout allSets[3] = {
            phys.descLayout,              /* set 0: particle SSBOs */
            phys.siphonDensitySetLayout,  /* set 1: grid_density + particle_cell */
            phys.gradedSetLayout          /* set 2: graded state */
        };

        /* Same push constants as original siphon */
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(SiphonPushConstants);
        VkPipelineLayoutCreateInfo plInfo = {};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 3;
        plInfo.pSetLayouts = allSets;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                               &phys.siphonGradedPipelineLayout);

        const char* siphon_spv = phys.headlessMode
            ? "siphon_graded_headless.spv" : "siphon_graded.spv";
        auto code = readShaderFile(siphon_spv);
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.siphonGradedPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                 &phys.siphonGradedPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        printf("[vk-compute] Graded siphon pipeline created\n");
    }

    /* --- Graded constraint pipeline (Phase 3.3): set 0 + set 1 + set 2 ---
     * Only created if constraints are enabled (initConstraintCompute already ran). */
    if (phys.constraintEnabled) {
        VkDescriptorSetLayout allSets[3] = {
            phys.descLayout,           /* set 0: siphon particle layout (compat) */
            phys.constraintSet1Layout, /* set 1: pairs + rest + inv_m */
            phys.gradedSetLayout       /* set 2: graded particle state */
        };
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(ConstraintPushConstants);
        VkPipelineLayoutCreateInfo plInfo = {};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 3;
        plInfo.pSetLayouts = allSets;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                               &phys.constraintGradedPipelineLayout);

        auto code = readShaderFile("constraint_solve_graded.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo cpInfo = {};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName = "main";
        cpInfo.layout = phys.constraintGradedPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                 &phys.constraintGradedPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
        printf("[vk-compute] Graded constraint pipeline created\n");
    }

    /* --- Graded collision pipelines (Phase 3.4): set 0 + set 1 (collision) + set 2 ---
     * Resolve uses CollisionResolvePushConstants, Apply uses CollisionApplyPushConstants.
     * Both share the same 3-set layout with the collision set 1. */
    if (phys.collisionEnabled) {
        VkDescriptorSetLayout allSets[3] = {
            phys.descLayout,          /* set 0 */
            phys.collisionSet1Layout, /* set 1: collision data */
            phys.gradedSetLayout      /* set 2 */
        };

        /* Resolve pipeline */
        {
            VkPushConstantRange pcRange = {};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.size = sizeof(CollisionResolvePushConstants);
            VkPipelineLayoutCreateInfo plInfo = {};
            plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plInfo.setLayoutCount = 3;
            plInfo.pSetLayouts = allSets;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges = &pcRange;
            vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                                   &phys.collisionResolveGradedPipelineLayout);

            auto code = readShaderFile("collision_resolve_graded.spv");
            VkShaderModule mod = createShaderModule(ctx.device, code);
            VkComputePipelineCreateInfo cpInfo = {};
            cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpInfo.stage.module = mod;
            cpInfo.stage.pName = "main";
            cpInfo.layout = phys.collisionResolveGradedPipelineLayout;
            vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                     &phys.collisionResolveGradedPipeline);
            vkDestroyShaderModule(ctx.device, mod, nullptr);
        }

        /* Apply pipeline */
        {
            VkPushConstantRange pcRange = {};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.size = sizeof(CollisionApplyPushConstants);
            VkPipelineLayoutCreateInfo plInfo = {};
            plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plInfo.setLayoutCount = 3;
            plInfo.pSetLayouts = allSets;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges = &pcRange;
            vkCreatePipelineLayout(ctx.device, &plInfo, nullptr,
                                   &phys.collisionApplyGradedPipelineLayout);

            auto code = readShaderFile("collision_apply_graded.spv");
            VkShaderModule mod = createShaderModule(ctx.device, code);
            VkComputePipelineCreateInfo cpInfo = {};
            cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpInfo.stage.module = mod;
            cpInfo.stage.pName = "main";
            cpInfo.layout = phys.collisionApplyGradedPipelineLayout;
            vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr,
                                     &phys.collisionApplyGradedPipeline);
            vkDestroyShaderModule(ctx.device, mod, nullptr);
        }

        printf("[vk-compute] Graded collision pipelines created (resolve + apply)\n");
    }

    /* graded is now the only path (Phase 3.5) */
    printf("[vk-compute] Grade-separated buffers initialized (%d bindings, %.1f MB)\n",
           VK_GRADED_NUM_BINDINGS,
           (float)VK_GRADED_NUM_BINDINGS * N * sizeof(float) / (1024.0f * 1024.0f));
}

/* ========================================================================
 * COUNTING SORT — replaces shard-based atomic scatter
 * ======================================================================== */

void initCountingSortCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    /* --- Allocate buffers --- */
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.cellOffsetBuffer, phys.cellOffsetMemory,
               (size_t)V21_GRID_CELLS * sizeof(uint32_t));
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.scanBlockSumsBuffer, phys.scanBlockSumsMemory,
               1024 * sizeof(uint32_t));
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.writeCounterBuffer, phys.writeCounterMemory,
               (size_t)V21_GRID_CELLS * sizeof(uint32_t));

    /* --- Descriptor pool (all 3 pipelines share one pool) --- */
    VkDescriptorPoolSize poolSizes[1] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 20;  /* generous for 3 sets */
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 3;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &phys.countingSortDescPool);

    /* --- Scan pipeline: set 0 = {data, block_sums} ---
     * data = gridDensityBuffer (histogram, scanned in-place to become offsets)
     * Actually we need cell_offset as a COPY of the density, so scan doesn't
     * destroy the density. We'll copy gridDensity → cellOffset, then scan cellOffset. */
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.scanSetLayout);

        VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScanPushConstants)};
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &phys.scanSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.scanPipelineLayout);

        auto code = readShaderFile("scatter_scan.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.scanPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.scanPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        /* Allocate + write descriptor set */
        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = phys.countingSortDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &phys.scanSetLayout;
        vkAllocateDescriptorSets(ctx.device, &dsai, &phys.scanSet);

        VkDescriptorBufferInfo bufs[2] = {
            { phys.cellOffsetBuffer, 0, VK_WHOLE_SIZE },
            { phys.scanBlockSumsBuffer, 0, VK_WHOLE_SIZE }
        };
        VkWriteDescriptorSet writes[2] = {};
        for (int i = 0; i < 2; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = phys.scanSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufs[i];
        }
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }

    /* --- Histogram pipeline: set 0 = {particle_cell, cell_count(=gridDensity)} --- */
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.histogramSetLayout);

        VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HistogramPushConstants)};
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &phys.histogramSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.histogramPipelineLayout);

        auto code = readShaderFile("scatter_histogram.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.histogramPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.histogramPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = phys.countingSortDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &phys.histogramSetLayout;
        vkAllocateDescriptorSets(ctx.device, &dsai, &phys.histogramSet);

        VkDescriptorBufferInfo bufs[2] = {
            { phys.particleCellBuffer, 0, VK_WHOLE_SIZE },
            { phys.gridDensityBuffer, 0, VK_WHOLE_SIZE }
        };
        VkWriteDescriptorSet writes[2] = {};
        for (int i = 0; i < 2; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = phys.histogramSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufs[i];
        }
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }

    /* --- Reorder pipeline: set 0 (Cartesian write) + set 2 (graded read)
     *     + set 3 = {particle_cell, cell_offset, write_counter} --- */
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 3;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.reorderSetLayout);

        /* Pipeline layout: set 0 (Cartesian), set 1 (empty), set 2 (graded), set 3 (sort) */
        VkDescriptorSetLayoutCreateInfo emptyInfo = {};
        emptyInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        emptyInfo.bindingCount = 0;
        VkDescriptorSetLayout emptyLayout;
        vkCreateDescriptorSetLayout(ctx.device, &emptyInfo, nullptr, &emptyLayout);

        VkDescriptorSetLayout allSets[4] = {
            phys.descLayout,        /* set 0: Cartesian (write pos) */
            emptyLayout,            /* set 1: unused */
            phys.gradedSetLayout,   /* set 2: graded (read) */
            phys.reorderSetLayout   /* set 3: sort buffers */
        };
        VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReorderPushConstants)};
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 4;
        pli.pSetLayouts = allSets;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.reorderPipelineLayout);
        vkDestroyDescriptorSetLayout(ctx.device, emptyLayout, nullptr);

        auto code = readShaderFile("scatter_reorder.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.reorderPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.reorderPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = phys.countingSortDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &phys.reorderSetLayout;
        vkAllocateDescriptorSets(ctx.device, &dsai, &phys.reorderSet);

        VkDescriptorBufferInfo bufs[3] = {
            { phys.particleCellBuffer, 0, VK_WHOLE_SIZE },
            { phys.cellOffsetBuffer, 0, VK_WHOLE_SIZE },
            { phys.writeCounterBuffer, 0, VK_WHOLE_SIZE }
        };
        VkWriteDescriptorSet writes[3] = {};
        for (int i = 0; i < 3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = phys.reorderSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufs[i];
        }
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }

    /* --- Packed siphon pipeline: single AoS struct + fused Cartesian projection ---
     * Skipped when forward siphon is active — saves ~1.5 GB at 20M particles.
     * Descriptor set 0 layout:
     *   binding 0: Particle[N] (packed struct, 80 bytes each)
     *   binding 1-6: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z (Cartesian output)
     * Uses the same SiphonPushConstants as the graded siphon. */
    if (!phys.forwardSiphonEnabled)
    {
        size_t packed_size = (size_t)phys.N * 80;  /* 80 bytes per particle (std430 padded) */
        createSSBO(ctx.device, ctx.physicalDevice,
                   phys.packedParticleBuffer, phys.packedParticleMemory, packed_size);

        /* Descriptor set layout: 7 bindings (1 packed + 6 Cartesian out) */
        VkDescriptorSetLayoutBinding bindings[7] = {};
        for (int b = 0; b < 7; b++) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 7;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.packedSiphonSetLayout);

        VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SiphonPushConstants)};
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &phys.packedSiphonSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.packedSiphonPipelineLayout);

        auto code = readShaderFile("siphon_packed.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.packedSiphonPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.packedSiphonPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        /* Descriptor pool */
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &ps;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &phys.packedSiphonDescPool);

        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = phys.packedSiphonDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &phys.packedSiphonSetLayout;
        vkAllocateDescriptorSets(ctx.device, &dsai, &phys.packedSiphonSet);

        /* Write descriptors: binding 0 = packed, binding 1-6 = Cartesian pos/vel */
        VkDescriptorBufferInfo bufInfos[7] = {
            { phys.packedParticleBuffer, 0, VK_WHOLE_SIZE },
            { phys.soa_buffers[0], 0, VK_WHOLE_SIZE },  /* pos_x */
            { phys.soa_buffers[1], 0, VK_WHOLE_SIZE },  /* pos_y */
            { phys.soa_buffers[2], 0, VK_WHOLE_SIZE },  /* pos_z */
            { phys.soa_buffers[3], 0, VK_WHOLE_SIZE },  /* vel_x */
            { phys.soa_buffers[4], 0, VK_WHOLE_SIZE },  /* vel_y */
            { phys.soa_buffers[5], 0, VK_WHOLE_SIZE },  /* vel_z */
        };
        VkWriteDescriptorSet writes[7] = {};
        for (int b = 0; b < 7; b++) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = phys.packedSiphonSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bufInfos[b];
        }
        vkUpdateDescriptorSets(ctx.device, 7, writes, 0, nullptr);

        /* AoS packing is SLOWER than SoA on GPU (breaks warp coalescing).
         * Packed siphon measured 13.44 ms vs SoA 12.08 ms at 20M particles.
         * Keep infrastructure for reference but disable by default. */
        phys.packedSiphonEnabled = false;
        printf("[vk-compute] Packed siphon pipeline created (%.1f MB packed buffer)\n",
               (float)packed_size / (1024.0f * 1024.0f));
    }

    /* Counting sort is slower than atomic scatter at ≤20M particles on Turing.
     * Atomic scatter: 1.986 ms. Counting sort: 33 ms (reorder bandwidth dominates).
     * Keep infrastructure for future use at higher particle counts or different
     * GPU architectures where atomic contention becomes the bottleneck.
     * Toggle via CLI flag --counting-sort when needed. */
    phys.countingSortEnabled = false;
    printf("[vk-compute] Counting sort initialized (disabled — atomic scatter faster at this scale)\n");
}

void dispatchCartesianToGraded(PhysicsCompute& phys, VkCommandBuffer cmd) {


    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      phys.cartToGradedPipeline);

    VkDescriptorSet sets[3] = { phys.descSet, VK_NULL_HANDLE, phys.gradedSet };
    /* Bind set 0 and set 2 only (set 1 is unused). */
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cartToGradedPipelineLayout, 0, 1, &sets[0], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cartToGradedPipelineLayout, 2, 1, &sets[2], 0, nullptr);

    CartesianToGradedPushConstants pc = {};
    pc.N = phys.N;
    pc.BH_MASS = 100.0f;
    vkCmdPushConstants(cmd, phys.cartToGradedPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Barrier: graded writes must complete before any reader. */
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void dispatchGradedToCartesian(PhysicsCompute& phys, VkCommandBuffer cmd) {


    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      phys.gradedToCartPipeline);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.gradedToCartPipelineLayout, 0, 1, &phys.descSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.gradedToCartPipelineLayout, 2, 1, &phys.gradedSet, 0, nullptr);

    GradedToCartesianPushConstants pc = {};
    pc.N = phys.N;
    vkCmdPushConstants(cmd, phys.gradedToCartPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Barrier: Cartesian writes must complete before oracle/rendering. */
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
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
                         const float* viewProj, int render_mode) {
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
    projPush.render_mode = render_mode;
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

    /* Projection end */
    {
        uint32_t base = phys.queryFrame * 9;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            phys.queryPool, base + 7);
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
    /* Diagnostic modes use distinct colormaps for visual clarity */
    switch (render_mode) {
        case 1: tonePush.colormap = 2; break;  /* alive = blue-white */
        case 2: tonePush.colormap = 0; break;  /* nova = thermal (red/orange) */
        case 3: tonePush.colormap = 4; break;  /* crystal = new ice colormap */
        default: tonePush.colormap = 3; break;  /* all = hopf (normal) */
    }
    vkCmdPushConstants(cmd, phys.tonePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(tonePush), &tonePush);

    vkCmdDraw(cmd, 3, 1, 0, 0);  /* Fullscreen triangle */

    vkCmdEndRenderPass(cmd);

    /* Tone-map end */
    {
        uint32_t base = phys.queryFrame * 9;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            phys.queryPool, base + 8);
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
                    double* out_collision_ms,
                    double* out_siphon_ms,
                    double* out_project_ms,
                    double* out_tonemap_ms) {
    uint32_t slot = phys.queryFrame ^ 1;
    if (!phys.queryValid[slot]) return false;

    uint32_t base = slot * 9;
    uint64_t ts[9];
    VkResult r = vkGetQueryPoolResults(device, phys.queryPool, base, 9,
        sizeof(ts), ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return false;

    double nsPerTick = (double)phys.timestampPeriodNs;
    *out_scatter_ms    = (double)(ts[1] - ts[0]) * nsPerTick / 1.0e6;
    *out_stencil_ms    = (double)(ts[2] - ts[1]) * nsPerTick / 1.0e6;
    *out_gather_ms     = (double)(ts[3] - ts[2]) * nsPerTick / 1.0e6;
    *out_constraint_ms = (double)(ts[4] - ts[3]) * nsPerTick / 1.0e6;
    *out_collision_ms  = (double)(ts[5] - ts[4]) * nsPerTick / 1.0e6;
    *out_siphon_ms     = (double)(ts[6] - ts[5]) * nsPerTick / 1.0e6;
    *out_project_ms    = (double)(ts[7] - ts[6]) * nsPerTick / 1.0e6;
    *out_tonemap_ms    = (double)(ts[8] - ts[7]) * nsPerTick / 1.0e6;
    return true;
}

/* ========================================================================
 * READBACK — copy subset to CPU for oracle
 * ======================================================================== */

/* ========================================================================
 * ASYNC READBACK — record copies into existing command buffer
 * ======================================================================== */

void recordReadbackCopies(PhysicsCompute& phys, VkCommandBuffer cmd, int count) {
    if (count <= 0) return;
    if (count > ORACLE_SUBSET_SIZE) count = ORACLE_SUBSET_SIZE;

    size_t sz = (size_t)count * sizeof(float);

    /* Barrier: ensure siphon writes are visible to transfer reads */
    VkMemoryBarrier bar = {};
    bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &bar, 0, nullptr, 0, nullptr);

    const int src_bindings[8] = {0, 1, 2, 3, 4, 5, 12, 14};
    size_t offset = 0;
    for (int i = 0; i < 8; i++) {
        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size = sz;
        vkCmdCopyBuffer(cmd, phys.soa_buffers[src_bindings[i]],
                        phys.staging, 1, &region);
        offset += sz;
    }

    /* Graded-state copies (r, vel_r, phi, omega_orb) */
    const int graded_bindings[4] = {0, 3, 5, 6};
    for (int i = 0; i < 4; i++) {
        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size = sz;
        vkCmdCopyBuffer(cmd, phys.graded_buffers[graded_bindings[i]],
                        phys.staging, 1, &region);
        offset += sz;
    }
}

void consumeReadback(PhysicsCompute& phys,
                     float* out_pos_x, float* out_pos_y, float* out_pos_z,
                     float* out_vel_x, float* out_vel_y, float* out_vel_z,
                     float* out_theta, float* out_pump_scale,
                     uint8_t* out_flags, int count,
                     float* out_r, float* out_vel_r,
                     float* out_phi, float* out_omega_orb) {
    if (count <= 0) return;
    if (count > ORACLE_SUBSET_SIZE) count = ORACLE_SUBSET_SIZE;

    size_t sz = (size_t)count * sizeof(float);
    char* mapped = (char*)phys.stagingMapped;

    memcpy(out_pos_x, mapped + 0 * sz, sz);
    memcpy(out_pos_y, mapped + 1 * sz, sz);
    memcpy(out_pos_z, mapped + 2 * sz, sz);
    memcpy(out_vel_x, mapped + 3 * sz, sz);
    memcpy(out_vel_y, mapped + 4 * sz, sz);
    memcpy(out_vel_z, mapped + 5 * sz, sz);
    memcpy(out_theta, mapped + 6 * sz, sz);

    /* pump_scale is a stub — fill with 1.0 */
    for (int i = 0; i < count; i++) out_pump_scale[i] = 1.0f;

    /* Flags from binding 14 (slot 7), uint32 → uint8 */
    const uint32_t* flags32 = (const uint32_t*)(mapped + 7 * sz);
    for (int i = 0; i < count; i++) {
        out_flags[i] = (uint8_t)(flags32[i] & 0xFF);
    }

    /* Graded-state arrays */
    if (out_r && out_vel_r && out_phi && out_omega_orb) {
        memcpy(out_r,         mapped +  8 * sz, sz);
        memcpy(out_vel_r,     mapped +  9 * sz, sz);
        memcpy(out_phi,       mapped + 10 * sz, sz);
        memcpy(out_omega_orb, mapped + 11 * sz, sz);
    }
}

/* ========================================================================
 * LEGACY SYNC READBACK — kept for headless mode
 * ======================================================================== */

void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       float* out_theta, float* out_pump_scale,
                       uint8_t* out_flags,
                       int count,
                       float* out_r, float* out_vel_r,
                       float* out_phi, float* out_omega_orb) {
    if (count <= 0) return;
    if (count > ORACLE_SUBSET_SIZE) count = ORACLE_SUBSET_SIZE;

    size_t sz = (size_t)count * sizeof(float);

    /* Source bindings: pos(0-2), vel(3-5), theta(12), flags(14).
     * Binding 7 (pump_scale) and 6 (packed_meta) are stubs — skip them.
     * Flags come from binding 14 directly (uint32, bit 0 = active). */
    const int src_bindings[8] = {0, 1, 2, 3, 4, 5, 12, 14};

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
    for (int i = 0; i < 8; i++) {
        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size = sz;
        vkCmdCopyBuffer(cmd, phys.soa_buffers[src_bindings[i]],
                        phys.staging, 1, &region);
        offset += sz;
    }

    /* Optional graded-state readback for physics diagnostics.
     * Copies r, vel_r, phi, omega_orb from graded_buffers (set 2)
     * into staging offsets 9*sz..12*sz. */
    bool do_graded = (out_r && out_vel_r && out_phi && out_omega_orb);
    if (do_graded) {
        const int graded_bindings[4] = {0, 3, 5, 6};  /* r, vel_r, phi, omega_orb */
        for (int i = 0; i < 4; i++) {
            VkBufferCopy region = {};
            region.srcOffset = 0;
            region.dstOffset = offset;
            region.size = sz;
            vkCmdCopyBuffer(cmd, phys.graded_buffers[graded_bindings[i]],
                            phys.staging, 1, &region);
            offset += sz;
        }
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

    /* pump_scale is a stub — fill with 1.0 for diagnostics compatibility */
    for (int i = 0; i < count; i++) out_pump_scale[i] = 1.0f;

    /* Flags from binding 14 (slot 7 in readback), uint32 → uint8 */
    const uint32_t* flags32 = (const uint32_t*)(mapped + 7 * sz);
    for (int i = 0; i < count; i++) {
        out_flags[i] = (uint8_t)(flags32[i] & 0xFF);
    }

    /* Copy graded-state arrays if requested */
    if (do_graded) {
        memcpy(out_r,         mapped +  8 * sz, sz);
        memcpy(out_vel_r,     mapped +  9 * sz, sz);
        memcpy(out_phi,       mapped + 10 * sz, sz);
        memcpy(out_omega_orb, mapped + 11 * sz, sz);
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

    /* Binding 6 is a stub — pump_state is not part of allocator state.
     * Return zeros instead of reading from a 4-byte buffer. */
    memset(out_states, 0, (size_t)count * sizeof(int));
}

/* ========================================================================
 * FOURIER RENDERER — sparse sampling + analytic reconstruction
 * ======================================================================== */

void initFourierRender(PhysicsCompute& phys, VulkanContext& ctx) {
    size_t coeff_sz = FOURIER_TOTAL_COEFFS * sizeof(uint32_t);
    size_t shell_sz = FOURIER_TOTAL_CELLS * sizeof(uint32_t);

    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fourierCosAccBuffer, phys.fourierCosAccMemory, coeff_sz);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fourierSinAccBuffer, phys.fourierSinAccMemory, coeff_sz);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fourierShellCountBuffer, phys.fourierShellCountMemory, shell_sz);

    /* Sample pass: set 0 = particle SSBOs (shared), set 1 = accumulators */
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        for (int b = 0; b < 3; b++) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 3;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fourierSampleSetLayout);

        VkDescriptorSetLayout sets[2] = { phys.descLayout, phys.fourierSampleSetLayout };
        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(FourierSamplePushConstants);
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 2;
        pli.pSetLayouts = sets;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.fourierSamplePipelineLayout);

        auto code = readShaderFile("fourier_sample.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.fourierSamplePipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.fourierSamplePipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* Render pass: set 0 = accumulators + density image + camera */
    {
        VkDescriptorSetLayoutBinding bindings[5] = {};
        /* 0: density image (storage image) */
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        /* 1: camera UBO */
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fourierRenderSetLayout);

        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(FourierRenderPushConstants);
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &phys.fourierRenderSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.fourierRenderPipelineLayout);

        auto code = readShaderFile("fourier_render.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.fourierRenderPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.fourierRenderPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* Descriptor pool: sample set (3 SSBOs) + render set (3 SSBOs + 1 image + 1 UBO) */
    {
        VkDescriptorPoolSize poolSizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},  /* sample set only */
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
        };
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 2;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = poolSizes;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &phys.fourierRenderDescPool);
    }

    /* Sample descriptor set (set 1: cos_acc, sin_acc, shell_count) */
    {
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.fourierRenderDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &phys.fourierSampleSetLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &phys.fourierSampleSet);

        VkDescriptorBufferInfo infos[3] = {
            {phys.fourierCosAccBuffer, 0, VK_WHOLE_SIZE},
            {phys.fourierSinAccBuffer, 0, VK_WHOLE_SIZE},
            {phys.fourierShellCountBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet ws[3] = {};
        for (int b = 0; b < 3; b++) {
            ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[b].dstSet = phys.fourierSampleSet;
            ws[b].dstBinding = b;
            ws[b].descriptorCount = 1;
            ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(ctx.device, 3, ws, 0, nullptr);
    }

    /* Render descriptor set (set 0: accumulators + density image + camera) */
    {
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.fourierRenderDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &phys.fourierRenderSetLayout;
        vkAllocateDescriptorSets(ctx.device, &ai, &phys.fourierRenderSet);

        VkDescriptorImageInfo imgInfo = {};
        imgInfo.imageView = phys.densityView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo camInfo = {phys.cameraUBO, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet ws[2] = {};
        ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet = phys.fourierRenderSet;
        ws[0].dstBinding = 0;
        ws[0].descriptorCount = 1;
        ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ws[0].pImageInfo = &imgInfo;
        ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[1].dstSet = phys.fourierRenderSet;
        ws[1].dstBinding = 1;
        ws[1].descriptorCount = 1;
        ws[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ws[1].pBufferInfo = &camInfo;
        vkUpdateDescriptorSets(ctx.device, 2, ws, 0, nullptr);
    }

    phys.fourierRenderEnabled = true;
    printf("[vk-compute] Fourier renderer initialized: %d modes × %d shells, "
           "accumulators %.1f KB\n",
           FOURIER_NUM_MODES, V21_CYL_NR,
           (float)(2 * coeff_sz + shell_sz) / 1024.0f);
}

void recordFourierRender(PhysicsCompute& phys, VkCommandBuffer cmd,
                         VulkanContext& ctx, uint32_t imageIndex,
                         const float* viewProj, int frame) {
    /* Transition density image to GENERAL for compute write */
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = phys.densityImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Clear density image + accumulator buffers */
    VkClearColorValue clearColor = {};
    clearColor.uint32[0] = 0;
    VkImageSubresourceRange clearRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, phys.densityImage, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &clearRange);
    vkCmdFillBuffer(cmd, phys.fourierCosAccBuffer, 0,
                    FOURIER_TOTAL_COEFFS * sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, phys.fourierSinAccBuffer, 0,
                    FOURIER_TOTAL_COEFFS * sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, phys.fourierShellCountBuffer, 0,
                    FOURIER_TOTAL_CELLS * sizeof(uint32_t), 0);

    /* Barrier: clear → compute */
    {
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 1, &barrier);
    }

    /* Update camera UBO */
    memcpy(phys.cameraMapped, viewProj, 16 * sizeof(float));
    /* Write width/height after the 4x4 matrix + zoom + aspect */
    float* camData = (float*)phys.cameraMapped;
    /* ProjectCameraUBO layout: float[16] viewProj, float zoom, float aspect, int width, int height */
    camData[16] = 1.0f;  /* zoom */
    camData[17] = (float)ctx.swapchainExtent.width / (float)ctx.swapchainExtent.height;
    ((int*)phys.cameraMapped)[18] = (int)ctx.swapchainExtent.width;
    ((int*)phys.cameraMapped)[19] = (int)ctx.swapchainExtent.height;

    /* Analytic hopfion field — no sampling pass needed */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.fourierRenderPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.fourierRenderPipelineLayout, 0, 1, &phys.fourierRenderSet, 0, nullptr);

    FourierRenderPushConstants renderPush = {};
    renderPush.brightness = 500.0f;
    renderPush.time = (float)frame * (1.0f / 60.0f);
    renderPush.omega_pump = 0.125f;
    vkCmdPushConstants(cmd, phys.fourierRenderPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(renderPush), &renderPush);

    uint32_t gx = (ctx.swapchainExtent.width + 15) / 16;
    uint32_t gy = (ctx.swapchainExtent.height + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);

    /* Barrier: compute write → fragment read */
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Tone-map render pass (same as recordDensityRender) */
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
    viewport.width = (float)ctx.swapchainExtent.width;
    viewport.height = (float)ctx.swapchainExtent.height;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, ctx.swapchainExtent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, phys.tonePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        phys.tonePipelineLayout, 0, 1, &phys.toneDescSet, 0, nullptr);

    ToneMapPushConstants tmPush = {};
    tmPush.peak_density = (float)phys.N * 0.001f;
    tmPush.exposure = 5.0f;
    tmPush.gamma = 2.2f;
    tmPush.colormap = 3;  /* hopf */
    vkCmdPushConstants(cmd, phys.tonePipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(tmPush), &tmPush);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

/* ========================================================================
 * FORWARD SIPHON — V8-inspired double-buffered, no RMW
 * ======================================================================== */

void initForwardSiphonCompute(PhysicsCompute& phys, VulkanContext& ctx) {
    int N = phys.N;
    size_t float_sz = (size_t)N * sizeof(float);
    size_t uint_sz  = (size_t)N * sizeof(uint32_t);

    /* Buffer field sizes: r, delta_y, vel_r, vel_y, phi, omega_orb, theta (float), meta (uint) */
    size_t field_sizes[FWD_SIPHON_NUM_FIELDS] = {
        float_sz, float_sz, float_sz, float_sz,
        float_sz, float_sz, float_sz, uint_sz
    };

    /* Allocate A and B copies */
    for (int f = 0; f < FWD_SIPHON_NUM_FIELDS; f++) {
        createSSBO(ctx.device, ctx.physicalDevice,
                   phys.fwdBuffersA[f], phys.fwdMemoryA[f], field_sizes[f]);
        createSSBO(ctx.device, ctx.physicalDevice,
                   phys.fwdBuffersB[f], phys.fwdMemoryB[f], field_sizes[f]);
    }

    /* Shared read-only buffers (not double-buffered) */
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fwdDeltaRBuffer, phys.fwdDeltaRMemory, float_sz);
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fwdOmegaNatBuffer, phys.fwdOmegaNatMemory, float_sz);

    /* History accumulator (single buffer, RMW) */
    createSSBO(ctx.device, ctx.physicalDevice,
               phys.fwdHistoryBuffer, phys.fwdHistoryMemory, float_sz);

    /* Copy initial state from graded set 2 into buffer A + shared buffers.
     * Graded layout: 0=r, 1=delta_r, 2=delta_y, 3=vel_r, 4=vel_y,
     *                5=phi, 6=omega_orb, 7=theta, 8=omega_nat, 9=L_tilt
     * Forward A layout: 0=r, 1=delta_y, 2=vel_r, 3=vel_y,
     *                   4=phi, 5=omega_orb, 6=theta, 7=meta */
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba = {};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = ctx.commandPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cba, &cmd);
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    /* Graded set 2 index → Forward A index mapping */
    struct { int graded_idx; int fwd_idx; size_t sz; } copies[] = {
        {0, 0, float_sz},  /* r → A[0] */
        {2, 1, float_sz},  /* delta_y → A[1] */
        {3, 2, float_sz},  /* vel_r → A[2] */
        {4, 3, float_sz},  /* vel_y → A[3] */
        {5, 4, float_sz},  /* phi → A[4] */
        {6, 5, float_sz},  /* omega_orb → A[5] */
        {7, 6, float_sz},  /* theta → A[6] */
    };
    for (auto& c : copies) {
        VkBufferCopy region = {0, 0, c.sz};
        vkCmdCopyBuffer(cmd, phys.graded_buffers[c.graded_idx],
                        phys.fwdBuffersA[c.fwd_idx], 1, &region);
    }
    /* Bindings 6 (packed_meta) and 11 (pump_history) are stubs —
     * zero-fill the forward siphon targets instead of copying. */
    vkCmdFillBuffer(cmd, phys.fwdBuffersA[7], 0, uint_sz, 0);
    /* Shared readonly: delta_r from graded[1], omega_nat from graded[8] */
    {
        VkBufferCopy region = {0, 0, float_sz};
        vkCmdCopyBuffer(cmd, phys.graded_buffers[1],
                        phys.fwdDeltaRBuffer, 1, &region);
    }
    {
        VkBufferCopy region = {0, 0, float_sz};
        vkCmdCopyBuffer(cmd, phys.graded_buffers[8],
                        phys.fwdOmegaNatBuffer, 1, &region);
    }
    vkCmdFillBuffer(cmd, phys.fwdHistoryBuffer, 0, float_sz, 0);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue);
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);

    /* --- Descriptor set layouts --- */
    /* Set 0 (input): 10 readonly = 8 fields + delta_r + omega_nat */
    {
        VkDescriptorSetLayoutBinding bindings[10] = {};
        for (int b = 0; b < 10; b++) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 10;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fwdInputSetLayout);
    }

    /* Set 1 (output): 8 writeonly */
    {
        VkDescriptorSetLayoutBinding bindings[8] = {};
        for (int b = 0; b < 8; b++) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 8;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fwdOutputSetLayout);
    }

    /* Set 2 (accumulator): 1 RMW buffer (history) */
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &binding;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fwdAccumSetLayout);
    }

    /* Set 3 (density): 1 readonly (cyl_density) */
    {
        VkDescriptorSetLayoutBinding binding = {};
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (int b = 0; b < 4; b++) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 4;
        li.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &phys.fwdDensitySetLayout);
    }

    /* Pipeline layout: 4 descriptor sets + push constants */
    {
        VkDescriptorSetLayout allSets[4] = {
            phys.fwdInputSetLayout, phys.fwdOutputSetLayout,
            phys.fwdAccumSetLayout, phys.fwdDensitySetLayout
        };
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.size = sizeof(ForwardSiphonPushConstants);
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 4;
        pli.pSetLayouts = allSets;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.fwdSiphonPipelineLayout);
    }

    /* Compute pipeline */
    {
        auto code = readShaderFile("siphon_forward.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.fwdSiphonPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.fwdSiphonPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }

    /* Descriptor pool: need 2 directions × 4 sets = 8 sets total
     * SSBOs: 2 × (10 + 8) + 1 + 1 = 38 */
    {
        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 46};
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 8;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &poolSize;
        vkCreateDescriptorPool(ctx.device, &pi, nullptr, &phys.fwdDescPool);
    }

    /* Allocate and write descriptor sets for both directions.
     *
     * Shader input layout (set 0, 10 bindings):
     *   0: r          = fields[0]     (double-buffered)
     *   1: delta_r    = fwdDeltaRBuffer  (shared readonly)
     *   2: delta_y    = fields[1]     (double-buffered)
     *   3: vel_r      = fields[2]
     *   4: vel_y      = fields[3]
     *   5: phi        = fields[4]
     *   6: omega_orb  = fields[5]
     *   7: theta      = fields[6]
     *   8: omega_nat  = fwdOmegaNatBuffer (shared readonly)
     *   9: meta       = fields[7]     (double-buffered)
     */
    auto writeInputSet = [&](VkDescriptorSet set, VkBuffer fields[FWD_SIPHON_NUM_FIELDS]) {
        /* Map fields[0..7] → shader bindings, skipping 1 (delta_r) and 8 (omega_nat) */
        const int field_to_binding[FWD_SIPHON_NUM_FIELDS] = {0, 2, 3, 4, 5, 6, 7, 9};

        VkDescriptorBufferInfo infos[10];
        VkWriteDescriptorSet writes[10] = {};
        for (int f = 0; f < FWD_SIPHON_NUM_FIELDS; f++) {
            int b = field_to_binding[f];
            infos[f] = {fields[f], 0, VK_WHOLE_SIZE};
            writes[f].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[f].dstSet = set;
            writes[f].dstBinding = b;
            writes[f].descriptorCount = 1;
            writes[f].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[f].pBufferInfo = &infos[f];
        }
        /* Binding 1: delta_r (shared readonly) */
        infos[8] = {phys.fwdDeltaRBuffer, 0, VK_WHOLE_SIZE};
        writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[8].dstSet = set;
        writes[8].dstBinding = 1;
        writes[8].descriptorCount = 1;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[8].pBufferInfo = &infos[8];
        /* Binding 8: omega_nat (shared readonly) */
        infos[9] = {phys.fwdOmegaNatBuffer, 0, VK_WHOLE_SIZE};
        writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[9].dstSet = set;
        writes[9].dstBinding = 8;
        writes[9].descriptorCount = 1;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[9].pBufferInfo = &infos[9];
        vkUpdateDescriptorSets(ctx.device, 10, writes, 0, nullptr);
    };

    auto writeOutputSet = [&](VkDescriptorSet set, VkBuffer fields[FWD_SIPHON_NUM_FIELDS]) {
        VkDescriptorBufferInfo infos[8];
        VkWriteDescriptorSet writes[8] = {};
        for (int b = 0; b < FWD_SIPHON_NUM_FIELDS; b++) {
            infos[b] = {fields[b], 0, VK_WHOLE_SIZE};
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(ctx.device, 8, writes, 0, nullptr);
    };

    auto writeAccumSet = [&](VkDescriptorSet set) {
        VkDescriptorBufferInfo info = {phys.fwdHistoryBuffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    };

    auto writeDensitySet = [&](VkDescriptorSet set) {
        VkDescriptorBufferInfo infos[4] = {
            {phys.cylDensityBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureRBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressurePhiBuffer, 0, VK_WHOLE_SIZE},
            {phys.cylPressureYBuffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet ws[4] = {};
        for (int b = 0; b < 4; b++) {
            ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[b].dstSet = set;
            ws[b].dstBinding = b;
            ws[b].descriptorCount = 1;
            ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(ctx.device, 4, ws, 0, nullptr);
    };

    /* Direction A→B: input=A, output=B */
    {
        VkDescriptorSetLayout layouts[4] = {
            phys.fwdInputSetLayout, phys.fwdOutputSetLayout,
            phys.fwdAccumSetLayout, phys.fwdDensitySetLayout
        };
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.fwdDescPool;
        ai.descriptorSetCount = 4;
        ai.pSetLayouts = layouts;
        vkAllocateDescriptorSets(ctx.device, &ai, phys.fwdSetA2B);
        writeInputSet(phys.fwdSetA2B[0], phys.fwdBuffersA);
        writeOutputSet(phys.fwdSetA2B[1], phys.fwdBuffersB);
        writeAccumSet(phys.fwdSetA2B[2]);
        writeDensitySet(phys.fwdSetA2B[3]);
    }

    /* Direction B→A: input=B, output=A */
    {
        VkDescriptorSetLayout layouts[4] = {
            phys.fwdInputSetLayout, phys.fwdOutputSetLayout,
            phys.fwdAccumSetLayout, phys.fwdDensitySetLayout
        };
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = phys.fwdDescPool;
        ai.descriptorSetCount = 4;
        ai.pSetLayouts = layouts;
        vkAllocateDescriptorSets(ctx.device, &ai, phys.fwdSetB2A);
        writeInputSet(phys.fwdSetB2A[0], phys.fwdBuffersB);
        writeOutputSet(phys.fwdSetB2A[1], phys.fwdBuffersA);
        writeAccumSet(phys.fwdSetB2A[2]);
        writeDensitySet(phys.fwdSetB2A[3]);
    }

    /* --- Projection pipeline: graded → Cartesian --- */
    {
        /* Set 0 (input): 7 buffers (6 from forward output + delta_r shared) */
        VkDescriptorSetLayoutBinding ib[7] = {};
        for (int b = 0; b < 7; b++) {
            ib[b].binding = b;
            ib[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ib[b].descriptorCount = 1;
            ib[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ili = {};
        ili.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ili.bindingCount = 7;
        ili.pBindings = ib;
        vkCreateDescriptorSetLayout(ctx.device, &ili, nullptr, &phys.fwdProjInputSetLayout);

        /* Set 1 (output): 6 Cartesian buffers */
        VkDescriptorSetLayoutBinding ob[6] = {};
        for (int b = 0; b < 6; b++) {
            ob[b].binding = b;
            ob[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ob[b].descriptorCount = 1;
            ob[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo oli = {};
        oli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        oli.bindingCount = 6;
        oli.pBindings = ob;
        vkCreateDescriptorSetLayout(ctx.device, &oli, nullptr, &phys.fwdProjOutputSetLayout);

        /* Pipeline layout */
        VkDescriptorSetLayout projSets[2] = {
            phys.fwdProjInputSetLayout, phys.fwdProjOutputSetLayout
        };
        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = sizeof(int);
        VkPipelineLayoutCreateInfo pli = {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 2;
        pli.pSetLayouts = projSets;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(ctx.device, &pli, nullptr, &phys.fwdProjPipelineLayout);

        /* Pipeline */
        auto code = readShaderFile("graded_project.spv");
        VkShaderModule mod = createShaderModule(ctx.device, code);
        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = phys.fwdProjPipelineLayout;
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &phys.fwdProjPipeline);
        vkDestroyShaderModule(ctx.device, mod, nullptr);

        /* Allocate projection descriptor sets (4 total: 2 per direction × 2 sets each).
         * Reuse the forward desc pool — add capacity. Actually, allocate from a small
         * separate pool to keep things clean. */
        VkDescriptorPoolSize projPoolSz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * (7 + 6)};
        VkDescriptorPoolCreateInfo projPoolInfo = {};
        projPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        projPoolInfo.maxSets = 4;
        projPoolInfo.poolSizeCount = 1;
        projPoolInfo.pPoolSizes = &projPoolSz;
        VkDescriptorPool projPool;
        vkCreateDescriptorPool(ctx.device, &projPoolInfo, nullptr, &projPool);
        /* Stash the pool handle — we'll leak it into the fwdDescPool cleanup for simplicity.
         * TODO: proper lifetime management if this persists. */

        /* Helper to write projection input set */
        auto writeProjInput = [&](VkDescriptorSet set, VkBuffer fields[FWD_SIPHON_NUM_FIELDS]) {
            /* Bindings 0-5: r, delta_y, vel_r, vel_y, phi, omega_orb from forward output */
            /* Binding 6: delta_r (shared) */
            VkDescriptorBufferInfo infos[7];
            VkWriteDescriptorSet ws[7] = {};
            for (int b = 0; b < 6; b++) {
                infos[b] = {fields[b], 0, VK_WHOLE_SIZE};
                ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[b].dstSet = set;
                ws[b].dstBinding = b;
                ws[b].descriptorCount = 1;
                ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ws[b].pBufferInfo = &infos[b];
            }
            infos[6] = {phys.fwdDeltaRBuffer, 0, VK_WHOLE_SIZE};
            ws[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[6].dstSet = set;
            ws[6].dstBinding = 6;
            ws[6].descriptorCount = 1;
            ws[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[6].pBufferInfo = &infos[6];
            vkUpdateDescriptorSets(ctx.device, 7, ws, 0, nullptr);
        };

        auto writeProjOutput = [&](VkDescriptorSet set) {
            VkDescriptorBufferInfo infos[6] = {
                {phys.soa_buffers[0], 0, VK_WHOLE_SIZE},  /* pos_x */
                {phys.soa_buffers[1], 0, VK_WHOLE_SIZE},  /* pos_y */
                {phys.soa_buffers[2], 0, VK_WHOLE_SIZE},  /* pos_z */
                {phys.soa_buffers[3], 0, VK_WHOLE_SIZE},  /* vel_x */
                {phys.soa_buffers[4], 0, VK_WHOLE_SIZE},  /* vel_y */
                {phys.soa_buffers[5], 0, VK_WHOLE_SIZE},  /* vel_z */
            };
            VkWriteDescriptorSet ws[6] = {};
            for (int b = 0; b < 6; b++) {
                ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[b].dstSet = set;
                ws[b].dstBinding = b;
                ws[b].descriptorCount = 1;
                ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ws[b].pBufferInfo = &infos[b];
            }
            vkUpdateDescriptorSets(ctx.device, 6, ws, 0, nullptr);
        };

        /* A: data in buffers A → project from A */
        {
            VkDescriptorSetLayout ls[2] = { phys.fwdProjInputSetLayout, phys.fwdProjOutputSetLayout };
            VkDescriptorSetAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = projPool;
            ai.descriptorSetCount = 2;
            ai.pSetLayouts = ls;
            vkAllocateDescriptorSets(ctx.device, &ai, phys.fwdProjSetA);
            writeProjInput(phys.fwdProjSetA[0], phys.fwdBuffersA);
            writeProjOutput(phys.fwdProjSetA[1]);
        }
        /* B: data in buffers B → project from B */
        {
            VkDescriptorSetLayout ls[2] = { phys.fwdProjInputSetLayout, phys.fwdProjOutputSetLayout };
            VkDescriptorSetAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = projPool;
            ai.descriptorSetCount = 2;
            ai.pSetLayouts = ls;
            vkAllocateDescriptorSets(ctx.device, &ai, phys.fwdProjSetB);
            writeProjInput(phys.fwdProjSetB[0], phys.fwdBuffersB);
            writeProjOutput(phys.fwdProjSetB[1]);
        }

        printf("[vk-compute] Forward projection pipeline created\n");
    }

    phys.forwardSiphonEnabled = true;
    phys.forwardPingPong = 0;

    size_t total = 0;
    for (int f = 0; f < FWD_SIPHON_NUM_FIELDS; f++)
        total += field_sizes[f] * 2;  /* A + B */
    total += float_sz * 3;  /* delta_r + omega_nat + history */
    printf("[vk-compute] Forward siphon initialized: %d fields × 2 + 3 shared = %.1f MB\n",
           FWD_SIPHON_NUM_FIELDS, (float)total / (1024*1024));
}

/* Dispatch cylindrical scatter + stencil passes (density grid update).
 * Reads Cartesian positions from set 0, populates cyl_density + pressure fields.
 * Reusable by both graded and forward siphon paths. */
void dispatchCylDensity(PhysicsCompute& phys, VkCommandBuffer cmd) {
    /* Clear cylindrical density grid */
    vkCmdFillBuffer(cmd, phys.cylDensityBuffer, 0,
                    V21_CYL_CELLS * sizeof(uint32_t), 0);
    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    /* Cylindrical scatter: particles → (r, phi, y) bins */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.cylScatterPipeline);
    VkDescriptorSet cylScSets[2] = { phys.descSet, phys.cylScatterSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cylScatterPipelineLayout, 0, 2, cylScSets, 0, nullptr);
    int cylScN = phys.N;
    vkCmdPushConstants(cmd, phys.cylScatterPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &cylScN);
    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    /* Cylindrical stencil: density → pressure gradients in (r, phi, y) */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.cylStencilPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.cylStencilPipelineLayout, 0, 1, &phys.cylStencilSet, 0, nullptr);
    float cylPressureK = 0.01f;
    vkCmdPushConstants(cmd, phys.cylStencilPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &cylPressureK);
    vkCmdDispatch(cmd, (V21_CYL_CELLS + 255) / 256, 1, 1);

    {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }
}

void dispatchForwardProject(PhysicsCompute& phys, VkCommandBuffer cmd) {
    /* After dispatchForwardSiphon flipped pingPong, the fresh data is in
     * the buffer that pingPong NOW points to as "input" (i.e., the one
     * that was just written to). pingPong==0 → data in A, pingPong==1 → data in B. */
    VkDescriptorSet* sets = (phys.forwardPingPong == 0)
        ? phys.fwdProjSetA : phys.fwdProjSetB;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.fwdProjPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.fwdProjPipelineLayout, 0, 2, sets, 0, nullptr);

    int n = phys.N;
    vkCmdPushConstants(cmd, phys.fwdProjPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &n);
    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);
}

void dispatchForwardSiphon(PhysicsCompute& phys, VkCommandBuffer cmd,
                           float sim_time, float dt) {
    VkDescriptorSet* sets = (phys.forwardPingPong == 0) ? phys.fwdSetA2B : phys.fwdSetB2A;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, phys.fwdSiphonPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        phys.fwdSiphonPipelineLayout, 0, 4, sets, 0, nullptr);

    ForwardSiphonPushConstants push = {};
    push.N = phys.N;
    push.time = sim_time;
    push.dt = dt;
    push.BH_MASS = 100.0f;
    push.FIELD_STRENGTH = 0.01f;
    push.bias = 0.75f;
    vkCmdPushConstants(cmd, phys.fwdSiphonPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (phys.N + 255) / 256, 1, 1);

    /* Flip for next frame */
    phys.forwardPingPong ^= 1;
}

/* ========================================================================
 * CLEANUP
 * ======================================================================== */

void cleanupPhysicsCompute(PhysicsCompute& phys, VkDevice device) {
    if (!phys.initialized) return;

    /* Forward siphon (optional) */
    if (phys.forwardSiphonEnabled) {
        for (int f = 0; f < FWD_SIPHON_NUM_FIELDS; f++) {
            vkDestroyBuffer(device, phys.fwdBuffersA[f], nullptr);
            vkFreeMemory(device, phys.fwdMemoryA[f], nullptr);
            vkDestroyBuffer(device, phys.fwdBuffersB[f], nullptr);
            vkFreeMemory(device, phys.fwdMemoryB[f], nullptr);
        }
        vkDestroyBuffer(device, phys.fwdDeltaRBuffer, nullptr);
        vkFreeMemory(device, phys.fwdDeltaRMemory, nullptr);
        vkDestroyBuffer(device, phys.fwdOmegaNatBuffer, nullptr);
        vkFreeMemory(device, phys.fwdOmegaNatMemory, nullptr);
        vkDestroyBuffer(device, phys.fwdHistoryBuffer, nullptr);
        vkFreeMemory(device, phys.fwdHistoryMemory, nullptr);
        vkDestroyPipeline(device, phys.fwdSiphonPipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.fwdSiphonPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.fwdInputSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.fwdOutputSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.fwdAccumSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.fwdDensitySetLayout, nullptr);
        vkDestroyDescriptorPool(device, phys.fwdDescPool, nullptr);
    }

    for (int i = 0; i < VK_COMPUTE_NUM_BINDINGS; i++) {
        vkDestroyBuffer(device, phys.soa_buffers[i], nullptr);
    }
    vkFreeMemory(device, phys.soa_unified_memory, nullptr);
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

    /* Siphon set 1 (pressure grid descriptors) */
    vkDestroyDescriptorSetLayout(device, phys.siphonSet1Layout, nullptr);
    vkDestroyDescriptorPool(device, phys.siphonSet1DescPool, nullptr);

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

    /* Collision pipeline + collision SSBOs (optional, Phase 2.2) */
    if (phys.collisionEnabled) {
        vkDestroyPipeline(device, phys.collisionApplyPipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.collisionApplyPipelineLayout, nullptr);
        vkDestroyPipeline(device, phys.collisionResolvePipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.collisionResolvePipelineLayout, nullptr);
        vkDestroyPipeline(device, phys.collisionSyncPipeline, nullptr);
        vkDestroyPipeline(device, phys.collisionSnapshotPipeline, nullptr);
        /* collisionSyncPipelineLayout is shared with apply — don't double-destroy */
        vkDestroyDescriptorSetLayout(device, phys.collisionSet1Layout, nullptr);
        vkDestroyDescriptorPool(device, phys.collisionDescPool, nullptr);
        vkDestroyBuffer(device, phys.rigidBodyIdBuffer, nullptr);
        vkFreeMemory(device, phys.rigidBodyIdMemory, nullptr);
        vkDestroyBuffer(device, phys.velDeltaBuffer, nullptr);
        vkFreeMemory(device, phys.velDeltaMemory, nullptr);
        vkDestroyBuffer(device, phys.contactCountBuffer, nullptr);
        vkFreeMemory(device, phys.contactCountMemory, nullptr);
        vkDestroyBuffer(device, phys.firstContactFrameBuffer, nullptr);
        vkFreeMemory(device, phys.firstContactFrameMemory, nullptr);
        vkDestroyBuffer(device, phys.posPrevBuffer, nullptr);
        vkFreeMemory(device, phys.posPrevMemory, nullptr);
    }

    /* Cylindrical density grid */
    vkDestroyPipeline(device, phys.cylScatterPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.cylScatterPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.cylScatterSetLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.cylScatterDescPool, nullptr);
    vkDestroyPipeline(device, phys.cylStencilPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.cylStencilPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.cylStencilSetLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.cylStencilDescPool, nullptr);
    vkDestroyBuffer(device, phys.cylDensityBuffer, nullptr);
    vkFreeMemory(device, phys.cylDensityMemory, nullptr);
    vkDestroyBuffer(device, phys.cylPressureRBuffer, nullptr);
    vkFreeMemory(device, phys.cylPressureRMemory, nullptr);
    vkDestroyBuffer(device, phys.cylPressurePhiBuffer, nullptr);
    vkFreeMemory(device, phys.cylPressurePhiMemory, nullptr);
    vkDestroyBuffer(device, phys.cylPressureYBuffer, nullptr);
    vkFreeMemory(device, phys.cylPressureYMemory, nullptr);

    /* Graded siphon pipeline + density feedback set */
    vkDestroyPipeline(device, phys.siphonGradedPipeline, nullptr);
    vkDestroyPipelineLayout(device, phys.siphonGradedPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, phys.siphonDensitySetLayout, nullptr);
    vkDestroyDescriptorPool(device, phys.siphonDensityDescPool, nullptr);

    /* Projection pipeline (only if density rendering was initialized) */
    if (phys.projPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, phys.projPipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.projPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.projDescLayout, nullptr);
        vkDestroyDescriptorPool(device, phys.projDescPool, nullptr);
    }

    /* Tone-map pipeline */
    if (phys.tonePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, phys.tonePipeline, nullptr);
        vkDestroyPipelineLayout(device, phys.tonePipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, phys.toneDescLayout, nullptr);
        vkDestroyDescriptorPool(device, phys.toneDescPool, nullptr);
    }

    /* Density image */
    if (phys.densityImage != VK_NULL_HANDLE) {
        vkDestroySampler(device, phys.densitySampler, nullptr);
        vkDestroyImageView(device, phys.densityView, nullptr);
        vkDestroyImage(device, phys.densityImage, nullptr);
        vkFreeMemory(device, phys.densityMemory, nullptr);
    }

    /* Camera UBO */
    if (phys.cameraUBO != VK_NULL_HANDLE) {
        vkUnmapMemory(device, phys.cameraMemory);
        vkDestroyBuffer(device, phys.cameraUBO, nullptr);
        vkFreeMemory(device, phys.cameraMemory, nullptr);
    }

    /* Query pool */
    vkDestroyQueryPool(device, phys.queryPool, nullptr);

    phys.initialized = false;
    printf("[vk-compute] Physics compute cleaned up\n");
}
