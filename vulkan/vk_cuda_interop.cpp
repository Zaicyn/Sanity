// V20 CUDA-Vulkan Interop Implementation
// ========================================
// Creates shared GPU buffers between CUDA physics and Vulkan rendering
// Uses VK_KHR_external_memory_fd to export Vulkan buffers to CUDA

#include "vk_cuda_interop.h"
#include "vk_types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

// Helper to find memory type with required properties
static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

int createSharedBuffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    uint32_t particleCount,
    SharedBuffer* outBuffer
) {
    VkResult result;
    outBuffer->particleCount = particleCount;
    outBuffer->size = particleCount * sizeof(ParticleVertex);

    // === Create Vulkan buffer with external memory flag ===
    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalMemoryBufferInfo;
    bufferInfo.size = outBuffer->size;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, &bufferInfo, nullptr, &outBuffer->vkBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create shared buffer\n";
        return -1;
    }

    // === Get memory requirements ===
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, outBuffer->vkBuffer, &memRequirements);

    // === Allocate with export capability ===
    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        std::cerr << "[interop] Failed to find suitable memory type\n";
        vkDestroyBuffer(device, outBuffer->vkBuffer, nullptr);
        return -1;
    }

    result = vkAllocateMemory(device, &allocInfo, nullptr, &outBuffer->vkMemory);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to allocate shared memory\n";
        vkDestroyBuffer(device, outBuffer->vkBuffer, nullptr);
        return -1;
    }

    // === Bind buffer to memory ===
    result = vkBindBufferMemory(device, outBuffer->vkBuffer, outBuffer->vkMemory, 0);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to bind buffer memory\n";
        vkFreeMemory(device, outBuffer->vkMemory, nullptr);
        vkDestroyBuffer(device, outBuffer->vkBuffer, nullptr);
        return -1;
    }

    // === Export memory as file descriptor ===
    VkMemoryGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = outBuffer->vkMemory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    // Get the function pointer (extension function)
    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        std::cerr << "[interop] vkGetMemoryFdKHR not available\n";
        vkFreeMemory(device, outBuffer->vkMemory, nullptr);
        vkDestroyBuffer(device, outBuffer->vkBuffer, nullptr);
        return -1;
    }

    result = vkGetMemoryFdKHR(device, &getFdInfo, &outBuffer->vkMemoryFd);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to export memory FD\n";
        vkFreeMemory(device, outBuffer->vkMemory, nullptr);
        vkDestroyBuffer(device, outBuffer->vkBuffer, nullptr);
        return -1;
    }

    std::cout << "[interop] Created shared buffer: " << (outBuffer->size / 1024.0f / 1024.0f)
              << " MB for " << particleCount << " particles, fd=" << outBuffer->vkMemoryFd << "\n";

    return 0;
}

int importBufferToCUDA(SharedBuffer* buffer) {
    cudaError_t err;

    // === Import external memory into CUDA ===
    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memHandleDesc.handle.fd = buffer->vkMemoryFd;
    memHandleDesc.size = buffer->size;
    memHandleDesc.flags = 0;

    err = cudaImportExternalMemory(&buffer->cudaExtMem, &memHandleDesc);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to import external memory to CUDA: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Note: After import, the fd is owned by CUDA and should not be closed

    // === Get mapped buffer from external memory ===
    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = buffer->size;
    bufferDesc.flags = 0;

    err = cudaExternalMemoryGetMappedBuffer(&buffer->cudaPtr, buffer->cudaExtMem, &bufferDesc);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to get mapped buffer: " << cudaGetErrorString(err) << "\n";
        cudaDestroyExternalMemory(buffer->cudaExtMem);
        return -1;
    }

    std::cout << "[interop] Imported to CUDA: ptr=" << buffer->cudaPtr << "\n";

    return 0;
}

int createSharedSemaphore(
    VkDevice device,
    SharedSemaphore* outSem
) {
    VkResult result;

    // === Create timeline semaphore with export capability ===
    VkSemaphoreTypeCreateInfo typeInfo = {};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = 0;

    VkExportSemaphoreCreateInfo exportInfo = {};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.pNext = &typeInfo;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &exportInfo;

    result = vkCreateSemaphore(device, &semInfo, nullptr, &outSem->vkSemaphore);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create timeline semaphore\n";
        return -1;
    }

    // === Export semaphore as file descriptor ===
    VkSemaphoreGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    getFdInfo.semaphore = outSem->vkSemaphore;
    getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!vkGetSemaphoreFdKHR) {
        std::cerr << "[interop] vkGetSemaphoreFdKHR not available\n";
        vkDestroySemaphore(device, outSem->vkSemaphore, nullptr);
        return -1;
    }

    result = vkGetSemaphoreFdKHR(device, &getFdInfo, &outSem->vkSemaphoreFd);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to export semaphore FD\n";
        vkDestroySemaphore(device, outSem->vkSemaphore, nullptr);
        return -1;
    }

    std::cout << "[interop] Created timeline semaphore, fd=" << outSem->vkSemaphoreFd << "\n";

    return 0;
}

int importSemaphoreToCUDA(SharedSemaphore* sem) {
    cudaError_t err;

    cudaExternalSemaphoreHandleDesc semHandleDesc = {};
    semHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    semHandleDesc.handle.fd = sem->vkSemaphoreFd;
    semHandleDesc.flags = 0;

    err = cudaImportExternalSemaphore(&sem->cudaExtSem, &semHandleDesc);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to import semaphore to CUDA: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    std::cout << "[interop] Imported semaphore to CUDA\n";

    return 0;
}

void destroySharedBuffer(VkDevice device, SharedBuffer* buffer) {
    if (buffer->cudaPtr) {
        cudaDestroyExternalMemory(buffer->cudaExtMem);
        buffer->cudaPtr = nullptr;
    }
    if (buffer->vkBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer->vkBuffer, nullptr);
        buffer->vkBuffer = VK_NULL_HANDLE;
    }
    if (buffer->vkMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer->vkMemory, nullptr);
        buffer->vkMemory = VK_NULL_HANDLE;
    }
}

void destroySharedSemaphore(VkDevice device, SharedSemaphore* sem) {
    if (sem->cudaExtSem) {
        cudaDestroyExternalSemaphore(sem->cudaExtSem);
    }
    if (sem->vkSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, sem->vkSemaphore, nullptr);
        sem->vkSemaphore = VK_NULL_HANDLE;
    }
}

// === CUDA Signal/Wait Helpers ===

void cudaSignalSemaphore(cudaExternalSemaphore_t sem, uint64_t value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&sem, &params, 1, stream);
}

void cudaWaitSemaphore(cudaExternalSemaphore_t sem, uint64_t value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};
    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&sem, &params, 1, stream);
}

// ============================================================================
// Shared Density Grid (128³ 3D texture for hybrid LOD)
// ============================================================================
// Note: 3D image interop is complex. For simplicity, we use a linear buffer
// that CUDA writes to, then upload to Vulkan 3D texture via staging buffer.
// This approach is more portable and easier to debug.

int createSharedDensityGrid(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    SharedDensityGrid* outGrid
) {
    VkResult result;
    outGrid->dim = DENSITY_GRID_DIM;
    // 128³ voxels × 4 floats (RGBA32F) × 4 bytes/float = 32 MB
    outGrid->size = DENSITY_GRID_DIM * DENSITY_GRID_DIM * DENSITY_GRID_DIM * 4 * sizeof(float);

    // === Create Vulkan 3D Image ===
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;  // RGBA32F
    imageInfo.extent.width = DENSITY_GRID_DIM;
    imageInfo.extent.height = DENSITY_GRID_DIM;
    imageInfo.extent.depth = DENSITY_GRID_DIM;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    result = vkCreateImage(device, &imageInfo, nullptr, &outGrid->vkImage);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create density grid image\n";
        return -1;
    }

    // === Allocate memory ===
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, outGrid->vkImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    result = vkAllocateMemory(device, &allocInfo, nullptr, &outGrid->vkMemory);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to allocate density grid memory\n";
        vkDestroyImage(device, outGrid->vkImage, nullptr);
        return -1;
    }

    vkBindImageMemory(device, outGrid->vkImage, outGrid->vkMemory, 0);

    // === Create image view ===
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = outGrid->vkImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    result = vkCreateImageView(device, &viewInfo, nullptr, &outGrid->vkImageView);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create density grid image view\n";
        vkFreeMemory(device, outGrid->vkMemory, nullptr);
        vkDestroyImage(device, outGrid->vkImage, nullptr);
        return -1;
    }

    // === Create sampler ===
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    result = vkCreateSampler(device, &samplerInfo, nullptr, &outGrid->vkSampler);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create density grid sampler\n";
        vkDestroyImageView(device, outGrid->vkImageView, nullptr);
        vkFreeMemory(device, outGrid->vkMemory, nullptr);
        vkDestroyImage(device, outGrid->vkImage, nullptr);
        return -1;
    }

    std::cout << "[interop] Created density grid: " << DENSITY_GRID_DIM << "³ ("
              << (outGrid->size / 1024.0f / 1024.0f) << " MB)\n";

    return 0;
}

int importDensityGridToCUDA(SharedDensityGrid* grid) {
    cudaError_t err;

    // For simplicity, allocate a separate CUDA buffer for accumulation
    // This will be copied to the Vulkan texture via staging buffer
    // True image interop is more complex and less portable

    size_t linearSize = grid->dim * grid->dim * grid->dim * 4 * sizeof(float);
    err = cudaMalloc(&grid->cudaLinearPtr, linearSize);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to allocate CUDA density buffer: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Initialize to zero
    err = cudaMemset(grid->cudaLinearPtr, 0, linearSize);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to clear CUDA density buffer: " << cudaGetErrorString(err) << "\n";
        cudaFree(grid->cudaLinearPtr);
        grid->cudaLinearPtr = nullptr;
        return -1;
    }

    std::cout << "[interop] Allocated CUDA density buffer: " << (linearSize / 1024.0f / 1024.0f) << " MB\n";

    return 0;
}

void destroySharedDensityGrid(VkDevice device, SharedDensityGrid* grid) {
    if (grid->cudaLinearPtr) {
        cudaFree(grid->cudaLinearPtr);
        grid->cudaLinearPtr = nullptr;
    }
    if (grid->vkSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, grid->vkSampler, nullptr);
        grid->vkSampler = VK_NULL_HANDLE;
    }
    if (grid->vkImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, grid->vkImageView, nullptr);
        grid->vkImageView = VK_NULL_HANDLE;
    }
    if (grid->vkImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, grid->vkImage, nullptr);
        grid->vkImage = VK_NULL_HANDLE;
    }
    if (grid->vkMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, grid->vkMemory, nullptr);
        grid->vkMemory = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Shared Indirect Draw Buffers for Stream Compaction
// ============================================================================
// This enables true vertex culling: CUDA compacts visible particles into a
// contiguous buffer and writes the count to an indirect draw command buffer.
// Vulkan then uses vkCmdDrawIndirect to draw only the visible particles.

// Helper to create a shared buffer with specific usage flags
static int createSharedBufferWithUsage(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer* outBuffer,
    VkDeviceMemory* outMemory,
    int* outFd
) {
    VkResult result;

    // Create buffer with external memory flag
    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalMemoryBufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, &bufferInfo, nullptr, outBuffer);
    if (result != VK_SUCCESS) {
        std::cerr << "[interop] Failed to create buffer\n";
        return -1;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, *outBuffer, &memRequirements);

    // Allocate with export capability
    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device, *outBuffer, nullptr);
        return -1;
    }

    result = vkAllocateMemory(device, &allocInfo, nullptr, outMemory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, *outBuffer, nullptr);
        return -1;
    }

    result = vkBindBufferMemory(device, *outBuffer, *outMemory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, *outMemory, nullptr);
        vkDestroyBuffer(device, *outBuffer, nullptr);
        return -1;
    }

    // Export memory as file descriptor
    VkMemoryGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = *outMemory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        vkFreeMemory(device, *outMemory, nullptr);
        vkDestroyBuffer(device, *outBuffer, nullptr);
        return -1;
    }

    result = vkGetMemoryFdKHR(device, &getFdInfo, outFd);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, *outMemory, nullptr);
        vkDestroyBuffer(device, *outBuffer, nullptr);
        return -1;
    }

    return 0;
}

// Helper to import a buffer to CUDA
static int importBufferToCUDAHelper(int fd, size_t size, cudaExternalMemory_t* outExtMem, void** outPtr) {
    cudaError_t err;

    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memHandleDesc.handle.fd = fd;
    memHandleDesc.size = size;
    memHandleDesc.flags = 0;

    err = cudaImportExternalMemory(outExtMem, &memHandleDesc);
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to import buffer to CUDA: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    err = cudaExternalMemoryGetMappedBuffer(outPtr, *outExtMem, &bufferDesc);
    if (err != cudaSuccess) {
        cudaDestroyExternalMemory(*outExtMem);
        return -1;
    }

    return 0;
}

int createSharedIndirectDraw(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    uint32_t maxParticles,
    SharedIndirectDraw* outIndirect
) {
    int ret;
    outIndirect->maxParticles = maxParticles;
    outIndirect->compactedSize = maxParticles * sizeof(ParticleVertex);

    // === Create compacted particle buffer ===
    // This holds the stream-compacted visible particles
    ret = createSharedBufferWithUsage(
        device, physicalDevice,
        outIndirect->compactedSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &outIndirect->compactedBuffer,
        &outIndirect->compactedMemory,
        &outIndirect->compactedMemoryFd
    );
    if (ret != 0) {
        std::cerr << "[interop] Failed to create compacted particle buffer\n";
        return -1;
    }

    // === Create indirect draw command buffer ===
    // This holds the VkDrawIndirectCommand written by CUDA
    ret = createSharedBufferWithUsage(
        device, physicalDevice,
        sizeof(IndirectDrawCommand),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &outIndirect->indirectBuffer,
        &outIndirect->indirectMemory,
        &outIndirect->indirectMemoryFd
    );
    if (ret != 0) {
        std::cerr << "[interop] Failed to create indirect draw buffer\n";
        vkFreeMemory(device, outIndirect->compactedMemory, nullptr);
        vkDestroyBuffer(device, outIndirect->compactedBuffer, nullptr);
        return -1;
    }

    std::cout << "[interop] Created indirect draw buffers: compacted="
              << (outIndirect->compactedSize / 1024.0f / 1024.0f)
              << " MB, indirect=" << sizeof(IndirectDrawCommand) << " bytes\n";

    return 0;
}

int importIndirectDrawToCUDA(SharedIndirectDraw* indirect) {
    cudaError_t err;

    // Import compacted particle buffer
    int ret = importBufferToCUDAHelper(
        indirect->compactedMemoryFd,
        indirect->compactedSize,
        &indirect->cudaCompactedExtMem,
        &indirect->cudaCompactedPtr
    );
    if (ret != 0) {
        std::cerr << "[interop] Failed to import compacted buffer to CUDA\n";
        return -1;
    }

    // Import indirect draw buffer
    ret = importBufferToCUDAHelper(
        indirect->indirectMemoryFd,
        sizeof(IndirectDrawCommand),
        &indirect->cudaIndirectExtMem,
        (void**)&indirect->cudaIndirectPtr
    );
    if (ret != 0) {
        std::cerr << "[interop] Failed to import indirect buffer to CUDA\n";
        cudaDestroyExternalMemory(indirect->cudaCompactedExtMem);
        return -1;
    }

    // Allocate atomic counter for stream compaction
    err = cudaMalloc(&indirect->cudaWriteIndex, sizeof(unsigned int));
    if (err != cudaSuccess) {
        std::cerr << "[interop] Failed to allocate write index: " << cudaGetErrorString(err) << "\n";
        cudaDestroyExternalMemory(indirect->cudaIndirectExtMem);
        cudaDestroyExternalMemory(indirect->cudaCompactedExtMem);
        return -1;
    }

    // Initialize to zero
    err = cudaMemset(indirect->cudaWriteIndex, 0, sizeof(unsigned int));

    std::cout << "[interop] Imported indirect draw to CUDA: compactedPtr="
              << indirect->cudaCompactedPtr << ", indirectPtr=" << indirect->cudaIndirectPtr << "\n";

    return 0;
}

void destroySharedIndirectDraw(VkDevice device, SharedIndirectDraw* indirect) {
    // CUDA cleanup
    if (indirect->cudaWriteIndex) {
        cudaFree(indirect->cudaWriteIndex);
        indirect->cudaWriteIndex = nullptr;
    }
    if (indirect->cudaIndirectPtr) {
        cudaDestroyExternalMemory(indirect->cudaIndirectExtMem);
        indirect->cudaIndirectPtr = nullptr;
    }
    if (indirect->cudaCompactedPtr) {
        cudaDestroyExternalMemory(indirect->cudaCompactedExtMem);
        indirect->cudaCompactedPtr = nullptr;
    }

    // Vulkan cleanup
    if (indirect->indirectBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, indirect->indirectBuffer, nullptr);
        indirect->indirectBuffer = VK_NULL_HANDLE;
    }
    if (indirect->indirectMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, indirect->indirectMemory, nullptr);
        indirect->indirectMemory = VK_NULL_HANDLE;
    }
    if (indirect->compactedBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, indirect->compactedBuffer, nullptr);
        indirect->compactedBuffer = VK_NULL_HANDLE;
    }
    if (indirect->compactedMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, indirect->compactedMemory, nullptr);
        indirect->compactedMemory = VK_NULL_HANDLE;
    }
}
