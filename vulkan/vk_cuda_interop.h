// V20 CUDA-Vulkan Interop Header
// ================================
// Shares GPU buffers between CUDA physics and Vulkan rendering
#pragma once

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

// Particle data that CUDA writes and Vulkan reads
// This must match the layout expected by the vertex shader
struct CUDAParticleData {
    float position[3];    // xyz world position
    float pump_scale;     // 1.0 = 12D, ~1.33 = 16D
    float pump_residual;  // 0-1, dissolution at >0.95
    float temp;           // temperature
    float velocity[3];    // xyz velocity
    float elongation;     // streak length
};

// CUDA-Vulkan shared buffer
struct SharedBuffer {
    // Vulkan side
    VkBuffer vkBuffer;
    VkDeviceMemory vkMemory;
    int vkMemoryFd;  // File descriptor for sharing

    // CUDA side
    cudaExternalMemory_t cudaExtMem;
    void* cudaPtr;

    // Size
    size_t size;
    uint32_t particleCount;
};

// External semaphore for synchronization
struct SharedSemaphore {
    VkSemaphore vkSemaphore;
    int vkSemaphoreFd;
    cudaExternalSemaphore_t cudaExtSem;
};

#ifdef __cplusplus
extern "C" {
#endif

// Create a buffer that can be shared between CUDA and Vulkan
// Returns 0 on success
int createSharedBuffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    uint32_t particleCount,
    SharedBuffer* outBuffer
);

// Import the Vulkan buffer into CUDA
int importBufferToCUDA(SharedBuffer* buffer);

// Create a single shared semaphore
int createSharedSemaphore(
    VkDevice device,
    SharedSemaphore* outSem
);

// Import semaphore to CUDA
int importSemaphoreToCUDA(SharedSemaphore* sem);

// Cleanup
void destroySharedBuffer(VkDevice device, SharedBuffer* buffer);
void destroySharedSemaphore(VkDevice device, SharedSemaphore* sem);

// CUDA signal/wait helpers
void cudaSignalSemaphore(cudaExternalSemaphore_t sem, uint64_t value, cudaStream_t stream);
void cudaWaitSemaphore(cudaExternalSemaphore_t sem, uint64_t value, cudaStream_t stream);

// ============================================================================
// Density Grid for Hybrid LOD Rendering
// ============================================================================
// A shared 128³ 3D texture for volumetric far-field rendering
// CUDA accumulates particle data into this grid, Vulkan raymarches it

#define DENSITY_GRID_DIM 128

struct SharedDensityGrid {
    // Vulkan side
    VkImage vkImage;
    VkDeviceMemory vkMemory;
    VkImageView vkImageView;
    VkSampler vkSampler;
    int vkMemoryFd;

    // CUDA side
    cudaExternalMemory_t cudaExtMem;
    cudaMipmappedArray_t cudaMipArray;
    cudaSurfaceObject_t cudaSurface;  // For writing
    cudaTextureObject_t cudaTexture;  // For reading (optional)

    // For direct memory access (linear buffer alternative)
    float* cudaLinearPtr;  // 128³ × 4 floats = 32MB

    // Size
    uint32_t dim;  // 128
    size_t size;   // 128³ × 4 × sizeof(float)
};

// Create a 128³ 3D texture that can be shared between CUDA and Vulkan
// Format: RGBA32F (scale_sum, temp_sum, count, coherence)
int createSharedDensityGrid(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    SharedDensityGrid* outGrid
);

// Import the Vulkan 3D image into CUDA for writing
int importDensityGridToCUDA(SharedDensityGrid* grid);

// Cleanup
void destroySharedDensityGrid(VkDevice device, SharedDensityGrid* grid);

// ============================================================================
// Indirect Draw Support for Stream Compaction
// ============================================================================
// CUDA performs stream compaction: only visible particles are packed into
// the compacted buffer, and an indirect draw command is updated with the count.
// This allows Vulkan to skip vertex processing for FAR particles entirely.

// VkDrawIndirectCommand structure (matches Vulkan spec)
// CUDA writes this to tell Vulkan how many particles to draw
struct IndirectDrawCommand {
    uint32_t vertexCount;    // Always 1 for instanced points
    uint32_t instanceCount;  // Number of visible particles (written by CUDA)
    uint32_t firstVertex;    // Always 0
    uint32_t firstInstance;  // Always 0
};

// Shared indirect draw resources
struct SharedIndirectDraw {
    // Compacted particle buffer (Vulkan-visible, CUDA-writable)
    VkBuffer compactedBuffer;
    VkDeviceMemory compactedMemory;
    int compactedMemoryFd;
    cudaExternalMemory_t cudaCompactedExtMem;
    void* cudaCompactedPtr;  // CUDA writes compacted particles here

    // Indirect draw command buffer (Vulkan-visible, CUDA-writable)
    VkBuffer indirectBuffer;
    VkDeviceMemory indirectMemory;
    int indirectMemoryFd;
    cudaExternalMemory_t cudaIndirectExtMem;
    IndirectDrawCommand* cudaIndirectPtr;  // CUDA writes draw command here

    // Atomic counter for stream compaction (device memory)
    unsigned int* cudaWriteIndex;  // Atomic counter for compaction

    // Capacity
    uint32_t maxParticles;
    size_t compactedSize;
};

// Create shared buffers for indirect draw
int createSharedIndirectDraw(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    uint32_t maxParticles,
    SharedIndirectDraw* outIndirect
);

// Import indirect draw buffers to CUDA
int importIndirectDrawToCUDA(SharedIndirectDraw* indirect);

// Cleanup
void destroySharedIndirectDraw(VkDevice device, SharedIndirectDraw* indirect);

#ifdef __cplusplus
}
#endif
