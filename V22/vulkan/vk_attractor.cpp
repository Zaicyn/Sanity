// vk_attractor.cpp — Density-Based Galaxy Rendering Pipeline Implementation
// ==========================================================================

#include "vk_attractor.h"
#include <fstream>
#include <cstring>
#include <cmath>

namespace vk {

// ============================================================================
// Shader Loading
// ============================================================================
static std::vector<char> readShaderFile(const std::string& filename) {
    // Try multiple paths
    std::vector<std::string> paths = {
        "shaders/",
        "vulkan/shaders/",
        "vulkan/build/shaders/",
        "../shaders/",
    };

    for (const auto& path : paths) {
        std::ifstream file(path + filename, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t size = (size_t)file.tellg();
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }
    }
    throw std::runtime_error("Failed to load shader: " + filename);
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return module;
}

// ============================================================================
// Density Buffer Creation
// ============================================================================
static void createDensityBuffer(VulkanContext& ctx, AttractorPipeline& attractor) {
    // R32_UINT storage image, same size as swapchain
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R32_UINT;
    imageInfo.extent = { ctx.swapchainExtent.width, ctx.swapchainExtent.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(ctx.device, &imageInfo, nullptr, &attractor.densityImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create density image");
    }

    // Allocate memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(ctx.device, attractor.densityImage, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &attractor.densityMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate density image memory");
    }
    vkBindImageMemory(ctx.device, attractor.densityImage, attractor.densityMemory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = attractor.densityImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32_UINT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &attractor.densityView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create density image view");
    }

    // Create sampler for tone mapping
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &attractor.densitySampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create density sampler");
    }

    std::cout << "[attractor] Density buffer: " << ctx.swapchainExtent.width << "x"
              << ctx.swapchainExtent.height << " R32_UINT\n";
}

// ============================================================================
// Uniform Buffer Creation
// ============================================================================
static void createUniformBuffers(VulkanContext& ctx, AttractorPipeline& attractor) {
    // Camera UBO
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeof(AttractorCameraUBO);
    bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    vkCreateBuffer(ctx.device, &bufInfo, nullptr, &attractor.cameraUBO);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, attractor.cameraUBO, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(ctx.device, &allocInfo, nullptr, &attractor.cameraUBOMemory);
    vkBindBufferMemory(ctx.device, attractor.cameraUBO, attractor.cameraUBOMemory, 0);
    vkMapMemory(ctx.device, attractor.cameraUBOMemory, 0, sizeof(AttractorCameraUBO), 0,
                &attractor.cameraUBOMapped);

    // Initialize with defaults
    AttractorCameraUBO* cam = (AttractorCameraUBO*)attractor.cameraUBOMapped;
    memset(cam->view_proj, 0, sizeof(cam->view_proj));
    cam->view_proj[0] = 1.0f / (80.0f * ((float)ctx.swapchainExtent.width / ctx.swapchainExtent.height));
    cam->view_proj[5] = 1.0f / 80.0f;
    cam->view_proj[10] = -0.001f;
    cam->view_proj[15] = 1.0f;
    cam->zoom = 1.0f;
    cam->aspect = (float)ctx.swapchainExtent.width / ctx.swapchainExtent.height;
    cam->width = ctx.swapchainExtent.width;
    cam->height = ctx.swapchainExtent.height;

    // State UBO (for pure attractor mode)
    bufInfo.size = sizeof(AttractorStateUBO);
    vkCreateBuffer(ctx.device, &bufInfo, nullptr, &attractor.stateUBO);

    vkGetBufferMemoryRequirements(ctx.device, attractor.stateUBO, &memReqs);
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(ctx.device, &allocInfo, nullptr, &attractor.stateUBOMemory);
    vkBindBufferMemory(ctx.device, attractor.stateUBO, attractor.stateUBOMemory, 0);
    vkMapMemory(ctx.device, attractor.stateUBOMemory, 0, sizeof(AttractorStateUBO), 0,
                &attractor.stateUBOMapped);

    // Initialize state UBO with defaults
    AttractorStateUBO* state = (AttractorStateUBO*)attractor.stateUBOMapped;
    state->w = 0.0f;
    state->s_theta = 1.0f;  // sqrt(1 - 0^2) = 1
    state->phase = 0.0f;
    state->residual = 0.0f;
}

// ============================================================================
// Compute Pipeline (Particle Density Accumulation)
// ============================================================================
static void createComputePipeline(VulkanContext& ctx, AttractorPipeline& attractor) {
    // Descriptor set layout
    VkDescriptorSetLayoutBinding bindings[3] = {};

    // Binding 0: Particle buffer (SSBO)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 1: Density buffer (storage image)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 2: Camera UBO
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
            &attractor.computeDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }

    // Push constants
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(AttractorComputePush);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &attractor.computeDescLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
            &attractor.computePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline layout");
    }

    // Load position-primary shader (harmonic_sample.comp)
    auto shaderCode = readShaderFile("harmonic_sample.spv");
    VkShaderModule shaderModule = createShaderModule(ctx.device, shaderCode);

    // Create position-primary pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = attractor.computePipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
            &attractor.computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);
    std::cout << "[attractor] Position-primary pipeline created\n";

    // Load phase-primary shader (harmonic_phase.comp)
    // Uses same descriptor layout but reconstructs position from phase on GPU
    try {
        auto phaseCode = readShaderFile("harmonic_phase.spv");
        VkShaderModule phaseModule = createShaderModule(ctx.device, phaseCode);

        pipelineInfo.stage.module = phaseModule;

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                &attractor.phasePipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create phase-primary pipeline");
        }

        vkDestroyShaderModule(ctx.device, phaseModule, nullptr);
        std::cout << "[attractor] Phase-primary pipeline created (toggle with 'P')\n";
    } catch (const std::exception& e) {
        std::cout << "[attractor] Phase-primary shader not found, skipping: " << e.what() << "\n";
        attractor.phasePipeline = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Pure Attractor Pipeline (Parametric Sampling - No Particles)
// ============================================================================
static void createPureAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor) {
    // Descriptor set layout for pure attractor mode
    // Different from particle mode: binding 0 is UBO not SSBO
    VkDescriptorSetLayoutBinding bindings[3] = {};

    // Binding 0: Attractor State UBO (instead of particle SSBO)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 1: Density buffer (storage image)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 2: Camera UBO
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
            &attractor.pureDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pure attractor descriptor set layout");
    }

    // Push constants for pure mode
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(AttractorPurePush);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &attractor.pureDescLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
            &attractor.purePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pure attractor pipeline layout");
    }

    // Load pure attractor shader
    try {
        auto pureCode = readShaderFile("harmonic_attractor.spv");
        VkShaderModule pureModule = createShaderModule(ctx.device, pureCode);

        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = pureModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = attractor.purePipelineLayout;

        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                &attractor.purePipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pure attractor pipeline");
        }

        vkDestroyShaderModule(ctx.device, pureModule, nullptr);
        std::cout << "[attractor] Pure attractor pipeline created (toggle with 'P')\n";
    } catch (const std::exception& e) {
        std::cout << "[attractor] Pure attractor shader not found, skipping: " << e.what() << "\n";
        attractor.purePipeline = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Graphics Pipeline (Tone Mapping)
// ============================================================================
static void createGraphicsPipeline(VulkanContext& ctx, AttractorPipeline& attractor) {
    // Descriptor set layout
    VkDescriptorSetLayoutBinding bindings[2] = {};

    // Binding 0: Density buffer (sampled image)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 1: Sampler
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
            &attractor.graphicsDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics descriptor set layout");
    }

    // Push constants
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(AttractorTonePush);

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &attractor.graphicsDescLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr,
            &attractor.graphicsPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline layout");
    }

    // Load shaders
    auto vertCode = readShaderFile("tone_map_vert.spv");
    auto fragCode = readShaderFile("tone_map_frag.spv");
    VkShaderModule vertModule = createShaderModule(ctx.device, vertCode);
    VkShaderModule fragModule = createShaderModule(ctx.device, fragCode);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // No vertex input (fullscreen triangle)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport/scissor
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

    // No depth test
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

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = attractor.graphicsPipelineLayout;
    pipelineInfo.renderPass = ctx.renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
            &attractor.graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    std::cout << "[attractor] Graphics pipeline created\n";
}

// ============================================================================
// Descriptor Pool & Sets
// ============================================================================
static void createDescriptorSets(VulkanContext& ctx, AttractorPipeline& attractor) {
    // Create dedicated descriptor pool
    // Need: 1 storage buffer (particles), 3 uniform buffers (camera, state x2), 2 storage images,
    //       1 sampled image (graphics), 1 sampler (graphics)
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },  // camera + state + camera for pure
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },   // density for compute + pure
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1 }
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 5;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 3;  // compute set, graphics set, pure set

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &attractor.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create attractor descriptor pool");
    }

    // Compute descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = attractor.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &attractor.computeDescLayout;

    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &attractor.computeDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate compute descriptor set");
    }

    // Graphics descriptor set
    allocInfo.pSetLayouts = &attractor.graphicsDescLayout;
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &attractor.graphicsDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate graphics descriptor set");
    }

    // Pure attractor descriptor set (if pipeline was created)
    if (attractor.pureDescLayout != VK_NULL_HANDLE) {
        allocInfo.pSetLayouts = &attractor.pureDescLayout;
        if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &attractor.pureDescSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate pure attractor descriptor set");
        }
    }

    // Update compute descriptor set
    // Binding 0: Particle buffer (from ctx.particleBuffer - the CUDA-shared buffer)
    VkDescriptorBufferInfo particleInfo = { ctx.particleBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo densityInfo = { VK_NULL_HANDLE, attractor.densityView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo cameraInfo = { attractor.cameraUBO, 0, sizeof(AttractorCameraUBO) };

    VkWriteDescriptorSet computeWrites[3] = {};
    computeWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    computeWrites[0].dstSet = attractor.computeDescSet;
    computeWrites[0].dstBinding = 0;
    computeWrites[0].descriptorCount = 1;
    computeWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    computeWrites[0].pBufferInfo = &particleInfo;

    computeWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    computeWrites[1].dstSet = attractor.computeDescSet;
    computeWrites[1].dstBinding = 1;
    computeWrites[1].descriptorCount = 1;
    computeWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    computeWrites[1].pImageInfo = &densityInfo;

    computeWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    computeWrites[2].dstSet = attractor.computeDescSet;
    computeWrites[2].dstBinding = 2;
    computeWrites[2].descriptorCount = 1;
    computeWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    computeWrites[2].pBufferInfo = &cameraInfo;

    vkUpdateDescriptorSets(ctx.device, 3, computeWrites, 0, nullptr);

    // Update graphics descriptor set
    VkDescriptorImageInfo sampledInfo = { VK_NULL_HANDLE, attractor.densityView,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo samplerInfo = { attractor.densitySampler, VK_NULL_HANDLE,
                                          VK_IMAGE_LAYOUT_UNDEFINED };

    VkWriteDescriptorSet graphicsWrites[2] = {};
    graphicsWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    graphicsWrites[0].dstSet = attractor.graphicsDescSet;
    graphicsWrites[0].dstBinding = 0;
    graphicsWrites[0].descriptorCount = 1;
    graphicsWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    graphicsWrites[0].pImageInfo = &sampledInfo;

    graphicsWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    graphicsWrites[1].dstSet = attractor.graphicsDescSet;
    graphicsWrites[1].dstBinding = 1;
    graphicsWrites[1].descriptorCount = 1;
    graphicsWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    graphicsWrites[1].pImageInfo = &samplerInfo;

    vkUpdateDescriptorSets(ctx.device, 2, graphicsWrites, 0, nullptr);

    // Update pure attractor descriptor set (if created)
    if (attractor.pureDescSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo stateInfo = { attractor.stateUBO, 0, sizeof(AttractorStateUBO) };

        VkWriteDescriptorSet pureWrites[3] = {};
        // Binding 0: State UBO
        pureWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        pureWrites[0].dstSet = attractor.pureDescSet;
        pureWrites[0].dstBinding = 0;
        pureWrites[0].descriptorCount = 1;
        pureWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pureWrites[0].pBufferInfo = &stateInfo;

        // Binding 1: Density image
        pureWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        pureWrites[1].dstSet = attractor.pureDescSet;
        pureWrites[1].dstBinding = 1;
        pureWrites[1].descriptorCount = 1;
        pureWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pureWrites[1].pImageInfo = &densityInfo;

        // Binding 2: Camera UBO
        pureWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        pureWrites[2].dstSet = attractor.pureDescSet;
        pureWrites[2].dstBinding = 2;
        pureWrites[2].descriptorCount = 1;
        pureWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pureWrites[2].pBufferInfo = &cameraInfo;

        vkUpdateDescriptorSets(ctx.device, 3, pureWrites, 0, nullptr);
    }
}

// ============================================================================
// Public API
// ============================================================================

void createAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor) {
    std::cout << "[attractor] Initializing density rendering pipeline...\n";

    createDensityBuffer(ctx, attractor);
    createUniformBuffers(ctx, attractor);
    createComputePipeline(ctx, attractor);
    createPureAttractorPipeline(ctx, attractor);  // Add pure attractor pipeline
    createGraphicsPipeline(ctx, attractor);
    createDescriptorSets(ctx, attractor);

    attractor.enabled = true;
    attractor.particleCount = ctx.particleCount;

    std::cout << "[attractor] Pipeline ready (" << ctx.particleCount << " particles)\n";
}

void recordAttractorCommands(
    VulkanContext& ctx,
    AttractorPipeline& attractor,
    VkCommandBuffer cmd,
    uint32_t imageIndex)
{
    if (!attractor.enabled || attractor.particleCount == 0) return;

    // Transition density image to GENERAL for compute write
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = attractor.densityImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Clear density buffer
    VkClearColorValue clearColor = { .uint32 = {0, 0, 0, 0} };
    VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cmd, attractor.densityImage, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &clearRange);

    // Barrier: clear -> compute
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Compute pass: accumulate density based on mode
    if (attractor.mode == AttractorMode::PURE_ATTRACTOR && attractor.purePipeline != VK_NULL_HANDLE) {
        // Pure attractor mode - parametric sampling on GPU, no particle data
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, attractor.purePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            attractor.purePipelineLayout, 0, 1, &attractor.pureDescSet, 0, nullptr);

        // Use particle count as sample count for comparison
        int n_samples = attractor.particleCount;
        AttractorPurePush purePush = {
            n_samples,
            1.0f,  // shell_bias
            attractor.brightness,
            attractor.time
        };
        vkCmdPushConstants(cmd, attractor.purePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AttractorPurePush), &purePush);

        vkCmdDispatch(cmd, (n_samples + 255) / 256, 1, 1);
    } else {
        // Particle-based modes (position-primary or phase-primary)
        VkPipeline computePipeline = (attractor.mode == AttractorMode::PHASE_PRIMARY && attractor.phasePipeline)
                                     ? attractor.phasePipeline
                                     : attractor.computePipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            attractor.computePipelineLayout, 0, 1, &attractor.computeDescSet, 0, nullptr);

        AttractorComputePush computePush = {
            attractor.particleCount,
            attractor.brightness,
            attractor.temp_scale,
            attractor.time  // Pass elapsed time for phase evolution
        };
        vkCmdPushConstants(cmd, attractor.computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AttractorComputePush), &computePush);

        // Dispatch one thread per particle
        vkCmdDispatch(cmd, (attractor.particleCount + 255) / 256, 1, 1);
    }

    // Barrier: compute -> fragment read
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void updateAttractorCamera(
    AttractorPipeline& attractor,
    const float* viewProj,
    float zoom, float aspect,
    int width, int height)
{
    AttractorCameraUBO* cam = (AttractorCameraUBO*)attractor.cameraUBOMapped;
    memcpy(cam->view_proj, viewProj, 16 * sizeof(float));
    cam->zoom = zoom;
    cam->aspect = aspect;
    cam->width = width;
    cam->height = height;
}

void updateAttractorState(
    AttractorPipeline& attractor,
    float w, float phase, float residual)
{
    if (attractor.stateUBOMapped == nullptr) return;

    AttractorStateUBO* state = (AttractorStateUBO*)attractor.stateUBOMapped;
    state->w = w;
    state->s_theta = sqrtf(1.0f - w * w);  // Projection factor
    state->phase = phase;
    state->residual = residual;
}

void destroyAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor) {
    vkDestroyPipeline(ctx.device, attractor.graphicsPipeline, nullptr);
    vkDestroyPipeline(ctx.device, attractor.computePipeline, nullptr);
    if (attractor.phasePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, attractor.phasePipeline, nullptr);
    }
    if (attractor.purePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, attractor.purePipeline, nullptr);
    }
    vkDestroyPipelineLayout(ctx.device, attractor.graphicsPipelineLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, attractor.computePipelineLayout, nullptr);
    if (attractor.purePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, attractor.purePipelineLayout, nullptr);
    }
    vkDestroyDescriptorSetLayout(ctx.device, attractor.graphicsDescLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, attractor.computeDescLayout, nullptr);
    if (attractor.pureDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, attractor.pureDescLayout, nullptr);
    }

    // Destroy dedicated descriptor pool (frees descriptor sets automatically)
    vkDestroyDescriptorPool(ctx.device, attractor.descriptorPool, nullptr);

    vkDestroySampler(ctx.device, attractor.densitySampler, nullptr);
    vkDestroyImageView(ctx.device, attractor.densityView, nullptr);
    vkDestroyImage(ctx.device, attractor.densityImage, nullptr);
    vkFreeMemory(ctx.device, attractor.densityMemory, nullptr);

    vkDestroyBuffer(ctx.device, attractor.cameraUBO, nullptr);
    vkFreeMemory(ctx.device, attractor.cameraUBOMemory, nullptr);

    if (attractor.stateUBO != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, attractor.stateUBO, nullptr);
        vkFreeMemory(ctx.device, attractor.stateUBOMemory, nullptr);
    }

    std::cout << "[attractor] Pipeline destroyed\n";
}

} // namespace vk
