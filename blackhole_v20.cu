// blackhole_v20.cu — V20 Siphon Pump Integration
// ================================================
//
// Realtime Hopfion Lattice Black Hole + Siphon Pump State Machine
//
// Compile:
//   make  (uses Makefile — Vulkan interop build)
//
// Controls:
//   Left drag  — orbit camera     Space — pause/resume
//   Scroll     — zoom             C     — cycle color modes
//   R          — reset camera     H     — toggle topology
//   A          — toggle arms      L     — toggle shell lensing
//   E          — inject entropy   ESC   — quit

// ============================================================================
// Standard Library
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// Core Primitives
// ============================================================================
#include "squaragon.h"                // O(1) cuboctahedral primitive
#include "siphon_pump.h"              // 12↔16 dimensional siphon state machine

// ============================================================================
// CUDA LUT (fast trigonometry — quarter-sector sine table in constant memory)
// ============================================================================
#define CUDA_LUT_IMPLEMENTATION
#include "cuda_lut.cuh"

// ============================================================================
// Modular Physics Headers (Math.md compliant)
// ============================================================================
#include "disk.cuh"                   // GPUDisk struct, constants, inline compute
#include "harmonic.cuh"               // Heartbeat cos(θ)cos(3θ), coherence filter
#include "forces.cuh"                 // Viviani field, angular momentum, ion kick
#include "siphon_pump.cuh"            // 8-state pump machine, ejection
#include "aizawa.cuh"                 // Phase-breathing attractor for jets
#include "sun_trace.cuh"              // VulkanSunTrace, d_shell_radii[8]
#include "passive_advection.cuh"      // Passive Keplerian advection kernel
#include "active_region.cuh"          // ActiveRegion struct + in-region mask kernel
#include "hopfion.cuh"                // Topological reaction algebra + Q LUT

// ============================================================================
// Spatial Data Structures
// ============================================================================
#include "cuda_primitives.cuh"        // GPU scan/sort/count utilities
#include "octree.cuh"                 // OctreeNode, Morton encode/decode, XOR corner
#include "cell_grid.cuh"              // CellGrid struct, grid constants

// ============================================================================
// Validation & Topology
// ============================================================================
#include "validator/frame_export.cuh" // Frame export for offline validation
#include "topology_recorder.cuh"      // Ring buffer for crystal detection
#include "mip_tree.cuh"              // Hierarchical coherence mip-tree

// ============================================================================
// Vulkan + CUDA Interop
// ============================================================================
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "vulkan/vk_types.h"          // ParticleVertex, VulkanContext
#include "vulkan/vk_cuda_interop.h"   // Shared CUDA-Vulkan buffer management
#include "vulkan/vk_attractor.h"      // Attractor pipeline

// ============================================================================
// Runtime Configuration & VRAM Management
// ============================================================================
#include "vram_config.cuh"            // initVRAMConfig(), canOctreeFit(), grid constants
#include "math_types.cuh"             // Vec3, Mat4 (minimal, no GLM)

// ============================================================================
// Rendering
// ============================================================================
#include "render_color.cuh"           // blackbody(), mix(), device color helpers
#include "render_fill.cuh"            // Vulkan fill/compact kernels (LOD, compaction)

// ============================================================================
// Simulation Globals & CLI
// ============================================================================
#include "sim_globals.h"              // Camera, physics flags, test suite globals
#include "cli_args.cuh"               // parseCLI() — command line argument handling
#include "diagnostics.cuh"            // StressCounters, PumpMetrics, sampling kernel
#include "sim_context.h"              // SimulationContext — backend-agnostic state bundle
#include "sim_init.cuh"               // initParticles(), initDiagnostics(), initOctree(), initGrid(), initTopology()
#include "sim_cleanup.cuh"            // cleanupSimulation()

// ============================================================================
// Global Instances
// ============================================================================
TopologyRecorder g_topo_recorder = {};
AttractorPipeline g_attractor;

struct ValidationContext {
    GPUDisk* d_disk = nullptr;
    int N_current = 0;
    float sim_time = 0.0f;
    float heartbeat = 1.0f;
    float avg_scale = 0.0f;
    float avg_residual = 0.0f;
    int export_frame_id = 0;
};
static ValidationContext g_validation_ctx;

// ============================================================================
// Vulkan Forward Declarations
// ============================================================================
namespace vk {
    void initWindow(VulkanContext& ctx);
    void initVulkan(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createDescriptorSetLayout(VulkanContext& ctx);
    void createGraphicsPipeline(VulkanContext& ctx);
    void createSwapchain(VulkanContext& ctx);
    void createFramebuffers(VulkanContext& ctx);
    void createCommandPool(VulkanContext& ctx);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects(VulkanContext& ctx);
    void createUniformBuffers(VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx);
    void createDepthResources(VulkanContext& ctx);
    void createVolumeDescriptorSetLayout(VulkanContext& ctx);
    void createVolumePipeline(VulkanContext& ctx);
    void createVolumeUniformBuffers(VulkanContext& ctx);
    void createVolumeDescriptorSets(VulkanContext& ctx);
    void createAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);
    void destroyAttractorPipeline(VulkanContext& ctx, AttractorPipeline& attractor);
    void drawFrame(VulkanContext& ctx);
    void cleanup(VulkanContext& ctx);
    void updateUniformBuffer(VulkanContext& ctx, uint32_t currentImage);
    void recordCommandBuffer(VulkanContext& ctx, VkCommandBuffer commandBuffer, uint32_t imageIndex);
}

// ============================================================================
// Rendering Configuration
// ============================================================================
#define WIDTH   1280
#define HEIGHT  720

#ifndef ENABLE_PASSIVE_ADVECTION
#define ENABLE_PASSIVE_ADVECTION 1
#endif

// Host-side grid macros (device code must use d_grid_* directly)
#define GRID_DIM        g_grid_dim
#define GRID_CELLS      g_grid_cells
#define GRID_CELL_SIZE  g_grid_cell_size
#define GRID_STRIDE_Y   g_grid_dim
#define GRID_STRIDE_Z   (g_grid_dim * g_grid_dim)

#include "physics_constants.cuh"  // d_PI, d_BH_MASS, d_PHI, arm constants, atomic counters
#include "topology.cuh"           // Must follow arm constants from physics_constants.cuh

// Cell grid device functions now in cell_grid.cuh (included above)

// ============================================================================
// Kernel Includes (order matters — each may depend on constants above)
// ============================================================================
#include "physics.cu"                 // siphonDiskKernel (main per-particle physics)
#include "spawn.cuh"                  // spawnParticlesKernel, injectEntropyCluster
#include "octree_kernels.cuh"         // Morton, tree build, leaf phase, pressure+vorticity
#include "cell_grid_kernels.cuh"      // scatter/field/gather 3-pass grid physics
#include "kuramoto.cuh"               // R_cell, global R reduction, phase histogram
#include "active_compact.cuh"         // Active compaction + sparse tile flags
#include "sim_dispatch.cuh"           // dispatchCorePhysics()

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Parse command-line arguments (sets all g_* globals, returns particle count)
    int num_particles = parseCLI(argc, argv);
    if (num_particles == 0) return 0;  // --help was printed

    // [CLI parsing loop removed — now in cli_args.cuh]

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       BLACKHOLE V20 - Siphon Pump State Machine              ║\n");
    printf("║       12↔16 Dimensional Circulation Visualizer               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // === DYNAMIC VRAM CONFIGURATION ===
    // Query GPU memory and set grid size + particle cap BEFORE any allocations
    // This must happen first so all subsequent code uses correct grid dimensions
    initVRAMConfig();

    // === CUDA LUT INITIALIZATION ===
    // Initialize lookup tables for fast trigonometry in hot loops
    cuda_lut_init();                    // Quarter-sector sine table (2KB)
    cuda_lut_gaussian_init(12.0f);      // Gaussian exp(-r²/2σ²) for density (σ=12)
    cuda_lut_repulsion_init(6.0f);      // Repulsion exp(-r/λ) for soft forces (λ=6)
    init_Q_lut();                       // Hopfion discrete helicity LUT (256 bytes)

    // Clamp user-requested particles to VRAM-safe limit
    if (num_particles > g_runtime_particle_cap) {
        printf("[config] Requested %d particles, clamping to VRAM-safe cap %d\n",
               num_particles, g_runtime_particle_cap);
        num_particles = g_runtime_particle_cap;
    }

    printf("[config] Particles: %d\n", num_particles);
    printf("[config] Topology: %s\n", g_use_hopfion_topology ? "Hopfion shells (discrete)" : "Smooth gradient (continuous)");
    printf("[config] Mode: %s\n", g_headless ? "HEADLESS (physics + logging only)" : "INTERACTIVE (with rendering)");

    // === SIMULATION CONTEXT (backend-agnostic state bundle) ===
    SimulationContext ctx = {};
    ctx.particles.N_seed = num_particles;
    ctx.particles.particle_cap = g_runtime_particle_cap;
    ctx.timing.threads = 256;

    // === CONDITIONAL RENDERING SETUP ===
#ifdef VULKAN_INTEROP
    // === VULKAN INITIALIZATION ===
    VulkanContext vkCtx;
    // Allocate for full growth potential — spawning can grow 5-10x from seed
    // Using MAX_DISK_PTS ensures buffer never overflows regardless of growth
    int max_render_particles = g_runtime_particle_cap;  // VRAM-safe cap, not compile-time MAX_DISK_PTS
    vkCtx.particleCount = max_render_particles;

    // Initialize GLFW for Vulkan (no OpenGL context)
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // No OpenGL
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);   // Fixed size (helps with tiling WMs)
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);     // Request floating window (Wayland hint)

    vkCtx.window = glfwCreateWindow(WIDTH, HEIGHT, "Siphon Pump Black Hole — V20 (Vulkan)", NULL, NULL);
    if (!vkCtx.window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }

    // Mouse callbacks for camera control
    glfwSetWindowUserPointer(vkCtx.window, &vkCtx);
    glfwSetMouseButtonCallback(vkCtx.window, [](GLFWwindow* w, int button, int action, int) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            ctx->mousePressed = (action == GLFW_PRESS);
            if (ctx->mousePressed) glfwGetCursorPos(w, &ctx->lastMouseX, &ctx->lastMouseY);
        }
    });
    glfwSetCursorPosCallback(vkCtx.window, [](GLFWwindow* w, double x, double y) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        if (ctx->mousePressed) {
            float dx = (float)(x - ctx->lastMouseX);
            float dy = (float)(y - ctx->lastMouseY);
            ctx->cameraYaw += dx * 0.005f;
            ctx->cameraPitch += dy * 0.005f;
            ctx->cameraPitch = fmaxf(-1.5f, fminf(1.5f, ctx->cameraPitch));
            ctx->lastMouseX = x;
            ctx->lastMouseY = y;
        }
    });
    glfwSetScrollCallback(vkCtx.window, [](GLFWwindow* w, double, double yoff) {
        auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
        ctx->cameraRadius *= (yoff > 0) ? 0.9f : 1.1f;
        ctx->cameraRadius = fmaxf(5.0f, fminf(5000.0f, ctx->cameraRadius));
    });
    glfwSetKeyCallback(vkCtx.window, [](GLFWwindow* w, int key, int, int action, int mods) {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, 1);
            else if (key == GLFW_KEY_SPACE) g_cam.paused = !g_cam.paused;
            else if (key == GLFW_KEY_R) {
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                ctx->cameraYaw = 0; ctx->cameraPitch = 0.3f; ctx->cameraRadius = 800.0f;
            }
            else if (key == GLFW_KEY_H) {
                extern bool g_use_hopfion_topology;
                g_use_hopfion_topology = !g_use_hopfion_topology;
                printf("[toggle] Radial topology: %s\n", g_use_hopfion_topology ? "Hopfion shells" : "Smooth gradient");
            }
            else if (key == GLFW_KEY_L) {
                // Toggle hybrid LOD (culling) at runtime for perf testing
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                ctx->useIndirectDraw = !ctx->useIndirectDraw;
                printf("[toggle] Hybrid LOD (particle culling): %s\n", ctx->useIndirectDraw ? "ON" : "OFF");
            }
            else if (key == GLFW_KEY_V) {
                // Cycle shell brightness: 100% → 50% → 25% → OFF → 100%
                // Allows viewing jets without volumetric shell glare
                auto* ctx = (VulkanContext*)glfwGetWindowUserPointer(w);
                if (ctx->shellBrightness > 0.9f) {
                    ctx->shellBrightness = 0.5f;
                    printf("[shells] Brightness: 50%%\n");
                } else if (ctx->shellBrightness > 0.4f) {
                    ctx->shellBrightness = 0.25f;
                    printf("[shells] Brightness: 25%%\n");
                } else if (ctx->shellBrightness > 0.1f) {
                    ctx->shellBrightness = 0.0f;
                    printf("[shells] Brightness: OFF (particles only)\n");
                } else {
                    ctx->shellBrightness = 1.0f;
                    printf("[shells] Brightness: 100%%\n");
                }
            }
            else if (key == GLFW_KEY_P) {
                extern AttractorPipeline g_attractor;
                bool shift_held = (mods & GLFW_MOD_SHIFT) != 0;

                if (shift_held && g_attractor.purePipeline != VK_NULL_HANDLE) {
                    // Shift+P: toggle pure attractor mode (easter egg)
                    if (g_attractor.mode == AttractorMode::PURE_ATTRACTOR) {
                        g_attractor.mode = AttractorMode::POSITION_PRIMARY;
                    } else {
                        g_attractor.mode = AttractorMode::PURE_ATTRACTOR;
                    }
                } else {
                    // P: toggle between position-primary and phase-primary
                    if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                        g_attractor.mode = AttractorMode::POSITION_PRIMARY;
                    } else if (g_attractor.phasePipeline != VK_NULL_HANDLE) {
                        g_attractor.mode = AttractorMode::PHASE_PRIMARY;
                    }
                }

                const char* modeNames[] = {
                    "POSITION-PRIMARY (reads CUDA particle xyz)",
                    "PHASE-PRIMARY (flattened disk, shows shell rings)",
                    "PURE ATTRACTOR (parametric GPU sampling)"
                };
                printf("[V20] Render mode: %s\n", modeNames[static_cast<int>(g_attractor.mode)]);
            }
            else if (key == GLFW_KEY_X) {
                // Export single frame for validation (uses global validation context)
                if (g_validation_ctx.d_disk) {
                    system("mkdir -p frames/");
                    exportFrameBinary(g_validation_ctx.d_disk, g_validation_ctx.N_current,
                                      g_validation_ctx.export_frame_id);
                    exportValidationMetadata(g_validation_ctx.export_frame_id,
                                             g_validation_ctx.N_current,
                                             g_validation_ctx.sim_time,
                                             g_validation_ctx.heartbeat,
                                             g_validation_ctx.avg_scale,
                                             g_validation_ctx.avg_residual,
                                             g_grid_dim, 500.0f);
                    g_validation_ctx.export_frame_id++;
                } else {
                    printf("[validator] Error: Simulation not yet initialized\n");
                }
            }
            else if (key == GLFW_KEY_F) {
                // Start stack capture (F = Frames) - dumps 64 consecutive frames
                if (!isStackCaptureActive()) {
                    startStackCapture();
                } else {
                    printf("[validator] Stack capture already in progress (%d remaining)\n",
                           g_stack_capture_remaining);
                }
            }
            else if (key == GLFW_KEY_T) {
                // Manual topology ring buffer dump (T = Topology)
                printf("[topo] Manual dump triggered...\n");
                topology_recorder_dump("manual");
            }
        }
    });

    // Initialize Vulkan
    printf("[vulkan] Initializing Vulkan renderer...\n");
    vk::createInstance(vkCtx);
    vk::setupDebugMessenger(vkCtx);
    if (glfwCreateWindowSurface(vkCtx.instance, vkCtx.window, nullptr, &vkCtx.surface) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create window surface\n"); return 1;
    }
    vk::pickPhysicalDevice(vkCtx);
    vk::createLogicalDevice(vkCtx);
    vk::createSwapchain(vkCtx);
    vk::createImageViews(vkCtx);
    vk::createRenderPass(vkCtx);
    vk::createDescriptorSetLayout(vkCtx);
    vk::createGraphicsPipeline(vkCtx);
    vk::createDepthResources(vkCtx);
    vk::createFramebuffers(vkCtx);
    vk::createCommandPool(vkCtx);
    vk::createUniformBuffers(vkCtx);
    vk::createDescriptorPool(vkCtx);
    vk::createDescriptorSets(vkCtx);

    // Volume rendering pipeline (analytic far-field shells)
    vk::createVolumeDescriptorSetLayout(vkCtx);
    vk::createVolumePipeline(vkCtx);
    vk::createVolumeUniformBuffers(vkCtx);
    vk::createVolumeDescriptorSets(vkCtx);

    vk::createCommandBuffers(vkCtx);
    vk::createSyncObjects(vkCtx);

    // === CREATE SHARED CUDA-VULKAN BUFFER ===
    // Size for growth capacity (2x initial), not just initial count
    printf("[vulkan] Creating CUDA-Vulkan shared buffer for %d particles (capacity for growth)...\n", max_render_particles);
    SharedBuffer sharedParticleBuffer;
    if (createSharedBuffer(vkCtx.device, vkCtx.physicalDevice, max_render_particles, &sharedParticleBuffer) != 0) {
        fprintf(stderr, "Failed to create shared buffer\n"); return 1;
    }
    if (importBufferToCUDA(&sharedParticleBuffer) != 0) {
        fprintf(stderr, "Failed to import buffer to CUDA\n"); return 1;
    }

    // Store the shared buffer in context for rendering
    vkCtx.particleBuffer = sharedParticleBuffer.vkBuffer;
    vkCtx.particleBufferMemory = sharedParticleBuffer.vkMemory;

    // Get the CUDA pointer for the fill kernel
    ParticleVertex* d_vkParticles = (ParticleVertex*)sharedParticleBuffer.cudaPtr;

    printf("[vulkan] Initialization complete! Shared buffer at CUDA ptr=%p\n", d_vkParticles);

    // Density rendering pipeline (now that particle buffer exists)
    // Press 'M' to toggle density mode
    try {
        vk::createAttractorPipeline(vkCtx, g_attractor);
    } catch (const std::exception& e) {
        fprintf(stderr, "[V20] Warning: Density pipeline not available: %s\n", e.what());
    }

    // === CREATE DENSITY GRID FOR HYBRID LOD ===
    SharedDensityGrid densityGrid = {};
    float* d_densityGrid = nullptr;
    unsigned int* d_nearCount = nullptr;
    // Hybrid LOD is disabled by default until volume rendering is fully implemented
    // The LOD kernel adds overhead without benefit until we can skip far vertices
    extern bool g_hybrid_lod;
    bool hybridLODEnabled = g_hybrid_lod;  // Use --hybrid flag to enable

    // === CREATE INDIRECT DRAW RESOURCES FOR STREAM COMPACTION ===
    // Double-buffered: physics writes to back buffer, renderer reads front buffer
    SharedIndirectDraw indirectDrawBuffers[2] = {};
    int frontBuffer = 0;  // Renderer reads this
    int backBuffer = 1;   // Compaction writes to this

    // Current frame's CUDA pointers (updated each frame based on backBuffer)
    ParticleVertex* d_compactedParticles = nullptr;
    CUDADrawIndirectCommand* d_drawCommand = nullptr;
    unsigned int* d_writeIndex = nullptr;
    bool doubleBufferEnabled = false;

    if (hybridLODEnabled) {
        printf("[hybrid] Creating density grid for volumetric far-field...\n");
        if (createSharedDensityGrid(vkCtx.device, vkCtx.physicalDevice, &densityGrid) != 0) {
            printf("[hybrid] WARNING: Failed to create density grid, falling back to points-only\n");
            hybridLODEnabled = false;
        } else if (importDensityGridToCUDA(&densityGrid) != 0) {
            printf("[hybrid] WARNING: Failed to import density grid to CUDA, falling back to points-only\n");
            destroySharedDensityGrid(vkCtx.device, &densityGrid);
            hybridLODEnabled = false;
        } else {
            d_densityGrid = densityGrid.cudaLinearPtr;
            // Allocate counter for near particles
            cudaMalloc(&d_nearCount, sizeof(unsigned int));
            cudaMemset(d_nearCount, 0, sizeof(unsigned int));
            printf("[hybrid] Density grid created successfully\n");
        }

        // Create DOUBLE-BUFFERED indirect draw buffers for stream compaction
        // This decouples physics from rendering - no sync needed
        if (hybridLODEnabled) {
            printf("[hybrid] Creating double-buffered indirect draw buffers...\n");
            bool buffersOK = true;

            for (int i = 0; i < 2 && buffersOK; i++) {
                if (createSharedIndirectDraw(vkCtx.device, vkCtx.physicalDevice, num_particles, &indirectDrawBuffers[i]) != 0) {
                    printf("[hybrid] WARNING: Failed to create indirect draw buffer %d\n", i);
                    buffersOK = false;
                } else if (importIndirectDrawToCUDA(&indirectDrawBuffers[i]) != 0) {
                    printf("[hybrid] WARNING: Failed to import indirect draw buffer %d to CUDA\n", i);
                    destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
                    buffersOK = false;
                }
            }

            if (!buffersOK) {
                // Cleanup any partially created buffers
                for (int i = 0; i < 2; i++) {
                    if (indirectDrawBuffers[i].compactedBuffer != VK_NULL_HANDLE) {
                        destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
                    }
                }
                printf("[hybrid] Falling back to single-buffered mode\n");
            } else {
                doubleBufferEnabled = true;

                // Initialize CUDA pointers to back buffer (where compaction will write)
                d_compactedParticles = (ParticleVertex*)indirectDrawBuffers[backBuffer].cudaCompactedPtr;
                d_drawCommand = (CUDADrawIndirectCommand*)indirectDrawBuffers[backBuffer].cudaIndirectPtr;
                d_writeIndex = indirectDrawBuffers[backBuffer].cudaWriteIndex;

                // Wire up Vulkan context to front buffer (where renderer will read)
                vkCtx.compactedParticleBuffer = indirectDrawBuffers[frontBuffer].compactedBuffer;
                vkCtx.compactedParticleBufferMemory = indirectDrawBuffers[frontBuffer].compactedMemory;
                vkCtx.indirectDrawBuffer = indirectDrawBuffers[frontBuffer].indirectBuffer;
                vkCtx.indirectDrawBufferMemory = indirectDrawBuffers[frontBuffer].indirectMemory;
                vkCtx.useIndirectDraw = true;

                printf("[hybrid] Double-buffered stream compaction enabled!\n");
                printf("[hybrid]   Buffer A: compacted=%p indirect=%p\n",
                       indirectDrawBuffers[0].cudaCompactedPtr, indirectDrawBuffers[0].cudaIndirectPtr);
                printf("[hybrid]   Buffer B: compacted=%p indirect=%p\n",
                       indirectDrawBuffers[1].cudaCompactedPtr, indirectDrawBuffers[1].cudaIndirectPtr);
                printf("[hybrid] Physics writes to back, renderer reads front - zero sync\n");
            }
        }
    }

    // Dummy window pointer for compatibility with headless checks
    GLFWwindow* window = vkCtx.window;

#else
    // === OPENGL INITIALIZATION ===
    GLFWwindow* window = nullptr;
    if (!g_headless) {
        // GLFW init
        // Force X11 platform for GLX-based GLEW compatibility on Wayland systems
        #if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 4
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        #endif
        if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        window = glfwCreateWindow(WIDTH, HEIGHT,
                                  "Siphon Pump Black Hole — V20", NULL, NULL);
        if (!window) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);  // Disable vsync - uncapped FPS

        // GLEW must init after context is current — loads all GL 3.3+ function pointers
        glewExperimental = GL_TRUE;
        GLenum glew_err = glewInit();
        if (glew_err != GLEW_OK) {
            fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(glew_err));
            return 1;
        }
        // glewInit sometimes triggers a benign GL_INVALID_ENUM — clear it
        glGetError();

        glfwSetMouseButtonCallback(window, mouseButtonCB);
        glfwSetCursorPosCallback(window, cursorPosCB);
        glfwSetScrollCallback(window, scrollCB);
        glfwSetKeyCallback(window, keyCB);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_MULTISAMPLE);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Shaders (only in interactive mode)
    GLuint bhProgram = 0, diskProgram = 0;
    if (!g_headless) {
        bhProgram = linkProgram(quadVS, bhFS);
        diskProgram = linkProgram(diskVS, diskFS);
    }
#endif

    // Initialize particles and upload to GPU
    int N = num_particles;
    GPUDisk* d_disk = initParticles(ctx, N, g_rng_seed);

    // Initialize diagnostics, sampling, async streams, topology recorder
    DiagnosticLocals diag = initDiagnostics(ctx, N);

    // Diagnostic locals still referenced by main loop (octree, grid, diagnostics output)
    // These alias the struct fields — will be eliminated when those sections are extracted too.
    StressCounters* d_stress = diag.d_stress;
    StressCounters* d_stress_async = diag.d_stress_async;
    float* d_kr_sin_sum = diag.d_kr_sin_sum;
    float* d_kr_cos_sum = diag.d_kr_cos_sum;
    int* d_kr_count = diag.d_kr_count;
    int kr_max_blocks = diag.kr_max_blocks;
    int* d_phase_hist = diag.d_phase_hist;
    float* d_phase_omega_sum = diag.d_phase_omega_sum;
    float* d_phase_omega_sq = diag.d_phase_omega_sq;
    int* d_sample_indices = diag.d_sample_indices;
    SampleMetrics* d_sample_metrics[2] = { diag.d_sample_metrics[0], diag.d_sample_metrics[1] };
    SampleMetrics* h_sample_metrics = diag.h_sample_metrics;
    StressCounters h_stats_cache = diag.h_stats_cache;
    cudaStream_t sample_stream = diag.sample_stream;
    cudaStream_t stats_stream = diag.stats_stream;
    cudaEvent_t stats_ready = diag.stats_ready;
    bool stats_pending = false;
    bool spawn_pending = false;
    int current_buffer = 0;
    int N_current = N;
    float R_global_cached = 0.0f;
    const int KR_THREADS = 256;

    // Host readback vectors
    std::vector<float> h_kr_sin_sum(kr_max_blocks);
    std::vector<float> h_kr_cos_sum(kr_max_blocks);
    std::vector<int> h_kr_count(kr_max_blocks);
    std::vector<int> h_phase_hist(PHASE_HIST_BINS);
    std::vector<float> h_phase_omega_sum(PHASE_HIST_BINS);
    std::vector<float> h_phase_omega_sq(PHASE_HIST_BINS);

    // Initialize octree (conditional — only if --octree-rebuild or --octree-render)
    OctreeLocals octree = initOctree(ctx);

    // Unpack locals still referenced by the main loop
    uint64_t* d_morton_keys = octree.d_morton_keys;
    uint32_t* d_xor_corners = octree.d_xor_corners;
    uint32_t* d_particle_ids = octree.d_particle_ids;
    OctreeNode* d_octree_nodes = octree.d_octree_nodes;
    uint32_t* d_node_count = octree.d_node_count;
    uint32_t* d_leaf_counts = octree.d_leaf_counts;
    uint32_t* d_leaf_counts_culled = octree.d_leaf_counts_culled;
    uint32_t* d_leaf_offsets = octree.d_leaf_offsets;
    uint32_t* d_leaf_node_indices = octree.d_leaf_node_indices;
    uint32_t* d_leaf_node_count = octree.d_leaf_node_count;
    const bool octreeEnabled = octree.octreeEnabled;
    bool useOctreeTraversal = octree.useOctreeTraversal;
    bool useOctreePhysics = octree.useOctreePhysics;
    uint32_t h_analytic_node_count = octree.h_analytic_node_count;
    uint32_t h_total_node_count = octree.h_total_node_count;
    uint32_t h_leaf_node_count = octree.h_leaf_node_count;
    uint32_t h_cached_total_particles = octree.h_cached_total_particles;
    uint32_t h_culled_total_particles = octree.h_culled_total_particles;
    uint32_t h_num_active = octree.h_num_active;
    float* d_leaf_vel_x = octree.d_leaf_vel_x;
    float* d_leaf_vel_y = octree.d_leaf_vel_y;
    float* d_leaf_vel_z = octree.d_leaf_vel_z;
    float* d_leaf_phase = octree.d_leaf_phase;
    float* d_leaf_frequency = octree.d_leaf_frequency;
    float* d_leaf_coherence = octree.d_leaf_coherence;
    uint64_t* d_leaf_hash_keys = octree.d_leaf_hash_keys;
    uint32_t* d_leaf_hash_values = octree.d_leaf_hash_values;
    uint32_t h_leaf_hash_size = octree.h_leaf_hash_size;
    int morton_capacity = octree.morton_capacity;
    float pressure_k = octree.pressure_k;
    float vorticity_k = octree.vorticity_k;
    float substrate_k = octree.substrate_k;
    float phase_coupling_k = octree.phase_coupling_k;

    // Initialize grid physics + sparse flags + active compaction
    GridLocals grid = initGrid(ctx);
    float* d_grid_density = grid.d_grid_density;
    float* d_grid_momentum_x = grid.d_grid_momentum_x;
    float* d_grid_momentum_y = grid.d_grid_momentum_y;
    float* d_grid_momentum_z = grid.d_grid_momentum_z;
    float* d_grid_phase_sin = grid.d_grid_phase_sin;
    float* d_grid_phase_cos = grid.d_grid_phase_cos;
    float* d_grid_pressure_x = grid.d_grid_pressure_x;
    float* d_grid_pressure_y = grid.d_grid_pressure_y;
    float* d_grid_pressure_z = grid.d_grid_pressure_z;
    float* d_grid_vorticity_x = grid.d_grid_vorticity_x;
    float* d_grid_vorticity_y = grid.d_grid_vorticity_y;
    float* d_grid_vorticity_z = grid.d_grid_vorticity_z;
    float* d_grid_R_cell = grid.d_grid_R_cell;
    float* d_rc_bin_R = grid.d_rc_bin_R;
    float* d_rc_bin_W = grid.d_rc_bin_W;
    float* d_rc_bin_N = grid.d_rc_bin_N;
    uint32_t* d_particle_cell = grid.d_particle_cell;
    uint8_t* d_active_flags = grid.d_active_flags;
    uint8_t* d_tile_flags = grid.d_tile_flags;
    uint32_t* d_compact_active_list = grid.d_compact_active_list;
    uint32_t* d_compact_active_count = grid.d_compact_active_count;
    uint32_t* d_active_tiles = grid.d_active_tiles;
    uint32_t* d_active_tile_count = grid.d_active_tile_count;
    uint32_t h_compact_active_count = 0;
    uint32_t h_active_tile_count = 0;
    const int RC_RADIAL_BINS = 16;
    std::vector<float> h_rc_bin_R(RC_RADIAL_BINS);
    std::vector<float> h_rc_bin_W(RC_RADIAL_BINS);
    std::vector<float> h_rc_bin_N(RC_RADIAL_BINS);
    extern bool g_grid_physics;
    extern bool g_grid_flags;

    // Initialize topology (passive/active mask, hopfion, ActiveRegion bootstrap)
    TopologyLocals topo = initTopology(ctx);
    // Topology locals used by diagnostics readback in main loop
    int h_Q_sum = 0;
    int h_operator_counts[5] = {};
    int g_Q_target = 0;
    float g_hopfion_flip_scale = 1.0f;

    // Context fully wired by init functions above — no manual shadow wiring needed.

    // Timing
    float sim_time = 0.0f;
    int frame = 0;
    int threads = 256;
    // blocks computed dynamically each frame based on N_current
    auto t0 = std::chrono::steady_clock::now();
    double fps_acc = 0; int fps_frames = 0;

    // === BRIDGE METRICS: Smoothed pump state for raymarcher ===
    // These are exponentially smoothed to prevent visual jitter
    PumpMetrics pump_bridge = {1.0f, 0.0f, 0.0f, 0.0f};

    printf("[run] Controls: drag=orbit, scroll=zoom, R=reset, Space=pause, C=color, V=shell brightness\n");
    printf("[run] Seam: 1=closed 2=up 3=down 4=full | Bias: [/] or T=turbo\n");
    printf("[run] PURE PHYSICS MODE - no template forcing\n");

    // Render loop
    // Main loop (headless runs until frame limit, interactive until window closes)
    bool running = true;
    while (running) {
        // Compute blocks based on current particle count (grows via spawning)
        int blocks = (N_current + threads - 1) / threads;

        if (!g_headless && glfwWindowShouldClose(window)) {
            running = false;
            break;
        }
        auto t1 = std::chrono::steady_clock::now();
        float dt_wall = std::chrono::duration<float>(t1 - t0).count();
        t0 = t1;

        // === FIXED TIMESTEP PHYSICS ===
        // Decouple simulation time from wall-clock time to prevent
        // "Aizawa fling" when frame export causes slowdown spikes.
        // Physics always advances in fixed increments regardless of render lag.
        //
        // Simple approach: dt_sim is ALWAYS FIXED_DT, regardless of wall time.
        // During stalls (frame export), simulation effectively pauses rather
        // than trying to "catch up" which would cause instability.
        constexpr float FIXED_DT = 1.0f / 60.0f;  // 60 Hz physics tick

        // dt_sim is ALWAYS fixed - never variable, never scaled by wall time
        float dt_sim = FIXED_DT;

        // Advance sim_time only when not paused (one tick per frame)
        // During export stalls, frames still advance but wall-clock is ignored
        bool should_simulate = g_headless || !g_cam.paused;
        if (should_simulate) {
            sim_time += FIXED_DT;
        }

        // === KERNEL TIMING (every 900 frames) ===
        static cudaEvent_t t_start, t_siphon, t_octree, t_physics, t_render;
        static bool timing_init = false;
        static float ms_siphon = 0, ms_physics = 0, ms_render = 0;
        bool do_timing = (frame > 0 && frame % 900 == 0);
        if (!timing_init) {
            cudaEventCreate(&t_start); cudaEventCreate(&t_siphon);
            cudaEventCreate(&t_octree); cudaEventCreate(&t_physics);
            cudaEventCreate(&t_render);
            timing_init = true;
        }

        // Simulate (fixed timestep - one physics step per frame, dt always constant)
        if (should_simulate) {
            if (do_timing) cudaEventRecord(t_start);

            // Update arm topology device constants ONLY when changed
            // (cudaMemcpyToSymbol is expensive - ~1ms per call, 5 calls = 5ms/frame wasted)
            static int cached_num_arms = -1;
            static bool cached_use_topology = false;
            static float cached_boost_override = -999.0f;
            static bool arm_constants_dirty = true;  // First frame always updates

            int h_num_arms = g_enable_arms ? NUM_ARMS : 0;
            extern float g_arm_boost_override;

            if (arm_constants_dirty ||
                h_num_arms != cached_num_arms ||
                g_use_arm_topology != cached_use_topology ||
                g_arm_boost_override != cached_boost_override) {

                float h_arm_width = ARM_WIDTH_DEG;
                float h_arm_trap = ARM_TRAP_STRENGTH;
                cudaMemcpyToSymbol(d_NUM_ARMS, &h_num_arms, sizeof(int));
                cudaMemcpyToSymbol(d_ARM_WIDTH_DEG, &h_arm_width, sizeof(float));
                cudaMemcpyToSymbol(d_ARM_TRAP_STRENGTH, &h_arm_trap, sizeof(float));
                cudaMemcpyToSymbol(d_USE_ARM_TOPOLOGY, &g_use_arm_topology, sizeof(bool));
                cudaMemcpyToSymbol(d_ARM_BOOST_OVERRIDE, &g_arm_boost_override, sizeof(float));

                cached_num_arms = h_num_arms;
                cached_use_topology = g_use_arm_topology;
                cached_boost_override = g_arm_boost_override;
                arm_constants_dirty = false;
            }

            // Core physics dispatch: mask → siphon → passive → hopfion → spawn
            int spawn_blocks = (N_current + threads - 1) / threads;
            dispatchCorePhysics(ctx, sim_time, dt_sim, frame,
                                g_cam.seam_bits, g_cam.bias,
                                N_current, spawn_blocks, threads,
                                spawn_pending, g_hopfion_flip_scale);
            if (do_timing) cudaEventRecord(t_siphon);

            // === OCTREE UPDATE (every N frames) ===
            // Rebuild Morton-sorted tree for active particles
            // The octree is the crystallization — 24 (LAMBDA_OCTREE) is the finished stone layer.
            // DISABLED BY DEFAULT: Mip-tree provides hierarchical coherence without Morton sort
            // Use --octree-rebuild to re-enable if needed for comparison
            extern bool g_octree_rebuild;
            static int octreeRebuildCounter = 0;
            const int OCTREE_REBUILD_INTERVAL = 30;  // Rebuild every 30 frames (~0.2s at 150 FPS)
            static int octreeSkipCount = 0;          // Track consecutive skips

            if (g_octree_rebuild && octreeEnabled && ++octreeRebuildCounter >= OCTREE_REBUILD_INTERVAL) {
                // VRAM-aware check: test if rebuild will fit before attempting
                // This replaces the hard threshold (was 200k) with dynamic VRAM test
                if (!canOctreeFit(N_current)) {
                    // Skip this rebuild — not enough VRAM for thrust temp buffers
                    octreeSkipCount++;
                    if (octreeSkipCount == 1 || (octreeSkipCount % 100) == 0) {
                        size_t free_now = 0, total = 0;
                        cudaError_t memErr = cudaMemGetInfo(&free_now, &total);
                        cudaError_t lastErr = cudaGetLastError();  // Check if CUDA is in error state
                        printf("[octree] Skip #%d: %d particles, %.1f MB free (need ~%.1f MB), CUDA err=%d/%d\n",
                               octreeSkipCount, N_current, free_now / 1e6,
                               (float)N_current * 28 / 0.75 / 1e6, (int)memErr, (int)lastErr);
                    }
                    // Retry in 10 frames instead of full interval
                    octreeRebuildCounter = OCTREE_REBUILD_INTERVAL - 10;
                } else {
                    octreeSkipCount = 0;  // Reset skip counter on successful rebuild
                    octreeRebuildCounter = 0;

                // 1. Assign Morton keys to all particles (active inner get real keys)
                float boxSize = 500.0f;
                assignMortonKeys<<<blocks, threads>>>(
                    d_disk, d_morton_keys, d_xor_corners, d_particle_ids, N_current, boxSize
                );

                // 2. Count active particles directly (no sort needed)
                // Keys < 0xFFFF... are active (inner particles)
                h_num_active = gpuCountLessThan(d_morton_keys, N_current, 0xFFFFFFFFFFFFFFFFULL);

                // Octree enabled for all particle counts (morton sort restored)

                // 4. Reset node count to analytic tree size (discard old stochastic)
                cudaMemcpy(d_node_count, &h_analytic_node_count,
                           sizeof(uint32_t), cudaMemcpyHostToDevice);

                // 5. Build stochastic tree (levels 6-13) from sorted active particles
                if (h_num_active > 0) {
                    int stochBlocks = (h_num_active + 255) / 256;
                    buildStochasticTree<<<stochBlocks, 256>>>(
                        d_octree_nodes,
                        d_node_count,
                        d_morton_keys,
                        h_num_active,
                        boxSize,    // 500.0f
                        6,          // start_level (first stochastic)
                        13          // max_level
                    );
                }

                // 6. Get total node count for render traversal
                cudaMemcpy(&h_total_node_count, d_node_count,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost);

                // 7. Pre-compute leaf node count for traversal and physics
                // Needed for both render traversal AND physics neighbor lookup
                if ((useOctreeTraversal || useOctreePhysics) && h_total_node_count > h_analytic_node_count) {
                    cudaMemsetAsync(d_leaf_node_count, 0, sizeof(uint32_t));
                    uint32_t stochastic_count = h_total_node_count - h_analytic_node_count;
                    int extractBlocks = (stochastic_count + 255) / 256;
                    extractLeafNodeCounts<<<extractBlocks, 256>>>(
                        d_leaf_counts,
                        d_leaf_node_indices,
                        d_leaf_node_count,
                        d_octree_nodes,
                        h_total_node_count,
                        h_analytic_node_count,
                        13
                    );
                    cudaMemcpy(&h_leaf_node_count, d_leaf_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

                    // Safety check: warn if leaf count exceeds hash capacity
                    if (h_leaf_node_count > h_leaf_hash_size / 2) {
                        printf("[octree] WARNING: leaf count %u exceeds 50%% hash capacity (%u) — increase h_leaf_hash_size\n",
                               h_leaf_node_count, h_leaf_hash_size / 2);
                    }

                    // Build hash table for O(1) neighbor lookup (replaces binary search)
                    if (h_leaf_node_count > 0) {
                        // Clear hash table (set all keys to UINT64_MAX = empty)
                        cudaMemsetAsync(d_leaf_hash_keys, 0xFF, h_leaf_hash_size * sizeof(uint64_t));
                        // Build hash table from leaf indices
                        int hashBlocks = (h_leaf_node_count + 255) / 256;
                        buildLeafHashTable<<<hashBlocks, 256>>>(
                            d_leaf_hash_keys,
                            d_leaf_hash_values,
                            h_leaf_hash_size,
                            h_leaf_hash_size - 1,  // hash_mask
                            d_octree_nodes,
                            d_leaf_node_indices,
                            h_leaf_node_count
                        );
                    }

                    // Cache total particle count for V3 flat dispatch (avoids per-frame sync)
                    if (h_leaf_node_count > 0) {
                        h_cached_total_particles = gpuReduceSum(d_leaf_counts, h_leaf_node_count);
                    } else {
                        h_cached_total_particles = 0;
                    }
                }

                // 8. Print stats occasionally (every 30 rebuilds = 900 frames)
                static int stochastic_rebuild_count = 0;
                stochastic_rebuild_count++;
                if (stochastic_rebuild_count == 1 || stochastic_rebuild_count % 30 == 0) {
                    printf("[octree] Tree: %u analytic + %u stochastic = %u total, %u active, %u leaves\n",
                           h_analytic_node_count, h_total_node_count - h_analytic_node_count,
                           h_total_node_count, h_num_active, h_leaf_node_count);
                }

                // 9. Initialize phase state on first rebuild (once we have leaves)
                static bool phase_initialized = false;
                if (!phase_initialized && h_leaf_node_count > 0) {
                    int leafBlocks = (h_leaf_node_count + 255) / 256;
                    float base_frequency = 0.1f;  // Base oscillation frequency
                    initializeLeafPhase<<<leafBlocks, 256>>>(
                        d_leaf_phase,
                        d_leaf_frequency,
                        d_octree_nodes,
                        d_leaf_node_indices,
                        h_leaf_node_count,
                        base_frequency
                    );
                    phase_initialized = true;
                    printf("[phase] S3 phase state initialized: %u leaves, ω_base=%.2f, coupling=%.3f\n",
                           h_leaf_node_count, base_frequency, phase_coupling_k);
                }
                }  // end else (canOctreeFit succeeded)
            }

            // === OCTREE PHYSICS — Pressure + Vorticity Forces ===
            // 1. Pressure: F_p = -k_p ∇ρ (radial balance, shell formation)
            // 2. Vorticity: F_ω = k_ω (ω × v) (spiral arms, rotation)
            if (useOctreePhysics && octreeEnabled && h_leaf_node_count > 0 && h_num_active > 0) {
                // Step 1: Accumulate average velocity per leaf node (for vorticity)
                accumulateLeafVelocities<<<h_leaf_node_count, 256>>>(
                    d_leaf_vel_x,
                    d_leaf_vel_y,
                    d_leaf_vel_z,
                    d_octree_nodes,
                    d_leaf_node_indices,
                    d_particle_ids,
                    d_disk,
                    h_leaf_node_count
                );

                // Step 2: Apply pressure + vorticity forces to all active particles
                // Phase modulates pressure via sin(θ) - zero-cost oscillation
                int particleBlocks = (h_num_active + 255) / 256;
                applyPressureVorticityKernel<<<particleBlocks, 256>>>(
                    d_disk,
                    d_morton_keys,
                    d_particle_ids,
                    d_octree_nodes,
                    d_leaf_node_indices,
                    d_leaf_vel_x,
                    d_leaf_vel_y,
                    d_leaf_vel_z,
                    d_leaf_phase,  // Direct phase read (no coherence lookup)
                    d_leaf_hash_keys,     // Hash table for O(1) neighbor lookup
                    d_leaf_hash_values,
                    h_leaf_hash_size - 1, // Hash mask
                    h_leaf_node_count,
                    h_num_active,
                    topo.d_in_active_region,   // Step 3: skip passive particles
                    dt_sim,
                    pressure_k,
                    vorticity_k
                );

                // Step 3: Evolve S3 phase state (Kuramoto model)
                // Phase coupling creates temporal coherence and enables resonance
                // Update every 10 frames - neighbor lookups only here, not in pressure
                // Can be disabled with --no-octree-phase to use mip-tree for coherence instead
                extern bool g_octree_phase;
                if (g_octree_phase && frame % 10 == 0) {
                    int leafBlocks = (h_leaf_node_count + 255) / 256;

                    // Evolve phase with Kuramoto coupling (neighbor lookups here only)
                    evolveLeafPhase<<<leafBlocks, 256>>>(
                        d_leaf_phase,
                        d_leaf_frequency,
                        d_octree_nodes,
                        d_leaf_node_indices,
                        d_leaf_hash_keys,      // Hash table for O(1) neighbor lookup
                        d_leaf_hash_values,
                        h_leaf_hash_size - 1,  // Hash mask
                        h_leaf_node_count,
                        dt_sim * 10.0f,  // Compensate for reduced update rate
                        phase_coupling_k
                    );
                }
            }

            // === GRID PHYSICS — DNA/RNA Streaming Forward-Pass Model ===
            // Four modes:
            //   Cadence mode (--grid-physics): scatter/stencil every 30 frames, gather every frame
            //   Flags mode (--grid-flags): presence flags, no lists, no sort, no dedup (optimal)
            extern bool g_grid_physics;
            extern bool g_grid_flags;
            static int gridRebuildCounter = 0;
            const int GRID_REBUILD_INTERVAL = 30;

            // Flags mode state
            static bool flagsInitialized = false;

            if (g_grid_physics && d_grid_density != nullptr) {
                int clearBlocks = (GRID_CELLS + 255) / 256;

                // Per-pass timing events (static to avoid allocation overhead)
                static cudaEvent_t e_scatter_start, e_scatter_end;
                static cudaEvent_t e_compact_end, e_pressure_end, e_gather_end;
                static bool timing_events_init = false;
                static float ms_scatter = 0, ms_compact = 0, ms_pressure = 0, ms_gather = 0;
                if (!timing_events_init) {
                    cudaEventCreate(&e_scatter_start);
                    cudaEventCreate(&e_scatter_end);
                    cudaEventCreate(&e_compact_end);
                    cudaEventCreate(&e_pressure_end);
                    cudaEventCreate(&e_gather_end);
                    timing_events_init = true;
                }

                if (g_grid_flags && d_active_flags != nullptr && d_compact_active_list != nullptr) {
                    // === SPARSE FLAGS + O(n) COMPACTION — Transcription Pattern ===
                    // 1. Scatter marks flags (duplicates collapse)
                    // 2. Compact flags → active_list (O(n), no sort)
                    // 3. Process only active_list (~4k cells, not 2M)

                    // Single flags buffer - no double-buffering needed without propagation
                    // Pipeline: scatter→flags, compact, sparse_clear, repeat

                    // Initialize on first frame
                    if (!flagsInitialized) {
                        // Clear grid
                        clearCellGrid<<<clearBlocks, 256>>>(
                            d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z
                        );

                        flagsInitialized = true;
                        printf("[flags+tiles] Initialized with hierarchical tiled compaction\n");
                    }

                    // Full cadence: only scatter+compact+decay every N frames
                    // Between rebuilds, only gather (like cadence mode)
                    static const int FLAGS_CADENCE = 30;
                    static int flagsCadenceCounter = 0;
                    bool doRebuild = (flagsCadenceCounter == 0);

                    if (doRebuild) {
                        // === ACTIVE PARTICLE COMPACTION (when locked) ===
                        // Instead of scatter(N), do scatter(active_N) + baked static grid
                        extern bool g_active_compaction;
                        extern ActiveParticleState g_active_particles;
                        extern HarmonicLock g_harmonic_lock;
                        bool use_active_compact = g_active_compaction &&
                                                  g_active_particles.initialized &&
                                                  g_harmonic_lock.locked;

                        // Force full scatter if static grid needs rebaking
                        if (use_active_compact && !g_active_particles.static_baked) {
                            use_active_compact = false;  // Will bake on this frame
                        }

                        // Periodic rebake even when locked (every 256 frames)
                        if (use_active_compact &&
                            (frame - g_active_particles.bake_frame) >= ActiveParticleState::REBAKE_INTERVAL) {
                            use_active_compact = false;  // Force rebake
                        }

                        cudaEventRecord(e_scatter_start);

                        if (use_active_compact) {
                            // === ACTIVE COMPACTION PATH ===
                            // 1. Compute activity mask
                            computeParticleActivityMask<<<blocks, threads>>>(
                                d_disk, d_particle_cell, g_active_particles.d_prev_cell,
                                g_active_particles.d_active_mask, N_current,
                                ActiveParticleState::VELOCITY_THRESHOLD
                            );

                            // 2. Compact active particles
                            cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));
                            compactActiveParticles<<<blocks, threads>>>(
                                g_active_particles.d_active_mask,
                                g_active_particles.d_active_list,
                                g_active_particles.d_active_count,
                                N_current
                            );
                            cudaMemcpy(&g_active_particles.h_active_count,
                                       g_active_particles.d_active_count,
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost);

                            // 3. Copy static grid to working grid (base layer)
                            int gridCopyBlocks = (GRID_CELLS + 255) / 256;
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_density, d_grid_density, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_x, d_grid_momentum_x, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_y, d_grid_momentum_y, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_momentum_z, d_grid_momentum_z, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_phase_sin, d_grid_phase_sin, GRID_CELLS);
                            copyStaticToWorkingGrid<<<gridCopyBlocks, 256>>>(
                                g_active_particles.d_static_phase_cos, d_grid_phase_cos, GRID_CELLS);

                            // 4. Scatter ONLY active particles on top
                            if (g_active_particles.h_active_count > 0) {
                                int activeBlks = (g_active_particles.h_active_count + 255) / 256;
                                scatterActiveParticles<<<activeBlks, 256>>>(
                                    d_disk, d_grid_density,
                                    d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                    d_grid_phase_sin, d_grid_phase_cos, d_particle_cell,
                                    g_active_particles.d_active_list, g_active_particles.h_active_count
                                );
                            }

                            // 5. Update prev_cell for next frame
                            copyCurrentToPrevCell<<<blocks, threads>>>(
                                d_particle_cell, g_active_particles.d_prev_cell, N_current
                            );

                            // Mark tile flags for sparse pressure (estimate from active particles)
                            // For now, use full tile marking - could optimize later
                            cudaMemset(d_tile_flags, 0, NUM_TILES * sizeof(uint8_t));
                            cudaMemset(d_active_flags, 0, GRID_CELLS * sizeof(uint8_t));

                            // Log active compaction stats occasionally
                            static int active_log_counter = 0;
                            if (++active_log_counter >= 30) {
                                active_log_counter = 0;
                                float active_pct = 100.0f * g_active_particles.h_active_count / N_current;
                                printf("[active-compact] frame=%d | active=%u (%.1f%%) | saved %.1f%% scatter ops\n",
                                       frame, g_active_particles.h_active_count, active_pct, 100.0f - active_pct);
                            }
                        } else {
                            // === FULL SCATTER PATH (or baking) ===
                            // Pass 1: Scatter particles → marks BOTH cell and tile flags
                            scatterWithTileFlags<<<blocks, threads>>>(
                                d_disk, d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                d_grid_phase_sin, d_grid_phase_cos, d_particle_cell,
                                d_active_flags, d_tile_flags, N_current, 1.0f
                            );

                            // === BAKE STATIC GRID when lock just engaged ===
                            if (g_active_compaction && g_active_particles.initialized &&
                                g_harmonic_lock.locked && !g_active_particles.static_baked) {
                                // First time locked: bake static grid

                                // Compute activity mask for baking
                                computeParticleActivityMask<<<blocks, threads>>>(
                                    d_disk, d_particle_cell, g_active_particles.d_prev_cell,
                                    g_active_particles.d_active_mask, N_current,
                                    ActiveParticleState::VELOCITY_THRESHOLD
                                );

                                // Clear static grid
                                cudaMemset(g_active_particles.d_static_density, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_x, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_y, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_momentum_z, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_phase_sin, 0, GRID_CELLS * sizeof(float));
                                cudaMemset(g_active_particles.d_static_phase_cos, 0, GRID_CELLS * sizeof(float));

                                // Scatter static (non-active) particles to static grid
                                scatterStaticParticles<<<blocks, threads>>>(
                                    d_disk, g_active_particles.d_active_mask,
                                    g_active_particles.d_static_density,
                                    g_active_particles.d_static_momentum_x,
                                    g_active_particles.d_static_momentum_y,
                                    g_active_particles.d_static_momentum_z,
                                    g_active_particles.d_static_phase_sin,
                                    g_active_particles.d_static_phase_cos,
                                    d_particle_cell, N_current
                                );

                                // Count static vs active
                                cudaMemset(g_active_particles.d_active_count, 0, sizeof(uint32_t));
                                compactActiveParticles<<<blocks, threads>>>(
                                    g_active_particles.d_active_mask,
                                    g_active_particles.d_active_list,
                                    g_active_particles.d_active_count,
                                    N_current
                                );
                                cudaMemcpy(&g_active_particles.h_active_count,
                                           g_active_particles.d_active_count,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost);

                                g_active_particles.static_baked = true;
                                g_active_particles.bake_frame = frame;

                                uint32_t static_count = N_current - g_active_particles.h_active_count;
                                printf("[active-compact] BAKED: %u static (%.1f%%) + %u active (%.1f%%)\n",
                                       static_count, 100.0f * static_count / N_current,
                                       g_active_particles.h_active_count,
                                       100.0f * g_active_particles.h_active_count / N_current);
                            }

                            // Update prev_cell for activity tracking
                            if (g_active_compaction && g_active_particles.initialized) {
                                copyCurrentToPrevCell<<<blocks, threads>>>(
                                    d_particle_cell, g_active_particles.d_prev_cell, N_current
                                );
                            }

                            // Invalidate bake when lock breaks
                            if (!g_harmonic_lock.locked) {
                                g_active_particles.static_baked = false;
                            }
                        }
                        cudaEventRecord(e_scatter_end);

                        // Pass 2a: Compact tiles — O(4096) instead of O(2M)!
                        cudaMemset(d_active_tile_count, 0, sizeof(uint32_t));
                        int tileBlocks = (NUM_TILES + 255) / 256;
                        compactActiveTiles<<<tileBlocks, 256>>>(
                            d_tile_flags, d_active_tiles, d_active_tile_count
                        );
                        cudaMemcpy(&h_active_tile_count, d_active_tile_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

                        // Pass 2b: Compact cells within active tiles — O(active_tiles × 512)
                        // Each block handles one tile (512 threads max per tile)
                        cudaMemset(d_compact_active_count, 0, sizeof(uint32_t));
                        if (h_active_tile_count > 0) {
                            compactCellsInTiles<<<h_active_tile_count, CELLS_PER_TILE>>>(
                                d_active_flags, d_active_tiles, h_active_tile_count,
                                d_compact_active_list, d_compact_active_count
                            );
                        }
                        cudaMemcpy(&h_compact_active_count, d_compact_active_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                        cudaEventRecord(e_compact_end);

                        // Sparse clear: only clear flags for active cells (also clears tile flags)
                        int activeBlocks = (h_compact_active_count + 255) / 256;
                        if (activeBlocks == 0) activeBlocks = 1;
                        sparseClearTileAndCellFlags<<<activeBlocks, 256>>>(
                            d_active_flags, d_tile_flags, d_compact_active_list, h_compact_active_count
                        );

                        // Pass 3: Compute Pressure for active cells
                        decayAndComputePressure<<<activeBlocks, 256>>>(
                            d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_compact_active_list, h_compact_active_count,
                            1.0f, pressure_k
                        );
                        cudaEventRecord(e_pressure_end);

                        // Pass 3b: Build mip-tree hierarchy for scale coupling
                        // This replaces Morton-sorted octree for coherence
                        // PREDICTIVE LOCKING: Skip when shells are locked (m=0 ground state)
                        extern bool g_predictive_locking;
                        extern HarmonicLock g_harmonic_lock;
                        bool skip_mip = g_predictive_locking && g_harmonic_lock.locked;

                        // Periodic recheck: rebuild every RECHECK_INTERVAL frames even when locked
                        if (skip_mip && g_harmonic_lock.lock_recheck_counter >= HarmonicLock::RECHECK_INTERVAL) {
                            skip_mip = false;  // Force rebuild for verification
                            g_harmonic_lock.lock_recheck_counter = 0;
                        }

                        if (g_mip_tree.initialized && !skip_mip) {
                            mip_tree_from_grid(d_grid_density, d_grid_momentum_x, d_grid_momentum_y, d_grid_momentum_z,
                                               d_grid_phase_sin, d_grid_phase_cos);
                            mip_tree_build_up();
                            mip_tree_propagate_down(0.05f);  // Mild coupling
                        }
                    }

                    flagsCadenceCounter = (flagsCadenceCounter + 1) % FLAGS_CADENCE;

                    // Pass 4: Gather cell forces to particles
                    // Half-rate gather: when locked, only gather every other frame
                    // (Keplerian orbits are smooth enough to extrapolate between frames)
                    extern bool g_active_compaction;
                    extern ActiveParticleState g_active_particles;
                    extern HarmonicLock g_harmonic_lock;

                    static int gather_phase = 0;
                    bool use_active_gather = g_active_compaction &&
                                             g_active_particles.initialized &&
                                             g_harmonic_lock.locked &&
                                             g_active_particles.static_baked &&
                                             g_active_particles.h_active_count > 0;

                    // Half-rate: skip gather on odd frames when locked
                    bool skip_gather = use_active_gather && (gather_phase & 1);
                    gather_phase++;

                    // Scale-invariant reference density for shear weighting:
                    // mean density × 8 (shells are ~8× denser than grid average)
                    float shear_rho_ref = (float)N_current / (float)GRID_CELLS * 8.0f;

                    if (skip_gather) {
                        // Skip gather this frame - particles extrapolate from last frame's forces
                        // (Static counter to log occasionally)
                        static int skip_count = 0;
                        if (++skip_count % 500 == 1) {
                            printf("[half-rate] Skipped %d gathers (locked, extrapolating)\n", skip_count);
                        }
                    } else if (use_active_gather) {
                        // Gather only to active particles (static particles don't need updates)
                        int activeBlks = (g_active_particles.h_active_count + 255) / 256;
                        gatherToActiveParticles<<<activeBlks, 256>>>(
                            d_disk, d_grid_density,
                            d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_particle_cell, g_active_particles.d_active_list,
                            g_active_particles.h_active_count, dt_sim, substrate_k, g_shear_k, shear_rho_ref,
                            g_kuramoto_k, g_n12_envelope ? 1 : 0, g_envelope_scale
                        );
                    } else {
                        // Full gather to all particles (Step 3: passive particles skipped inside kernel)
                        gatherCellForcesToParticles<<<blocks, threads>>>(
                            d_disk, d_grid_density, d_grid_pressure_x, d_grid_pressure_y, d_grid_pressure_z,
                            d_grid_vorticity_x, d_grid_vorticity_y, d_grid_vorticity_z,
                            d_grid_phase_sin, d_grid_phase_cos,
                            d_particle_cell, topo.d_in_active_region,
                            N_current, dt_sim, substrate_k, g_shear_k, shear_rho_ref,
                            g_kuramoto_k, g_n12_envelope ? 1 : 0, g_envelope_scale
                        );
                    }
                    cudaEventRecord(e_gather_end);

                    // Collect timing every rebuild cycle
                    if (doRebuild) {
                        cudaEventSynchronize(e_gather_end);
                        cudaEventElapsedTime(&ms_scatter, e_scatter_start, e_scatter_end);
                        cudaEventElapsedTime(&ms_compact, e_scatter_end, e_compact_end);
                        cudaEventElapsedTime(&ms_pressure, e_compact_end, e_pressure_end);
                        cudaEventElapsedTime(&ms_gather, e_pressure_end, e_gather_end);
                    }

                    // Debug stats on rebuild frames, every 30 rebuilds (~900 frames)
                    static int rebuild_count = 0;
                    if (doRebuild) {
                        rebuild_count++;
                        if (rebuild_count % 30 == 0 || rebuild_count == 1) {
                            printf("[flags+tiles] frame=%u rebuild=%d | Active: %u cells in %u tiles (%.2f%% of grid)\n",
                                   frame, rebuild_count, h_compact_active_count, h_active_tile_count,
                                   100.0f * h_compact_active_count / GRID_CELLS);
                            printf("[grid timing] scatter=%.2fms compact=%.2fms pressure=%.2fms gather=%.2fms total=%.2fms\n",
                                   ms_scatter, ms_compact, ms_pressure, ms_gather,
                                   ms_scatter + ms_compact + ms_pressure + ms_gather);
                        }
                    }
                } else {
                    // === CADENCE MODE — Rebuild every 30 frames ===

                    // Pass 0+1+2: Rebuild cell state every 30 frames (amortized cost)
                    if (++gridRebuildCounter >= GRID_REBUILD_INTERVAL) {
                        gridRebuildCounter = 0;

                        // Pass 0: Clear cell state
                        clearCellGrid<<<clearBlocks, 256>>>(
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_phase_sin,
                            d_grid_phase_cos,
                            d_grid_pressure_x,
                            d_grid_pressure_y,
                            d_grid_pressure_z,
                            d_grid_vorticity_x,
                            d_grid_vorticity_y,
                            d_grid_vorticity_z
                        );

                        // Pass 1: Scatter particles to cells (atomic accumulation)
                        scatterParticlesToCells<<<blocks, threads>>>(
                            d_disk,
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_phase_sin,
                            d_grid_phase_cos,
                            d_particle_cell,
                            N
                        );

                        // Pass 2: Compute cell fields (fixed 6-neighbor stencil)
                        computeCellFields<<<clearBlocks, 256>>>(
                            d_grid_density,
                            d_grid_momentum_x,
                            d_grid_momentum_y,
                            d_grid_momentum_z,
                            d_grid_pressure_x,
                            d_grid_pressure_y,
                            d_grid_pressure_z,
                            d_grid_vorticity_x,
                            d_grid_vorticity_y,
                            d_grid_vorticity_z,
                            pressure_k,
                            vorticity_k
                        );
                    }

                    // Pass 3: Gather cell forces to particles (every frame, O(1) lookup)
                    // Step 3: passive particles skipped inside kernel via in_active_region check.
                    float shear_rho_ref_cadence = (float)N / (float)GRID_CELLS * 8.0f;
                    gatherCellForcesToParticles<<<blocks, threads>>>(
                        d_disk,
                        d_grid_density,
                        d_grid_pressure_x,
                        d_grid_pressure_y,
                        d_grid_pressure_z,
                        d_grid_vorticity_x,
                        d_grid_vorticity_y,
                        d_grid_vorticity_z,
                        d_grid_phase_sin,
                        d_grid_phase_cos,
                        d_particle_cell,
                        topo.d_in_active_region,
                        N,
                        dt_sim,
                        substrate_k,
                        g_shear_k,
                        shear_rho_ref_cadence,
                        g_kuramoto_k,
                        g_n12_envelope ? 1 : 0,
                        g_envelope_scale
                    );

                    // Debug stats every 900 frames
                    if (frame % 900 == 0) {
                        printf("[grid] Cadence mode: gather every frame, scatter/stencil every %d frames\n",
                               GRID_REBUILD_INTERVAL);
                    }
                }
            }

            if (do_timing) cudaEventRecord(t_physics);

            // === ENTROPY INJECTION TEST ===
            // Inject high-entropy cluster when E key is pressed
            if (g_inject_entropy) {
                injectEntropyCluster<<<blocks, threads>>>(d_disk, N_current, sim_time);
                g_inject_entropy = false;  // One-shot injection
                printf("[ENTROPY] Cluster injected at r=200, falling toward core...\n");
            }
        }

        // === RENDERING: Fill instance buffers ===
        // Skip all rendering work in headless mode
        if (!g_headless) {
        // Fill Vulkan shared buffer with LOD-aware kernel
        // Camera position for LOD distance calculation
        float camX_fill = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * sinf(vkCtx.cameraYaw);
        float camY_fill = vkCtx.cameraRadius * sinf(vkCtx.cameraPitch);
        float camZ_fill = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * cosf(vkCtx.cameraYaw);

        // LOD thresholds (adjust with camera distance for infinite zoom effect)
        // Near threshold scales with camera radius: closer camera = tighter near zone
        float baseFactor = fminf(vkCtx.cameraRadius / 400.0f, 2.0f);  // Scale factor 0.5-2x
        float nearThreshold = vkCtx.lodConfig.nearThreshold * baseFactor;
        float farThreshold = vkCtx.lodConfig.farThreshold * baseFactor;
        float volumeScale = vkCtx.lodConfig.volumeScale;

        // Check if we should use stream compaction (runtime toggleable via L key)
        bool useCompaction = hybridLODEnabled && d_densityGrid != nullptr &&
                             vkCtx.useIndirectDraw && d_compactedParticles != nullptr;

        if (useCompaction) {
            // === HYBRID LOD WITH STREAM COMPACTION ===
            // True vertex culling: only visible particles go through the pipeline

            // 1. Clear density grid
            int gridVoxels = LOD_GRID_SIZE * LOD_GRID_SIZE * LOD_GRID_SIZE;
            clearDensityGrid<<<(gridVoxels * 4 + 255) / 256, 256>>>(d_densityGrid, gridVoxels);

            // 2. Reset write index for compaction
            cudaMemsetAsync(d_writeIndex, 0, sizeof(unsigned int));

            // 3. Compact visible particles (two paths: flat scan or octree traversal)
            // Adaptive: only use octree if previous frame had >10% culling
            // Probe every 30 frames to check if culling rate changed
            static float last_cull_ratio = 0.0f;
            static int probe_counter = 0;
            bool shouldProbe = (probe_counter++ % 30 == 0);
            bool useOctreePath = useOctreeTraversal && octreeEnabled && h_leaf_node_count > 0
                                 && (last_cull_ratio > 0.10f || shouldProbe);

            if (useOctreePath) {
                // === OCTREE TRAVERSAL PATH WITH FRUSTUM CULLING ===

                // Build view-projection matrix for frustum extraction
                Vec3 eye = {camX_fill, camY_fill, camZ_fill};
                Vec3 center = {0, 0, 0};
                Mat4 view = Mat4::lookAt(eye, center, {0, 1, 0});

                int fb_w, fb_h;
                glfwGetFramebufferSize(vkCtx.window, &fb_w, &fb_h);
                float aspect = (float)fb_w / (float)fb_h;
                Mat4 proj = Mat4::perspective(PI / 4.0f, aspect, 0.1f, 2000.0f);
                proj.m[5] *= -1.0f;  // Vulkan Y inversion
                Mat4 vp = Mat4::mul(proj, view);

                // Extract frustum planes
                FrustumPlanes frustum;
                extractFrustumPlanes(vp.m, frustum);

                // Copy pristine leaf counts to working buffer
                cudaMemcpyAsync(d_leaf_counts_culled, d_leaf_counts,
                               h_leaf_node_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

                // Cull leaves against frustum (zeros out particle_count for culled nodes)
                int cullBlocks = (h_leaf_node_count + 255) / 256;
                cullLeafNodesFrustum<<<cullBlocks, 256>>>(
                    d_leaf_counts_culled,
                    d_leaf_node_indices,
                    d_octree_nodes,
                    h_leaf_node_count,
                    frustum
                );

                // Exclusive scan on culled counts to get output offsets (CUB)
                gpuExclusiveScan(d_leaf_counts_culled, d_leaf_offsets, h_leaf_node_count);

                // Compute total from scan tail: total = offset[last] + count[last]
                // Read both values to compute total
                uint32_t tail_values[2];
                cudaMemcpy(&tail_values[0], d_leaf_offsets + h_leaf_node_count - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(&tail_values[1], d_leaf_counts_culled + h_leaf_node_count - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost);
                h_culled_total_particles = tail_values[0] + tail_values[1];

                // Update culling ratio for adaptive fallback
                last_cull_ratio = 1.0f - (float)h_culled_total_particles / (float)h_cached_total_particles;

                // Print culling stats occasionally
                static int cull_stat_frame = 0;
                if (cull_stat_frame++ % 300 == 0) {
                    printf("[frustum] %u → %u particles (%.1f%% culled), %d blocks\n",
                           h_cached_total_particles, h_culled_total_particles, last_cull_ratio * 100.0f,
                           (h_culled_total_particles + 255) / 256);
                }

                // Dispatch V3 traversal with culled particle count
                if (h_culled_total_particles > 0) {
                    int v3Blocks = (h_culled_total_particles + 255) / 256;
                    octreeRenderTraversalV3<<<v3Blocks, 256>>>(
                        d_compactedParticles,
                        d_leaf_offsets,
                        d_leaf_node_indices,
                        d_octree_nodes,
                        d_particle_ids,
                        d_disk,
                        h_leaf_node_count,
                        h_culled_total_particles
                    );
                }

                // Set total (culled count)
                cudaMemcpyAsync(d_writeIndex, &h_culled_total_particles,
                               sizeof(unsigned int), cudaMemcpyHostToDevice);
            } else {
                // === FLAT SCAN PATH (original) ===
                compactVisibleParticles<<<blocks, threads>>>(
                    d_compactedParticles,    // Output: compacted visible particles
                    d_drawCommand,           // Unused in this kernel
                    d_densityGrid,           // Output: density grid for volume
                    d_disk,
                    N,
                    camX_fill, camY_fill, camZ_fill,
                    nearThreshold,
                    farThreshold,
                    volumeScale,
                    d_writeIndex,            // Atomic counter
                    nullptr                  // maxRadius disabled for performance
                );
            }

            // 4. Update indirect draw command with final count
            updateIndirectDrawCommand<<<1, 1>>>(d_drawCommand, d_writeIndex);

            // 5. DOUBLE BUFFER SWAP: back buffer is now ready, swap with front
            // After swap: renderer reads just-written data, compaction writes to old front
            if (doubleBufferEnabled) {
                // Swap indices
                int temp = frontBuffer;
                frontBuffer = backBuffer;
                backBuffer = temp;

                // Update Vulkan context to point to new front buffer (for renderer)
                vkCtx.compactedParticleBuffer = indirectDrawBuffers[frontBuffer].compactedBuffer;
                vkCtx.compactedParticleBufferMemory = indirectDrawBuffers[frontBuffer].compactedMemory;
                vkCtx.indirectDrawBuffer = indirectDrawBuffers[frontBuffer].indirectBuffer;
                vkCtx.indirectDrawBufferMemory = indirectDrawBuffers[frontBuffer].indirectMemory;

                // Update CUDA pointers to new back buffer (for next frame's compaction)
                d_compactedParticles = (ParticleVertex*)indirectDrawBuffers[backBuffer].cudaCompactedPtr;
                d_drawCommand = (CUDADrawIndirectCommand*)indirectDrawBuffers[backBuffer].cudaIndirectPtr;
                d_writeIndex = indirectDrawBuffers[backBuffer].cudaWriteIndex;
            }

            // Visible count readback removed — caused stutter from GPU sync
        } else if (hybridLODEnabled && d_densityGrid != nullptr && !vkCtx.useIndirectDraw) {
            // === HYBRID LOD WITHOUT COMPACTION (toggle OFF for perf comparison) ===
            // Fill ALL particles to main buffer - no culling, for baseline comparison
            if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                fillVulkanSunTraceBuffer<<<blocks, threads>>>(
                    (VulkanSunTrace*)d_vkParticles, d_disk, N_current);
            } else {
                fillVulkanParticleBuffer<<<blocks, threads>>>(d_vkParticles, d_disk, N_current, g_ghost_projection);
            }
            vkCtx.nearParticleCount = 0;  // Clear count to indicate no culling
        } else if (hybridLODEnabled && d_densityGrid != nullptr) {
            // === HYBRID LOD WITHOUT COMPACTION (fallback) ===
            // LOD kernel sets alpha=0 for far particles but still processes all vertices
            int gridVoxels = LOD_GRID_SIZE * LOD_GRID_SIZE * LOD_GRID_SIZE;
            clearDensityGrid<<<(gridVoxels * 4 + 255) / 256, 256>>>(d_densityGrid, gridVoxels);

            cudaMemsetAsync(d_nearCount, 0, sizeof(unsigned int));

            fillVulkanParticleBufferLOD<<<blocks, threads>>>(
                d_vkParticles,
                d_densityGrid,
                d_disk,
                N_current,
                camX_fill, camY_fill, camZ_fill,
                nearThreshold,
                farThreshold,
                volumeScale,
                d_nearCount,
                g_shell_lensing,
                g_ghost_projection
            );

            // Near count readback removed — caused stutter from GPU sync
        } else {
            // Simple fill (no LOD)
            // When phase-primary mode is enabled, fill with SunTrace data instead
            // Both structs are 40 bytes so they share the same buffer
            if (g_attractor.mode == AttractorMode::PHASE_PRIMARY) {
                fillVulkanSunTraceBuffer<<<blocks, threads>>>(
                    (VulkanSunTrace*)d_vkParticles, d_disk, N_current);
            } else {
                fillVulkanParticleBuffer<<<blocks, threads>>>(d_vkParticles, d_disk, N_current, g_ghost_projection);
            }
        }
        // Sync CUDA before Vulkan reads the particle buffer
        cudaDeviceSynchronize();
        } // End of !g_headless check for Vulkan rendering

        // === ASYNC DOUBLE-BUFFERED BRIDGE METRICS ===
        // Zero-sync sampling: launch reduction on stream, read previous frame's result
        // This eliminates ALL pipeline stalls from GPU↔CPU synchronization

        // 1. Determine which buffer to write to (other one has previous frame's data)
        int write_buffer = 1 - current_buffer;

        // 2. Launch async reduction on the sample stream (128 particles only, O(1))
        sampleReductionKernel<<<1, 128, 0, sample_stream>>>(
            d_disk, d_sample_indices, SAMPLE_COUNT, d_sample_metrics[write_buffer], g_use_hopfion_topology);

        // 3. Async memcpy to pinned host memory
        static SampleMetrics* h_sample_metrics_back = nullptr;  // Back buffer
        static cudaEvent_t sample_copy_ready;
        static bool sample_event_init = false;
        static bool sample_copy_pending = false;
        if (!sample_event_init) {
            cudaMallocHost(&h_sample_metrics_back, sizeof(SampleMetrics));
            *h_sample_metrics_back = *h_sample_metrics;  // Initialize
            cudaEventCreate(&sample_copy_ready);
            sample_event_init = true;
        }

        // Check if previous copy is done before launching new one
        if (sample_copy_pending) {
            if (cudaEventQuery(sample_copy_ready) == cudaSuccess) {
                // Swap: back becomes front, front becomes back
                SampleMetrics* tmp = h_sample_metrics;
                h_sample_metrics = h_sample_metrics_back;
                h_sample_metrics_back = tmp;
                sample_copy_pending = false;
            }
            // If not done, skip this frame's copy (use stale data)
        }

        if (!sample_copy_pending) {
            cudaMemcpyAsync(h_sample_metrics_back, d_sample_metrics[write_buffer],
                            sizeof(SampleMetrics), cudaMemcpyDeviceToHost, sample_stream);
            cudaEventRecord(sample_copy_ready, sample_stream);
            sample_copy_pending = true;
        }

        // 4. Swap buffers for next frame
        current_buffer = write_buffer;

        // 5. Smooth the metrics using FRONT buffer (always safe to read, no sync)
        const float BRIDGE_SMOOTH = 0.25f;
        pump_bridge.avg_scale = pump_bridge.avg_scale * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->avg_scale * BRIDGE_SMOOTH;
        pump_bridge.avg_residual = pump_bridge.avg_residual * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->avg_residual * BRIDGE_SMOOTH;
        pump_bridge.total_work = pump_bridge.total_work * (1.0f - BRIDGE_SMOOTH)
        + h_sample_metrics->total_work * BRIDGE_SMOOTH;

        // === HEARTBEAT: Always updates (cheap CPU-side calculation) ===
        float scale_phase = pump_bridge.avg_scale * 0.3f + sim_time * 2.0f;
        pump_bridge.heartbeat = sinf(scale_phase);

        // === UPDATE VALIDATION CONTEXT (for key handler access) ===
        g_validation_ctx.d_disk = d_disk;
        g_validation_ctx.N_current = N_current;
        g_validation_ctx.sim_time = sim_time;
        g_validation_ctx.heartbeat = pump_bridge.heartbeat;
        g_validation_ctx.avg_scale = pump_bridge.avg_scale;
        g_validation_ctx.avg_residual = pump_bridge.avg_residual;

        // === VALIDATION FRAME EXPORT ===
        // Stack capture mode: exports 64 consecutive frames with sync
        if (isStackCaptureActive()) {
            cudaDeviceSynchronize();  // Ensure frame is complete before export
            maybeExportStackFrame(d_disk, N_current, sim_time,
                                  pump_bridge.heartbeat, pump_bridge.avg_scale,
                                  pump_bridge.avg_residual, g_grid_dim, 500.0f);
        }
        // Legacy continuous mode (every N frames)
        maybeExportFrame(d_disk, N_current, frame, sim_time,
                         pump_bridge.heartbeat, pump_bridge.avg_scale,
                         pump_bridge.avg_residual, g_grid_dim, 500.0f);

        // === TOPOLOGY RING BUFFER UPDATE ===
        // Records downsampled m-field to detect crystallization events
        // Uses h_stats_cache (may be 1-2 frames stale, acceptable for detection)
        {
            // Compute stability from cached shell data
            float stability = 0.0f;
            float mean_n = 1.0f;
            if (h_sample_metrics->num_shells > 0) {
                float n_min = h_sample_metrics->shell_n[0];
                float n_max = h_sample_metrics->shell_n[0];
                float n_sum = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float n = h_sample_metrics->shell_n[i];
                    if (n < n_min) n_min = n;
                    if (n > n_max) n_max = n;
                    n_sum += n;
                }
                mean_n = n_sum / h_sample_metrics->num_shells;
                float delta_n = n_max - n_min;
                stability = (mean_n > 1.001f) ? delta_n / (mean_n - 1.0f) : 1.0f;
            }

            // Get particle position/velocity device pointers via offsetof
            const float* d_pos_x = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_x));
            const float* d_pos_y = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_y));
            const float* d_pos_z = (const float*)((char*)d_disk + offsetof(GPUDisk, pos_z));
            const float* d_vel_x = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_x));
            const float* d_vel_y = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_y));
            const float* d_vel_z = (const float*)((char*)d_disk + offsetof(GPUDisk, vel_z));

            bool crystal_detected = topology_recorder_update(
                d_pos_x, d_pos_y, d_pos_z,
                d_vel_x, d_vel_y, d_vel_z,
                N_current,
                frame,
                sim_time,
                h_stats_cache.total_kinetic_energy,
                stability,
                pump_bridge.avg_scale,
                mean_n,
                250.0f,  // grid_half_size
                g_harmonic_lock.locked  // lock-aware topology gating
            );

            // Auto-dump on crystal detection and pause for user verification
            if (crystal_detected) {
                printf("\n");
                printf("╔══════════════════════════════════════════════════════════════╗\n");
                printf("║           *** CRYSTAL DETECTED ***                           ║\n");
                printf("║                                                              ║\n");
                printf("║   Topology ring buffer auto-dumped.                          ║\n");
                printf("║   Simulation PAUSED for visual verification.                 ║\n");
                printf("║                                                              ║\n");
                printf("║   Press SPACE to continue (disables further detection)       ║\n");
                printf("╚══════════════════════════════════════════════════════════════╝\n");
                printf("\n");
                topology_recorder_dump("auto_crystal");
                g_cam.paused = true;  // Pause simulation
            }
        }

        // Check if user has unpaused after crystal detection
        if (!g_cam.paused && topology_recorder_awaiting_continue()) {
            topology_recorder_acknowledge_crystal();
        }

        // === ASYNC STATS COLLECTION (fully non-blocking) ===
        // Check if previous async reduction AND copy are complete
        StressCounters sc = h_stats_cache;  // Use cached results by default
        static bool stats_copy_pending = false;
        static cudaEvent_t stats_copy_ready;
        static bool stats_copy_event_init = false;
        if (!stats_copy_event_init) {
            cudaEventCreate(&stats_copy_ready);
            stats_copy_event_init = true;
        }

        // Check if copy is done (non-blocking)
        if (stats_copy_pending) {
            if (cudaEventQuery(stats_copy_ready) == cudaSuccess) {
                sc = h_stats_cache;  // Now safe to use
                stats_copy_pending = false;
            }
        }

        // Check if reduction is done, then start async copy
        if (stats_pending && !stats_copy_pending) {
            if (cudaEventQuery(stats_ready) == cudaSuccess) {
                // Reduction done - launch async copy (NO sync!)
                cudaMemcpyAsync(&h_stats_cache, d_stress_async, sizeof(StressCounters),
                               cudaMemcpyDeviceToHost, stats_stream);
                cudaEventRecord(stats_copy_ready, stats_stream);
                stats_copy_pending = true;
                stats_pending = false;
            }
        }

        // === Dense R(t) logging (optional) ===
        // Compact per-frame Kuramoto order parameter print for time-series
        // analysis. Minimal overhead (one block reduce + small DtoH).
        if (g_r_log_interval > 0 && frame % g_r_log_interval == 0 && frame > 0) {
            int kr_blocks = (N_current + KR_THREADS - 1) / KR_THREADS;
            if (kr_blocks > kr_max_blocks) kr_blocks = kr_max_blocks;
            reduceKuramotoR<<<kr_blocks, KR_THREADS>>>(
                d_disk, N_current, d_kr_sin_sum, d_kr_cos_sum, d_kr_count);
            cudaMemcpy(h_kr_sin_sum.data(), d_kr_sin_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_cos_sum.data(), d_kr_cos_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_count.data(), d_kr_count, kr_blocks * sizeof(int), cudaMemcpyDeviceToHost);
            double sum_sin = 0.0, sum_cos = 0.0;
            long long total_count = 0;
            for (int b = 0; b < kr_blocks; b++) {
                sum_sin += h_kr_sin_sum[b];
                sum_cos += h_kr_cos_sum[b];
                total_count += h_kr_count[b];
            }
            if (total_count > 0) {
                double inv_n = 1.0 / (double)total_count;
                double mean_sin = sum_sin * inv_n;
                double mean_cos = sum_cos * inv_n;
                float R_t = (float)sqrt(mean_sin * mean_sin + mean_cos * mean_cos);
                printf("[rt] frame=%d R=%.6f\n", frame, R_t);
            }
        }

        // Print stats every 90 frames using sample metrics (no stall)
        if (frame % 90 == 0) {
            // === Compute Kuramoto order parameter R = |⟨e^{iθ}⟩| ===
            int kr_blocks = (N_current + KR_THREADS - 1) / KR_THREADS;
            if (kr_blocks > kr_max_blocks) kr_blocks = kr_max_blocks;
            reduceKuramotoR<<<kr_blocks, KR_THREADS>>>(
                d_disk, N_current, d_kr_sin_sum, d_kr_cos_sum, d_kr_count);
            cudaMemcpy(h_kr_sin_sum.data(), d_kr_sin_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_cos_sum.data(), d_kr_cos_sum, kr_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_kr_count.data(), d_kr_count, kr_blocks * sizeof(int), cudaMemcpyDeviceToHost);
            double sum_sin = 0.0, sum_cos = 0.0;
            long long total_count = 0;
            for (int b = 0; b < kr_blocks; b++) {
                sum_sin += h_kr_sin_sum[b];
                sum_cos += h_kr_cos_sum[b];
                total_count += h_kr_count[b];
            }
            if (total_count > 0) {
                double inv_n = 1.0 / (double)total_count;
                double mean_sin = sum_sin * inv_n;
                double mean_cos = sum_cos * inv_n;
                R_global_cached = (float)sqrt(mean_sin * mean_sin + mean_cos * mean_cos);
            }

            // === Compute per-cell R and radial profile ===
            float r_inner = 0.0f, r_mid = 0.0f, r_outer = 0.0f;
            float R_recon = 0.0f;  // grid-reconstructed global R for consistency check
            if (g_grid_physics && d_grid_R_cell != nullptr) {
                int cell_blocks = (GRID_CELLS + 255) / 256;
                computeRcell<<<cell_blocks, 256>>>(
                    d_grid_density, d_grid_phase_sin, d_grid_phase_cos,
                    d_grid_R_cell, GRID_CELLS);

                // === CONSISTENCY CHECK: R_recon from grid vector sums ===
                // Summing phase_sin[cell] across cells gives total Σ sin(θ_i).
                // Dividing by total density gives ⟨sin θ⟩. R_recon must equal
                // R_global (particle-level reduction) if the grid and particle
                // paths agree. Any discrepancy flags inconsistency (inactive
                // particles, double-counting, sampling bias, etc.).
                {
                    std::vector<float> h_grid_ps(GRID_CELLS);
                    std::vector<float> h_grid_pc(GRID_CELLS);
                    std::vector<float> h_grid_rho(GRID_CELLS);
                    cudaMemcpy(h_grid_ps.data(), d_grid_phase_sin, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_grid_pc.data(), d_grid_phase_cos, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_grid_rho.data(), d_grid_density, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    double tot_s = 0.0, tot_c = 0.0, tot_rho = 0.0;
                    for (int c = 0; c < GRID_CELLS; c++) {
                        tot_s += h_grid_ps[c];
                        tot_c += h_grid_pc[c];
                        tot_rho += h_grid_rho[c];
                    }
                    if (tot_rho > 0.0) {
                        double inv_rho = 1.0 / tot_rho;
                        double ms = tot_s * inv_rho;
                        double mc = tot_c * inv_rho;
                        R_recon = (float)sqrt(ms * ms + mc * mc);
                    }
                }

                // Radial profile: 16 bins from center to edge, noise-floor corrected
                cudaMemsetAsync(d_rc_bin_R, 0, RC_RADIAL_BINS * sizeof(float));
                cudaMemsetAsync(d_rc_bin_W, 0, RC_RADIAL_BINS * sizeof(float));
                cudaMemsetAsync(d_rc_bin_N, 0, RC_RADIAL_BINS * sizeof(float));
                reduceRcellRadialProfile<<<cell_blocks, 256>>>(
                    d_grid_R_cell, d_grid_density,
                    g_grid_dim, RC_RADIAL_BINS, g_grid_cell_size,
                    d_rc_bin_R, d_rc_bin_W, d_rc_bin_N);
                cudaMemcpy(h_rc_bin_R.data(), d_rc_bin_R, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rc_bin_W.data(), d_rc_bin_W, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rc_bin_N.data(), d_rc_bin_N, RC_RADIAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);

                // Collapse 16 bins into 3 zones (inner / mid / outer) for printing
                float sum_R[3] = {0.0f, 0.0f, 0.0f};
                float sum_W[3] = {0.0f, 0.0f, 0.0f};
                for (int b = 0; b < RC_RADIAL_BINS; b++) {
                    int zone = (b < 6) ? 0 : (b < 12) ? 1 : 2;
                    sum_R[zone] += h_rc_bin_R[b];
                    sum_W[zone] += h_rc_bin_W[b];
                }
                r_inner = (sum_W[0] > 0.0f) ? sum_R[0] / sum_W[0] : 0.0f;
                r_mid   = (sum_W[1] > 0.0f) ? sum_R[1] / sum_W[1] : 0.0f;
                r_outer = (sum_W[2] > 0.0f) ? sum_R[2] / sum_W[2] : 0.0f;

                // Optional: dump full R_cell grid to disk for offline analysis
                if (g_r_export_interval > 0 && (frame % g_r_export_interval == 0)) {
                    system("mkdir -p r_export");
                    char fname[256];
                    snprintf(fname, sizeof(fname), "r_export/frame_%05d.bin", frame);
                    std::vector<float> h_R(GRID_CELLS);
                    cudaMemcpy(h_R.data(), d_grid_R_cell, GRID_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
                    FILE* fp = fopen(fname, "wb");
                    if (fp) {
                        // Header: grid_dim (int), grid_cell_size (float), frame (int), 4 bytes pad
                        int hdr_dim = g_grid_dim;
                        float hdr_cell = g_grid_cell_size;
                        int hdr_frame = frame;
                        int hdr_pad = 0;
                        fwrite(&hdr_dim, sizeof(int), 1, fp);
                        fwrite(&hdr_cell, sizeof(float), 1, fp);
                        fwrite(&hdr_frame, sizeof(int), 1, fp);
                        fwrite(&hdr_pad, sizeof(int), 1, fp);
                        fwrite(h_R.data(), sizeof(float), GRID_CELLS, fp);
                        fclose(fp);
                        printf("[r-export] Wrote %s (%.1f MB)\n", fname, GRID_CELLS * sizeof(float) / 1.0e6);
                    }
                }
            }

            // === Phase histogram: multi-domain clustering check ===
            cudaMemsetAsync(d_phase_hist, 0, PHASE_HIST_BINS * sizeof(int));
            cudaMemsetAsync(d_phase_omega_sum, 0, PHASE_HIST_BINS * sizeof(float));
            cudaMemsetAsync(d_phase_omega_sq, 0, PHASE_HIST_BINS * sizeof(float));
            int hist_blocks = (N_current + 255) / 256;
            reducePhaseHistogram<<<hist_blocks, 256>>>(d_disk, N_current, d_phase_hist, d_phase_omega_sum, d_phase_omega_sq);
            cudaMemcpy(h_phase_hist.data(), d_phase_hist, PHASE_HIST_BINS * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_phase_omega_sum.data(), d_phase_omega_sum, PHASE_HIST_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_phase_omega_sq.data(), d_phase_omega_sq, PHASE_HIST_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            // Compute max bin for normalization
            int max_bin_count = 0;
            long long hist_total = 0;
            for (int b = 0; b < PHASE_HIST_BINS; b++) {
                if (h_phase_hist[b] > max_bin_count) max_bin_count = h_phase_hist[b];
                hist_total += h_phase_hist[b];
            }

            printf("[frame %5d] fps=%.0f | particles=%d "
            "avg_scale=%.2f (sample=%.2f) | bridge: s=%.2f r=%.3f hb=%.2f | R=%.4f/Rrec=%.4f [%.3f/%.3f/%.3f]",
                   frame, fps_acc > 0 ? fps_frames / fps_acc : 0.0,
                   N_current,
                   h_sample_metrics->avg_scale, h_sample_metrics->avg_scale,
                   pump_bridge.avg_scale, pump_bridge.avg_residual, pump_bridge.heartbeat,
                   R_global_cached, R_recon, r_inner, r_mid, r_outer);

#ifdef VULKAN_INTEROP
            // Show hybrid LOD stats (visible particle count and culling percentage)
            if (vkCtx.useIndirectDraw && vkCtx.nearParticleCount > 0) {
                float cull_pct = 100.0f * (1.0f - (float)vkCtx.nearParticleCount / (float)N);
                printf(" | LOD: %u visible (%.0f%% culled)", vkCtx.nearParticleCount, cull_pct);
            }
#endif

            // Entropy dissolution diagnostic
            if (sc.high_stress_count > 0) {
                float dissolution_pct = 100.0f * (float)sc.high_stress_count / (float)sc.active_count;
                printf(" | DISSOLVING: %u (%.1f%%) at >0.95 stress", sc.high_stress_count, dissolution_pct);
            }
            printf("\n");

            // Phase histogram (32 bins, ASCII heat map + numeric peak analysis).
            // Multi-peak → multi-domain clustering. Flat → uniform. Single
            // peak → global lock. Bin 0 = θ ∈ [0, 2π/32).
            //
            // n_peaks and peak_frac are hoisted to outer scope so the
            // Kuramoto × topology correlation dump below can read them.
            int n_peaks = 0;
            float peak_frac = 0.0f;
            if (hist_total > 0 && max_bin_count > 0) {
                printf("[phase-hist] ");
                const char* ramps = " .,-:;=+*#%@";
                int n_ramps = 12;
                float expected = (float)hist_total / (float)PHASE_HIST_BINS;  // flat baseline
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    float ratio = (float)h_phase_hist[b] / expected;
                    if (ratio > 3.0f) ratio = 3.0f;
                    int r = (int)(ratio * (n_ramps - 1) / 3.0f);
                    if (r < 0) r = 0;
                    if (r >= n_ramps) r = n_ramps - 1;
                    putchar(ramps[r]);
                }
                printf(" max/avg=%.2f\n", (float)max_bin_count / expected);

                // Peak detection: count local maxima that are ≥ 1.5× expected
                // and compute how much of total mass is in peaks vs background.
                long long peak_mass = 0;
                float peak_threshold = 1.5f * expected;
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    int prev = h_phase_hist[(b + PHASE_HIST_BINS - 1) % PHASE_HIST_BINS];
                    int curr = h_phase_hist[b];
                    int next = h_phase_hist[(b + 1) % PHASE_HIST_BINS];
                    if (curr > prev && curr > next && (float)curr > peak_threshold) {
                        n_peaks++;
                        peak_mass += curr;
                    }
                }
                peak_frac = (float)peak_mass / (float)hist_total;
                printf("[phase-hist] peaks=%d  peak_mass_frac=%.6f  total_hist=%lld\n", n_peaks, peak_frac, hist_total);

                // === Velocity-filter check (Gemini's "Sieve" hypothesis) ===
                // Do clustered particles have a different ω distribution than
                // background particles? If the clusters are selecting particles
                // whose ω matches the envelope beat frequency, the clustered
                // subset should have a tighter ω variance and a mean ω closer
                // to the envelope-determined attractor frequency.
                double cluster_w_sum = 0.0, cluster_w_sq = 0.0;
                long long cluster_n = 0;
                double bg_w_sum = 0.0, bg_w_sq = 0.0;
                long long bg_n = 0;
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    int prev = h_phase_hist[(b + PHASE_HIST_BINS - 1) % PHASE_HIST_BINS];
                    int curr = h_phase_hist[b];
                    int next = h_phase_hist[(b + 1) % PHASE_HIST_BINS];
                    bool is_peak = (curr > prev && curr > next && (float)curr > peak_threshold);
                    if (is_peak) {
                        cluster_w_sum += h_phase_omega_sum[b];
                        cluster_w_sq += h_phase_omega_sq[b];
                        cluster_n += curr;
                    } else {
                        bg_w_sum += h_phase_omega_sum[b];
                        bg_w_sq += h_phase_omega_sq[b];
                        bg_n += curr;
                    }
                }
                if (cluster_n > 0 && bg_n > 0) {
                    double cluster_mean = cluster_w_sum / cluster_n;
                    double cluster_var = cluster_w_sq / cluster_n - cluster_mean * cluster_mean;
                    double cluster_std = (cluster_var > 0) ? sqrt(cluster_var) : 0.0;
                    double bg_mean = bg_w_sum / bg_n;
                    double bg_var = bg_w_sq / bg_n - bg_mean * bg_mean;
                    double bg_std = (bg_var > 0) ? sqrt(bg_var) : 0.0;
                    printf("[omega-filter] cluster: μ=%.4f σ=%.4f n=%lld  bg: μ=%.4f σ=%.4f n=%lld\n",
                           cluster_mean, cluster_std, cluster_n,
                           bg_mean, bg_std, bg_n);
                }

                // Numeric dump: ratio per bin for quantitative inspection
                printf("[phase-hist-num] ");
                for (int b = 0; b < PHASE_HIST_BINS; b++) {
                    float ratio = (float)h_phase_hist[b] / expected;
                    printf("%.2f ", ratio);
                }
                printf("\n");
            }
            float latest_Q = topology_recorder_get_latest_Q();
            printf("[%s] num=%d Q=%.2f | ",
                   g_use_hopfion_topology ? "hopfion shells" : "smooth gradient",
                   h_sample_metrics->num_shells, latest_Q);
            for (int i = 0; i < h_sample_metrics->num_shells && i < 4; i++) {
                printf("r=%.1f n=%.3f | ", h_sample_metrics->shell_radii[i], h_sample_metrics->shell_n[i]);
            }
            printf("\n");

            // === Kuramoto × topology correlation dump ===
            // One CSV-friendly row per stats frame with all scalars needed to
            // correlate phase-cluster structure with Hopfion invariant Q.
            // Columns: frame, R_global, R_recon, n_peaks, peak_mass_frac,
            //          Q, num_shells, N, R_inner, R_mid, active_frac
            // Step 4: active_frac = fraction of alive particles classified as active (siphon).
            if (g_qr_corr_log) {
                // Step 4: count active-region particles via host readback.
                // 1MB every 90 frames = negligible bandwidth.
                float active_frac = 1.0f;
#if ENABLE_PASSIVE_ADVECTION
                {
                    uint32_t h_active_region_count = 0;
                    std::vector<uint8_t> h_region_mask(N_current);
                    cudaMemcpy(h_region_mask.data(), topo.d_in_active_region,
                               N_current * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                    for (int j = 0; j < N_current; j++)
                        if (h_region_mask[j]) h_active_region_count++;
                    active_frac = (float)h_active_region_count / (float)N_current;
                }
#endif
                printf("[QR-corr] %d %.6f %.6f %d %.6f %.4f %d %d %.4f %.4f %.4f\n",
                       frame,
                       R_global_cached,
                       R_recon,
                       n_peaks,
                       peak_frac,
                       latest_Q,
                       h_sample_metrics->num_shells,
                       N_current,
                       r_inner,
                       r_mid,
                       active_frac);

                // Hopfion Q_discrete readback (one-frame lag is fine)
                cudaMemcpy(&h_Q_sum, topo.d_Q_sum, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_operator_counts, topo.d_operator_counts, 5 * sizeof(int), cudaMemcpyDeviceToHost);
                if (frame == 0) g_Q_target = h_Q_sum;  // Set conservation target on first read
                if (h_Q_sum != g_Q_target) {
                    printf("[TOPO] Q DRIFT: %d → %d (Δ=%d) at frame %d\n",
                           g_Q_target, h_Q_sum, h_Q_sum - g_Q_target, frame);
                    g_Q_target = h_Q_sum;  // Update target (statistical conservation)
                }
                printf("[TOPO] frame %d Q_discrete=%d flips=%d freezes=%d fusions=%d tensions=%d vents=%d\n",
                       frame, h_Q_sum, h_operator_counts[0], h_operator_counts[1],
                       h_operator_counts[2], h_operator_counts[3], h_operator_counts[4]);

                // Recycling equilibrium: adjust phason flip rate to balance fusion/vent
                int fusions = h_operator_counts[2];
                int vents = h_operator_counts[4];
                if (fusions > 0 && vents > 0) {
                    float ratio = (float)fusions / (float)vents;
                    // If fusion >> vent, increase flips (more relaxation to shed complexity)
                    // If vent >> fusion, decrease flips (let complexity build)
                    g_hopfion_flip_scale = fminf(fmaxf(ratio, 0.1f), 10.0f);
                }
            }

            // === RING STABILITY DIAGNOSTIC ===
            // Compute observational stability: Δn / (n_avg - 1)
            if (h_sample_metrics->num_shells > 0) {
                // Find min and max refractive index across shells
                float n_min = h_sample_metrics->shell_n[0];
                float n_max = h_sample_metrics->shell_n[0];
                float n_sum = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float n = h_sample_metrics->shell_n[i];
                    if (n < n_min) n_min = n;
                    if (n > n_max) n_max = n;
                    n_sum += n;
                }
                float n_avg = n_sum / h_sample_metrics->num_shells;
                float delta_n = n_max - n_min;

                // Instability ratio (target: ~10% for EHT match)
                float stability_pct = 100.0f * delta_n / (n_avg - 1.0f);

                // === M_EFF CALCULATION (Option D: Active Region Only) ===
                // Only include shells within the active pumping region where mass
                // is being fed. Shells beyond M_EFF_ACTIVE_RADIUS are "fossil" structures
                // that propagate the refractive index wave but no longer accumulate mass.
                float M_eff = 0.0f;
                int active_shells = 0;
                float innermost_active_r = 0.0f;
                for (int i = 0; i < h_sample_metrics->num_shells; i++) {
                    float r = h_sample_metrics->shell_radii[i];
                    if (r < M_EFF_ACTIVE_RADIUS) {
                        M_eff += (h_sample_metrics->shell_n[i] - 1.0f) * r;
                        active_shells++;
                        if (innermost_active_r == 0.0f || r < innermost_active_r) {
                            innermost_active_r = r;
                        }
                    }
                }

                // Log shell migration for wave propagation analysis
                // (useful even when shells leave active region)
                float innermost_r = h_sample_metrics->shell_radii[0];
                static float prev_innermost_r = 0.0f;
                float migration_rate = 0.0f;
                if (prev_innermost_r > 0.0f && frame > 100) {
                    migration_rate = (innermost_r - prev_innermost_r);  // units per diagnostic interval
                }
                prev_innermost_r = innermost_r;

                printf("[lensing] stability=%.1f%% (target ~10%%) | M_eff=%.3f (%d active shells, r<%.0f) | n_avg=%.4f Δn=%.4f",
                       stability_pct, M_eff, active_shells, M_EFF_ACTIVE_RADIUS, n_avg, delta_n);

                // Show migration warning if shells are leaving active region
                if (active_shells < h_sample_metrics->num_shells && innermost_r > M_EFF_ACTIVE_RADIUS * 0.5f) {
                    printf(" | wave: r=%.0f (migration=%.2f/frame)", innermost_r, migration_rate);
                }
                printf("\n");

                // === PREDICTIVE LOCKING UPDATE ===
                // Check if we should enter/exit locked state based on stability metrics
                extern bool g_predictive_locking;
                if (g_predictive_locking) {
                    extern HarmonicLock g_harmonic_lock;
                    HarmonicLock& lock = g_harmonic_lock;

                    // Get current Q
                    float current_Q = latest_Q;
                    float Q_delta = fabsf(current_Q - lock.prev_Q);
                    int shell_count = h_sample_metrics->num_shells;

                    // Check lock criteria:
                    // 1. Shell count stable (8 shells)
                    // 2. Q variance low (not jumping around)
                    // 3. Stability low (shells well-formed)
                    bool shell_stable = (shell_count == lock.prev_shell_count && shell_count >= 6);
                    bool q_stable = (Q_delta < HarmonicLock::Q_VARIANCE_THRESHOLD);
                    bool stability_ok = (stability_pct < HarmonicLock::STABILITY_THRESHOLD * 100.0f);

                    if (shell_stable && q_stable && stability_ok) {
                        lock.stable_frames++;
                        if (!lock.locked && lock.stable_frames >= HarmonicLock::LOCK_THRESHOLD_FRAMES) {
                            lock.locked = true;
                            lock.lock_recheck_counter = 0;
                            printf("[lock] LOCKED: %d shells, Q=%.2f, stability=%.1f%% — skipping mip-tree rebuild\n",
                                   shell_count, current_Q, stability_pct);
                        }
                    } else {
                        // Lost stability
                        if (lock.locked) {
                            printf("[lock] UNLOCKED: shells=%d→%d, ΔQ=%.2f, stability=%.1f%%\n",
                                   lock.prev_shell_count, shell_count, Q_delta, stability_pct);
                        }
                        lock.stable_frames = 0;
                        lock.locked = false;
                    }

                    // Update previous values
                    lock.prev_shell_count = shell_count;
                    lock.prev_Q = current_Q;

                    // Periodic recheck when locked
                    if (lock.locked) {
                        lock.lock_recheck_counter++;
                    }
                }

                // === GEOMETRIC COHERENCE PUMP TRACKING ===
                // Power injected: P = (m·n)² — self-limiting
                // Equilibrium: E = (m·n)²/γ — bounded by |m|²/γ
                // With coherence filter: only aligned modes survive
                printf("[coherence] E_kin=%.2e E/N=%.3f | λ=%.3f γ=%.3f (bounded)\n",
                       sc.total_kinetic_energy, sc.energy_per_particle,
                       COHERENCE_LAMBDA, COHERENCE_GAMMA);

                // === SEAM DRIFT TRACKING (m=3 phase over time) ===
                // Log the m=3 phase angle to detect whether seam orientation drifts as M_eff grows
                if (sc.stress_sample_count > 10 && g_seam_drift_count < SEAM_DRIFT_LOG_SIZE) {
                    // Compute m=3 phase from circular mean: atan2(sin_sum, cos_sum)
                    float m3_phase = atan2f(sc.stress_sin_sum, sc.stress_cos_sum);
                    // Convert to degrees [0, 120) - the m=3 fundamental domain
                    float phase_deg = m3_phase * 180.0f / 3.14159265f;  // [-180, 180]
                    if (phase_deg < 0) phase_deg += 360.0f;             // [0, 360)
                    phase_deg = fmodf(phase_deg, 120.0f);               // [0, 120) - m=3 symmetry

                    g_seam_drift_log[g_seam_drift_count].frame = frame;
                    g_seam_drift_log[g_seam_drift_count].M_eff = M_eff;
                    g_seam_drift_log[g_seam_drift_count].phase_deg = phase_deg;
                    g_seam_drift_log[g_seam_drift_count].sample_count = sc.stress_sample_count;
                    g_seam_drift_count++;
                }

                // === PHOTON RING RADIUS TRACKING (EHT Observable) ===
                // Track the Einstein ring radius over time to verify geometric stability
                // EHT measurements show < 2% variation in M87* and Sgr A*
                static float ring_history[100] = {0};
                static int ring_idx = 0;
                static int ring_count = 0;

                float current_ring = h_sample_metrics->photon_ring_radius;
                ring_history[ring_idx] = current_ring;
                ring_idx = (ring_idx + 1) % 100;
                if (ring_count < 100) ring_count++;

                // Compute ring radius variation over last 100 frames
                if (ring_count > 10) {
                    float ring_min = ring_history[0];
                    float ring_max = ring_history[0];
                    float ring_sum = 0.0f;
                    for (int i = 0; i < ring_count; i++) {
                        if (ring_history[i] < ring_min) ring_min = ring_history[i];
                        if (ring_history[i] > ring_max) ring_max = ring_history[i];
                        ring_sum += ring_history[i];
                    }
                    float ring_avg = ring_sum / ring_count;
                    float ring_variation = 100.0f * (ring_max - ring_min) / ring_avg;

                    printf("[photon ring] R=%.2f | ΔR/R=%.2f%% (EHT target <2%%) | [%.2f ... %.2f]",
                           ring_avg, ring_variation, ring_min, ring_max);

                    // === SHELL-RING CONVERGENCE WARNING ===
                    // Gemini observed: innermost shell (r~90) expanding, ring (R~105) contracting
                    // When they meet, we test GPT's necessity question:
                    //   - Does topology cause graceful re-weaving? (proves necessity)
                    //   - Or does geometry collapse? (disproves necessity)
                    float r_inner = h_sample_metrics->shell_radii[h_sample_metrics->num_shells - 1];
                    float convergence_ratio = r_inner / ring_avg;

                    if (convergence_ratio > 0.95f) {
                        // CRITICAL: Shell has reached/passed the photon ring
                        printf(" | 🔥 TOPOLOGY TEST: Shell swallowing photon ring!");
                    } else if (convergence_ratio > 0.8f) {
                        // WARNING: Approaching convergence
                        printf(" | ⚠ CONVERGENCE: r_inner=%.1f (%.0f%% of R)", r_inner, convergence_ratio * 100.0f);

                        // Calculate shell spacing to predict re-weaving
                        if (h_sample_metrics->num_shells > 1) {
                            float r_next = h_sample_metrics->shell_radii[h_sample_metrics->num_shells - 2];
                            float shell_gap = r_next - r_inner;
                            printf(" | gap_to_next=%.1f", shell_gap);
                        }
                    }
                    printf("\n");

                    // === CONTROLLED EXPERIMENT TERMINATION ===
                    // Only auto-terminate in headless mode - interactive runs until user closes window
                    const int TARGET_FRAMES = g_target_frames;
                    const float TARGET_RADIUS = g_target_ring_radius;

                    bool termination_condition = false;
                    if (g_headless) {
                        if (g_terminate_on_radius) {
                            // Radius-based: terminate when ring reaches target (need at least 10 samples)
                            termination_condition = (ring_avg >= TARGET_RADIUS && ring_count >= 10);
                        } else {
                            // Frame-based: terminate at fixed frame count
                            termination_condition = (frame >= TARGET_FRAMES);
                        }
                    }

                    if (termination_condition) {
                        printf("\n");
                        printf("╔════════════════════════════════════════════════════════════════╗\n");
                        if (g_terminate_on_radius) {
                            printf("║  RADIUS-CONTROLLED EXPERIMENT COMPLETE                         ║\n");
                        } else {
                            printf("║  CONTROLLED EXPERIMENT COMPLETE (Equal Time)                   ║\n");
                        }
                        printf("╚════════════════════════════════════════════════════════════════╝\n");

                        if (g_terminate_on_radius) {
                            printf("[TERMINATION] Target ring radius reached:\n");
                            printf("  ✓ Ring R = %.2f (target: %.2f)\n", ring_avg, TARGET_RADIUS);
                            printf("  ✓ Frames = %d\n", frame);
                        } else {
                            printf("[TERMINATION] Target frames reached:\n");
                            printf("  ✓ Frames = %d (target: %d)\n", frame, TARGET_FRAMES);
                        }
                        printf("  ✓ M_eff = %.3f (compare mass retention)\n", M_eff);
                        printf("\nFinal Results:\n");
                        printf("  Radial Topology: %s\n", g_use_hopfion_topology ? "Hopfion shells" : "Smooth gradient");
                        printf("  Arm Mode: %s\n", !g_enable_arms ? "DISABLED" : g_use_arm_topology ? "Discrete boundaries" : "Smooth waves");
                        printf("  Photon Ring R: %.2f\n", ring_avg);
                        printf("  Ring Stability ΔR/R: %.2f%% (EHT target <2%%)\n", ring_variation);
                        printf("  Lensing Stability: %.1f%% (target ~10%%)\n", stability_pct);
                        printf("  Active Particles: %u\n", sc.active_count);
                        printf("  Ejected Particles: %u\n", sc.ejected_count);

                        // === AZIMUTHAL EJECTION DISTRIBUTION ===
                        // Tracks WHERE particles are ejected around the disk
                        // If pump creates m=3 pattern, we expect 3-fold symmetry in ejections
                        printf("\n");
                        printf("╔════════════════════════════════════════════════════════════╗\n");
                        printf("║  AZIMUTHAL EJECTION DISTRIBUTION (m=3 pump hypothesis)    ║\n");
                        printf("╚════════════════════════════════════════════════════════════╝\n");
                        printf("\n");

                        // Find max for scaling
                        unsigned int ejection_max = 0;
                        for (int b = 0; b < 16; b++) {
                            if (sc.ejection_bins[b] > ejection_max) ejection_max = sc.ejection_bins[b];
                        }

                        // Print histogram
                        printf("  φ (deg)   Count     Distribution\n");
                        printf("  ════════════════════════════════════════════════════\n");
                        unsigned int ejection_total = 0;
                        for (int b = 0; b < 16; b++) {
                            ejection_total += sc.ejection_bins[b];
                        }

                        for (int b = 0; b < 16; b++) {
                            float angle = b * 22.5f;
                            float pct = ejection_total > 0 ? 100.0f * sc.ejection_bins[b] / ejection_total : 0.0f;
                            int bar_len = ejection_max > 0 ? (sc.ejection_bins[b] * 40 / ejection_max) : 0;

                            printf("  %5.1f°  %7u   ", angle, sc.ejection_bins[b]);
                            for (int j = 0; j < bar_len; j++) printf("█");
                            printf(" %.1f%%\n", pct);
                        }

                        // Calculate m=3 asymmetry metric
                        // If pump creates 3 peaks, bins 0,5,10 (or shifted) should dominate
                        // Compute FFT-like m=3 component
                        float cos_sum = 0.0f, sin_sum = 0.0f;
                        for (int b = 0; b < 16; b++) {
                            float phi = b * 22.5f * 3.14159f / 180.0f;  // Convert to radians
                            float weight = ejection_total > 0 ? (float)sc.ejection_bins[b] / ejection_total : 0.0f;
                            cos_sum += weight * cosf(3.0f * phi);  // m=3 mode
                            sin_sum += weight * sinf(3.0f * phi);
                        }
                        float m3_amplitude = sqrtf(cos_sum * cos_sum + sin_sum * sin_sum);
                        float m3_phase_deg = atan2f(sin_sum, cos_sum) * 180.0f / 3.14159f;

                        printf("\n  m=3 Mode Analysis:\n");
                        printf("    Amplitude: %.3f (0=uniform, 1=perfect 3-fold)\n", m3_amplitude);
                        printf("    Phase: %.1f° (orientation of pattern)\n", m3_phase_deg);

                        if (m3_amplitude > 0.3f) {
                            printf("    → STRONG m=3 asymmetry detected! Pump-driven ejection.\n");
                        } else if (m3_amplitude > 0.15f) {
                            printf("    → Moderate m=3 signal. Partial pump influence.\n");
                        } else {
                            printf("    → Weak m=3. Ejections appear isotropic.\n");
                        }

                        // === BEAT FREQUENCY CROSSOVER TEST ===
                        // Tests whether m=3 clustering is due to beat frequency (ω_orb - ω_pump)
                        // Crossover radius r≈185: inner zone has ω_orb > ω_pump, outer has ω_orb < ω_pump
                        // If beat frequency model is correct: clustering differs between zones
                        if (sc.inner_ejection_total > 0 || sc.outer_ejection_total > 0) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  BEAT FREQUENCY CROSSOVER TEST (Claude's ω_orb vs ω_pump)     ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Crossover radius: r = %.0f (where ω_orb = ω_pump ≈ 0.125/step)\n", BEAT_CROSSOVER_RADIUS);
                            printf("Inner zone (r < %.0f): %u ejections\n", BEAT_CROSSOVER_RADIUS, sc.inner_ejection_total);
                            printf("Outer zone (r ≥ %.0f): %u ejections\n", BEAT_CROSSOVER_RADIUS, sc.outer_ejection_total);
                            printf("\n");

                            // Print side-by-side comparison
                            printf("  Sector    INNER ZONE (ω_orb > ω_pump)    OUTER ZONE (ω_orb < ω_pump)\n");
                            printf("  ═══════════════════════════════════════════════════════════════════\n");

                            // Find max for scaling
                            unsigned int inner_max = 0, outer_max = 0;
                            for (int b = 0; b < 16; b++) {
                                if (sc.inner_ejection_bins[b] > inner_max) inner_max = sc.inner_ejection_bins[b];
                                if (sc.outer_ejection_bins[b] > outer_max) outer_max = sc.outer_ejection_bins[b];
                            }

                            for (int b = 0; b < 16; b++) {
                                float angle = b * 22.5f;
                                float inner_pct = sc.inner_ejection_total > 0 ?
                                    100.0f * sc.inner_ejection_bins[b] / sc.inner_ejection_total : 0.0f;
                                float outer_pct = sc.outer_ejection_total > 0 ?
                                    100.0f * sc.outer_ejection_bins[b] / sc.outer_ejection_total : 0.0f;

                                // Bar charts (10 chars max each)
                                int inner_bar = inner_max > 0 ? (sc.inner_ejection_bins[b] * 10 / inner_max) : 0;
                                int outer_bar = outer_max > 0 ? (sc.outer_ejection_bins[b] * 10 / outer_max) : 0;

                                printf("  %3.0f°    %3u (%5.1f%%) ", angle, sc.inner_ejection_bins[b], inner_pct);
                                for (int i = 0; i < inner_bar; i++) printf("█");
                                for (int i = inner_bar; i < 10; i++) printf(" ");
                                printf("    %3u (%5.1f%%) ", sc.outer_ejection_bins[b], outer_pct);
                                for (int i = 0; i < outer_bar; i++) printf("█");
                                printf("\n");
                            }

                            // Analyze clustering in each zone
                            printf("\n");
                            printf("Analysis:\n");

                            // Find dominant sectors in each zone (m=3 means ~120° spacing)
                            int inner_peaks[3] = {-1, -1, -1};
                            int outer_peaks[3] = {-1, -1, -1};
                            unsigned int inner_peak_vals[3] = {0, 0, 0};
                            unsigned int outer_peak_vals[3] = {0, 0, 0};

                            for (int b = 0; b < 16; b++) {
                                // Insert into sorted top-3 for inner
                                if (sc.inner_ejection_bins[b] > inner_peak_vals[2]) {
                                    inner_peak_vals[2] = sc.inner_ejection_bins[b];
                                    inner_peaks[2] = b;
                                    // Bubble sort
                                    for (int i = 2; i > 0 && inner_peak_vals[i] > inner_peak_vals[i-1]; i--) {
                                        unsigned int tv = inner_peak_vals[i]; inner_peak_vals[i] = inner_peak_vals[i-1]; inner_peak_vals[i-1] = tv;
                                        int tp = inner_peaks[i]; inner_peaks[i] = inner_peaks[i-1]; inner_peaks[i-1] = tp;
                                    }
                                }
                                // Insert into sorted top-3 for outer
                                if (sc.outer_ejection_bins[b] > outer_peak_vals[2]) {
                                    outer_peak_vals[2] = sc.outer_ejection_bins[b];
                                    outer_peaks[2] = b;
                                    for (int i = 2; i > 0 && outer_peak_vals[i] > outer_peak_vals[i-1]; i--) {
                                        unsigned int tv = outer_peak_vals[i]; outer_peak_vals[i] = outer_peak_vals[i-1]; outer_peak_vals[i-1] = tv;
                                        int tp = outer_peaks[i]; outer_peaks[i] = outer_peaks[i-1]; outer_peaks[i-1] = tp;
                                    }
                                }
                            }

                            if (sc.inner_ejection_total >= 3 && inner_peaks[0] >= 0) {
                                printf("  Inner zone peaks: %d° (%u), %d° (%u), %d° (%u)\n",
                                       inner_peaks[0] * 22, inner_peak_vals[0],
                                       inner_peaks[1] * 22, inner_peak_vals[1],
                                       inner_peaks[2] * 22, inner_peak_vals[2]);
                                // Check for ~120° spacing
                                int d1 = abs(inner_peaks[1] - inner_peaks[0]);
                                int d2 = abs(inner_peaks[2] - inner_peaks[1]);
                                if (d1 >= 4 && d1 <= 6 && d2 >= 4 && d2 <= 6) {
                                    printf("    → m=3 clustering (~120° spacing) DETECTED\n");
                                }
                            }

                            if (sc.outer_ejection_total >= 3 && outer_peaks[0] >= 0) {
                                printf("  Outer zone peaks: %d° (%u), %d° (%u), %d° (%u)\n",
                                       outer_peaks[0] * 22, outer_peak_vals[0],
                                       outer_peaks[1] * 22, outer_peak_vals[1],
                                       outer_peaks[2] * 22, outer_peak_vals[2]);
                                int d1 = abs(outer_peaks[1] - outer_peaks[0]);
                                int d2 = abs(outer_peaks[2] - outer_peaks[1]);
                                if (d1 >= 4 && d1 <= 6 && d2 >= 4 && d2 <= 6) {
                                    printf("    → m=3 clustering (~120° spacing) DETECTED\n");
                                }
                            }

                            printf("\n");
                            printf("Interpretation:\n");
                            printf("  If INNER shows m=3 clustering but OUTER doesn't:\n");
                            printf("    → Beat frequency model CONFIRMED (clustering = ω_orb - ω_pump)\n");
                            printf("  If BOTH zones show same clustering pattern:\n");
                            printf("    → Clustering is arm-geometry driven, not beat frequency\n");
                            printf("\n");
                        }

                        // === HIGH-STRESS FIELD SPATIAL DISTRIBUTION ===
                        // Track pump_residual > 0.7 across the full disk by (r, θ)
                        // This reveals the spatial structure of pump instability across all radii
                        unsigned int total_high_stress = 0;
                        for (int r = 0; r < STRESS_RADIAL_BINS; r++) {
                            total_high_stress += sc.stress_radial_totals[r];
                        }

                        if (total_high_stress > 10) {  // Need sufficient data
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  HIGH-STRESS FIELD SPATIAL DISTRIBUTION (pump_residual > 0.7) ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");

                            // Radial bin boundaries (at ISCO = 6)
                            const char* radial_labels[STRESS_RADIAL_BINS] = {
                                "ISCO-2×ISCO  (6-12)",
                                "2×-4×ISCO   (12-24)",
                                "4×-8×ISCO   (24-48)",
                                "8×ISCO+     (48+)  "
                            };

                            // Print header with radial context
                            printf("Radial structure of high-stress particles:\n");
                            printf("  (ω_pump ≈ 0.125/step, crossover at r≈185 where ω_orb = ω_pump)\n\n");

                            // Print azimuthal distribution for each radial bin
                            for (int r = 0; r < STRESS_RADIAL_BINS; r++) {
                                if (sc.stress_radial_totals[r] < 3) continue;  // Skip sparse bins

                                printf("═══ %s: %u high-stress particles ═══\n",
                                       radial_labels[r], sc.stress_radial_totals[r]);

                                // Find max in this radial bin for scaling
                                unsigned int rmax = 0;
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    if (sc.stress_field[r][a] > rmax) rmax = sc.stress_field[r][a];
                                }

                                // Print angular distribution with bar chart
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    float angle = a * 22.5f;
                                    float pct = 100.0f * sc.stress_field[r][a] / sc.stress_radial_totals[r];
                                    int bar = rmax > 0 ? (sc.stress_field[r][a] * 15 / rmax) : 0;

                                    printf("  %5.1f° %4u (%5.1f%%) ", angle, sc.stress_field[r][a], pct);
                                    for (int i = 0; i < bar; i++) printf("█");
                                    printf("\n");
                                }

                                // Detect m-mode for this radial bin
                                // Find top 3 peaks
                                int peaks[3] = {-1, -1, -1};
                                unsigned int peak_vals[3] = {0, 0, 0};
                                for (int a = 0; a < STRESS_ANGULAR_BINS; a++) {
                                    if (sc.stress_field[r][a] > peak_vals[2]) {
                                        peak_vals[2] = sc.stress_field[r][a];
                                        peaks[2] = a;
                                        for (int i = 2; i > 0 && peak_vals[i] > peak_vals[i-1]; i--) {
                                            unsigned int tv = peak_vals[i]; peak_vals[i] = peak_vals[i-1]; peak_vals[i-1] = tv;
                                            int tp = peaks[i]; peaks[i] = peaks[i-1]; peaks[i-1] = tp;
                                        }
                                    }
                                }

                                if (peaks[0] >= 0 && peaks[1] >= 0 && peaks[2] >= 0) {
                                    // Sort peaks by angle for spacing calculation
                                    int sorted[3] = {peaks[0], peaks[1], peaks[2]};
                                    for (int i = 0; i < 2; i++) {
                                        for (int j = i+1; j < 3; j++) {
                                            if (sorted[j] < sorted[i]) {
                                                int t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                                            }
                                        }
                                    }

                                    // Calculate angular spacing (wrap-around aware)
                                    int d01 = sorted[1] - sorted[0];
                                    int d12 = sorted[2] - sorted[1];
                                    int d20 = (16 - sorted[2]) + sorted[0];  // wrap-around

                                    printf("  Peaks: %.0f°, %.0f°, %.0f° (spacing: %d, %d, %d sectors)\n",
                                           sorted[0] * 22.5f, sorted[1] * 22.5f, sorted[2] * 22.5f,
                                           d01, d12, d20);

                                    // Check for m=3 (~5-6 sectors = 112-135°)
                                    bool is_m3 = (d01 >= 4 && d01 <= 7) && (d12 >= 4 && d12 <= 7) && (d20 >= 4 && d20 <= 7);
                                    // Check for m=2 (~8 sectors = 180°)
                                    bool is_m2 = (d01 >= 7 && d01 <= 9) || (d12 >= 7 && d12 <= 9) || (d20 >= 7 && d20 <= 9);

                                    if (is_m3) {
                                        printf("  → m=3 mode detected (~120° spacing)\n");
                                    } else if (is_m2) {
                                        printf("  → m=2 mode detected (~180° spacing)\n");
                                    } else {
                                        printf("  → No clear m-mode (irregular spacing)\n");
                                    }
                                }
                                printf("\n");
                            }

                            // Summary: does m-mode change with radius?
                            printf("═══ M-MODE GRADIENT SUMMARY ═══\n");
                            printf("Beat frequency model predicts:\n");
                            printf("  Inner disk (ω_orb >> ω_pump): higher m-modes (m=3 or higher)\n");
                            printf("  Outer disk (ω_orb → ω_pump): lower m-modes (m=2 or m=1)\n");
                            printf("  Past crossover (ω_orb < ω_pump): azimuthally uniform\n");
                            printf("\n");
                        }

                        // === SEAM DRIFT TIME SERIES ===
                        // Output the logged m=3 phase over time to detect precession
                        if (g_seam_drift_count > 5) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  SEAM DRIFT TIME SERIES (m=3 phase vs M_eff)                   ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Tracking: Does the ~113° seam gap orientation drift as M_eff grows?\n");
                            printf("  If stationary: seam locked to arm geometry (fixed in lab frame)\n");
                            printf("  If drifting:   seam precessing on long timescale\n");
                            printf("\n");
                            printf("  Frame     M_eff    m=3 Phase   Samples\n");
                            printf("  ═════════════════════════════════════════\n");

                            // Print every Nth entry to keep output manageable
                            int stride = (g_seam_drift_count > 20) ? g_seam_drift_count / 20 : 1;
                            for (int i = 0; i < g_seam_drift_count; i += stride) {
                                printf("  %5d   %7.2f   %6.1f°     %d\n",
                                       g_seam_drift_log[i].frame,
                                       g_seam_drift_log[i].M_eff,
                                       g_seam_drift_log[i].phase_deg,
                                       g_seam_drift_log[i].sample_count);
                            }

                            // Compute phase drift statistics
                            float phase_sum = 0.0f, phase_sq_sum = 0.0f;
                            float phase_first = g_seam_drift_log[0].phase_deg;
                            float phase_last = g_seam_drift_log[g_seam_drift_count - 1].phase_deg;
                            for (int i = 0; i < g_seam_drift_count; i++) {
                                phase_sum += g_seam_drift_log[i].phase_deg;
                                phase_sq_sum += g_seam_drift_log[i].phase_deg * g_seam_drift_log[i].phase_deg;
                            }
                            float phase_mean = phase_sum / g_seam_drift_count;
                            float phase_var = phase_sq_sum / g_seam_drift_count - phase_mean * phase_mean;
                            float phase_std = sqrtf(fmaxf(phase_var, 0.0f));
                            float phase_drift = phase_last - phase_first;

                            // === LIMIT CYCLE VS FIXED POINT ANALYSIS ===
                            // Split data into thirds and compare oscillation amplitude
                            // If amplitude decreases: approaching fixed point (damped)
                            // If amplitude constant: limit cycle (persistent oscillation)
                            int third = g_seam_drift_count / 3;
                            float early_sum = 0.0f, early_sq = 0.0f, early_mean;
                            float mid_sum = 0.0f, mid_sq = 0.0f, mid_mean;
                            float late_sum = 0.0f, late_sq = 0.0f, late_mean;

                            for (int i = 0; i < third; i++) {
                                early_sum += g_seam_drift_log[i].phase_deg;
                            }
                            early_mean = early_sum / third;
                            for (int i = 0; i < third; i++) {
                                float d = g_seam_drift_log[i].phase_deg - early_mean;
                                early_sq += d * d;
                            }
                            float early_std = sqrtf(early_sq / third);

                            for (int i = third; i < 2*third; i++) {
                                mid_sum += g_seam_drift_log[i].phase_deg;
                            }
                            mid_mean = mid_sum / third;
                            for (int i = third; i < 2*third; i++) {
                                float d = g_seam_drift_log[i].phase_deg - mid_mean;
                                mid_sq += d * d;
                            }
                            float mid_std = sqrtf(mid_sq / third);

                            for (int i = 2*third; i < g_seam_drift_count; i++) {
                                late_sum += g_seam_drift_log[i].phase_deg;
                            }
                            int late_count = g_seam_drift_count - 2*third;
                            late_mean = late_sum / late_count;
                            for (int i = 2*third; i < g_seam_drift_count; i++) {
                                float d = g_seam_drift_log[i].phase_deg - late_mean;
                                late_sq += d * d;
                            }
                            float late_std = sqrtf(late_sq / late_count);

                            printf("\n");
                            printf("Summary:\n");
                            printf("  Phase mean: %.1f° ± %.1f° (std dev)\n", phase_mean, phase_std);
                            printf("  Net drift:  %.1f° (first→last)\n", phase_drift);
                            printf("\n");
                            printf("Oscillation Amplitude by Phase (limit cycle vs fixed point test):\n");
                            printf("  Early (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[third/2].M_eff, early_mean, early_std);
                            printf("  Mid   (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[third + third/2].M_eff, mid_mean, mid_std);
                            printf("  Late  (M_eff ~%.0f):  mean=%.1f° ± %.1f°\n",
                                   g_seam_drift_log[2*third + late_count/2].M_eff, late_mean, late_std);
                            printf("\n");

                            // Determine behavior
                            float amp_ratio = (early_std > 0.1f) ? late_std / early_std : 1.0f;
                            if (phase_std < 10.0f && fabsf(phase_drift) < 15.0f) {
                                printf("  → STATIONARY: Seam orientation is locked to arm geometry\n");
                            } else if (fabsf(phase_drift) > 30.0f) {
                                printf("  → PRECESSING: Seam drifting %.1f° over run\n", phase_drift);
                                if (amp_ratio < 0.5f) {
                                    printf("  → DAMPED: Oscillation amplitude decreasing (%.1f× early→late)\n", amp_ratio);
                                    printf("     Approaching FIXED POINT - seam will phase-lock\n");
                                } else if (amp_ratio > 0.8f && amp_ratio < 1.2f) {
                                    printf("  → LIMIT CYCLE: Oscillation amplitude stable (%.1f× early→late)\n", amp_ratio);
                                    printf("     Persistent oscillation - pump still driving seam\n");
                                } else if (amp_ratio > 1.5f) {
                                    printf("  → UNSTABLE: Oscillation amplitude growing (%.1f× early→late)\n", amp_ratio);
                                    printf("     System may be diverging\n");
                                }
                            } else {
                                printf("  → UNCERTAIN: Moderate scatter, needs longer run\n");
                            }
                            printf("\n");
                        }

                        // === RESIDENCE TIME DIAGNOSTIC (Test A) ===
                        extern bool g_test_residence_time;
                        if (g_test_residence_time && g_enable_arms && sc.active_count > 0) {
                            printf("\n");
                            printf("╔════════════════════════════════════════════════════════════════╗\n");
                            printf("║  RESIDENCE TIME DIAGNOSTIC (Test A: Weak Barrier Test)        ║\n");
                            printf("╚════════════════════════════════════════════════════════════════╝\n");
                            printf("\n");
                            printf("Accumulated residence time over %d frames:\n", frame);
                            printf("\n");

                            float total_time = sc.arm_residence_time + sc.gap_residence_time;
                            float arm_fraction = (total_time > 0) ? sc.arm_residence_time / total_time : 0.0f;
                            float gap_fraction = (total_time > 0) ? sc.gap_residence_time / total_time : 0.0f;

                            printf("  Arm Regions:  %.0f particle-frames (%.1f%% of total)\n",
                                   sc.arm_residence_time, arm_fraction * 100.0f);
                            printf("  Gap Regions:  %.0f particle-frames (%.1f%% of total)\n",
                                   sc.gap_residence_time, gap_fraction * 100.0f);
                            printf("\n");
                            printf("  Current distribution:\n");
                            printf("    Particles in arms: %u (%.1f%%)\n",
                                   sc.arm_particle_count,
                                   100.0f * sc.arm_particle_count / (float)sc.active_count);
                            printf("    Particles in gaps: %u (%.1f%%)\n",
                                   sc.gap_particle_count,
                                   100.0f * sc.gap_particle_count / (float)sc.active_count);
                            printf("\n");

                            float arm_width = ARM_WIDTH_DEG / 360.0f;  // Fraction of circle
                            float expected_arm_fraction = arm_width * NUM_ARMS;
                            float residence_enhancement = arm_fraction / expected_arm_fraction;

                            printf("  Expected arm fraction (geometric): %.1f%%\n", expected_arm_fraction * 100.0f);
                            printf("  Observed arm fraction (residence): %.1f%%\n", arm_fraction * 100.0f);
                            printf("  Residence enhancement factor: %.2fx\n", residence_enhancement);
                            printf("\n");

                            if (g_use_arm_topology) {
                                printf("Interpretation (DISCRETE ARMS):\n");
                                printf("  If residence enhancement > 1.5× → Weak barriers trap particles\n");
                                printf("  If residence enhancement ≈ 1.0× → No trapping effect\n");
                                printf("\n");
                                if (residence_enhancement > 1.5f) {
                                    printf("  ✓ BARRIER CONFIRMED: Particles accumulate in arms\n");
                                } else if (residence_enhancement > 1.2f) {
                                    printf("  ~ WEAK BARRIER: Modest accumulation in arms\n");
                                } else {
                                    printf("  ✗ NO BARRIER: Particles drift through arms freely\n");
                                }
                            } else {
                                printf("Interpretation (SMOOTH ARMS):\n");
                                printf("  Expected: residence ≈ geometric (no preferential trapping)\n");
                                printf("  If enhancement > 1.2× → Density gradient causes accumulation\n");
                            }
                            printf("\n");
                        }
                        printf("\n");

                        // Clean exit
                        if (!g_headless) {
                            glfwSetWindowShouldClose(window, 1);
                        }
                        running = false;
                    }
                }
            }

            fps_acc = 0; fps_frames = 0;
        }

        // === RENDERING ===
#ifdef VULKAN_INTEROP
        // Vulkan rendering
        {
            // NOTE: No cudaDeviceSynchronize() - CUDA and Vulkan work on same GPU
            // The shared buffer is written by CUDA kernel, read by Vulkan vertex shader
            // GPU naturally serializes operations on same memory

            // Update camera from Vulkan context
            float camX = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * sinf(vkCtx.cameraYaw);
            float camY = vkCtx.cameraRadius * sinf(vkCtx.cameraPitch);
            float camZ = vkCtx.cameraRadius * cosf(vkCtx.cameraPitch) * cosf(vkCtx.cameraYaw);

            // Build view-projection matrix
            Vec3 eye = {camX, camY, camZ};
            Vec3 center = {0, 0, 0};
            Mat4 view = Mat4::lookAt(eye, center, {0, 1, 0});

            int fb_w, fb_h;
            glfwGetFramebufferSize(vkCtx.window, &fb_w, &fb_h);
            float aspect = (float)fb_w / (float)fb_h;
            Mat4 proj = Mat4::perspective(PI / 4.0f, aspect, 0.1f, 2000.0f);

            // Vulkan uses inverted Y compared to OpenGL
            proj.m[5] *= -1.0f;

            Mat4 vp = Mat4::mul(proj, view);

            // Update uniform buffer
            GlobalUBO ubo;
            memcpy(ubo.viewProj, vp.m, sizeof(float) * 16);
            ubo.cameraPos[0] = eye.x;
            ubo.cameraPos[1] = eye.y;
            ubo.cameraPos[2] = eye.z;
            ubo.time = sim_time;
            ubo.avgScale = pump_bridge.avg_scale;
            ubo.avgResidual = pump_bridge.avg_residual;
            ubo.heartbeat = pump_bridge.heartbeat;
            ubo.pump_phase = sim_time * 0.125f * 2.0f * 3.14159f;  // ω_pump = 0.125

            // Copy to mapped uniform buffer
            memcpy(vkCtx.uniformBuffersMapped[vkCtx.currentFrame], &ubo, sizeof(ubo));

            // Update attractor pipeline (density rendering mode)
            if (g_attractor.enabled) {
                // Update particle count for compute dispatch
                g_attractor.particleCount = N_current;
                // Update time for phase evolution (in phase-primary mode)
                g_attractor.time = sim_time;

                // Update camera (viewProj already computed)
                float aspect = (float)vkCtx.swapchainExtent.width / (float)vkCtx.swapchainExtent.height;
                vk::updateAttractorCamera(g_attractor, ubo.viewProj, 1.0f, aspect,
                                          vkCtx.swapchainExtent.width, vkCtx.swapchainExtent.height);

                // Update attractor state for pure mode (use bridge state)
                // w oscillates based on heartbeat, phase rotates with time
                float w = fmaxf(0.0f, fminf(0.5f, pump_bridge.heartbeat * 0.25f + 0.25f));
                float phase = fmodf(sim_time * 0.5f, 6.2831853f);  // Slow rotation
                vk::updateAttractorState(g_attractor, w, phase, pump_bridge.avg_residual);
            }

            // Draw frame (skip in headless mode for pure physics benchmark)
            if (!g_headless) {
                vk::drawFrame(vkCtx);
            }

            // === KERNEL TIMING OUTPUT (async - no stall) ===
            // Record end event, but don't sync - we'll print NEXT time if ready
            static bool timing_pending = false;
            if (do_timing) {
                cudaEventRecord(t_render);
                timing_pending = true;
            }
            // Check if previous timing is ready (non-blocking)
            if (timing_pending) {
                cudaError_t status = cudaEventQuery(t_render);
                if (status == cudaSuccess) {
                    cudaEventElapsedTime(&ms_siphon, t_start, t_siphon);
                    cudaEventElapsedTime(&ms_physics, t_siphon, t_physics);
                    cudaEventElapsedTime(&ms_render, t_physics, t_render);
                    printf("[profile] siphon=%.2fms physics=%.2fms render=%.2fms total=%.2fms\n",
                           ms_siphon, ms_physics, ms_render, ms_siphon + ms_physics + ms_render);
                    timing_pending = false;
                }
            }

            glfwPollEvents();
        }
#else
        // === OPENGL RENDERING (only in interactive mode) ===
        if (!g_headless) {
            // Framebuffer
            int fb_w, fb_h;
            glfwGetFramebufferSize(window, &fb_w, &fb_h);
            glViewport(0, 0, fb_w, fb_h);
            float fb_aspect = (float)fb_w / (float)fb_h;

            // Camera
            float camX = g_cam.dist * cosf(g_cam.elevation) * sinf(g_cam.azimuth);
            float camY = g_cam.dist * sinf(g_cam.elevation);
            float camZ = g_cam.dist * cosf(g_cam.elevation) * cosf(g_cam.azimuth);
            Vec3 eye = {camX, camY, camZ};
            Vec3 center = {0,0,0};
            Vec3 fwd = (center - eye).norm();
            Vec3 right = fwd.cross({0,1,0}).norm();
            Vec3 camup = right.cross(fwd);

            Mat4 view = Mat4::lookAt(eye, center, {0,1,0});
            Mat4 proj = Mat4::perspective(PI/4.0f, fb_aspect, 0.1f, 1000.0f);  // Larger far plane
            Mat4 vp = Mat4::mul(proj, view);

            float fovTan = tanf(PI/8.0f);

            // Render BH background
            glDepthMask(GL_FALSE);
            glUseProgram(bhProgram);
            glUniform3f(u_camPos_bh, eye.x, eye.y, eye.z);
            glUniform3f(u_camFwd_bh, fwd.x, fwd.y, fwd.z);
            glUniform3f(u_camRight_bh, right.x, right.y, right.z);
            glUniform3f(u_camUp_bh, camup.x, camup.y, camup.z);
            glUniform1f(u_time_bh, sim_time);
            glUniform1f(u_aspect_bh, fb_aspect);
            glUniform1f(u_fov_bh, fovTan);

            // === BRIDGE UNIFORMS: Feed pump state to raymarcher ===
            glUniform1f(u_avgScale_bh, pump_bridge.avg_scale);
            glUniform1f(u_avgResidual_bh, pump_bridge.avg_residual);
            glUniform1f(u_pumpWork_bh, pump_bridge.total_work);
            glUniform1f(u_heartbeat_bh, pump_bridge.heartbeat);

            // === HOPFION SHELLS: Upload EM confinement layer structure ===
            glUniform1fv(u_shellRadii_bh, h_sample_metrics->num_shells, h_sample_metrics->shell_radii);
            glUniform1fv(u_shellN_bh, h_sample_metrics->num_shells, h_sample_metrics->shell_n);
            glUniform1i(u_numShells_bh, h_sample_metrics->num_shells);

            glBindVertexArray(fsVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glDepthMask(GL_TRUE);

            // Render disk
            glClear(GL_DEPTH_BUFFER_BIT);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glDepthMask(GL_FALSE);
            glUseProgram(diskProgram);
            glUniformMatrix4fv(u_viewProj, 1, GL_FALSE, vp.m);
            glUniform3f(u_camPos_d, eye.x, eye.y, eye.z);

            // === HOPFION SHELLS: Upload same shell structure to particle shader ===
            glUniform1fv(u_shellRadii_d, h_sample_metrics->num_shells, h_sample_metrics->shell_radii);
            glUniform1fv(u_shellN_d, h_sample_metrics->num_shells, h_sample_metrics->shell_n);
            glUniform1i(u_numShells_d, h_sample_metrics->num_shells);

            glBindVertexArray(quadVAO);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, N_current);
            glDepthMask(GL_TRUE);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }  // End of !g_headless rendering
#endif

        frame++;
        fps_acc += dt_wall;
        fps_frames++;
    }

    // === CLEANUP: Graceful shutdown for 3.5M particles ===
    printf("\n[shutdown] Beginning cleanup...\n");

    // 1. Synchronize all CUDA operations before cleanup
    cudaDeviceSynchronize();
    printf("[shutdown] CUDA synchronized\n");

#ifdef VULKAN_INTEROP
    // 2. Wait for Vulkan to finish all GPU work BEFORE destroying shared buffers
    // This prevents "buffer in use by command buffer" errors
    vkDeviceWaitIdle(vkCtx.device);
    printf("[shutdown] Vulkan device idle\n");

    // Vulkan cleanup - now safe to destroy shared resources
    destroySharedBuffer(vkCtx.device, &sharedParticleBuffer);
    // Clear handles to prevent double-free in vk::cleanup
    vkCtx.particleBuffer = VK_NULL_HANDLE;
    vkCtx.particleBufferMemory = VK_NULL_HANDLE;
    printf("[shutdown] Shared buffer destroyed\n");

    // Clean up hybrid LOD resources
    if (hybridLODEnabled) {
        // Clean up double-buffered indirect draw resources
        if (doubleBufferEnabled) {
            for (int i = 0; i < 2; i++) {
                destroySharedIndirectDraw(vkCtx.device, &indirectDrawBuffers[i]);
            }
            // Clear handles to prevent double-free
            vkCtx.compactedParticleBuffer = VK_NULL_HANDLE;
            vkCtx.compactedParticleBufferMemory = VK_NULL_HANDLE;
            vkCtx.indirectDrawBuffer = VK_NULL_HANDLE;
            vkCtx.indirectDrawBufferMemory = VK_NULL_HANDLE;
            printf("[shutdown] Double-buffered indirect draw resources destroyed\n");
        }

        destroySharedDensityGrid(vkCtx.device, &densityGrid);
        printf("[shutdown] Density grid destroyed\n");
        if (d_nearCount) {
            cudaFree(d_nearCount);
            printf("[shutdown] Near count buffer freed\n");
        }
    }

    // Cleanup attractor pipeline
    if (g_attractor.enabled) {
        vk::destroyAttractorPipeline(vkCtx, g_attractor);
    }

    vk::cleanup(vkCtx);
    printf("[shutdown] Vulkan cleaned up\n");

    glfwDestroyWindow(vkCtx.window);
    glfwTerminate();
    printf("[shutdown] Window closed, GLFW terminated\n");
#else
    // 2. Unregister graphics resources (only if rendering was enabled)
    if (!g_headless) {
        cudaGraphicsUnregisterResource(posRes);
        cudaGraphicsUnregisterResource(colorRes);
        cudaGraphicsUnregisterResource(sizeRes);
        cudaGraphicsUnregisterResource(velocityRes);
        cudaGraphicsUnregisterResource(elongationRes);
        printf("[shutdown] Graphics resources unregistered\n");
    }

    // Cleanup Vulkan shared memory (file-based IPC mode)
    cleanupVulkanSharedMemory();

    // Delete OpenGL resources (only if rendering was enabled)
    if (!g_headless) {
        glDeleteProgram(bhProgram);
        glDeleteProgram(diskProgram);
        printf("[shutdown] Shaders deleted\n");

        glfwDestroyWindow(window);
        glfwTerminate();
        printf("[shutdown] Window closed, GLFW terminated\n");
    }
#endif

    // 3. Free all CUDA resources via context
    cleanupSimulation(ctx, diag, octree);

    return 0;
}
