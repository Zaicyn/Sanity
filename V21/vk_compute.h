/*
 * V21 VULKAN COMPUTE — GPU Physics Dispatch
 * ==========================================
 *
 * Loads siphon.spv, creates compute pipeline, dispatches GPU physics.
 * Replaces CPU physics_step() with parallel GPU execution.
 * Includes staging buffer for CPU oracle readback.
 *
 * The GPU's rasterizer does the rendering (additive blend = harmonic sum).
 * The GPU's compute unit does the physics (SPIRV = vendor-neutral).
 * The CPU's oracle validates the invariants.
 */

#ifndef V21_VK_COMPUTE_H
#define V21_VK_COMPUTE_H

#include "../vulkan/vk_types.h"
#include <vulkan/vulkan.h>

/* Push constants for siphon.comp */
struct SiphonPushConstants {
    int   N;
    float time;
    float dt;
    float BH_MASS;
    float FIELD_STRENGTH;
    float FIELD_FALLOFF;
    float TANGENT_SCALE;
    uint32_t seam_bits;
    float bias;
};

/* Push constants for scatter.comp */
struct ScatterPushConstants {
    int   N;
    int   grid_dim;
    float grid_cell_size;
    float grid_half_size;
};

/* Push constants for scatter_reduce.comp */
struct ScatterReducePushConstants {
    int total_cells;
};

/* Push constants for stencil.comp (Pass 2) */
struct StencilPushConstants {
    int   grid_dim;
    int   total_cells;
    float grid_cell_size;
    float pressure_k;
};

/* Push constants for gather_measure.comp (Pass 3, measurement-only) */
struct GatherMeasurePushConstants {
    int   N;
    int   total_cells;
    float dt;
};

/* Push constants for constraint_solve.comp (Pass 4, rigid-body MVP) */
struct ConstraintPushConstants {
    uint32_t constraint_offset;   /* start index of current bucket in pairs[] */
    uint32_t constraint_count;    /* size of current bucket */
    uint32_t rigid_base;          /* = N_field, first lattice particle index */
    uint32_t rigid_count;         /* = N_rigid */
    float    beta;                /* Baumgarte coefficient (0.2 for MVP) */
    float    compliance;          /* XPBD compliance (0.0 placeholder) */
    float    dt;
    uint32_t _pad;
};

/* Push constants for collision_apply.comp (Phase 2.2 — velocity writeback + position integrate). */
struct CollisionApplyPushConstants {
    uint32_t rigid_base;          /* = N_field, first lattice particle index */
    uint32_t rigid_count;         /* = N_rigid */
    float    dt;                  /* timestep for position integration */
    uint32_t _pad;
};

/* Push constants for collision_resolve.comp (Phase 2.2 C2 — fused broadphase + impulse). */
struct CollisionResolvePushConstants {
    uint32_t rigid_base;          /* = N_field, first lattice particle index */
    uint32_t rigid_count;         /* = N_rigid */
    float    dt;                  /* simulation timestep */
    uint32_t frame_number;        /* for first_contact_frame probe */
};

/* Push constants for scatter_scan.comp */
struct ScanPushConstants {
    uint32_t N;
    uint32_t mode;   /* 0=local, 1=block, 2=propagate */
};

/* Push constants for scatter_histogram.comp */
struct HistogramPushConstants {
    uint32_t N;
};

/* Push constants for scatter_reorder.comp */
struct ReorderPushConstants {
    uint32_t N;
};

/* Cell grid constants (must match scatter.comp) */
#define V21_GRID_DIM          64
#define V21_GRID_CELLS        (V21_GRID_DIM * V21_GRID_DIM * V21_GRID_DIM)
#define V21_GRID_HALF_SIZE    250.0f
#define V21_GRID_CELL_SIZE    (2.0f * V21_GRID_HALF_SIZE / (float)V21_GRID_DIM)
#define V21_GRID_SHARD_COUNT  8

/* Cylindrical density grid — aligned with disk geometry, no Cartesian artifacts */
#define V21_CYL_NR            64
#define V21_CYL_NPHI          96
#define V21_CYL_NY            32
#define V21_CYL_CELLS         (V21_CYL_NR * V21_CYL_NPHI * V21_CYL_NY)
#define V21_CYL_R_MAX         200.0f
#define V21_CYL_Y_HALF        100.0f
#define V21_CYL_DR            (V21_CYL_R_MAX / (float)V21_CYL_NR)
#define V21_CYL_DPHI          (6.28318530718f / (float)V21_CYL_NPHI)
#define V21_CYL_DY            (2.0f * V21_CYL_Y_HALF / (float)V21_CYL_NY)

/* Scatter contention-mode selector (A/B/C test for Squaragon thesis). */
enum ScatterMode {
    SCATTER_MODE_BASELINE  = 0,  /* 1 shard, maximum atomic contention */
    SCATTER_MODE_UNIFORM   = 1,  /* 8 shards, uniform lane & 7 selector */
    SCATTER_MODE_SQUARAGON = 2   /* 8 shards, V21_SCATTER_LUT[lane&31] & 7 */
};

/* Push constants for project.comp */
struct ProjectPushConstants {
    int   particle_count;
    float brightness;
    float temp_scale;
};

/* Push constants for tone_map.frag */
struct ToneMapPushConstants {
    float peak_density;
    float exposure;
    float gamma;
    int   colormap;
};

/* Camera UBO for projection */
struct ProjectCameraUBO {
    float view_proj[16];
    float zoom;
    float aspect;
    int   width;
    int   height;
};

/* Number of SoA arrays bound as SSBOs */
#define VK_COMPUTE_NUM_BINDINGS 16

/* Number of grade-separated buffers (set 2) */
#define VK_GRADED_NUM_BINDINGS 10

/* Push constants for cartesian_to_graded.comp */
struct CartesianToGradedPushConstants {
    int   N;
    float BH_MASS;
};

/* Push constants for graded_to_cartesian.comp */
struct GradedToCartesianPushConstants {
    int N;
};

/* Physics compute state */
struct PhysicsCompute {
    /* Per-array SSBOs */
    VkBuffer      soa_buffers[VK_COMPUTE_NUM_BINDINGS];
    VkDeviceMemory soa_memory[VK_COMPUTE_NUM_BINDINGS];
    size_t         soa_sizes[VK_COMPUTE_NUM_BINDINGS];

    /* Siphon compute pipeline (physics) */
    VkDescriptorSetLayout descLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;

    /* Scatter compute pipeline (particles → cell grid, Pass 1) */
    ScatterMode           scatterMode;           /* 0=baseline, 1=uniform, 2=squaragon */
    VkBuffer              gridDensityBuffer;     /* uint[V21_GRID_CELLS] — canonical, after reduce */
    VkDeviceMemory        gridDensityMemory;
    VkBuffer              gridDensityShardsBuffer; /* uint[SHARD_COUNT * V21_GRID_CELLS] */
    VkDeviceMemory        gridDensityShardsMemory;
    VkBuffer              particleCellBuffer;    /* uint[N] */
    VkDeviceMemory        particleCellMemory;
    VkDescriptorSetLayout scatterSet1Layout;     /* layout for set 1 (shards + particle_cell) */
    VkPipelineLayout      scatterPipelineLayout; /* uses desc set 0 (shared with siphon) + set 1 */
    VkPipeline            scatterPipeline;
    VkDescriptorPool      scatterDescPool;
    VkDescriptorSet       scatterSet1;

    /* Scatter reduce pipeline (8 shards → 1 canonical density) */
    VkDescriptorSetLayout scatterReduceSetLayout;
    VkPipelineLayout      scatterReducePipelineLayout;
    VkPipeline            scatterReducePipeline;
    VkDescriptorPool      scatterReduceDescPool;
    VkDescriptorSet       scatterReduceSet;

    /* Stencil pipeline (Pass 2 — density → pressure gradient) */
    VkBuffer              pressureXBuffer;        /* float[V21_GRID_CELLS] */
    VkDeviceMemory        pressureXMemory;
    VkBuffer              pressureYBuffer;
    VkDeviceMemory        pressureYMemory;
    VkBuffer              pressureZBuffer;
    VkDeviceMemory        pressureZMemory;
    VkDescriptorSetLayout stencilSetLayout;
    VkPipelineLayout      stencilPipelineLayout;
    VkPipeline            stencilPipeline;
    VkDescriptorPool      stencilDescPool;
    VkDescriptorSet       stencilSet;

    /* Gather-measure pipeline (Pass 3 — pressure at cell, write to scratch) */
    VkBuffer              gatherScratchBuffer;    /* float[N], write-only sink */
    VkDeviceMemory        gatherScratchMemory;
    VkDescriptorSetLayout gatherMeasureSet1Layout; /* set 1: particle_cell + pressure + scratch */
    VkPipelineLayout      gatherMeasurePipelineLayout;
    VkPipeline            gatherMeasurePipeline;
    VkDescriptorPool      gatherMeasureDescPool;
    VkDescriptorSet       gatherMeasureSet1;

    /* Constraint solver pipeline (Pass 4 — PBD distance constraints, rigid-body MVP).
     * All buffers and pipeline state are only allocated when constraintEnabled is
     * true (i.e. when --rigid-body != off). */
    VkBuffer              constraintPairsBuffer;   /* uvec2[M], static upload */
    VkDeviceMemory        constraintPairsMemory;
    VkBuffer              restLengthsBuffer;       /* float[M] */
    VkDeviceMemory        restLengthsMemory;
    VkBuffer              invMassesBuffer;         /* float[N_rigid] */
    VkDeviceMemory        invMassesMemory;
    VkDescriptorSetLayout constraintSet1Layout;    /* set 1: pairs + rest + inv_m */
    VkPipelineLayout      constraintPipelineLayout;
    VkPipeline            constraintPipeline;
    VkDescriptorPool      constraintDescPool;
    VkDescriptorSet       constraintSet1;

    /* Constraint solver config (set by initConstraintCompute) */
    bool     constraintEnabled;
    uint32_t rigidBaseIndex;            /* = N_field (first lattice particle index) */
    uint32_t rigidCount;                /* = N_rigid */
    uint32_t constraintBucketOffsets[7];/* start offset of each bucket in pairs[] */
    uint32_t constraintBucketCounts[7]; /* count per (axis × parity) bucket; [6] reserved for joint edges */
    uint32_t constraintIterations;      /* solver iterations per frame */

    /* Collision pipeline (Phase 2.2 — dynamic contact constraints, velocity-level
     * impulses). Allocated when collisionEnabled is true (set by initCollisionCompute).
     *
     * C1 ships only the apply kernel + buffer infrastructure; the fused broadphase
     * + resolve kernel lands in C2. Buffers are sized for the rigid lattice range
     * only (vel_delta is N_rigid*3 int32 = 24 KB for cube2-collide).
     *
     * Descriptor set layout (per-pipeline set 1):
     *   binding 0  rigid_body_id[N]                uint32, init-time
     *   binding 1  vel_delta[N_rigid * 3]          int32,  zeroed/written per frame
     *   binding 2  contact_count[1]                uint32, probe, reset per frame
     *   binding 3  first_contact_frame[1]          uint32, probe, reset per frame
     */
    bool                  collisionEnabled;
    VkBuffer              rigidBodyIdBuffer;        /* uint32[N], init-time only */
    VkDeviceMemory        rigidBodyIdMemory;
    VkBuffer              velDeltaBuffer;           /* int32[N_rigid * 3] */
    VkDeviceMemory        velDeltaMemory;
    VkBuffer              contactCountBuffer;       /* uint32[1] */
    VkDeviceMemory        contactCountMemory;
    VkBuffer              firstContactFrameBuffer;  /* uint32[1] */
    VkDeviceMemory        firstContactFrameMemory;
    VkBuffer              posPrevBuffer;            /* float[N_rigid * 3], position snapshot */
    VkDeviceMemory        posPrevMemory;
    VkDescriptorSetLayout collisionSet1Layout;      /* set 1: 5 SSBOs */
    VkPipelineLayout      collisionApplyPipelineLayout;
    VkPipeline            collisionApplyPipeline;
    VkPipelineLayout      collisionResolvePipelineLayout;
    VkPipeline            collisionResolvePipeline;
    VkPipelineLayout      collisionSyncPipelineLayout;
    VkPipeline            collisionSyncPipeline;
    VkPipeline            collisionSnapshotPipeline;
    VkDescriptorPool      collisionDescPool;
    VkDescriptorSet       collisionSet1;

    /* Projection compute pipeline (rendering) */
    VkDescriptorSetLayout projDescLayout;
    VkPipelineLayout projPipelineLayout;
    VkPipeline projPipeline;
    VkDescriptorPool projDescPool;
    VkDescriptorSet projDescSet;

    /* Tone-map graphics pipeline */
    VkDescriptorSetLayout toneDescLayout;
    VkPipelineLayout tonePipelineLayout;
    VkPipeline tonePipeline;
    VkDescriptorPool toneDescPool;
    VkDescriptorSet toneDescSet;

    /* Density image (R32_UINT) */
    VkImage densityImage;
    VkDeviceMemory densityMemory;
    VkImageView densityView;
    VkSampler densitySampler;

    /* Camera UBO */
    VkBuffer cameraUBO;
    VkDeviceMemory cameraMemory;
    void* cameraMapped;

    /* Staging buffer for CPU oracle readback */
    VkBuffer staging;
    VkDeviceMemory stagingMemory;
    void* stagingMapped;
    size_t stagingSize;

    /* GPU timestamp profiling — double-buffered.
     * Each slot holds 4 timestamps: siphon_begin, siphon_end,
     * project_end, tonemap_end. Reads previous frame's results
     * non-blocking to avoid stalling the queue. */
    VkQueryPool queryPool;
    float timestampPeriodNs;   /* ns per timestamp tick */
    bool queryValid[2];        /* whether slot N has been written yet */
    uint32_t queryFrame;       /* 0 or 1, current write slot */

    /* Particle count */
    int N;
    bool initialized;

    /* ---- Grade-separated state (Phase 3.1 scaffolding) ----
     * 10 SSBOs in descriptor set 2, mirroring Cartesian set 0.
     * Binding 0: r[N]          (grade 0, cylindrical radius)
     * Binding 1: delta_r[N]    (grade 1, radial offset from shell)
     * Binding 2: delta_y[N]    (grade 1, vertical offset)
     * Binding 3: vel_r[N]      (grade 1, radial velocity)
     * Binding 4: vel_y[N]      (grade 1, vertical velocity)
     * Binding 5: phi[N]        (grade 2, orbital azimuthal phase)
     * Binding 6: omega_orb[N]  (grade 2, orbital angular velocity)
     * Binding 7: theta[N]      (grade 2, Viviani phase)
     * Binding 8: omega_nat[N]  (grade 2, natural frequency)
     * Binding 9: L_tilt[N]     (grade 2, angular momentum tilt) */
    /* gradedEnabled removed in Phase 3.5 — graded is the only path */
    VkBuffer              graded_buffers[VK_GRADED_NUM_BINDINGS];
    VkDeviceMemory        graded_memory[VK_GRADED_NUM_BINDINGS];
    VkDescriptorSetLayout gradedSetLayout;
    VkPipelineLayout      cartToGradedPipelineLayout;
    VkPipeline            cartToGradedPipeline;
    VkPipelineLayout      gradedToCartPipelineLayout;
    VkPipeline            gradedToCartPipeline;
    VkDescriptorPool      gradedDescPool;
    VkDescriptorSet       gradedSet;

    /* Graded siphon pipeline (Phase 3.2) — reads set 0 (pump) + set 1 (density) + set 2 (graded) */
    VkPipelineLayout      siphonGradedPipelineLayout;
    VkPipeline            siphonGradedPipeline;
    VkDescriptorSetLayout siphonDensitySetLayout;  /* set 1: cyl density + pressure */
    VkDescriptorPool      siphonDensityDescPool;
    VkDescriptorSet       siphonDensitySet;

    /* Cylindrical density grid — replaces Cartesian grid for density feedback.
     * Aligned with disk geometry: Nr × Nphi × Ny. No Cartesian artifacts. */
    VkBuffer              cylDensityBuffer;         /* uint[CYL_CELLS] */
    VkDeviceMemory        cylDensityMemory;
    VkBuffer              cylPressureRBuffer;       /* float[CYL_CELLS] — d(rho)/dr */
    VkDeviceMemory        cylPressureRMemory;
    VkBuffer              cylPressurePhiBuffer;     /* float[CYL_CELLS] — (1/r)*d(rho)/dphi */
    VkDeviceMemory        cylPressurePhiMemory;
    VkBuffer              cylPressureYBuffer;       /* float[CYL_CELLS] — d(rho)/dy */
    VkDeviceMemory        cylPressureYMemory;
    VkDescriptorSetLayout cylScatterSetLayout;
    VkPipelineLayout      cylScatterPipelineLayout;
    VkPipeline            cylScatterPipeline;
    VkDescriptorPool      cylScatterDescPool;
    VkDescriptorSet       cylScatterSet;
    VkDescriptorSetLayout cylStencilSetLayout;
    VkPipelineLayout      cylStencilPipelineLayout;
    VkPipeline            cylStencilPipeline;
    VkDescriptorPool      cylStencilDescPool;
    VkDescriptorSet       cylStencilSet;

    /* Graded constraint pipeline (Phase 3.3) — set 0 (compat) + set 1 (pairs) + set 2 (graded) */
    VkPipelineLayout      constraintGradedPipelineLayout;
    VkPipeline            constraintGradedPipeline;

    /* Graded collision pipelines (Phase 3.4) — set 0 (compat) + set 1 (collision) + set 2 (graded) */
    VkPipelineLayout      collisionResolveGradedPipelineLayout;
    VkPipeline            collisionResolveGradedPipeline;
    VkPipelineLayout      collisionApplyGradedPipelineLayout;
    VkPipeline            collisionApplyGradedPipeline;

    /* Counting sort infrastructure (replaces shard-based atomic scatter) */
    VkBuffer              cellOffsetBuffer;         /* uint[GRID_CELLS] — prefix sum of cell counts */
    VkDeviceMemory        cellOffsetMemory;
    VkBuffer              scanBlockSumsBuffer;      /* uint[1024] — temp for hierarchical scan */
    VkDeviceMemory        scanBlockSumsMemory;
    VkBuffer              writeCounterBuffer;       /* uint[GRID_CELLS] — per-cell atomic for reorder */
    VkDeviceMemory        writeCounterMemory;
    VkDescriptorSetLayout scanSetLayout;            /* scan: data + block_sums */
    VkDescriptorSet       scanSet;
    VkPipelineLayout      scanPipelineLayout;
    VkPipeline            scanPipeline;
    VkDescriptorSetLayout histogramSetLayout;       /* histogram: particle_cell + cell_count */
    VkDescriptorSet       histogramSet;
    VkPipelineLayout      histogramPipelineLayout;
    VkPipeline            histogramPipeline;
    VkDescriptorSetLayout reorderSetLayout;         /* reorder: sort buffers (set 3) */
    VkDescriptorSet       reorderSet;
    VkPipelineLayout      reorderPipelineLayout;
    VkPipeline            reorderPipeline;
    VkDescriptorPool      countingSortDescPool;
    bool                  countingSortEnabled;

    /* Packed siphon (bandwidth optimization) — single AoS struct buffer
     * + fused Cartesian projection. Eliminates graded_to_cartesian dispatch. */
    VkBuffer              packedParticleBuffer;     /* Particle[N], 80 bytes each */
    VkDeviceMemory        packedParticleMemory;
    VkDescriptorSetLayout packedSiphonSetLayout;    /* set 0: packed + Cartesian out */
    VkDescriptorSet       packedSiphonSet;
    VkPipelineLayout      packedSiphonPipelineLayout;
    VkPipeline            packedSiphonPipeline;
    VkDescriptorPool      packedSiphonDescPool;
    bool                  packedSiphonEnabled;
};

/* Initialize compute pipeline + SSBOs, upload initial particle state.
 * scatterMode selects the scatter variant (baseline / uniform / squaragon). */
void initPhysicsCompute(PhysicsCompute& phys, VulkanContext& ctx,
                         const float* pos_x, const float* pos_y, const float* pos_z,
                         const float* vel_x, const float* vel_y, const float* vel_z,
                         const float* pump_scale, const float* pump_residual,
                         const float* pump_history, const int* pump_state,
                         const float* theta, const float* omega_nat,
                         const uint8_t* flags, const uint8_t* topo_state,
                         int N, ScatterMode scatterMode);

/* Record compute dispatch commands into command buffer */
void dispatchPhysicsCompute(PhysicsCompute& phys, VkCommandBuffer cmd,
                            int frame, float sim_time, float dt);

/* Initialize scatter compute pipeline + grid SSBOs (Pass 1 of streaming arch).
 * Loads the SPIRV corresponding to phys.scatterMode. */
void initScatterCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* Initialize scatter reduce compute pipeline (8 shards → canonical density).
 * Must be called after initScatterCompute (depends on the shards buffer). */
void initScatterReduceCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* Initialize stencil compute pipeline (Pass 2 — density → pressure gradient).
 * Must be called after initScatterReduceCompute (depends on gridDensity). */
void initStencilCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* Initialize gather-measure compute pipeline (Pass 3, measurement-only).
 * Must be called after initStencilCompute (depends on pressure buffers). */
void initGatherMeasureCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* Initialize the constraint-solver pipeline (Pass 4, PBD distance constraints).
 * Uploads the static constraint data (pair list, rest lengths, inverse masses)
 * to device-local SSBOs, builds the dual-set pipeline layout sharing set 0 with
 * siphon's particle layout, and stores bucket metadata needed by the dispatcher.
 * M = total constraint count (sum of bucket_counts[0..5]).
 * Must be called after initPhysicsCompute (depends on phys.descLayout). */
void initConstraintCompute(PhysicsCompute& phys, VulkanContext& ctx,
                           const uint32_t* pair_indices,      /* uvec2[M] flattened */
                           const float*    rest_lengths,      /* float[M] */
                           const float*    inv_masses,        /* float[N_rigid] */
                           const uint32_t* bucket_offsets,    /* uint[6] */
                           const uint32_t* bucket_counts,     /* uint[6] */
                           uint32_t        rigid_base,
                           uint32_t        rigid_count,
                           uint32_t        iterations);

/* Initialize the collision pipeline (Phase 2.2 — dynamic contact constraints).
 * C1 builds the descriptor set, allocates the rigid_body_id / vel_delta /
 * probe SSBOs, uploads rigid_body_id once, and creates the collision_apply
 * pipeline. The fused broadphase+resolve kernel is added in C2.
 *
 * rigid_body_ids must point to a host array of length phys.N: 0 for field
 * particles, 1 for cube 0, 2 for cube 1, etc. (Whatever scheme the rigid
 * scenario init function uses.)
 *
 * Must be called after initConstraintCompute (depends on phys.descLayout
 * for set 0). */
void initCollisionCompute(PhysicsCompute& phys, VulkanContext& ctx,
                          const uint32_t* rigid_body_ids,    /* uint32[N] */
                          uint32_t        rigid_base,
                          uint32_t        rigid_count);

/* Initialize grade-separated buffers + conversion pipelines (Phase 3.1).
 * Allocates 10 SSBOs in descriptor set 2, creates cartesian_to_graded
 * and graded_to_cartesian compute pipelines. Must be called after
 * initPhysicsCompute (depends on phys.descLayout for set 0). */
void initGradedCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* One-shot: convert Cartesian set 0 → graded set 2.
 * Run once after initGradedCompute to populate graded buffers. */
void dispatchCartesianToGraded(PhysicsCompute& phys, VkCommandBuffer cmd);

/* Per-frame: reconstruct Cartesian set 0 from graded set 2.
 * Run each frame before oracle readback / rendering. */
void dispatchGradedToCartesian(PhysicsCompute& phys, VkCommandBuffer cmd);

/* Initialize counting sort infrastructure (histogram + Blelloch scan + reorder).
 * Replaces the shard-based atomic scatter for the density grid.
 * Must be called after initGradedCompute (uses graded set 2). */
void initCountingSortCompute(PhysicsCompute& phys, VulkanContext& ctx);

/* Initialize density rendering pipeline (projection + tone-map) */
void initDensityRender(PhysicsCompute& phys, VulkanContext& ctx);

/* Record density rendering commands: clear → project → barrier → tone-map */
void recordDensityRender(PhysicsCompute& phys, VkCommandBuffer cmd,
                         VulkanContext& ctx, uint32_t imageIndex,
                         const float* viewProj);

/* Oracle subset size — fixed cap so staging buffer stays small (~3.2 MB) */
#define ORACLE_SUBSET_SIZE 100000

/* Read back particle subset to CPU for oracle validation.
 * Copies the first `count` entries of each of 8 SoA arrays (pos_xyz, vel_xyz,
 * theta, pump_scale) plus flags as uint32 from device-local SSBOs into the
 * host-visible staging buffer, then memcpy's into the output arrays.
 * `count` must be <= ORACLE_SUBSET_SIZE. */
void readbackForOracle(PhysicsCompute& phys, VulkanContext& ctx,
                       float* out_pos_x, float* out_pos_y, float* out_pos_z,
                       float* out_vel_x, float* out_vel_y, float* out_vel_z,
                       float* out_theta, float* out_pump_scale,
                       uint8_t* out_flags,
                       int count,
                       float* out_r = nullptr, float* out_vel_r = nullptr,
                       float* out_phi = nullptr, float* out_omega_orb = nullptr);

/* Read back GPU timestamps from the previous frame (non-blocking).
 * Breakdown:
 *   scatter_ms    = Pass 1 (scatter) + reduce
 *   stencil_ms    = Pass 2 (density → pressure gradient)
 *   gather_ms     = Pass 3 measurement-only gather
 *   constraint_ms = Pass 4 distance-constraint solver (0 if --rigid-body off)
 *   collision_ms  = Phase 2.2 collision pipeline (0 if collisionEnabled false)
 *   siphon_ms     = main physics kernel
 *   project_ms    = density projection to screen
 *   tonemap_ms    = fragment shader tone mapping
 * Returns true if results are valid, false if not yet available. */
bool readTimestamps(PhysicsCompute& phys, VkDevice device,
                    double* out_scatter_ms,
                    double* out_stencil_ms,
                    double* out_gather_ms,
                    double* out_constraint_ms,
                    double* out_collision_ms,
                    double* out_siphon_ms,
                    double* out_project_ms,
                    double* out_tonemap_ms);

/* Diagnostic: read back pump_state prefix (binding 6, int array).
 * Used to inspect how particles are distributed across the 8 pump
 * states. out_states must have at least `count` int slots. */
void readbackPumpStateSample(PhysicsCompute& phys, VulkanContext& ctx,
                             int* out_states, int count);

/* Upload data to a device-local SSBO via temporary staging buffer */
void uploadToSSBO(VulkanContext& ctx, VkBuffer dst, const void* src, size_t size);

/* Cleanup */
void cleanupPhysicsCompute(PhysicsCompute& phys, VkDevice device);

#endif /* V21_VK_COMPUTE_H */
