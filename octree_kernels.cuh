// octree_kernels.cuh
// ============================================================
// Octree kernel definitions extracted from blackhole_v20.cu
// Includes: Morton key assignment, tree build (analytic +
// stochastic), leaf extraction/hashing, velocity accumulation,
// frustum culling, render traversal, leaf phase evolution.
//
// Dependencies (must be included before this file):
//   disk.cuh            — GPUDisk struct
//   octree.cuh          — OctreeNode, REGIME_*, HYBRID_R
//   cuda_primitives.cuh — FrustumPlanes
//   vulkan/vk_types.h   — ParticleVertex (VULKAN_INTEROP only)
// ============================================================

#pragma once

// ============================================================================
// Octree Kernels - Morton key assignment and tree building
// ============================================================================

// Assign Morton keys to particles for spatial sorting
// Active inner particles (r < HYBRID_R) get real keys, outer particles get max key
__global__ void assignMortonKeys(
    GPUDisk* disk,
    uint64_t* morton_keys,
    uint32_t* xor_corners,
    uint32_t* particle_ids,
    int N,
    float boxSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = disk->pos_x[i];
    float py = disk->pos_y[i];
    float pz = disk->pos_z[i];
    float r  = sqrtf(px*px + py*py + pz*pz);

    // Outer/inactive particles get max key — sort to end
    if (!particle_active(disk, i) || r >= HYBRID_R) {
        morton_keys[i]  = 0xFFFFFFFFFFFFFFFFULL;
        xor_corners[i]  = 0xFFFFFFFF;
        particle_ids[i] = i;
        return;
    }

    // Map position to integer coordinates
    uint32_t ix = (uint32_t)fminf(fmaxf((px / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iy = (uint32_t)fminf(fmaxf((py / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iz = (uint32_t)fminf(fmaxf((pz / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));

    morton_keys[i]  = (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
    xor_corners[i]  = ix ^ iy ^ iz;
    particle_ids[i] = i;
}

// Build frozen analytic tree (levels 0-5) - run once at init
// Creates nodes for outer region (r > HYBRID_R) with field-derived energy
__global__ void buildAnalyticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    float boxSize,
    float pump_phase,
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all potential nodes at all analytic levels
    for (int level = 0; level <= max_level; level++) {
        int nodes_per_axis = 1 << level;
        int total = nodes_per_axis * nodes_per_axis * nodes_per_axis;
        if (idx >= total) continue;

        // Decode node coordinates from linear index
        int iz =  idx % nodes_per_axis;
        int iy = (idx / nodes_per_axis) % nodes_per_axis;
        int ix =  idx / (nodes_per_axis * nodes_per_axis);

        // Compute node center and half-size
        float half = boxSize / (2.0f * nodes_per_axis);
        float cx = (-boxSize * 0.5f) + (ix + 0.5f) * 2.0f * half;
        float cy = (-boxSize * 0.5f) + (iy + 0.5f) * 2.0f * half;
        float cz = (-boxSize * 0.5f) + (iz + 0.5f) * 2.0f * half;
        float r  = sqrtf(cx*cx + cy*cy + cz*cz);

        // Skip nodes entirely inside stochastic region (r + diagonal < HYBRID_R)
        if (r + half * 1.732f < HYBRID_R) continue;

        // Classify regime based on relationship to HYBRID_R boundary
        uint8_t regime;
        if (r - half * 1.732f > HYBRID_R)
            regime = REGIME_ANALYTIC;      // Entirely outside
        else
            regime = REGIME_BOUNDARY;       // Straddles boundary

        // Atomically allocate node slot
        uint32_t node_idx = atomicAdd(node_count, 1);
        if (node_idx >= OCTREE_MAX_NODES) return;

        // Populate node
        OctreeNode& node = nodes[node_idx];
        node.morton_key     = (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
        node.xor_corner     = ix ^ iy ^ iz;
        node.particle_start = 0;
        node.particle_count = 0;
        node.energy         = fieldEnergy(cx, cy, cz, pump_phase);
        node.center_x       = cx;
        node.center_y       = cy;
        node.center_z       = cz;
        node.half_size      = half;
        node.level          = (uint8_t)level;
        node.regime         = regime;
        node.children_mask  = 0;  // Will be filled in linking pass
        node.padding        = 0;
    }
}

// ============================================================================
// STOCHASTIC TREE BUILD — Levels 6-13 from Morton-sorted particle spans
// Rebuilt every 30 frames alongside Morton sort
// ============================================================================
__global__ void buildStochasticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    const uint64_t* morton_keys,  // Sorted
    uint32_t num_active,
    float boxSize,
    int start_level,              // 6
    int max_level                 // 13
) {
    // Each thread handles one sorted particle position
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_active) return;

    uint64_t my_key = morton_keys[i];

    // For each stochastic level, check if this is a boundary
    for (int level = start_level; level <= max_level; level++) {
        // Parent mask: keep bits for levels 0 to level-1
        uint64_t parent_mask = ~((1ULL << (3 * level)) - 1);
        uint64_t my_parent = my_key & parent_mask;

        // Am I the first particle in my level-L cell?
        bool is_boundary = (i == 0) ||
            ((morton_keys[i - 1] & parent_mask) != my_parent);

        if (!is_boundary) continue;

        // Find span end: where parent key changes
        uint32_t span_end = i + 1;
        while (span_end < num_active &&
               (morton_keys[span_end] & parent_mask) == my_parent) {
            span_end++;
        }

        // Allocate node
        uint32_t node_idx = atomicAdd(node_count, 1);
        if (node_idx >= OCTREE_MAX_NODES) return;

        // Decode ix, iy, iz from Morton key at this level
        // Morton key encodes x,y,z interleaved: ...x2y2z2 x1y1z1 x0y0z0
        // Extract level bits by masking and deinterleaving
        uint32_t ix = 0, iy = 0, iz = 0;
        for (int b = 0; b < level; b++) {
            int bit_pos = 3 * b;
            iz |= ((my_key >> bit_pos) & 1) << b;
            iy |= ((my_key >> (bit_pos + 1)) & 1) << b;
            ix |= ((my_key >> (bit_pos + 2)) & 1) << b;
        }

        int nodes_per_axis = 1 << level;
        float half = boxSize / (2.0f * nodes_per_axis);
        float cx = (-boxSize * 0.5f) + (ix + 0.5f) * 2.0f * half;
        float cy = (-boxSize * 0.5f) + (iy + 0.5f) * 2.0f * half;
        float cz = (-boxSize * 0.5f) + (iz + 0.5f) * 2.0f * half;

        OctreeNode& node = nodes[node_idx];
        node.morton_key     = my_parent;  // Parent-level key for this cell
        node.xor_corner     = ix ^ iy ^ iz;
        node.particle_start = i;
        node.particle_count = span_end - i;
        node.energy         = 0.0f;  // Stochastic: energy from particles, not field
        node.center_x       = cx;
        node.center_y       = cy;
        node.center_z       = cz;
        node.half_size      = half;
        node.level          = (uint8_t)level;
        node.regime         = REGIME_STOCHASTIC;
        node.children_mask  = 0;
        node.padding        = 0;
    }
}

// NOTE: octreeRenderTraversal V1 removed — use octreeRenderTraversalV3 instead

// ============================================================================
// EXTRACT LEAF NODE COUNTS — For prefix scan to eliminate atomic contention
// Writes particle_count and node index for each level-13 node into compact arrays
// ============================================================================
__global__ void extractLeafNodeCounts(
    uint32_t* leaf_counts,           // Output: particle counts for level-13 nodes
    uint32_t* leaf_node_indices,     // Output: original node indices for level-13 nodes
    uint32_t* leaf_node_count,       // Output: number of level-13 nodes found
    const OctreeNode* nodes,
    uint32_t total_nodes,
    uint32_t analytic_count,
    int target_level                 // 13
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x + analytic_count;
    if (node_idx >= total_nodes) return;

    const OctreeNode& node = nodes[node_idx];

    // Only count stochastic nodes at target level
    if (node.regime != REGIME_STOCHASTIC) return;
    if (node.level != target_level) return;

    // Allocate slot in leaf array
    uint32_t leaf_idx = atomicAdd(leaf_node_count, 1);
    leaf_counts[leaf_idx] = node.particle_count;
    leaf_node_indices[leaf_idx] = node_idx;
}

// ============================================================================
// BUILD LEAF HASH TABLE — O(1) neighbor lookup (replaces binary search)
// ============================================================================
// Inserts morton_key → leaf_idx mappings into a hash table with linear probing.
// Call after extractLeafNodeCounts populates leaf_node_indices.
// Hash table must be pre-cleared to 0xFF (UINT64_MAX = empty marker).

__global__ void buildLeafHashTable(
    uint64_t* hash_keys,             // Output: hash table keys
    uint32_t* hash_values,           // Output: hash table values (leaf indices)
    uint32_t hash_size,              // Hash table size (power of 2)
    uint32_t hash_mask,              // hash_size - 1 (for fast modulo)
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    uint32_t num_leaves
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    unsigned long long key = (unsigned long long)nodes[node_idx].morton_key;

    // Linear probing insert
    uint32_t slot = (uint32_t)(key & hash_mask);
    for (int probe = 0; probe < 64; probe++) {  // Max 64 probes
        unsigned long long old = atomicCAS((unsigned long long*)&hash_keys[slot],
                                            (unsigned long long)UINT64_MAX, key);
        if (old == (unsigned long long)UINT64_MAX || old == key) {
            // Successfully inserted or key already exists
            hash_values[slot] = leaf_idx;
            return;
        }
        // Collision - try next slot
        slot = (slot + 1) & hash_mask;
    }
    // Should never reach here with 50% load factor
}

// ============================================================================
// LEAF VELOCITY ACCUMULATION — Computes average velocity per leaf node
// ============================================================================
// For vorticity computation, we need velocity at each cell. This kernel
// accumulates particle velocities per leaf using atomics, then divides by count.

__global__ void accumulateLeafVelocities(
    float* leaf_vel_x,               // Output: sum of vx per leaf (then averaged)
    float* leaf_vel_y,
    float* leaf_vel_z,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint32_t* particle_ids,
    const GPUDisk* disk,
    uint32_t num_leaves
) {
    // Each block handles one leaf, threads handle particles within
    int leaf_idx = blockIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    if (node.particle_count == 0) return;

    // Shared memory for warp reduction
    __shared__ float s_vx[256];
    __shared__ float s_vy[256];
    __shared__ float s_vz[256];

    int tid = threadIdx.x;
    float vx_sum = 0.0f, vy_sum = 0.0f, vz_sum = 0.0f;

    // Each thread accumulates multiple particles if needed
    for (int p = tid; p < node.particle_count; p += blockDim.x) {
        uint32_t sorted_pos = node.particle_start + p;
        uint32_t orig_idx = particle_ids[sorted_pos];

        vx_sum += disk->vel_x[orig_idx];
        vy_sum += disk->vel_y[orig_idx];
        vz_sum += disk->vel_z[orig_idx];
    }

    s_vx[tid] = vx_sum;
    s_vy[tid] = vy_sum;
    s_vz[tid] = vz_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_vx[tid] += s_vx[tid + stride];
            s_vy[tid] += s_vy[tid + stride];
            s_vz[tid] += s_vz[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes average
    if (tid == 0) {
        float inv_count = 1.0f / (float)node.particle_count;
        leaf_vel_x[leaf_idx] = s_vx[0] * inv_count;
        leaf_vel_y[leaf_idx] = s_vy[0] * inv_count;
        leaf_vel_z[leaf_idx] = s_vz[0] * inv_count;
    }
}

// FrustumPlanes struct, extractFrustumPlanes, sphereInFrustum now in octree.cuh

// Cull leaf nodes against frustum, zero out particle_count for culled nodes
// Operates on the leaf_counts array (copied from extractLeafNodeCounts output)
__global__ void cullLeafNodesFrustum(
    uint32_t* leaf_counts,              // In/out: particle counts (zeroed if culled)
    const uint32_t* leaf_node_indices,  // Node indices for each leaf
    const OctreeNode* nodes,
    uint32_t num_leaves,
    FrustumPlanes frustum
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    // Bounding sphere: center at node center, radius = half_size * sqrt(3)
    float radius = node.half_size * 1.732051f;

    if (!sphereInFrustum(node.center_x, node.center_y, node.center_z, radius, frustum)) {
        leaf_counts[leaf_idx] = 0;  // Cull this node
    }
    // Note: if visible, leaf_counts already has correct particle_count from extraction
}

#ifdef VULKAN_INTEROP
// ============================================================================
// OCTREE RENDER TRAVERSAL V3 — Warp-cooperative with frustum culling support
// Binary search finds leaf for each output position, handles culled leaves.
// ============================================================================
__global__ void octreeRenderTraversalV3(
    ParticleVertex* compactedOutput,
    const uint32_t* leaf_offsets,        // Exclusive scan of culled counts
    const uint32_t* leaf_node_indices,   // Node index for each leaf
    const OctreeNode* nodes,
    const uint32_t* particle_ids,        // Morton-sorted particle indices
    const GPUDisk* disk,
    uint32_t num_leaves,
    uint32_t total_particles             // Sum of visible particles after culling
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_particles) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp_mask = __activemask();

    // Warp-cooperative: lane 0 searches, broadcasts result
    int warp_base_out = out_idx - lane;
    int leaf_idx;

    if (lane == 0) {
        // Binary search: find leaf where offset <= warp_base_out < next_offset
        int lo = 0, hi = num_leaves;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (leaf_offsets[mid] <= warp_base_out) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        leaf_idx = lo - 1;
    }
    leaf_idx = __shfl_sync(warp_mask, leaf_idx, 0);

    // Linear search forward from warp's base leaf
    while (leaf_idx + 1 < (int)num_leaves && leaf_offsets[leaf_idx + 1] <= out_idx) {
        leaf_idx++;
    }

    // Get node and compute particle offset within node
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];
    uint32_t p = out_idx - leaf_offsets[leaf_idx];

    // Lookup original particle index
    uint32_t sorted_pos = node.particle_start + p;
    uint32_t orig_idx = particle_ids[sorted_pos];

    // Read and write particle data
    float px = disk->pos_x[orig_idx];
    float pz = disk->pos_z[orig_idx];
    compactedOutput[out_idx].position[0] = px;
    compactedOutput[out_idx].position[1] = disk->pos_y[orig_idx];
    compactedOutput[out_idx].position[2] = pz;
    compactedOutput[out_idx].pump_scale = disk->pump_scale[orig_idx];
    compactedOutput[out_idx].pump_residual = disk->pump_residual[orig_idx];
    // Compute temp on-demand (saves 4 bytes/particle storage)
    compactedOutput[out_idx].temp = compute_temp(compute_disk_r(px, pz));
    float vx = disk->vel_x[orig_idx];
    float vy = disk->vel_y[orig_idx];
    float vz = disk->vel_z[orig_idx];
    compactedOutput[out_idx].velocity[0] = vx;
    compactedOutput[out_idx].velocity[1] = vy;
    compactedOutput[out_idx].velocity[2] = vz;
    compactedOutput[out_idx].elongation = 1.0f + sqrtf(vx*vx + vy*vy + vz*vz) * 0.01f;
}
#endif  // VULKAN_INTEROP (octree render traversal kernels)

// ============================================================================
// OCTREE PHYSICS KERNEL — XOR neighbor lookup for local stress gradients
// ============================================================================
// For each leaf node, computes stress gradient from 6 face-adjacent neighbors.
// Stress gradient drives particle interactions (pressure, viscosity, etc.)
//
// This kernel operates on nodes, not particles. Each node accumulates:
//   - Neighbor density differences → pressure gradient
//   - Neighbor energy differences → heat flow
//   - Neighbor velocity differences → shear stress
//
// The gradients are stored back in the node for the particle physics kernel
// to use when updating individual particle velocities.

// Compute density gradient for a single leaf node
// Returns gradient vector via output parameters
// Uses O(1) hash lookup instead of O(log N) binary search
__device__ void computeLeafGradient(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx,
    float* grad_x, float* grad_y, float* grad_z
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    *grad_x = 0.0f;
    *grad_y = 0.0f;
    *grad_z = 0.0f;

    if (node.regime != REGIME_STOCHASTIC || node.particle_count == 0) return;

    // Get 6 face-adjacent neighbor keys
    uint64_t neighbor_keys[6];
    getNeighborKeys(node.morton_key, node.level, neighbor_keys);

    float our_density = (float)node.particle_count;
    int neighbor_count = 0;

    // Accumulate gradient: +X, -X, +Y, -Y, +Z, -Z
    float grad[3] = {0.0f, 0.0f, 0.0f};

    for (int n = 0; n < 6; n++) {
        if (neighbor_keys[n] == UINT64_MAX) continue;  // Boundary

        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf_idx = findLeafByHash(
            hash_keys, hash_values, hash_mask, neighbor_keys[n]
        );

        if (neighbor_leaf_idx == UINT32_MAX) continue;  // Empty cell

        uint32_t neighbor_node_idx = leaf_node_indices[neighbor_leaf_idx];
        float their_density = (float)nodes[neighbor_node_idx].particle_count;

        neighbor_count++;

        // n=0: +X, n=1: -X, n=2: +Y, n=3: -Y, n=4: +Z, n=5: -Z
        int axis = n / 2;
        int sign = (n % 2 == 0) ? 1 : -1;
        grad[axis] += sign * (their_density - our_density);
    }

    // Normalize by cell size
    if (neighbor_count > 0) {
        float cell_size = node.half_size * 2.0f;
        float inv_2h = 1.0f / (2.0f * cell_size);
        *grad_x = grad[0] * inv_2h;
        *grad_y = grad[1] * inv_2h;
        *grad_z = grad[2] * inv_2h;
    }
}

// ============================================================================
// VORTICITY COMPUTATION — Curl of velocity field: ω = ∇ × v
// ============================================================================
// Vorticity measures local rotation in the velocity field.
// ω = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
// This enables spiral arm formation, circulation, and turbulence.

__device__ void computeLeafVorticity(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const float* leaf_vel_x,
    const float* leaf_vel_y,
    const float* leaf_vel_z,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx,
    float* omega_x, float* omega_y, float* omega_z
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    *omega_x = 0.0f;
    *omega_y = 0.0f;
    *omega_z = 0.0f;

    if (node.regime != REGIME_STOCHASTIC || node.particle_count == 0) return;

    // Get 6 face-adjacent neighbor keys
    uint64_t neighbor_keys[6];
    getNeighborKeys(node.morton_key, node.level, neighbor_keys);

    // Velocity at this cell
    float vx0 = leaf_vel_x[leaf_idx];
    float vy0 = leaf_vel_y[leaf_idx];
    float vz0 = leaf_vel_z[leaf_idx];

    // Neighbor velocities: [+X, -X, +Y, -Y, +Z, -Z]
    float vx[6], vy[6], vz[6];
    bool have[6] = {false, false, false, false, false, false};

    for (int n = 0; n < 6; n++) {
        if (neighbor_keys[n] == UINT64_MAX) continue;  // Boundary

        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf_idx = findLeafByHash(
            hash_keys, hash_values, hash_mask, neighbor_keys[n]
        );

        if (neighbor_leaf_idx == UINT32_MAX) continue;  // Empty cell

        vx[n] = leaf_vel_x[neighbor_leaf_idx];
        vy[n] = leaf_vel_y[neighbor_leaf_idx];
        vz[n] = leaf_vel_z[neighbor_leaf_idx];
        have[n] = true;
    }

    float cell_size = node.half_size * 2.0f;
    float inv_2h = 1.0f / (2.0f * cell_size);

    // Compute partial derivatives using central differences where possible
    // ∂v/∂x = (v[+X] - v[-X]) / (2h)
    // Only compute off-diagonal terms needed for curl (vorticity)
    float dvy_dx = 0.0f, dvz_dx = 0.0f;
    float dvx_dy = 0.0f, dvz_dy = 0.0f;
    float dvx_dz = 0.0f, dvy_dz = 0.0f;

    // X derivatives (neighbors 0=+X, 1=-X) - need dvy_dx, dvz_dx for curl
    if (have[0] && have[1]) {
        dvy_dx = (vy[0] - vy[1]) * inv_2h;
        dvz_dx = (vz[0] - vz[1]) * inv_2h;
    } else if (have[0]) {
        dvy_dx = (vy[0] - vy0) / cell_size;
        dvz_dx = (vz[0] - vz0) / cell_size;
    } else if (have[1]) {
        dvy_dx = (vy0 - vy[1]) / cell_size;
        dvz_dx = (vz0 - vz[1]) / cell_size;
    }

    // Y derivatives (neighbors 2=+Y, 3=-Y) - need dvx_dy, dvz_dy for curl
    if (have[2] && have[3]) {
        dvx_dy = (vx[2] - vx[3]) * inv_2h;
        dvz_dy = (vz[2] - vz[3]) * inv_2h;
    } else if (have[2]) {
        dvx_dy = (vx[2] - vx0) / cell_size;
        dvz_dy = (vz[2] - vz0) / cell_size;
    } else if (have[3]) {
        dvx_dy = (vx0 - vx[3]) / cell_size;
        dvz_dy = (vz0 - vz[3]) / cell_size;
    }

    // Z derivatives (neighbors 4=+Z, 5=-Z) - need dvx_dz, dvy_dz for curl
    if (have[4] && have[5]) {
        dvx_dz = (vx[4] - vx[5]) * inv_2h;
        dvy_dz = (vy[4] - vy[5]) * inv_2h;
    } else if (have[4]) {
        dvx_dz = (vx[4] - vx0) / cell_size;
        dvy_dz = (vy[4] - vy0) / cell_size;
    } else if (have[5]) {
        dvx_dz = (vx0 - vx[5]) / cell_size;
        dvy_dz = (vy0 - vy[5]) / cell_size;
    }

    // Curl: ω = ∇ × v
    // ωx = ∂vz/∂y - ∂vy/∂z
    // ωy = ∂vx/∂z - ∂vz/∂x
    // ωz = ∂vy/∂x - ∂vx/∂y
    *omega_x = dvz_dy - dvy_dz;
    *omega_y = dvx_dz - dvz_dx;
    *omega_z = dvy_dx - dvx_dy;
}

// Compute phase coherence with neighbors (for pressure modulation)
// Returns average cos(Δphase): 1 = in sync, 0 = random, -1 = anti-phase
__device__ float computePhaseCoherence(
    const float* leaf_phase,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx
) {
    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    float my_phase = leaf_phase[leaf_idx];
    uint64_t my_key = node.morton_key;
    int level = node.level;

    uint64_t neighbor_keys[6];
    getNeighborKeys(my_key, level, neighbor_keys);

    float coherence_sum = 0.0f;
    int neighbor_count = 0;

    for (int n = 0; n < 6; n++) {
        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf = findLeafByHash(hash_keys, hash_values,
                                                 hash_mask, neighbor_keys[n]);
        if (neighbor_leaf != UINT32_MAX) {
            float neighbor_phase = leaf_phase[neighbor_leaf];
            coherence_sum += cuda_lut_cos(neighbor_phase - my_phase);
            neighbor_count++;
        }
    }

    if (neighbor_count == 0) return 0.0f;
    return coherence_sum / (float)neighbor_count;
}


// ============================================================================
// S3 PHASE STATE — Temporal Coherence Layer
// ============================================================================
// Phase tracking enables:
//   - Resonance detection (nodes with matching phase → standing waves)
//   - Temporal coherence (neighboring phases couple and synchronize)
//   - Wave patterns (phase gradients create propagating structures)
//
// Local frequency ω is derived from density: high density → high frequency
// This creates natural chirp patterns as matter falls inward.

// Initialize phase from spatial position (creates coherent seed pattern)
__global__ void initializeLeafPhase(
    float* leaf_phase,
    float* leaf_frequency,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    uint32_t num_leaves,
    float base_frequency   // Base oscillation frequency
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    // Seed phase from position: creates radial wave fronts
    float cx = node.center_x;
    float cy = node.center_y;
    float cz = node.center_z;
    float r = sqrtf(cx*cx + cy*cy + cz*cz);

    // Phase = radial distance mod 2π (creates concentric rings)
    // Add angular component for spiral pattern
    float theta = cuda_fast_atan2(cz, cx);
    float initial_phase = fmodf(r * 0.1f + theta * 0.5f, 2.0f * M_PI);
    if (initial_phase < 0.0f) initial_phase += 2.0f * M_PI;

    leaf_phase[leaf_idx] = initial_phase;

    // Frequency from density: ω = ω_base * (1 + log(ρ))
    // Denser regions oscillate faster (gravitational time dilation analog)
    float density = (float)node.particle_count + 1.0f;
    float freq = base_frequency * (1.0f + 0.1f * logf(density));
    leaf_frequency[leaf_idx] = freq;
}

// Evolve phase with local frequency and neighbor coupling (Kuramoto model)
__global__ void evolveLeafPhase(
    float* leaf_phase,
    float* leaf_frequency,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves,
    float dt,
    float coupling_k       // Phase coupling strength
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    uint32_t node_idx = leaf_node_indices[leaf_idx];
    const OctreeNode& node = nodes[node_idx];

    float my_phase = leaf_phase[leaf_idx];
    float my_freq = leaf_frequency[leaf_idx];

    // === PHASE COUPLING (Kuramoto model) ===
    // dθ/dt = ω + (K/N) * Σ sin(θ_j - θ_i)
    // Neighbors with similar phase reinforce; different phases repel

    uint64_t my_key = node.morton_key;
    int level = node.level;

    // Get 6 face-adjacent neighbors
    uint64_t neighbor_keys[6];
    getNeighborKeys(my_key, level, neighbor_keys);

    float coupling_sum = 0.0f;
    int neighbor_count = 0;

    for (int n = 0; n < 6; n++) {
        // O(1) hash lookup instead of O(log N) binary search
        uint32_t neighbor_leaf = findLeafByHash(hash_keys, hash_values,
                                                 hash_mask, neighbor_keys[n]);
        if (neighbor_leaf != UINT32_MAX) {
            float neighbor_phase = leaf_phase[neighbor_leaf];
            // Kuramoto coupling: sin(θ_neighbor - θ_self)
            coupling_sum += cuda_lut_sin(neighbor_phase - my_phase);
            neighbor_count++;
        }
    }

    // Average coupling contribution
    float coupling_term = 0.0f;
    if (neighbor_count > 0) {
        coupling_term = coupling_k * coupling_sum / (float)neighbor_count;
    }

    // Update phase: θ += (ω + coupling) * dt
    float new_phase = my_phase + (my_freq + coupling_term) * dt;

    // Wrap to [0, 2π]
    new_phase = fmodf(new_phase, 2.0f * M_PI);
    if (new_phase < 0.0f) new_phase += 2.0f * M_PI;

    leaf_phase[leaf_idx] = new_phase;

    // Update frequency from current density (tracks changing conditions)
    float density = (float)node.particle_count + 1.0f;
    float base_freq = 0.1f;  // Base frequency
    leaf_frequency[leaf_idx] = base_freq * (1.0f + 0.1f * logf(density));
}

// Pre-compute coherence for all leaves (reduces per-particle neighbor lookups)
__global__ void computeLeafCoherence(
    float* leaf_coherence,
    const float* leaf_phase,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves
) {
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= num_leaves) return;

    // Just call the device function and cache the result
    float coherence = computePhaseCoherence(leaf_phase, nodes, leaf_node_indices,
                                             hash_keys, hash_values, hash_mask, leaf_idx);
    leaf_coherence[leaf_idx] = coherence;
}


// ============================================================================
// Octree Pressure + Vorticity Kernel (from blackhole_v20.cu)
// ============================================================================
__global__ void applyPressureVorticityKernel(
    GPUDisk* disk,
    const uint64_t* morton_keys,
    const uint32_t* particle_ids,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const float* leaf_vel_x,
    const float* leaf_vel_y,
    const float* leaf_vel_z,
    const float* leaf_phase,  // S3 phase for direct modulation (zero-cost)
    const uint64_t* hash_keys,    // Hash table for O(1) neighbor lookup
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t num_leaves,
    uint32_t num_active,
    const uint8_t* __restrict__ in_active_region,  // Step 3: skip passive particles
    float dt,
    float pressure_k,    // Pressure coefficient (~0.03)
    float vorticity_k    // Vorticity coefficient (~0.01)
) {
    int sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= num_active) return;

    uint32_t orig_idx = particle_ids[sorted_idx];
    if (!particle_active(disk, orig_idx)) return;
    // Step 3: passive particles get physics from advectPassiveParticles.
    if (in_active_region && !in_active_region[orig_idx]) return;

    uint64_t my_key = morton_keys[sorted_idx];

    // Find containing leaf node via O(1) hash lookup
    uint64_t level13_mask = ~((1ULL << 39) - 1);
    uint64_t parent_key = my_key & level13_mask;

    uint32_t leaf_idx = findLeafByHash(hash_keys, hash_values, hash_mask, parent_key);
    if (leaf_idx == UINT32_MAX) return;

    // Read current velocity
    float vx = disk->vel_x[orig_idx];
    float vy = disk->vel_y[orig_idx];
    float vz = disk->vel_z[orig_idx];

    // ========================================================================
    // 0. PHASE MODULATION: derive from phase directly (no neighbor lookup)
    // ========================================================================
    // Use sin(phase) for oscillation - creates standing wave patterns
    // Single memory read, already coalesced with leaf access
    // sin(θ) oscillates [-1, 1], so mod ranges [0.5, 1.5]
    float phase_mod = 1.0f;
    if (leaf_phase != nullptr) {
        float phase = leaf_phase[leaf_idx];
        phase_mod = 1.0f + 0.5f * cuda_lut_sin(phase);
    }

    // ========================================================================
    // 1. PRESSURE FORCE: F_p = -k_p ∇ρ × phase_mod
    // ========================================================================
    float grad_x, grad_y, grad_z;
    computeLeafGradient(nodes, leaf_node_indices, hash_keys, hash_values, hash_mask, leaf_idx,
                        &grad_x, &grad_y, &grad_z);

    // rsqrtf pattern: get both grad_mag and inv_grad_mag from single SFU call
    float grad_sq = grad_x*grad_x + grad_y*grad_y + grad_z*grad_z;

    if (grad_sq > 1e-6f) {
        float inv_grad_mag = rsqrtf(grad_sq);
        float grad_mag = grad_sq * inv_grad_mag;
        uint32_t node_idx = leaf_node_indices[leaf_idx];
        float local_density = (float)nodes[node_idx].particle_count + 1.0f;
        // Phase coherence modulates pressure coupling
        float force_scale = pressure_k * phase_mod * grad_mag / local_density;

        // Pressure pushes toward lower density
        vx += -grad_x * inv_grad_mag * force_scale * dt;
        vy += -grad_y * inv_grad_mag * force_scale * dt;
        vz += -grad_z * inv_grad_mag * force_scale * dt;
    }

    // ========================================================================
    // 2. VORTICITY FORCE: F_ω = k_ω (ω × v)
    // ========================================================================
    // Vorticity confinement: amplifies existing rotation
    // The cross product ω × v produces a force perpendicular to both,
    // which induces spiral motion and maintains angular momentum.
    float omega_x, omega_y, omega_z;
    computeLeafVorticity(nodes, leaf_node_indices, leaf_vel_x, leaf_vel_y, leaf_vel_z,
                         hash_keys, hash_values, hash_mask, leaf_idx,
                         &omega_x, &omega_y, &omega_z);

    // rsqrtf pattern: get both omega_mag and inv_omega from single SFU call
    float omega_sq = omega_x*omega_x + omega_y*omega_y + omega_z*omega_z;

    if (omega_sq > 1e-8f && vorticity_k > 0.0f) {
        float inv_omega = rsqrtf(omega_sq);
        float omega_mag = omega_sq * inv_omega;
        float ox = omega_x * inv_omega;
        float oy = omega_y * inv_omega;
        float oz = omega_z * inv_omega;

        // Cross product: ω × v
        // This produces a force perpendicular to both, inducing rotation
        float cross_x = oy * vz - oz * vy;
        float cross_y = oz * vx - ox * vz;
        float cross_z = ox * vy - oy * vx;

        // Scale by vorticity magnitude and coefficient
        float vort_force = vorticity_k * omega_mag;
        vx += cross_x * vort_force * dt;
        vy += cross_y * vort_force * dt;
        vz += cross_z * vort_force * dt;
    }

    // ========================================================================
    // 3. VELOCITY DAMPING: v *= (1 - c)
    // ========================================================================
    const float damping = 0.999f;
    vx *= damping;
    vy *= damping;
    vz *= damping;

    // Write back
    disk->vel_x[orig_idx] = vx;
    disk->vel_y[orig_idx] = vy;
    disk->vel_z[orig_idx] = vz;
}
