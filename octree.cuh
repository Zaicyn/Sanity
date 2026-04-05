// octree.cuh — Spatial Octree for Hybrid Simulation
// ==================================================
//
// Morton-encoded octree with XOR neighbor lookup:
//   - expandBits21/morton64: 63-bit Morton encoding
//   - mortonNeighbor: O(1) adjacent cell lookup via bit manipulation
//   - findLeafByHash: O(1) hash table lookup
//
// Tree structure:
//   - Levels 0-5: Frozen analytic nodes (r > HYBRID_R)
//   - Levels 6-13: Stochastic nodes rebuilt every 30 frames
//
// Render kernels:
//   - octreeRenderTraversalV2/V3: Fill Vulkan particle buffers
//   - cullLeafNodesFrustum: View frustum culling
//
// Physics kernels:
//   - computeLeafGradient/computeLeafVorticity: Field derivatives
//   - computePhaseCoherence: Kuramoto-style phase coupling
//
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Forward declarations
struct GPUDisk;
struct ParticleVertex;

// LUT forward declarations
__device__ float cuda_lut_sin(float x);
__device__ float cuda_lut_cos(float x);
__device__ float cuda_lut_cos3(float x);
__device__ float cuda_fast_atan2(float y, float x);

// ============================================================================
// Octree Node Structure
// ============================================================================

struct OctreeNode {
    uint64_t  morton_key;        // 8 bytes - sort key, memory layout
    uint32_t  xor_corner;        // 4 bytes - neighbor lookup (flip bit i → move axis i)
    uint32_t  particle_start;    // 4 bytes - index into sorted particle list
    uint32_t  particle_count;    // 4 bytes
    float     energy;            // 4 bytes - field energy at node center
    float     center_x;          // 4 bytes
    float     center_y;          // 4 bytes
    float     center_z;          // 4 bytes
    float     half_size;         // 4 bytes
    uint8_t   level;             // 1 byte
    uint8_t   regime;            // 1 byte - 0=ANALYTIC, 1=BOUNDARY, 2=STOCHASTIC
    uint16_t  children_mask;     // 2 bytes - which of 8 children exist
    uint32_t  padding;           // 4 bytes - alignment to 48 bytes total
};

#define REGIME_ANALYTIC    0
#define REGIME_BOUNDARY    1
#define REGIME_STOCHASTIC  2
#define OCTREE_MAX_DEPTH   13
#define OCTREE_MAX_NODES   (1 << 20)   // 1M nodes max
#define ANALYTIC_MAX_LEVEL 5

// Physical constants for field energy calculation
#define LAMBDA_OCTREE      24.0f       // 4 × ISCO shell spacing
#define ARM_AMP_OCTREE     0.3f        // m=3 arm modulation depth
#define N_EXCESS_OCTREE    0.12f       // n_avg - 1.0

// ============================================================================
// Morton Key / Octree Device Functions
// ============================================================================

// Spread 21-bit integer into 63 bits with 2 gaps between each
__device__ __forceinline__ uint64_t expandBits21(uint32_t v) {
    uint64_t x = v & 0x1fffff;  // 21 bits
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8)  & 0x100f00f00f00f00fULL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2)  & 0x1249249249249249ULL;
    return x;
}

// Interleave 3 coordinates into 63-bit Morton code
__device__ __forceinline__ uint64_t morton64(float px, float py, float pz, float boxSize) {
    // Map position to [0, 2^21) integer range
    uint32_t ix = (uint32_t)fminf(fmaxf((px / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iy = (uint32_t)fminf(fmaxf((py / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    uint32_t iz = (uint32_t)fminf(fmaxf((pz / boxSize + 0.5f) * (float)(1 << 21), 0.f), (float)((1<<21)-1));
    return (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
}

// Parity-based node identity for O(1) neighbor lookup
__device__ __forceinline__ uint32_t xorCorner(uint32_t ix, uint32_t iy, uint32_t iz) {
    return ix ^ iy ^ iz;
}

// Field energy for analytic nodes - deterministic from position and pump phase
__device__ __forceinline__ float fieldEnergy(float cx, float cy, float cz, float pump_phase) {
    float r = sqrtf(cx*cx + cy*cy + cz*cz);
    float shell = cuda_lut_sin(r / LAMBDA_OCTREE);
    float theta = cuda_fast_atan2(cz, cx);
    float arms  = 1.0f + ARM_AMP_OCTREE * cuda_lut_cos3(theta);
    float pump  = 0.5f + 0.5f * cuda_lut_sin(pump_phase);
    return shell * shell * arms * pump * N_EXCESS_OCTREE;
}

// ============================================================================
// XOR Neighbor Lookup — O(1) Morton key manipulation for adjacent cells
// ============================================================================
// Morton key bit layout: ...x2y2z2 x1y1z1 x0y0z0
// At level L, to move ±1 along axis:
//   X: flip bit at position 3*L + 2
//   Y: flip bit at position 3*L + 1
//   Z: flip bit at position 3*L + 0

// Compute neighbor Morton key by flipping axis bit at specified level
// axis: 0=Z, 1=Y, 2=X (matches Morton interleaving order)
// Returns neighbor key, or UINT64_MAX if at boundary
__device__ uint64_t mortonNeighbor(uint64_t key, int level, int axis, int dir) {
    // Bit position for this axis at this level
    int bit_pos = 3 * level + axis;

    // Check if we're at boundary (would wrap)
    // Extract the coordinate bits for this axis across all levels
    uint64_t coord_mask = 0;
    for (int l = 0; l <= level; l++) {
        coord_mask |= (1ULL << (3 * l + axis));
    }
    uint64_t coord_bits = key & coord_mask;

    // For +1 direction: if all coord bits set at this level, we'd overflow
    // For -1 direction: if all coord bits zero at this level, we'd underflow
    if (dir > 0) {
        // Check if we're at max coordinate (all bits set up to this level)
        uint64_t max_for_level = 0;
        for (int l = 0; l <= level; l++) {
            max_for_level |= (1ULL << (3 * l + axis));
        }
        if (coord_bits == max_for_level) {
            return UINT64_MAX;  // At boundary
        }
    } else {
        // Check if we're at min coordinate (all bits zero)
        if (coord_bits == 0) {
            return UINT64_MAX;  // At boundary
        }
    }

    // Simple case: just flip the bit (works for single-cell moves without carry)
    // For proper neighbor lookup with carry propagation:
    if (dir > 0) {
        // Add 1 to coordinate: propagate carry through interleaved bits
        uint64_t carry = 1ULL << bit_pos;
        while (carry && bit_pos < 63) {
            if (key & carry) {
                key ^= carry;  // Clear this bit
                bit_pos += 3;  // Move to next level for this axis
                carry = 1ULL << bit_pos;
            } else {
                key |= carry;  // Set this bit
                break;
            }
        }
    } else {
        // Subtract 1 from coordinate: borrow through interleaved bits
        uint64_t borrow = 1ULL << bit_pos;
        while (bit_pos >= 0) {
            if (key & borrow) {
                key ^= borrow;  // Clear this bit
                break;
            } else {
                key |= borrow;  // Set this bit (borrowing)
                bit_pos -= 3;   // Move to previous level for this axis
                if (bit_pos < axis) break;  // Underflow
                borrow = 1ULL << bit_pos;
            }
        }
    }

    return key;
}

// Get all 6 face-adjacent neighbor keys for a cell at given level
// neighbors[0..5] = +X, -X, +Y, -Y, +Z, -Z
// UINT64_MAX indicates boundary (no neighbor)
__device__ void getNeighborKeys(uint64_t key, int level, uint64_t* neighbors) {
    neighbors[0] = mortonNeighbor(key, level, 2, +1);  // +X
    neighbors[1] = mortonNeighbor(key, level, 2, -1);  // -X
    neighbors[2] = mortonNeighbor(key, level, 1, +1);  // +Y
    neighbors[3] = mortonNeighbor(key, level, 1, -1);  // -Y
    neighbors[4] = mortonNeighbor(key, level, 0, +1);  // +Z
    neighbors[5] = mortonNeighbor(key, level, 0, -1);  // -Z
}

// O(1) hash lookup for leaf node by Morton key
// Uses linear probing hash table built by buildLeafHashTable kernel
// Returns leaf array index, or UINT32_MAX if not found
__device__ uint32_t findLeafByHash(
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,              // hash_size - 1
    uint64_t target_key
) {
    uint32_t slot = (uint32_t)(target_key & hash_mask);
    for (int probe = 0; probe < 32; probe++) {  // Max 32 probes (typically 1-2)
        uint64_t key = hash_keys[slot];
        if (key == target_key) {
            return hash_values[slot];
        }
        if (key == UINT64_MAX) {
            return UINT32_MAX;  // Empty slot - key not in table
        }
        slot = (slot + 1) & hash_mask;
    }
    return UINT32_MAX;  // Not found after max probes
}

// Binary search for leaf node by Morton key in the leaf_node_indices array
// DEPRECATED: Use findLeafByHash for O(1) lookup instead of O(log N)
// Kept as fallback for debugging
__device__ uint32_t findLeafByMorton(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,  // Array of node indices for level-13 leaves
    uint32_t num_leaves,
    uint64_t target_key
) {
    // Binary search in leaf array (leaves are Morton-ordered from sorted particles)
    int lo = 0, hi = (int)num_leaves;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        uint32_t node_idx = leaf_node_indices[mid];
        uint64_t mid_key = nodes[node_idx].morton_key;

        if (mid_key < target_key) {
            lo = mid + 1;
        } else if (mid_key > target_key) {
            hi = mid;
        } else {
            return mid;  // Found - return leaf array index
        }
    }
    return UINT32_MAX;  // Not found
}

// ============================================================================
// Octree Build Kernels
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
);

// Build frozen analytic tree (levels 0-5) - run once at init
// Creates nodes for outer region (r > HYBRID_R) with field-derived energy
__global__ void buildAnalyticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    float boxSize,
    float pump_phase,
    int max_level
);

// Build stochastic tree (levels 6-13) from Morton-sorted particle spans
// Rebuilt every 30 frames alongside Morton sort
__global__ void buildStochasticTree(
    OctreeNode* nodes,
    uint32_t* node_count,
    const uint64_t* morton_keys,  // Sorted
    uint32_t num_active,
    float boxSize,
    int start_level,              // 6
    int max_level                 // 13
);

// Extract leaf node counts for prefix scan (eliminates atomic contention)
__global__ void extractLeafNodeCounts(
    uint32_t* leaf_counts,           // Output: particle counts for level-13 nodes
    uint32_t* leaf_node_indices,     // Output: original node indices for level-13 nodes
    uint32_t* leaf_node_count,       // Output: number of level-13 nodes found
    const OctreeNode* nodes,
    uint32_t total_nodes,
    uint32_t analytic_count,
    int target_level                 // 13
);

// Build O(1) hash table for leaf lookup (replaces binary search)
__global__ void buildLeafHashTable(
    uint64_t* hash_keys,             // Output: hash table keys
    uint32_t* hash_values,           // Output: hash table values (leaf indices)
    uint32_t hash_size,              // Hash table size (power of 2)
    uint32_t hash_mask,              // hash_size - 1 (for fast modulo)
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    uint32_t num_leaves
);

// Accumulate particle velocities per leaf for vorticity computation
__global__ void accumulateLeafVelocities(
    float* leaf_vel_x,               // Output: sum of vx per leaf (then averaged)
    float* leaf_vel_y,
    float* leaf_vel_z,
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint32_t* particle_ids,
    const GPUDisk* disk,
    uint32_t num_leaves
);

// ============================================================================
// Frustum Culling
// ============================================================================

// Frustum plane: ax + by + cz + d >= 0 means point is inside
struct FrustumPlanes {
    float planes[6][4];  // 6 planes, each with (a, b, c, d)
};

// Extract frustum planes from view-projection matrix (row-major)
// Uses Gribb-Hartmann method: planes from VP matrix rows
__host__ inline void extractFrustumPlanes(const float* vp, FrustumPlanes& frustum) {
    // Left:   row3 + row0
    frustum.planes[0][0] = vp[3]  + vp[0];
    frustum.planes[0][1] = vp[7]  + vp[4];
    frustum.planes[0][2] = vp[11] + vp[8];
    frustum.planes[0][3] = vp[15] + vp[12];

    // Right:  row3 - row0
    frustum.planes[1][0] = vp[3]  - vp[0];
    frustum.planes[1][1] = vp[7]  - vp[4];
    frustum.planes[1][2] = vp[11] - vp[8];
    frustum.planes[1][3] = vp[15] - vp[12];

    // Bottom: row3 + row1
    frustum.planes[2][0] = vp[3]  + vp[1];
    frustum.planes[2][1] = vp[7]  + vp[5];
    frustum.planes[2][2] = vp[11] + vp[9];
    frustum.planes[2][3] = vp[15] + vp[13];

    // Top:    row3 - row1
    frustum.planes[3][0] = vp[3]  - vp[1];
    frustum.planes[3][1] = vp[7]  - vp[5];
    frustum.planes[3][2] = vp[11] - vp[9];
    frustum.planes[3][3] = vp[15] - vp[13];

    // Near:   row3 + row2
    frustum.planes[4][0] = vp[3]  + vp[2];
    frustum.planes[4][1] = vp[7]  + vp[6];
    frustum.planes[4][2] = vp[11] + vp[10];
    frustum.planes[4][3] = vp[15] + vp[14];

    // Far:    row3 - row2
    frustum.planes[5][0] = vp[3]  - vp[2];
    frustum.planes[5][1] = vp[7]  - vp[6];
    frustum.planes[5][2] = vp[11] - vp[10];
    frustum.planes[5][3] = vp[15] - vp[14];

    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float len = sqrtf(frustum.planes[i][0] * frustum.planes[i][0] +
                          frustum.planes[i][1] * frustum.planes[i][1] +
                          frustum.planes[i][2] * frustum.planes[i][2]);
        if (len > 1e-6f) {
            frustum.planes[i][0] /= len;
            frustum.planes[i][1] /= len;
            frustum.planes[i][2] /= len;
            frustum.planes[i][3] /= len;
        }
    }
}

// Test sphere against frustum (returns true if visible)
__device__ inline bool sphereInFrustum(float cx, float cy, float cz, float radius,
                                        const FrustumPlanes& frustum) {
    for (int i = 0; i < 6; i++) {
        float dist = frustum.planes[i][0] * cx +
                     frustum.planes[i][1] * cy +
                     frustum.planes[i][2] * cz +
                     frustum.planes[i][3];
        if (dist < -radius) return false;  // Sphere completely outside this plane
    }
    return true;
}

// Cull leaf nodes against frustum, zero out particle_count for culled nodes
__global__ void cullLeafNodesFrustum(
    uint32_t* leaf_counts,              // In/out: particle counts (zeroed if culled)
    const uint32_t* leaf_node_indices,  // Node indices for each leaf
    const OctreeNode* nodes,
    uint32_t num_leaves,
    FrustumPlanes frustum
);

// ============================================================================
// Octree Render Traversal (Vulkan)
// ============================================================================

#ifdef VULKAN_INTEROP
// V2: Uses pre-scanned offsets, no atomics. Each block handles one leaf.
__global__ void octreeRenderTraversalV2(
    ParticleVertex* compactedOutput,
    const uint32_t* leaf_offsets,      // Pre-scanned output positions
    const uint32_t* leaf_node_indices, // Node indices for each leaf
    const OctreeNode* nodes,
    const uint32_t* particle_ids,
    const GPUDisk* disk,
    uint32_t num_leaves
);

// V3: Warp-cooperative with frustum culling support.
// Binary search finds leaf for each output position, handles culled leaves.
__global__ void octreeRenderTraversalV3(
    ParticleVertex* compactedOutput,
    const uint32_t* leaf_offsets,        // Exclusive scan of culled counts
    const uint32_t* leaf_node_indices,   // Node index for each leaf
    const OctreeNode* nodes,
    const uint32_t* particle_ids,        // Morton-sorted particle indices
    const GPUDisk* disk,
    uint32_t num_leaves,
    uint32_t total_particles             // Sum of visible particles after culling
);
#endif  // VULKAN_INTEROP

// ============================================================================
// Octree Physics Kernels — XOR neighbor lookup for local stress gradients
// ============================================================================

// Compute density gradient for a single leaf node
// Uses O(1) hash lookup instead of O(log N) binary search
__device__ void computeLeafGradient(
    const OctreeNode* nodes,
    const uint32_t* leaf_node_indices,
    const uint64_t* hash_keys,
    const uint32_t* hash_values,
    uint32_t hash_mask,
    uint32_t leaf_idx,
    float* grad_x, float* grad_y, float* grad_z
);

// Compute vorticity (curl of velocity field): ω = ∇ × v
// Vorticity measures local rotation, enables spiral arm formation
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
);

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
);

// End of octree.cuh
