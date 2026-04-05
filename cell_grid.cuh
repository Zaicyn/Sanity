// cell_grid.cuh — Cell Grid (DNA/RNA Streaming Architecture)
// ============================================================
//
// Fixed-topology cell grid for forward-only physics passes.
// DNA layer: static grid geometry, O(1) cell lookup via position hash
// RNA layer: streaming particle state with atomic accumulation
//
// Three-pass model (no sorting, no binary search, no rebuild):
//   Pass 1: Particles → Cells (scatter with atomicAdd)
//   Pass 2: Cells → Cells (fixed 6-neighbor stencil)
//   Pass 3: Cells → Particles (direct gather)
//
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Grid Constants
// ============================================================================

// Compile-time max (for array sizing in structs that need fixed size)
// Actual runtime size is g_grid_dim (set by initVRAMConfig)
#define GRID_DIM_MAX    128
#define GRID_CELLS_MAX  (GRID_DIM_MAX * GRID_DIM_MAX * GRID_DIM_MAX)  // 2,097,152 cells
#define GRID_HALF_SIZE  250.0f                            // Simulation half-extent (fixed)

// Maximum active cells (with margin for propagation expansion)
// Typical active: ~3400 cells, max with propagation: ~20k
#define MAX_ACTIVE_CELLS 65536

// Activity threshold for flags (decays each frame, reactivated by scatter)
#define FLAG_ACTIVE_THRESHOLD 16   // Below this, cell is considered inactive
#define FLAG_INITIAL_VALUE 255     // Full activity when scatter marks cell

// ============================================================================
// Hierarchical Tiled Flags (Methylation Pattern)
// ============================================================================
// Instead of scanning 2M cells, scan tiles first then cells within active tiles
// This reduces O(GridSize) to O(Tiles) + O(ActiveTiles × CellsPerTile)
#define TILE_DIM 8                              // 8×8×8 cells per tile
#define TILES_PER_DIM (GRID_DIM_MAX / TILE_DIM) // 16 tiles per dimension (max)
#define NUM_TILES (TILES_PER_DIM * TILES_PER_DIM * TILES_PER_DIM)  // 4096 tiles (max)
#define CELLS_PER_TILE (TILE_DIM * TILE_DIM * TILE_DIM)            // 512 cells

// ============================================================================
// Cell Grid State (SoA layout for coalesced access)
// ============================================================================

struct CellGrid {
    // Accumulated fields (Pass 1 writes via atomicAdd)
    float* density;       // [GRID_CELLS] particle count
    float* momentum_x;    // [GRID_CELLS] sum of mass * vx
    float* momentum_y;    // [GRID_CELLS] sum of mass * vy
    float* momentum_z;    // [GRID_CELLS] sum of mass * vz
    float* phase_sin;     // [GRID_CELLS] sum of sin(phase) for Kuramoto
    float* phase_cos;     // [GRID_CELLS] sum of cos(phase) for Kuramoto

    // Derived fields (Pass 2 computes from accumulated)
    float* pressure_x;    // [GRID_CELLS] -∂ρ/∂x pressure gradient
    float* pressure_y;    // [GRID_CELLS] -∂ρ/∂y
    float* pressure_z;    // [GRID_CELLS] -∂ρ/∂z
    float* vorticity_x;   // [GRID_CELLS] (∂vz/∂y - ∂vy/∂z)
    float* vorticity_y;   // [GRID_CELLS] (∂vx/∂z - ∂vz/∂x)
    float* vorticity_z;   // [GRID_CELLS] (∂vy/∂x - ∂vx/∂y)

    // Per-particle cell assignment (computed in Pass 1, read in Pass 3)
    uint32_t* particle_cell;  // [N] which cell contains each particle

    // === SPARSE RNA ARCHITECTURE (List-based) ===
    // Active cell double-buffer: only process cells with particles
    // This is the true biological RNA model — sparse transcription, not global sweeps
    uint32_t* active_cells_A;     // [MAX_ACTIVE_CELLS] active cell indices (buffer A)
    uint32_t* active_cells_B;     // [MAX_ACTIVE_CELLS] active cell indices (buffer B)
    uint32_t* active_count_A;     // [1] count for buffer A
    uint32_t* active_count_B;     // [1] count for buffer B
    uint32_t* prev_cell_id;       // [N] previous cell ID per particle (for delta scatter)

    // === SPARSE FLAGS ARCHITECTURE (Optimal) ===
    // Instead of maintaining deduplicated lists, use presence flags.
    // Duplicates auto-collapse: writing 1 to the same cell 1000x is still 1.
    // No sorting, no thrust::unique, pure O(1) per cell.
    uint8_t* active_flags;        // [GRID_CELLS] presence flag (0=inactive, >0=active)
    uint8_t* next_flags;          // [GRID_CELLS] next frame's flags (double buffer)
};

// ============================================================================
// Cell Grid Device Functions
// ============================================================================
// Note: Device functions that depend on d_grid_* constants are implemented
// in blackhole_v20.cu since CUDA __constant__ variables can't be externed.
//
// Functions provided in .cu file:
//   - cellToTile(cell) -> tile index
//   - tileToFirstCell(tile) -> first cell in tile
//   - cellIndexFromPos(px, py, pz) -> cell index
//   - cellCoords(cell, &cx, &cy, &cz) -> extract coordinates
//   - neighborCellIndex(cell, dx, dy, dz) -> neighbor cell

// End of cell_grid.cuh
