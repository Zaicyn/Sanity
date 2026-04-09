# Resonance → Sanity: Allocation Pattern Learnings

## Context

Resonance (5,826 lines) implements a geometrically-grounded memory allocator with CPU (V9/aizawa) and GPU (V16/viviani) variants. Sanity currently uses direct cudaMalloc (177 calls) with no pooling or backend abstraction. This document captures what Resonance teaches us for when the CPU backend migration happens.

## What Resonance has that Sanity doesn't

### 1. Backend allocator interface
Resonance wraps allocation behind `v16_slab_init/destroy/reset` — host pointers to device memory with no hardcoded CUDA driver calls in the allocation hot path. Sanity's SimulationContext already has `void* buf_*` pointers (backend-agnostic by design). The missing piece is a function-pointer interface:

```c
struct SimulationBackend {
    void* (*alloc)(size_t size);
    void  (*free)(void* ptr);
    void  (*memset)(void* ptr, int val, size_t size);
    void  (*memcpy)(void* dst, const void* src, size_t size, int direction);
    void  (*sync)();
};
```

All 177 cudaMalloc calls in sim_init.cuh would go through `backend.alloc()`. CPU backend swaps to `malloc()`. Zero kernel changes needed.

### 2. Warp-level contention reduction (V16)
The V16 slab allocator uses exclusive-mode ranges per warp — non-overlapping memory regions that eliminate CAS contention. When ≥16 warps are active, contention drops 12-15%.

Sanity's scatter kernels (scatterParticlesToCells, scatterTopoToCells) have exactly this contention pattern: multiple particles in the same cell do atomicAdd to the same address. The V16 pattern could be adapted to scatter operations — partition cells across warps so no two warps write the same cell.

### 3. Two-tier allocation (large pools + small slab)
Resonance separates large allocations (direct pool) from small frequent ones (slab with 64/128/256B bins). Sanity allocates everything the same way — a 1.4GB GPUDisk struct and a 4-byte spawn counter both use cudaMalloc.

For CPU backend: large buffers should use mmap/VirtualAlloc (page-aligned), small buffers should use a slab or arena. The SimulationContext subsystem grouping (particles, diagnostics, grid, etc.) naturally maps to separate arenas.

### 4. Period-4 phase protection
Resonance uses quaternary encoding and 13-stride shadows to prevent pathological access patterns. The Viviani curve's period-4 structure (cos(θ)·cos(3θ) = ½[cos(2θ)+cos(4θ)]) is the same harmonic that drives the siphon pump. The allocation pattern and the physics share the same mathematical structure — not a coincidence.

## What Sanity has that Resonance doesn't

- Full physics simulation (Viviani field, Kuramoto, hopfion algebra)
- SimulationContext with backend-agnostic void* pointers (ready for abstraction)
- Dispatch layer (sim_dispatch.cuh) separating kernel launches from orchestration
- Analog nullable conventions (nullable.h)
- Photonic accumulators for coherent cluster compression
- Toomre Q instability-driven spawning

## When to integrate

**Trigger:** When the CPU backend work begins (replacing `<<<blocks, threads>>>` with OpenMP loops). At that point:

1. Create `sim_backend.h` with the allocator interface
2. Implement `sim_backend_cuda.cuh` (wraps cudaMalloc/Free/Memcpy)
3. Implement `sim_backend_cpu.h` (wraps malloc/free/memcpy)
4. Replace all cudaMalloc calls in sim_init.cuh with `backend.alloc()`
5. Consider V16 slab for small-buffer arena if profiling shows allocation overhead

## V21 Integration Status (completed April 2026)

All items extracted into `Sanity/V21/core/`:

| Item | V21 File | Status |
|------|----------|--------|
| Backend allocator interface | `v21_backend.h` | ✅ Done |
| CPU allocator (V9) | `v21_alloc_cpu.h` (410 lines) | ✅ Extracted + tested |
| GPU slab (V16) | `v21_alloc_gpu.h` (238 lines) | ✅ Extracted |
| Beta dual-strand | `v21_alloc_gpu.h` (extension) | ✅ Extracted |
| Warp contention reduction | `v21_alloc_gpu.h` (capped exclusive) | ✅ Extracted |
| Period-4 phase protection | `v21_types.h` + `v21_alloc_cpu.h` | ✅ Extracted |
| Rewire sim_init.cuh cudaMalloc | — | ⏳ Pending (mechanical) |

## Key files in Resonance (original reference)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `Resonance/aizawa.cuh` | 1,423 | CPU allocator: lock-free shadows, period-4 encoding, seam repair |
| `Resonance/viviani_v16_gpu.cuh` | 705 | GPU slab: warp contention reduction, half-step phase alternation |
| `Resonance/viviani_v16_beta.cuh` | 768 | Dual-strand (β-sheet): cross-link rungs, strand-independent cursors |
| `Resonance/aizawa_slab.cuh` | 475 | Slab utilities: pool init, superblock management |
