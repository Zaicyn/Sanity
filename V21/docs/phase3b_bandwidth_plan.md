# Phase 3B: Bandwidth Optimization Plan

## Context

Phase 3 shipped the grade-separated Clifford algebra state with fused
siphon + Cartesian projection. At 20M particles on RTX 2060:

  siphon = 10.69 ms (98% memory-bound, 2% compute)
  total  = 17.21 ms (58 FPS)
  theoretical minimum = 7.4 ms (2.48 GB / 336 GB/s)
  efficiency = ~69%

The remaining ~31% inefficiency comes from:
  - SSBO scatter access (not fully coalesced within warps)
  - Too many independent buffer streams (10 graded + 6 Cartesian + 7 pump = 23 total)
  - Writing fields that don't change (omega_nat, flags, in_active_region)
  - Register spill from fused kernel (increased live state)

Target: siphon ≤ 9.5 ms, total ≤ 16 ms (62+ FPS at 20M).

## What was tested and rejected (don't retry)

- **AoS struct packing**: 13.44 ms — breaks GPU warp coalescing. SoA is correct.
- **Counting sort scatter**: 33 ms — reorder bandwidth overwhelms atomic savings.
- **LUT trig**: V20 benchmark proved hardware sin/cos 5× faster than LUT on GPU.

## Task 1: Hot/Cold Buffer Split (est. 2–3 hours)

### Problem
The siphon reads 17 fields and writes 14 fields per particle. Several
fields are constant or rarely change:
  - omega_nat: set at init, never modified by siphon
  - in_active_region: set by tree architecture, not by siphon
  - flags: only modified on ejection (rare)
  - pump_work: accumulated but rarely read downstream

### Solution
Split graded set 2 into two descriptor sets:
  - **Set 2 (hot)**: r, delta_r, delta_y, vel_r, vel_y, phi, omega_orb, theta
    (8 bindings, read+write every frame)
  - **Set 4 (cold)**: omega_nat, L_tilt + pump fields + flags + in_active_region
    (read-only in siphon, written by separate low-frequency kernel)

The siphon reads set 2 (hot) + set 4 (cold, read-only). It writes ONLY
set 2 (hot) + the Cartesian projection (set 0 bindings 0-5).

### Expected savings
Current per-particle traffic:
  reads:  17 fields × 4B = 68B
  writes: 14 fields × 4B = 56B
  total: 124B

After hot/cold split:
  reads:  8 hot + 9 cold = 17 fields × 4B = 68B (unchanged — still need to read cold)
  writes: 8 hot + 6 Cartesian = 14 fields × 4B = 56B

Hmm — the read count doesn't change. But the WRITE count can be reduced:
  - Skip writing: omega_nat (never changes) = -4B
  - Skip writing: L_tilt (rarely changes) = -4B
  - Skip writing: pump_work (only write if pumping) = conditional -4B
  - Skip writing: pump_coherent (only write if pumping) = conditional -4B

**Net write savings: ~8–16B per particle (14–29% write reduction)**
At 20M particles: 160–320 MB less VRAM write traffic per frame.
At 336 GB/s: saves 0.5–1.0 ms.

### Implementation
1. Move omega_nat, L_tilt to a separate "cold" SSBO (or just stop writing them)
2. In siphon_graded.comp: remove the omega_nat write (it's never modified)
3. In siphon_graded.comp: conditionally write pump fields only when pump state changes
4. Regression test: oracle must still pass

### Files to modify
- kernels/siphon_graded.comp — remove cold field writes
- vk_compute.h — no structural changes needed (just stop writing)
- No new SSBOs needed — just stop writing existing ones

## Task 2: Field Compression (est. 2–3 hours)

### Problem
pump_state (8 values, needs 3 bits), pump_coherent (0-3, needs 2 bits),
flags (4 bits), in_active_region (1 bit) are each stored as separate
uint32 SSBOs. That's 16 bytes for 10 bits of information.

### Solution
Pack into a single uint32 bitfield (like the packed siphon meta field):
  bits 0-2:   pump_state (0-7)
  bits 3-4:   pump_coherent (0-3)
  bits 5-8:   flags (up to 16 flags)
  bit  9:     in_active_region
  bits 10-31: reserved

### Expected savings
Remove 3 SSBO bindings from set 0 (pump_state, pump_coherent, flags,
in_active_region merged into 1). Save 3 reads + 3 writes = 24B/particle.
At 20M: 480 MB less traffic. At 336 GB/s: saves ~1.4 ms.

### Implementation
1. Add a `packed_meta[N]` SSBO to set 0
2. Modify siphon_graded.comp to read/write packed_meta instead of 4 separate fields
3. Modify constraint/collision kernels similarly
4. Update initPhysicsCompute to pack initial values
5. Update readbackForOracle to unpack

### Risk
This changes the descriptor set layout for set 0, which affects ALL kernels
that declare the full 16-binding layout. Need to update every .comp file
that declares set 0 bindings 6, 8, 14, 15.

### Files to modify
- vk_compute.h — new packed_meta SSBO
- vk_compute.cpp — allocate, init, update descriptor writes
- kernels/siphon_graded.comp — pack/unpack meta
- kernels/constraint_solve_graded.comp — update set 0 declarations
- kernels/collision_*.comp — update set 0 declarations
- kernels/scatter.comp — update set 0 declarations
- blackhole_v21_visual.cpp — pack initial meta values

## Task 3: Register Pressure Audit (est. 1 hour)

### Problem
The fused siphon kernel has high register usage because it computes physics
+ Cartesian projection in one pass. High register usage reduces occupancy
(fewer warps per SM), which reduces the GPU's ability to hide memory latency.

### Diagnostic
Use `--nv-diag-all` or `nsight compute` to check:
  - Registers per thread
  - Achieved occupancy
  - Memory throughput utilization

If registers > 64 per thread, occupancy drops below 50% on Turing.

### Possible fixes
1. Move pump state machine to a separate kernel (it's cold-path logic)
2. Reduce live variable count by recomputing instead of storing
3. Use `__launch_bounds__` equivalent (GLSL doesn't have this directly,
   but reducing workgroup size from 256 to 128 can help if register-bound)

### Files to modify
- kernels/siphon_graded.comp — potential split or optimization

## Task 4: Verification

After all changes, run the standard regression suite:
1. Galaxy-only 10K: oracle pass, gather_ms flat
2. cube1000 11K: d(0,1) ≈ 0.500, oracle pass
3. cube2-collide 12K: d=0.500, d_centers=20.0, oracle pass
4. Galaxy 20M: siphon ≤ 9.5 ms, total ≤ 16 ms, oracle pass
5. Galaxy 50K frames: velocity sustained (energy balance)

## Reference: V8/V19 Shadow-Flyby Pattern

The V8 and V19 allocators use a "shadow + flyby detection" pattern for
write elision: compute an invariant (hash) of the current state, compare
against the stored shadow from the previous frame. If unchanged, skip
processing.

For the siphon, the simpler version is: just stop writing fields that
the kernel doesn't modify. omega_nat is never modified by siphon. flags
is only modified on ejection. These can be write-skipped trivially.

The shadow pattern could be useful for pump_history (exponential smoothing
converges → eventually stops changing), but the hash computation cost
may exceed the write savings at GPU scale.

## Execution Order

1. Task 1 (write elision) — lowest risk, immediate savings
2. Task 2 (field compression) — higher impact, more invasive
3. Task 3 (register audit) — diagnostic, informs whether Task 2 helps
4. Task 4 (verification) — after each task
