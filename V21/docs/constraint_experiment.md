# Constraint-Solver Experiment: Does GPU PBD Destroy Viviani Cache Coherence?

**Date:** 2026-04-11
**Hardware:** same GPU as the Viviani-sort retrospective baseline
**Scope:** one-shot experiment, no follow-up code expected from this document

## Question

Does adding a GPU distance-constraint solver over a 1000-particle lattice
embedded in the 20M-particle galaxy destroy the Viviani sort's gather-cache
coherence win (documented at `viviani_sort_retrospective.md`)?

## Experiment design

- **Lattice:** 10×10×10 cube (1000 particles) with 6-neighbor distance
  constraints (2700 total edges) dropped at r=50 in the disk plane, given
  the circular-orbit velocity a galaxy particle at the center would have
  so it feels the galactic environment rather than sitting inert.
- **Rest of the world:** 19,999,000 field particles running normal V21
  physics (scatter → reduce → stencil → gather_measure → siphon).
- **Solver:** GPU PBD distance constraints, 4 iterations per frame, no
  warm-starting, Baumgarte β=0.2. Constraints are pre-colored on the host
  into 6 vertex-disjoint buckets (axis × parity-of-starting-endpoint) so
  the kernel writes pos_x/y/z directly without atomics or contention.
  One dispatch per bucket, memory barrier between buckets → true
  sequential Gauss-Seidel on GPU.
- **Insertion point:** between gather_measure and siphon in the existing
  V21 dispatch pipeline. No change to siphon.comp or any other kernel.

## Three runs, ~5000 frames each at 20,000,000 particles

| Run | Config | Purpose |
|---|---|---|
| 1 | `--rigid-body off --init-sort viviani` | Control — reproduces retrospective baseline |
| 2 | `--rigid-body cube1000 --init-sort viviani` | Test — Viviani sort + constraint work |
| 3 | `--rigid-body cube1000 --init-sort none`    | Sanity — unsorted with constraints, should match unsorted retrospective baseline |

Steady-state means over 8 samples (excluding 2 warmup samples) per run:

| metric          | Run 1 (control) | Run 2 (test)   | Run 3 (sanity) |
|-----------------|----------------:|---------------:|---------------:|
| `scatter_ms`    |          1.979  |         2.003  |         1.736  |
| `stencil_ms`    |          0.012  |         0.012  |         0.012  |
| `gather_ms`     |      **1.391**  |     **1.395**  |     **2.930**  |
| `constraint_ms` |          0.001  |     **0.056**  |         0.056  |
| `siphon_ms`     |          8.085  |         8.065  |         8.066  |
| `project_ms`    |          1.683  |         1.686  |         1.762  |
| `tonemap_ms`    |          0.042  |         0.039  |         0.040  |
| `total_ms`      |     **13.192**  |    **13.258**  |    **14.601**  |

## Answer

**Run 2 `gather_ms` vs Run 1 `gather_ms`: +0.3% (+0.004 ms).**

Far inside the ±10% tolerance. The constraint solver does not disturb
the Viviani sort's cache coherence at this scale. Decision threshold
table from the plan:

| Δ gather_ms | Interpretation | Our result |
|---|---|---|
| ≤ ±10% | Door is open, unified substrate viable at small scale | **✓ +0.3%** |
| +10% to +30% | Marginal, needs investigation | — |
| +30% or more | Unified architecture wrong, pursue Bepu-alongside | — |

Run 3's `gather_ms = 2.930 ms` matches the retrospective's unsorted
baseline of 2.910 ms within 0.7%, confirming the measurement pipeline is
not confounded: the Viviani sort really is the thing keeping gather fast,
and the lattice/constraints aren't spuriously helping or hurting.

## Cost accounting

- **Solver cost**: `constraint_ms ≈ 0.056 ms/frame`. That's 24 dispatches
  (4 iters × 6 buckets × ~450 constraints each) running sub-millisecond.
  Essentially free relative to the 13 ms frame budget.
- **Total frame time delta** (Run 2 − Run 1): **+0.066 ms**, which equals
  `constraint_ms` within noise. **Zero cascading slowdown** — the rest of
  the pipeline pays nothing for the lattice's presence.

## Lattice stability (at 100K smoke test, not 20M)

At 100K particles (where lattice indices sit inside the 100K oracle
readback window), the lattice was verified to:

- **Orbit the black hole** as a rigid body: particle 0 moved from
  (47.8, -2.2, 0.1) at frame 500 to (31.1, -2.1, 37.3) at frame 9500,
  radius ≈ 48, fully consistent with a circular disk orbit at r=50.
- **Preserve all 2700 constraint distances** to within 0.04% of the
  0.5 rest length across 10000 frames under the full Viviani field
  (differential rotation trying to shear the cube). Max observed drift
  `d(0,1) = 0.5002` (X-neighbor), `d(0,10) = 0.4997` (Y-neighbor),
  `d(0,100) = 0.5000` (Z-neighbor). Far tighter than the ±10% bar
  the plan set.

## What this experiment does NOT answer

- Scaling to many rigid bodies (only 1 in this test). A real rigid-body
  engine needs thousands.
- Collision detection and response (none in MVP).
- Articulation (only distance constraints, no hinge/ball-socket).
- Determinism for networking (atomic ordering is OK here because the
  coloring is vertex-disjoint, but this is not a Gauss-Seidel proof of
  bit-identical reproducibility across runs).
- Friction, stacking, sleeping islands (all deferred).
- Constraint solver convergence quality (4 iters is not a fully
  converged solution; Baumgarte PBD is a latency test, not a physics
  validation).
- Anything at 100× or 1000× this many rigid-body particles.

These are all follow-up projects that depend on this MVP result being
encouraging. **The result is encouraging.** Phase 2 (collision detection,
multi-rigid-body scaling) is worth pursuing when the user wants to take
it there.

## Files touched (for future archaeology)

- `kernels/constraint_solve.comp` — new, 107 lines
- `vk_compute.h` — added `ConstraintPushConstants`, `PhysicsCompute`
  constraint fields, `initConstraintCompute` decl, `readTimestamps`
  signature with `out_constraint_ms`
- `vk_compute.cpp` — timestamp pool grew 7→8 slots, new
  `initConstraintCompute` (~200 lines), new constraint dispatch block
  in `dispatchPhysicsCompute` (6-bucket × 4-iter loop with barriers),
  `readTimestamps` now reports 7 values, cleanup destroys new resources
- `blackhole_v21_visual.cpp` — new `--rigid-body {off,cube1000}` flag,
  new `init_rigid_body_cube` helper with orbital velocity + 6-bucket
  coloring, `init_particles` takes an optional `n_generate` split,
  new `ConstraintLattice` struct, lattice-stability probe in the
  oracle-check block (no-ops at 20M because the lattice is outside
  the oracle window)
- `CMakeLists.txt` — added `constraint_solve.comp` to `KERNEL_SOURCES`

Original gameplan: `/home/zaiken/.claude/plans/wiggly-prancing-hamming.md`
