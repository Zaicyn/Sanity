# Phase 2.2 Handoff — Collision Detection

**Written:** 2026-04-11, at the end of the session that shipped Phase 2.1
through 2.3.1 Stage 6b (commits `eda621d` → `1521093`). This document is
the parking artifact for the *next* session, which will start with a
fresh context window and tackle collision detection.

**If you are that next-session Claude: read this file first.** Then read
the memory entry `project_constraint_solver_state.md` for the quick
architectural snapshot, then read the relevant `docs/constraint_experiment.md`
section for whichever Phase you care about. Do NOT read the full
`constraint_experiment.md` cold — it's 1300+ lines of accumulated narrative
and most of it isn't load-bearing for Phase 2.2.

---

## Where the rigid-body arc stands right now

Seven commits on master, all on top of the Viviani sort retrospective
(`9113944`):

```
1521093  Phase 2.3.1 Stage 6b — hinge beat-pattern sweep vs spin rate
e949414  Phase 2.3.1 Stage 6  — hinge DOF stress test (first spin data)
2cc6edd  Phase 2.3.1          — hinge MVP + noise floor methodology
1c76881  Phase 2.3            — ball-socket joint MVP
f855e9c  Phase 2.1            — N=1000 addendum (L2-exhaustion regime break)
56d2b23  Phase 2.1            — multi-body scaling sweep (N ∈ {1, 2, 10, 100})
eda621d  Pass 4               — constraint solver MVP (single cube, 2700 edges)
```

**Banked architectural claims, with evidence:**

1. **Cache-separability.** The constraint solver and the Viviani-sorted
   field particles operate in orthogonal memory regions. Validated across
   5 separate runs (N=1, 2, 10, 100, 1000 cubes) and 3 joint
   configurations (none, ball-socket, hinge). `gather_ms` is flat within
   ~3% run-to-run noise.
2. **Three-regime cost model** for static constraints:
   - N ≤ ~10:   launch-overhead-bound, ~0.055 ms flat
   - N ≈ 10–100: SM-saturated, L2-closed, ~1.12 µs/cube
   - N ≥ ~1000: L2-exhausted, memory-bandwidth-bound, ~2.64 µs/cube
3. **L2 boundary** is at ~3 MB pair list on RTX 2060 (3 MB L2). On larger
   L2 caches the regime-2 ceiling scales proportionally. Do NOT quote
   absolute cube counts as a general property.
4. **6-bucket vertex-disjoint coloring** (axis × parity-of-starting-
   endpoint) gives true GPU Gauss-Seidel without float atomics for any
   regular 3D lattice, and generalizes to 7 buckets with a joint bucket.
   Dispatch backend is fully parametric in bucket counts.
5. **Ball-socket and hinge joints work** as 1 and 2 cross-body distance
   constraints respectively. Joint stability holds under orbital shear;
   both edges track each other; internal cube fidelity preserved.
6. **The hinge DOF is real.** `--spin-rate 0.01` through `0.05` show cube 1
   actively rotating around the x-axis while cube 0 stays non-rotating,
   hinge absorbs the rotation without breaking.
7. **Viviani field does NOT damp external rotational input.** Rotation
   persists across all observed spin rates. Angular momentum is preserved
   on experimental timescales (up to ~22000 frames = 370 seconds sim time).
8. **Peak hinge stretch scales as ω^1.16** (Stage 6b, not ω² as naive
   centripetal would predict). The Baumgarte-PBD solver's per-iteration
   constraint-residual reduction gives a superlinear restoring response.
9. **Beat period is phase-locked to the orbit**, not to the spin rate.
   Observed ~5–7 orbital periods per beat cycle across a 10× spin range.
   This is a low-order commensurate resonance, not a driven oscillation.
10. **Run-to-run noise floor** on sub-ms GPU timings is ~3%. Single-run
    A/B comparisons below that bar need either multi-run averaging or
    in-process A/B scaffolding. This was a hard-earned lesson — the
    Phase 2.3 writeup initially attributed a noise effect to
    "co-location cache hotspot" before Phase 2.3.1 disconfirmed it.

## What GPT's framing gave us that matters for Phase 2.2

From the end-of-session critic review:

> **"You didn't just validate the architecture. You produced: a scaling
> law, a cache model, a dynamical law (ω^1.16), a resonance
> characterization, and a noise model (~3%). That's a complete
> experimental arc."**

The important part is the reframing: **we now have measured baselines for
how a physical system behaves under static constraints**, not just "it
runs." Phase 2.2 introduces dynamic topology, which is the first thing
that could break any of these baselines. So **Phase 2.2 success means
preserving the behaviors**, not just "collisions work."

Specifically, Phase 2.2 must preserve:
1. Cache-separability of the constraint solver from the Viviani field.
2. Bounded hinge/joint behavior under contact events.
3. The ω^1.16 response law (or a well-characterized deviation from it).
4. Angular momentum preservation.
5. Predictable cost scaling (maybe a *new* regime, but predictable).

And Phase 2.2 must **measure** (not just not-break):
1. What does the cost model look like for dynamic contact constraints?
2. What's the L2 behavior when the pair list changes every frame?
3. Does a collision event damp the hinge beat pattern, amplify it, or
   leave it alone?
4. Does dynamic bucket growth introduce any new coloring conflicts?

---

## Phase 2.2 scope — what the next session should build

### The one-question framing

Not "implement collision detection." The MVP question is sharper:

> **Can V21's constraint substrate host dynamic contact constraints
> without losing cache separability, solver stability, or the
> Hamiltonian-like phase-coherent behavior measured in Stage 6b?**

If yes, the unified substrate has a realistic path to shipping VR
physics. If no, we learn *where* the dynamic-topology failure mode lives
and get to choose between (a) isolating collisions in a separate solver
pass that doesn't touch the field particles, or (b) accepting the
performance/behavior cost and moving forward.

### The smallest meaningful test

**Two cubes on a collision course.** Same 10³ cube1000 lattices as Phase
2.1, placed at antipodal positions on the r=50 orbital circle but with
velocities that bring them toward each other. No joints. No hinges. Just
two rigid bodies orbiting, drifting, eventually touching.

- Cube 0 at (50, 0, 0), tangent velocity (0, 0, +v_orbit) *plus* a small
  radial velocity inward: add (−v_drift, 0, 0) where v_drift ~ 0.01.
- Cube 1 at (−50, 0, 0), symmetrically drifting inward: add (+v_drift, 0, 0).
- Over ~2500-5000 frames, the two cubes approach each other and collide
  at the origin region.

**What to measure:**
1. **Before contact:** `gather_ms`, `constraint_ms`, `total_ms` match
   Phase 2.1 N=2 baseline within noise. This is the "business as usual"
   segment.
2. **During contact:** how many contact constraints are generated per
   frame? What does the solver's dynamic bucket look like? Does the
   cost spike, stay flat, or rise smoothly?
3. **After contact:** do the cubes rebound cleanly, stick, interpenetrate,
   or oscillate? Does cube internal fidelity survive (d(0,1) etc. stays
   near 0.5)?
4. **Throughout:** does `gather_ms` drift during the contact event?
   This is the Q1 signal — dynamic topology trying to break cache
   separability.

### Design sketch (pre-digested; next session's plan will refine)

**Broadphase:** reuse the existing `particle_cell[]` SSBO from scatter.
Every particle already has its 64³ grid cell index computed each frame
by Pass 1 (scatter). For collision, the broadphase is essentially free:
for each lattice particle, scan its cell and 26 neighbors for other
lattice particles (from a DIFFERENT rigid body), and emit candidate
pairs. Crucial subtlety: lattice-lattice pairs within the same rigid
body don't need collision — they're already constrained by the lattice
edges. Broadphase must filter pairs by rigid-body ID.

- **New SSBO: `rigid_body_id[N]`**, uint32, one per particle. Zero for
  field particles, 1 for cube 0, 2 for cube 1, etc. Written once at init.
- **New kernel: `collision_broadphase.comp`**, dispatched per lattice
  particle, for each of the 27 cells (own + 26 neighbors), for each
  particle in that cell, if `rigid_body_id[other] != rigid_body_id[self]`
  and distance < 2·sphere_radius, emit to a contact pair buffer.
- **Contact pair buffer:** `uint2 contact_pairs[M_max]`, with `M_max`
  sized generously (~10000 pairs). Plus an atomic counter for how many
  pairs were emitted this frame.

**Narrow phase:** simplest possible. Each lattice particle is treated as
a sphere of radius `SPACING/2 = 0.25`. Contact distance = `2 × 0.25 = 0.5`.
If distance < 0.5, pair is a real contact. This collapses narrow phase
into the broadphase emission check — one pass, not two.

**Contact constraint generation:** each contact is a temporary distance
constraint with rest length = `2 × radius = 0.5` and *one-sided* (only
pushes apart when distance < rest length, doesn't pull together when
distance > rest length). This is a **unilateral constraint**, which is
new — all prior constraints were bilateral. The kernel change is trivial:
`if (C > 0) return;` after computing the constraint violation. If they're
already separated, do nothing.

**Dynamic bucket (bucket 7, or reuse bucket 6):** contact constraints
live in a new bucket that gets rebuilt every frame. Unlike buckets 0–5
(static lattice) and bucket 6 (static joints), bucket 7 has:
- Variable size each frame (could be 0 if no contacts, or thousands)
- Must be generated on-GPU (host can't see the contacts)
- Vertex-disjoint coloring is NOT automatic — multiple contacts can share
  a particle (e.g. corner of cube 0 touching face of cube 1 → 9 contacts
  all touching that one corner particle)

The vertex-disjoint problem is the main unknown. Three approaches:
- **(a) Use atomics for bucket 7 only.** Accept that contact resolution
  is Jacobi-style within this bucket, and serialize via barriers with
  the other buckets. Simpler but may converge poorly under pile-ups.
- **(b) GPU-side graph coloring.** Run a coloring pass after broadphase
  to partition contacts into sub-buckets that are vertex-disjoint. More
  complex, may not fit in regime 1 L2.
- **(c) Limit to 1 contact per particle via atomicMin/Max.** Each
  particle gets to "claim" at most one contact per frame using an atomic.
  Suboptimal physics but simple.

**Recommendation: start with (a).** It's the simplest, works for the
MVP, and characterizes the dynamic-topology cost baseline before we
optimize. If the MVP reveals bad convergence (cubes interpenetrating,
solver unstable), upgrade to (b).

**Dispatch ordering:**
```
... Pass 1 scatter → ... Pass 3 gather_measure →
Pass 4 constraint_solve:
  - static buckets 0..5 (lattice)
  - static bucket 6 (joints, may be empty)
  - Pass 4a: collision_broadphase → write to bucket 7 (dynamic)
  - Pass 4b: constraint_solve bucket 7 (contact resolution)
→ Pass 5 siphon
```

Between 4a and 4b, barrier. Between 4b and siphon, existing final
barrier.

### What already exists and can be reused

- **`particle_cell[N]`** from [vk_compute.cpp:422](../vk_compute.cpp#L422)
  — cell index per particle, updated every frame by scatter. Free broadphase.
- **`V21_GRID_DIM = 64`**, `V21_GRID_CELL_SIZE = 7.8125`, grid covers
  `[-250, 250]` in each axis. 10³ cube at SPACING=0.5 is 5 units wide,
  fits in ~1 cell or spans ~2 adjacent cells. A 27-neighbor search from
  any lattice particle's cell will cover the entire same-rigid-body's
  own cells plus any overlapping body.
- **`constraint_solve.comp`** kernel is already topology-agnostic. Adding
  a unilateral check (`if (C > 0) return;` for contact constraints) is
  either a new kernel variant or a push-constant flag.
- **`initConstraintCompute`** in [vk_compute.cpp:816](../vk_compute.cpp#L816)
  is fully parametric in bucket counts. Growing to 8 buckets is 15
  edit sites, same surgical pattern as Phase 2.3's 6→7 growth.
- **6-bucket coloring generalizes.** Adding a new dynamic bucket 7 at
  the end of the super-bucket list is architecturally free if we accept
  the atomics-in-bucket-7 compromise.

### What needs to be built

- **`collision_broadphase.comp`** — new kernel, ~80 lines. Per lattice
  particle, scan 27 cells, emit cross-body contact pairs to a dynamic
  buffer via atomic counter.
- **Contact pair SSBO + atomic counter.** Similar to the static
  `pairs[]` buffer but with variable length, reset each frame.
- **`rigid_body_id[N]`** SSBO. Init-time only.
- **Unilateral constraint mode** in `constraint_solve.comp`, gated by a
  new push-constant flag or a new kernel variant.
- **Dynamic bucket 7 dispatch.** Contact count is read back from the
  atomic counter and used to dispatch bucket 7 over `contact_count/64`
  workgroups. Needs indirect dispatch or a readback-then-dispatch with
  one frame of latency.
- **New CLI mode**: `--rigid-body cube2-collide` (antipodal with inward
  drift), or a `--collision on/off` orthogonal flag.
- **Measurement probe** extensions: contact count per frame, first-contact
  frame detection, per-frame `contact_ms` component of `constraint_ms`.

Total estimated new code: ~400 lines (one kernel, one init function,
host-side collision mode, probe extensions). Same order as the Phase 2.1
scaling refactor.

### What to measure, in priority order

1. **`gather_ms` through the contact event.** Does it drift when the
   dynamic bucket fills up? This is the cache-separability claim under
   its first real stress test.
2. **`constraint_ms` before, during, after contact.** How much does the
   broadphase add? How much does bucket 7 dispatch cost? Is there a
   nonlinearity as contact count grows?
3. **Contact count per frame** during the event. Expected shape: zero
   until approach, ramp up as cubes touch, peak at ~100-1000 contacts
   (cube 10×10 face has 100 particles, two faces → 200 candidates, 27
   cells means realistic overlap is ~50-500 contacts), ramp down as
   they rebound or stick.
4. **Cube internal fidelity during contact.** Do the cubes deform, or
   do they stay rigid? The contact constraints push particle pairs apart,
   which may fight against the lattice constraints that hold the cube
   together.
5. **Rebound or interpenetration?** Cube-cube collision should conserve
   momentum approximately; the two cubes should bounce off each other
   and continue their orbits with modified trajectories. If they
   interpenetrate or stick, the solver isn't keeping up.
6. **Post-contact gather_ms** back to baseline? If yes, the cache
   separability is robust. If not, contact events leave a cache shadow.

---

## Methodological guardrails (do not skip these)

These are hard-learned lessons from the current session. Ignoring them
will reproduce the Phase 2.3 noise artifact.

### Noise floor is ~3% on sub-ms GPU timings

Any A/B comparison below 3% on `gather_ms`, `constraint_ms`, or
`total_ms` is **not resolvable with a single run**. Either:
- Run both configurations 3+ times and compare means.
- Add an in-process A/B toggle that measures both paths in the same
  binary invocation.
- Accept that the effect is below the instrument noise floor.

Phase 2.3 spent 150 lines of doc explaining a "co-location cache
hotspot" that Phase 2.3.1 disconfirmed as noise. **Do not repeat this
mistake.** If the Phase 2.2 gather drift is small, run the comparison
twice before writing the interpretation.

### Each step is data; do not skip to the end

The Phase 2.1 → 2.3 → 2.3.1 → Stage 6 → Stage 6b arc worked because each
step produced trustworthy data that the next step built on. If Phase 2.2
tries to do "collision MVP + multi-body + friction + sleeping" in one
pass, the attribution of any failure will be ambiguous. Keep the MVP
scope tight: **two cubes, one collision event, no friction, no
persistent contact**.

If the two-cube MVP works, Phase 2.2.1 can add multi-cube collisions,
Phase 2.2.2 can add friction, Phase 2.2.3 can add persistent contact
manifolds, etc. Each step data.

### Predict first, reconcile honestly after

The Phase 2.1 and 2.3.1 writeups both included **explicit honest
predictions** before measurement, then **reconciled them against the
actual numbers** in the results section. This discipline is why the
session produced defensible physics. Keep doing it.

For Phase 2.2, useful predictions to write down before measuring:
- `gather_ms` during contact: expected within ±3% of baseline (same
  as all prior phases). If it drifts more, that's the breaking point.
- `constraint_ms` during contact: expected ~0.06 + (contact_count × 1.12
  µs), matching the Phase 2.1 regime 1 slope. If it's much higher, the
  dynamic-topology cost model is different from the static one.
- First contact frame: compute it from the approach geometry. Two cubes
  at ±50 drifting at ±0.01 toward origin → close the 100-unit gap at
  combined velocity 0.02 → contact at frame 5000. Verify the measurement
  matches the geometry.
- Post-contact rebound: if cubes are rigid and contact is elastic, they
  should rebound with reduced approach velocity. Note that V21's Viviani
  field preserves angular momentum but NOT energy — it's rotational
  transport, not inelastic damping — so the cubes should conserve
  linear momentum and the collision outcome depends on the solver's
  energy accounting.

### Commit discipline

The current session shipped 7 commits, each standalone, each with a
concrete result. The same pattern should apply to Phase 2.2:

- One commit for the collision infrastructure build-out (broadphase
  kernel, new SSBOs, dynamic bucket dispatch, CLI mode) + MVP smoke
  test passing at 100K.
- One commit for the 20M measurement run and Phase 2.2 writeup.
- If the MVP fails or reveals a regime change, one commit for the
  investigation and a plan for 2.2.1.

Do NOT commit work-in-progress or half-finished features. The tree
should be clean between atomic steps.

---

## Files you will read or modify

### Read-only references

- [Sanity/V21/docs/constraint_experiment.md](constraint_experiment.md)
  — the full experiment narrative, 1300+ lines. Read specific sections
  only. The Phase 2.1 section has the scaling model; Phase 2.3 has the
  ball-socket result; Phase 2.3.1 has the hinge and the methodology
  correction.
- `~/.claude/projects/-home-zaiken-sanity/memory/project_constraint_solver_state.md`
  — the updated memory file (should auto-load at session start).
- [Sanity/V21/kernels/constraint_solve.comp](../kernels/constraint_solve.comp)
  — the solver kernel, unchanged since Phase 2.1. Reference for how to
  add a unilateral constraint mode.
- [Sanity/V21/kernels/scatter.comp](../kernels/scatter.comp) — the
  scatter pass that writes `particle_cell[]`. Reference for how cells
  are computed and how the descriptor layout works.
- [Sanity/V21/vk_compute.cpp](../vk_compute.cpp) — the init and dispatch
  backend. `initScatterCompute`, `initConstraintCompute`, and
  `dispatchPhysicsCompute` are the templates for new passes.
- [Sanity/V21/vk_compute.h](../vk_compute.h) — grid constants,
  PhysicsCompute struct definition.

### Files likely to be modified

- `Sanity/V21/kernels/collision_broadphase.comp` — NEW kernel file
- `Sanity/V21/kernels/constraint_solve.comp` — extend with unilateral
  mode (push constant flag or new variant)
- `Sanity/V21/vk_compute.h` — new struct fields for collision pipeline,
  contact buffer, atomic counter
- `Sanity/V21/vk_compute.cpp` — new `initCollisionBroadphase` function,
  new dispatch in `dispatchPhysicsCompute`, bucket count 7→8 grow (or
  variable-size bucket 6)
- `Sanity/V21/blackhole_v21_visual.cpp` — new CLI mode for collision
  test, new `init_rigid_body_cube2_collide` function, probe extensions
  for contact count
- `Sanity/V21/CMakeLists.txt` — add new kernel to KERNEL_SOURCES
- `Sanity/V21/docs/constraint_experiment.md` — append Phase 2.2 section

---

## Open questions the next session needs to think about before coding

1. **Unilateral vs bilateral contact constraints.** Bilateral is easier
   (reuse existing kernel) but will pull cubes together when they
   separate. Unilateral is correct but requires a kernel branch. Which
   first? **Recommendation: unilateral from the start.** Bilateral
   contact is physically wrong and will produce confusing results.

2. **Contact resolution order vs solver iterations.** Should bucket 7
   (contacts) run *after* buckets 0-6 (lattice + joints) each iteration,
   or should contacts run *first* so lattice has a chance to propagate
   the contact response? Probably after, matching how most real-time
   PBD engines handle it, but worth thinking about.

3. **Atomics vs coloring for bucket 7.** The MVP recommendation is
   atomics (simpler), but if the scenario produces lots of contacts on
   shared particles (corner-face contact), atomics will converge badly.
   Decide based on the expected contact topology of the two-cube MVP.

4. **Per-particle radius.** Is 0.25 (= SPACING/2) the right collision
   radius? Too small and there's no contact (lattice particles slide
   past each other). Too large and the solver gets spurious contacts
   from self-cube particles in adjacent cells. Think about what "contact"
   actually means for a point-particle lattice representation of a rigid
   body.

5. **Timestep / contact velocity.** If the closing velocity is 0.02 and
   the particle radius is 0.25, the "contact window" is 25 frames wide.
   If it's 0.2, it's 2.5 frames — probably too fast for 4-iter PBD to
   resolve cleanly. Tune v_drift so the contact duration is ~50-100
   frames, enough for the solver to respond.

6. **What does a "successful" MVP look like physically?** Two cubes
   approach, touch, rebound, continue orbiting with modified
   trajectories. If they stick, interpenetrate, or shatter, the MVP
   failed at the physics level (even if the timing metrics are fine).
   Define success criteria *before* running.

---

## Explicit non-goals for the Phase 2.2 MVP

- **No friction.** Deferred to Phase 2.2.1 (or later). Friction requires
  persistent contact state and tangential impulse accumulation, which
  is a whole other design question.
- **No sleeping / islands.** Every body runs the full solver every
  frame. Optimization for later.
- **No multi-body pile-ups.** Two cubes, one contact event, that's it.
  Pile-up physics requires warm-starting and is Phase 2.2.2.
- **No persistent contact manifolds.** Contacts are generated fresh each
  frame, no memory. Real engines use manifold caching for stability;
  V21 MVP doesn't.
- **No tangential friction cones, no Coulomb friction.** All contacts
  are frictionless spheres bouncing off each other.
- **No convex-hull narrow phase.** Every lattice particle is a sphere.
  No box-box SAT, no GJK. The lattice representation *is* the collision
  shape.
- **No rigid body collisions with the field particles.** Only lattice-
  to-lattice (same rigid_body_id ≠ 0). Field particles (rigid_body_id=0)
  pass through lattice particles without collision. This is a
  substrate-level design choice that matches how the MVP lattices
  already coexist with field particles without interaction.

---

## If Phase 2.2 MVP fails or reveals a regime change

Three possible failure modes, each with a different follow-up:

**1. Gather_ms drifts during contact.** The cache-separability claim
breaks under dynamic topology. Investigate whether it's the broadphase
pass reading `particle_cell[]` in a way that competes with gather, or
the contact writes creating L2 contention. Potential fixes: run
collision_broadphase in a separate SM set, use a smaller contact
buffer, or isolate the collision particles from the field's cache
entirely.

**2. Constraint_ms grows nonlinearly with contact count.** The
dynamic-topology cost model is different from static. Characterize the
new slope. If it's still sub-quadratic, Phase 2.2 is just "expensive but
works." If it's catastrophic, the atomics-in-bucket-7 choice was wrong
and we need GPU graph coloring.

**3. Cubes interpenetrate or shatter.** The PBD solver isn't keeping up
with the closing velocity. Either increase iterations, decrease
timestep, or add velocity-level contact resolution before position
correction. This is a physics-fidelity problem, not a performance one.

**4. Hinge beat pattern changes character under contact.** This is
actually the most interesting failure mode from a physics standpoint.
If the Stage 6b Hamiltonian-like behavior breaks under contact events,
we've found the boundary of the "weakly-coupled energy-preserving"
regime. Deserves its own writeup and probably its own follow-up
experiment.

---

## Related memory files to consult or update

- `feedback_compile_constant_first.md` — "bake new params as #define,
  add CLI flags only after empirical sweep data." Applies to
  `SPACING`, collision radius, v_drift, etc.
- `feedback_measure_drift_first.md` — "run a long static-sort test
  before building re-sort infrastructure." The analogous lesson for
  Phase 2.2 is: run the MVP before building persistent contact
  manifolds or friction.
- `feedback_spirv_not_sass.md` — "NVIDIA driver already fuses FMul+FAdd."
  Don't hand-optimize the new kernel's arithmetic; profile first.
- `project_symmetric_sort_stability.md` — "Viviani sort self-stabilizes
  over 87k+ frames." Phase 2.2 contact events must not perturb this;
  verify with a long run after the MVP passes.

---

## Final synthesis from the end-of-session GPT review

> **"You didn't just prove your system works — you measured how it
> behaves as a physical system, and that's what makes the next phase
> meaningful."**

Phase 2.2 is no longer "can we add collisions?" It's:

> **"Does dynamic contact generation preserve the measured dynamical
> behavior of the static constraint system, or does it introduce
> damping, phase destruction, or cache interference?"**

That's a sharper question and it's the right one to carry into the
next session. Good luck, future-you.
