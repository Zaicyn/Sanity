# Active flag audit — Tree Architecture Step 2a

## Purpose

The upstream spec for Tree Architecture Step 2 assumed that the existing
`active` flag in the codebase had two *conflated* meanings — "is this
particle alive?" and "did this particle move cells this frame?" — and
needed to be *split* into `moved_cells_mask` and `in_active_region`.

**Code inspection shows the split already exists.** The two signals live
in completely separate storage, are read by disjoint sets of consumers,
and are never aliased in any call site. This audit enumerates every
reader and writer of each signal so that commit 2b can add a third
signal (`d_in_active_region[]`, the "inside an XOR corner" bit) as a
purely additive change. Nothing needs to be renamed. No call site
needs to combine the new bit with either of the existing signals in
Step 2.

The two existing signals are:

- **`PFLAG_ACTIVE`** — bit 0 of `disk->flags[i]`, set at
  `blackhole_v20.cu:4399` and `702`/`3698`, checked via
  `particle_active(disk, i)` ([disk.cuh:286](../disk.cuh#L286)). Persistent
  per-particle state meaning "this particle is alive / not deallocated."
- **`d_active_mask[]`** — a standalone `uint8_t*` buffer owned by
  `g_active_particles` ([blackhole_v20.cu:3609](../blackhole_v20.cu#L3609)),
  allocated at [4771](../blackhole_v20.cu#L4771), written by
  `computeParticleActivityMask` ([2353](../blackhole_v20.cu#L2353)), cleared
  each frame. Transient per-frame state meaning "this particle's cell
  changed since last frame (or velocity is above
  `VELOCITY_THRESHOLD = 10.0f`)."

The new signal added in Step 2b/2c/2d is:

- **`d_in_active_region[]`** — a new standalone `uint8_t*` buffer sized
  to `g_runtime_particle_cap`, initialized to all-0xFF in commit 2c. In
  Step 2 it is always all-1s (every particle is "in the all-encompassing
  bootstrap region"), so the passive kernel early-returns on every
  particle and does no writes. Step 3 makes this mask selective.

## Section 1 — Every consumer of `PFLAG_ACTIVE` / `particle_active()`

| file | line | consumer | read/write | current semantics | interaction with `in_active_region` |
| --- | --- | --- | --- | --- | --- |
| [disk.cuh:241](../disk.cuh#L241) | 241 | `PFLAG_ACTIVE` macro | def | bit 0 of `flags[]` = alive | NONE |
| [disk.cuh:286](../disk.cuh#L286) | 286 | `particle_active()` accessor | R helper | read bit 0 | NONE |
| [disk.cuh:294](../disk.cuh#L294) | 294 | `set_particle_active()` writer | W helper | set/clear bit 0 | NONE |
| [blackhole_v20.cu:577](../blackhole_v20.cu#L577) | 577 | `spawnParticlesKernel` host-file copy — entry guard | R | skip dead parents | NONE (Step 3 may want to also early-return on passive, but not Step 2) |
| [blackhole_v20.cu:702](../blackhole_v20.cu#L702) | 702 | spawn kernel (host copy) — child init | W | mark newborn alive | NONE |
| [blackhole_v20.cu:788](../blackhole_v20.cu#L788) | 788 | render instance fill — visibility | R | skip dead particles in render | NONE |
| [blackhole_v20.cu:842](../blackhole_v20.cu#L842) | 842 | render instance fill (secondary path) | R | skip dead particles in render | NONE |
| [blackhole_v20.cu:915](../blackhole_v20.cu#L915) | 915 | render kernel early return | R | skip dead | NONE |
| [blackhole_v20.cu:1062](../blackhole_v20.cu#L1062) | 1062 | render kernel early return | R | skip dead | NONE |
| [blackhole_v20.cu:1151](../blackhole_v20.cu#L1151) | 1151 | hybrid render path selection | R | skip dead OR route-by-radius | NONE |
| [blackhole_v20.cu:1826](../blackhole_v20.cu#L1826) | 1826 | octree kernel early return | R | skip dead | NONE |
| [blackhole_v20.cu:2283](../blackhole_v20.cu#L2283) | 2283 | Kuramoto global R reduction | R | include in R only if alive AND not ejected | NONE |
| [blackhole_v20.cu:2332](../blackhole_v20.cu#L2332) | 2332 | phase histogram kernel | R | skip dead or ejected | NONE in Step 2. (Step 3 may want to exclude passive from peak_frac — flagged below.) |
| [blackhole_v20.cu:2364](../blackhole_v20.cu#L2364) | 2364 | `computeParticleActivityMask` — gate writer | R | force `active_mask[i]=0` if dead | NONE (produces signal 2, doesn't consume signal 3) |
| [blackhole_v20.cu:2450](../blackhole_v20.cu#L2450) | 2450 | `scatterStaticParticles` — double gate | R | skip dead OR moved-this-frame | NONE (this row is the one place both signals 1 and 2 are read together, still independent of signal 3) |
| [blackhole_v20.cu:2661](../blackhole_v20.cu#L2661) | 2661 | `scatterWithTileFlags` — gate | R | skip dead | NONE |
| [blackhole_v20.cu:2875](../blackhole_v20.cu#L2875) | 2875 | `applyPressureVorticityKernel` (Morton-sorted) — gate | R | skip dead after Morton lookup | NONE |
| [blackhole_v20.cu:3252](../blackhole_v20.cu#L3252) | 3252 | `sampleReductionKernel` — stats gate | R | include in sample metrics only if alive | NONE |
| [blackhole_v20.cu:3601](../blackhole_v20.cu#L3601) | 3601 | `ActiveParticleState` comment block | doc | describes `d_particle_active` as uint8 mask | NONE (stale comment — the struct field is actually `d_active_mask`) |
| [blackhole_v20.cu:3698](../blackhole_v20.cu#L3698) | 3698 | init fallback — mark alive | W | set newborn alive | NONE |
| [blackhole_v20.cu:4328](../blackhole_v20.cu#L4328) | 4328 | host flags buffer declaration | doc comment | staging buffer for flags upload | NONE |
| [blackhole_v20.cu:4399](../blackhole_v20.cu#L4399) | 4399 | init — mark all particles alive | W | set PFLAG_ACTIVE for each particle at startup | NONE |
| [blackhole_v20.cu:4404](../blackhole_v20.cu#L4404) | 4404 | startup stats — count alive | R | count PFLAG_ACTIVE for init log | NONE |
| [physics.cu:105](../physics.cu#L105) | 105 | `siphonDiskKernel` entry guard | R | skip dead (Step 3 will add `\|\| in_active_region[i]` here) | NONE in Step 2 (Step 3 adds AND-with) |
| [physics.cu:378](../physics.cu#L378) | 378 | `spawnParticlesKernel` (actual def) entry guard | R | skip dead parents | NONE (Step 3 may want `&& !in_active_region` — R6 in plan) |
| [physics.cu:472](../physics.cu#L472) | 472 | spawn kernel — child init | W | mark newborn alive | NONE |

Total: 26 rows. Every row is `NONE` for Step 2.

## Section 2 — Every consumer of `d_active_mask[]` / `active_mask[]`

| file | line | consumer | read/write | current semantics | interaction with `in_active_region` |
| --- | --- | --- | --- | --- | --- |
| [blackhole_v20.cu:2357](../blackhole_v20.cu#L2357) | 2357 | `computeParticleActivityMask` signature | W (param) | writes `active_mask[i]` | NONE |
| [blackhole_v20.cu:2365](../blackhole_v20.cu#L2365) | 2365 | `computeParticleActivityMask` body — dead clear | W | `active_mask[i] = 0` for dead | NONE |
| [blackhole_v20.cu:2379](../blackhole_v20.cu#L2379) | 2379 | `computeParticleActivityMask` body — compute | W | `active_mask[i] = cell_changed \|\| moving` | NONE |
| [blackhole_v20.cu:2385](../blackhole_v20.cu#L2385) | 2385 | `compactActiveParticles` signature | R (param) | reads `active_mask[i]` to decide inclusion | NONE |
| [blackhole_v20.cu:2393](../blackhole_v20.cu#L2393) | 2393 | `compactActiveParticles` body | R | branch on `active_mask[i]` | NONE |
| [blackhole_v20.cu:2437](../blackhole_v20.cu#L2437) | 2437 | `scatterStaticParticles` signature | R (param) | reads mask (inverted usage) | NONE |
| [blackhole_v20.cu:2450](../blackhole_v20.cu#L2450) | 2450 | `scatterStaticParticles` body — inverted gate | R | skip if dead OR mask set | NONE (appears in Section 1 too because same line reads both signals) |
| [blackhole_v20.cu:3609](../blackhole_v20.cu#L3609) | 3609 | `ActiveParticleState::d_active_mask` field decl | decl | struct field | NONE |
| [blackhole_v20.cu:4771](../blackhole_v20.cu#L4771) | 4771 | allocation `cudaMalloc(&g_active_particles.d_active_mask, …)` | alloc | size `g_runtime_particle_cap * sizeof(uint8_t)` | NONE (sibling allocation point for `d_in_active_region`) |
| [blackhole_v20.cu:5245](../blackhole_v20.cu#L5245) | 5245 | main loop — active-compact path — call to `computeParticleActivityMask` | R/W | produce mask this frame | NONE |
| [blackhole_v20.cu:5252](../blackhole_v20.cu#L5252) | 5252 | main loop — active-compact path — call to `compactActiveParticles` | R | consume mask for compaction | NONE |
| [blackhole_v20.cu:5322](../blackhole_v20.cu#L5322) | 5322 | main loop — bake path — call to `computeParticleActivityMask` | R/W | produce mask for bake | NONE |
| [blackhole_v20.cu:5336](../blackhole_v20.cu#L5336) | 5336 | main loop — bake path — call to `scatterStaticParticles` | R | scatter inverted mask | NONE |
| [blackhole_v20.cu:5349](../blackhole_v20.cu#L5349) | 5349 | main loop — bake path — call to `compactActiveParticles` | R | compact for bake count | NONE |

Total: 14 rows. Every row is `NONE` for Step 2.

## Section 3 — Forward-declared Step 2 additions for `d_in_active_region[]`

| file | line | consumer | role | planned changes |
| --- | --- | --- | --- | --- |
| `blackhole_v20.cu` | ~170 (commit 2c) | `#ifndef ENABLE_PASSIVE_ADVECTION` guard macro | def | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~63 (commit 2b) | `#include "passive_advection.cuh"` | include | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~63 (commit 2d) | `#include "active_region.cuh"` | include | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~4667 (commit 2c) | `uint8_t* d_in_active_region = nullptr;` declaration | decl | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~4707 (commit 2c) | `cudaMalloc(&d_in_active_region, …)` + `cudaMemset(…, 0xFF, …)` | alloc | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~4707 (commit 2d) | `ActiveRegion* d_active_regions` allocation + bootstrap seed | alloc | PRODUCER OF NEW BUFFER |
| `blackhole_v20.cu` | ~4896 (commit 2c) | guarded `advectPassiveParticles<<<…>>>` launch | launch | PRODUCER OF NEW BUFFER (passive kernel writes it? no — reads it) |
| `blackhole_v20.cu` | ~4896 (commit 2d) | guarded `computeInActiveRegionMask<<<…>>>` launch | launch | PRODUCER OF NEW BUFFER (writes `d_in_active_region[]`) |
| `passive_advection.cuh` (new, commit 2b) | — | `advectPassiveParticles` kernel body — reads `in_active_region[i]` for early return | R | CONSUMER OF NEW BUFFER |
| `active_region.cuh` (new, commit 2d) | — | `computeInActiveRegionMask` kernel body — writes `in_active_region[i]` based on `ActiveRegion[]` bounding-box test | W | PRODUCER OF NEW BUFFER |

## Cross-signal invariants

- **Signal 1 dominates Signal 2 for dead particles.** `PFLAG_ACTIVE == 0
  ⇒ active_mask[i] == 0` (enforced at
  [blackhole_v20.cu:2364-2367](../blackhole_v20.cu#L2364-L2367)).
- **Signal 1 dominates Signal 3 for dead particles.** `PFLAG_ACTIVE == 0
  ⇒ passive kernel early-returns` (will be enforced by
  `passive_advection.cuh` in commit 2b).
- **Ejection dominates Signal 3.** `particle_ejected(disk, i)` ⇒ passive
  kernel early-returns (commit 2b). Ejected particles are owned by
  `siphonDiskKernel`'s Aizawa jet path and must not be moved by the
  passive advection kernel.
- **Step 2 bootstrap guarantees Signal 3 == 1 for every alive particle.**
  The all-encompassing `ActiveRegion` seeded in commit 2d means
  `computeInActiveRegionMask` always writes `1` to `in_active_region[i]`
  for alive particles, so the passive kernel's third early-return
  triggers on every particle, producing zero writes.
- **Signals 1, 2, and 3 are stored in three different locations.** Signal
  1 is `disk->flags[i]` (per-particle, bit-packed in the `GPUDisk`
  struct). Signal 2 is `g_active_particles.d_active_mask[i]` (standalone
  buffer owned by `ActiveParticleState`). Signal 3 will be
  `d_in_active_region[i]` (standalone buffer, local variable in
  `main()`). No aliasing risk.

## Step 3 follow-ups — resolution status

Items discovered during the Step 2 audit, now resolved in Step 3
(commits 3a–3e):

- **Phase histogram exclusion of passive particles.** ✅ RESOLVED:
  **no exclusion.** Passive particles remain in the histogram — they
  advance theta via `omega_nat` (passive_advection.cuh:144) and their
  contribution to `peak_frac` is physically meaningful. The grid
  scatter kernels (which feed the histogram) still include all alive
  particles.
- **`spawnParticlesKernel` `pump_history` freeze bias.** ✅ RESOLVED
  in commit 3d: passive parents skip spawning entirely via
  `if (!in_active_region[i]) return;` added to the spawn kernel. This
  prevents stale frozen `pump_history` from triggering spawning from
  settled particles and eliminates the velocity write race between
  `spawn_stream` (parent vel reduction) and default stream (passive
  kernel `vel_y` write).
- **Static-bake grid staleness (R1).** ✅ RESOLVED in commit 3c:
  `g_active_compaction = false` when `ENABLE_PASSIVE_ADVECTION = 1`.
  The passive kernel provides the dominant compute savings (siphon skip);
  the scatter-skip savings are dwarfed. The baked static grid is
  disabled to avoid staleness. Step 4 can re-enable with proper
  rebake logic if profiling shows scatter as a bottleneck.
- **`d_in_active_region[new_idx]` initialization on spawn.** ✅
  RESOLVED by design: newborns have `pump_history = parent * 0.3`
  (always < 0.7), so the mask kernel's `pump_history < 0.7f`
  forced-active condition classifies them as active on the next frame.
  No explicit write to `in_active_region[new_idx]` is needed.
- **Boundary recycle interaction with passive ownership.** ✅ RESOLVED
  by analysis: passive particles' radius is frozen by the passive
  kernel (advances azimuthally at fixed `r_cyl`) and the mask kernel's
  `r_cyl > PASSIVE_R_MAX` forces particles at the boundary to active.
  Passive particles can never drift past `ION_KICK_OUTER_R = 200`.
  No code change needed.
- **Velocity fix-up on passive→active promotion.** ✅ RESOLVED in
  commit 3b: the mask kernel snaps `vel_x/vel_z` to Keplerian
  tangential (`v_kep = √(GM/r)`, prograde direction) when a particle
  transitions from passive (old mask=0) to active (new mask=1).
- **Stale comment at [physics.cu:5-12](../physics.cu#L5-L12).** Still
  pending — trivial doc fix, not blocking any functionality.
- **Stale field comment at
  [blackhole_v20.cu:3601](../blackhole_v20.cu#L3601).** Still
  pending — trivial doc fix.

## Verification

- Grep across `Sanity/*.cu *.cuh validator/*.cuh` for
  `particle_active|set_particle_active|PFLAG_ACTIVE` produces 29 hits,
  and `active_mask` produces 14 hits. Section 1 has 26 rows — the
  difference from 29 is that accessor bodies collapse with their
  definitions (`disk.cuh:287` is the body of `particle_active()` at
  `286`, and `disk.cuh:295`-`296` are the two-line body of
  `set_particle_active()` at `294`), so 29 - 3 = 26. Section 2 has 14
  rows, matching the grep count exactly.
- No row in sections 1 or 2 is marked `AND-with`. Step 2 is purely
  additive.
- `Sanity/docs/baselines/qr_baseline.csv` is checked in as the
  reference trajectory, captured at HEAD `f6d0b70`. See
  `Sanity/docs/baselines/README.md` for the determinism envelope.
