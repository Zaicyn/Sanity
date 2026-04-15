# Squaragon V2 → V21 GPU Integration

## Status: Architecture Working, Force Tuning Needed

The Squaragon V2 geometry IS the ground truth. The allocator's math (47
tests, zero failures) defines the physics. The GPU simulation derives
from it, not the other way around.

## What Was Done This Session

### Squaragon V2 (allocator/V22/)
- Corrected scatter mixer (x-component, unified CPU/GPU)
- Corrected modulus (mod 8, power-of-2)
- Discovered Hopf shell structure, branching, torus closure
- Built DNA cell cycle (write, replicate, compare, repair, divide)
- Flow state soliton (3/9/27 harmonics, 8 windows per ring)
- Zero-sinf optimization (200 bytes of LUTs, all ops under 5 cycles)
- 47 tests, zero failures, complete documentation in squaragon_v2.md

### GPU Integration
- Found and fixed: siphon.comp was loaded but never dispatched
  (dispatch went through siphon_graded.comp — a completely different shader)
- Fixed dispatch in vk_compute.cpp to route through our clean shader
- Rewrote siphon.comp from 635 lines to ~200 lines of clean Squaragon physics
- Fixed BH_MASS mismatch (was 1.0 in GPU push constants, should be 100.0)
- Fixed init: Keplerian omega_nat, position-seeded theta
- Fixed far clip (5000 → 50000)
- Added missing pipeline barrier between siphon and projection

## Architecture (Ground Truth)

```
squaragon_v2.h                    ← THE SEED (read-only reference)
  │
  ├── Viviani curve:  x = sin(θ) - 0.5·sin(3θ)
  │                   y = -cos(θ) + 0.5·cos(3θ)
  │                   z = cos(θ)·cos(3θ)
  │
  ├── Tangent vector: dx/dθ = cos(θ) - 1.5·cos(3θ)
  │                   dy/dθ = sin(θ) - 1.5·sin(3θ)
  │                   dz/dθ = -sin(θ)·cos(3θ) - 3·cos(θ)·sin(3θ)
  │
  ├── Flow mode LUT:  32 positions, 0=COAST/1=ACTIVE/2=FLOW
  │                   10 coast + 14 active + 8 flow = 32
  │
  ├── Three force channels:
  │   EM (1θ):     tangent direction, always active
  │   Strong (3θ): z-coupling cos(θ)·cos(3θ), ACTIVE+ mode
  │   Weak (9θ):   curve point attractor, FLOW only
  │
  ├── Gravity = BH_MASS / r² (proper Keplerian)
  │
  ├── Torus: period-32, 8 bins, 512 states
  │
  └── 27/16 = soliton³ / frozen⁴ (coherent scale ratio)
```

## Current GPU Shader (siphon.comp)

Clean ~200-line shader implementing:

1. **Gravity**: `BH_MASS / r²` centripetal, always active
2. **EM channel**: Direction steering — rotates velocity toward Viviani
   tangent without changing speed. Flow-mode-modulated rate:
   COAST=0.1%, ACTIVE=0.5%, FLOW=2%
3. **Strong channel**: `cos(θ)·cos(3θ)` radial bunching, ACTIVE+ mode
4. **Weak channel**: Push toward Viviani curve point, FLOW only
5. **No damping**: Energy conserved by construction
6. **Per-particle fiber phase**: Keplerian omega_nat advances theta

## Files Modified

```
Sanity/V21/kernels/siphon.comp           ← clean Squaragon shader (THE physics)
Sanity/V21/vk_compute.cpp                ← dispatch routes to siphon.comp (not graded)
Sanity/V21/blackhole_v21.c               ← CPU reference (fiber phase + Keplerian)
Sanity/V21/blackhole_v21_visual.cpp      ← BH_MASS=100, Keplerian omega_nat, barrier
Sanity/vulkan/vk_buffer.cpp              ← far clip 50000
```

## Stale SPV Hazard

The build system compiles shaders to `V21/build/kernels/*.spv`. But stale
SPV files can exist in other locations that the loader searches first:

```
V21/kernels/*.spv                   ← STALE if present (source dir)
vulkan/shaders/compute/*.spv        ← STALE V20 artifacts
```

**Always delete stale SPVs before testing.** The loader searches:
`kernels/` → `../kernels/` → `shaders/compute/` → `vulkan/shaders/`

The `[shader] Loaded ...` log line (added this session) shows which file
was actually loaded. Check it.

## Known Issues (What Needs Tuning)

### 1. Rotation Curve is 9x Too Fast

```
v_circ = 4.07  vs  v_kep(diagnostic) = 0.44  at r=5.1
```

The diagnostic computes v_kep with BH_MASS=1.0 (hardcoded in the oracle).
The actual velocities match the gravity (`v = sqrt(100/r)`). The
diagnostic needs updating, not the physics.

**Fix**: Update the oracle's BH_MASS to match the push constant (100.0).

### 2. Inward Collapse (mean |p| shrinking over time)

The system slowly collapses inward. Energy is being drained somewhere.
Possible causes:

a) **Direction steering drains energy**: The perpendicular velocity
   removal in the EM channel (`vx -= perp_x * steer_rate`) reduces
   speed slightly when the perpendicular component is large. Over many
   frames this drains KE. Fix: after steering, rescale velocity to
   preserve original magnitude.

b) **Weak channel pushes inward**: The Viviani curve point at the
   particle's radius has a different direction than the particle's
   position. The acceleration toward the curve point creates a net
   inward bias because the curve point's radial component is often
   less than the particle's actual radius. Fix: project the weak
   channel force to be purely tangential/vertical, no radial component.

c) **Strong channel radial oscillation**: The `cos(θ)·cos(3θ)` bunching
   pushes particles in and out on the ring cycle. If the inward phase
   is slightly stronger than the outward phase (due to 1/r² weighting),
   there's a net inward drift. Fix: use a fixed weight instead of
   gravity-weighted.

### 3. No Bond Breaking (Aizawa Excursions)

The system has no mechanism for particles to deviate from their Viviani
tangent orbit. The steering locks them to the tangent direction. Without
deviations, there are no pressure gradients, no branching, no Aizawa
excursions.

**Fix**: The steering should have a maximum rate that decreases with flow
quality. In COAST mode, particles should drift freely (0% steering).
Only in FLOW mode should steering be strong enough to create structure.
Between soliton windows, particles coast on pure gravity — this creates
the natural variation that enables branching.

### 4. Initial Velocity Mismatch

Particles start with `v = sqrt(BH_MASS / max(r_xz, ISCO_R))` which
clamps at ISCO_R=6. For r < 6, all particles get the same initial speed
`sqrt(100/6) = 4.08`. This creates a sharp velocity discontinuity at
r=6. The core particles are all at the same speed regardless of radius.

**Fix**: Don't clamp at ISCO_R. Use `v = sqrt(BH_MASS / max(r_xz, 1.0))`
for a smoother velocity profile, or even better, initialize core
particles from the Viviani curve directly.

## Tuning Approach

The Squaragon allocator tests (47 tests) proved that the geometry works.
The issue is mapping abstract topological forces to Cartesian accelerations
in the sim. The tuning should follow this order:

### Step 1: Fix the energy drain
Add velocity magnitude preservation after direction steering:
```glsl
float speed_before = sqrt(vx*vx + vy*vy + vz*vz);
// ... do steering ...
float speed_after = sqrt(vx*vx + vy*vy + vz*vz);
float rescale = speed_before / (speed_after + 1e-8);
vx *= rescale; vy *= rescale; vz *= rescale;
```

### Step 2: Fix the weak channel
Remove radial component from weak channel force:
```glsl
// Project out radial component
float weak_dot_r = weak_ax*rx + weak_ay*ry + weak_az*rz;
weak_ax -= weak_dot_r * rx;
weak_ay -= weak_dot_r * ry;
weak_az -= weak_dot_r * rz;
```

### Step 3: Reduce steering in COAST mode
```glsl
float steer_rate = (mode == 2) ? 0.02 : (mode == 1) ? 0.002 : 0.0;
```
COAST = zero steering = pure gravity orbits = natural variation.

### Step 4: Verify energy conservation
After steps 1-3, total KE should be roughly constant (small fluctuations
OK, monotonic drain or growth = bug). The `[energy]` diagnostic line
tracks this.

### Step 5: Increase particle count
Once energy is stable at 100K, test at 1M and 10M. The structure should
scale — same geometry at all counts.

## Reference: Squaragon Constants

```
BH_MASS          = 100.0     (gravity weight)
BIAS             = 0.75      (25% headroom, branching threshold)
SCALE_RATIO      = 27/16     (coherent shell hop)
PHI              = 1.618...  (spillover hop)
TORUS_PERIOD     = 32        (seam shift cycle)
BINS             = 8         (Viviani mod-8 scatter)
FLOW_THRESHOLD   = 0.30      (soliton window entry)
FLOW_HARMONICS   = 3, 9, 27  (powers of 3)
FLOW_COEFFS      = 1/3, 1/9, 1/27
```

## File Locations

```
allocator/V22/                          ← Squaragon V2 reference (ground truth)
  squaragon_v2.h                        ← complete header
  squaragon_v2.md                       ← 3000+ line findings document
  squaragon_v2_dna.h                    ← DNA/allocator cell cycle
  squaragon_v2_forces.h                ← topological force kernels
  README.md                             ← overview

Sanity/V21/kernels/siphon.comp          ← GPU physics (clean Squaragon)
Sanity/V21/blackhole_v21.c              ← CPU reference implementation
Sanity/V21/blackhole_v21_visual.cpp     ← Vulkan rendering + CPU/GPU physics
Sanity/V21/vk_compute.cpp               ← Vulkan compute dispatch
Sanity/V21/vk_compute.h                 ← Vulkan compute types

test_squaragon_v3*.c                    ← 10 test files, 47 tests total
```
