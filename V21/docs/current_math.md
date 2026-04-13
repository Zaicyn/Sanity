# V21 Mathematical Summary

## Boundary Conditions (imposed geometry)

### Central Potential
Keplerian, `BH_MASS = 1.0`. Orbital velocity `v = sqrt(1/r)`, giving
differential rotation (inner particles orbit faster than outer).

### Viviani Curve Transport
The Viviani curve provides the transport field generator:

    c(theta) = (sin theta - 1/2 sin 3*theta,
               -cos theta + 1/2 cos 3*theta,
                cos theta * cos 3*theta)

The unit normal `n_hat = c / |c|` drives a double cross product transport:

    dv = FIELD_STRENGTH * (n_hat x v) x v

This rotates velocity toward the plane perpendicular to the Viviani
normal — the asymmetric coupling that converts radial motion into
tangential and back. The 12-fold closure of the Viviani curve creates
a topologically protected transport loop.

The curve's inherent 3-fold symmetry (from the sin(3*theta) and
cos(3*theta) terms) is the source of all m=3 structure in the system.
No external seed is required — the geometry IS the perturbation.

### Particle State (grade-separated cylindrical)

- **Grade 0 (scalars):** r, pump state machine fields (scale, residual,
  work, coherence memory)
- **Grade 1 (vectors):** delta_r, delta_y, vel_r, vel_y
- **Grade 2 (bivectors):** phi (orbital phase), omega_orb (orbital angular
  velocity), theta (Viviani phase), omega_nat = 0.377/r (natural frequency)

All packed into SoA SSBOs. Four scalar fields (pump_state, pump_coherent,
flags, in_active_region) compressed into a single uint32 packed_meta
bitfield at binding 6.

### Damping

- Anisotropic radial: `vel_r *= (1 - 0.02)` per frame
- Orbital-plane: `vel_y *= (1 - damping(r))` where
  `damping = min(2 * sqrt(BH_MASS / r^3), 0.50)`

### Angular Momentum Conservation

On radial displacement: `omega_orb *= (r_old / r_new)^2`

This preserves `L = r^2 * omega_orb` (Keplerian speedup/slowdown),
which Cartesian coordinates get for free from conservation of
tangential velocity.

## Arm Formation: Three Coupled Mechanisms

### 1. Viviani m=3 Transport Modulation
The transport's 3-fold symmetry is expressed through a rotating pattern:

    m3_phase = 3 * phi - r * 0.04 - 3 * omega_p * t

The transport derivatives are modulated multiplicatively:

    d_vel_r     *= (1 + DENSITY_GAIN * sin(m3_phase))
    d_omega_orb *= (1 + DENSITY_GAIN * sin(m3_phase))
    d_vel_y     *= (1 + DENSITY_GAIN * sin(m3_phase))

- `DENSITY_GAIN = 3.0`
- `omega_p = 0.004` (pattern speed, corotation at r ~ 40)
- `-r * 0.04` creates trailing spiral winding (arms tilt with radius)

This is multiplicative on the Viviani transport — it redistributes
transport strength azimuthally without injecting energy. Over a full
orbit, <sin> = 0. Perfectly energy-neutral.

### 2. Azimuthal Density Perturbation
Direct omega modulation bunches particles azimuthally:

    omega_orb += 3.0 * 0.00001 * sin(m3_phase) * dt

Particles in the "fast" phase advance (creating a gap); particles in
the "slow" phase fall behind (creating a pile-up). Differential
rotation winds this into trailing spirals. Small amplitude ensures
minimal energy injection.

### 3. Phase-Locked Bonding (Kuramoto synchronization)
Particles in overdense cells synchronize their orbital frequency to
the analytic m=3 pattern phase. This is the mechanism that makes
arms permanent — without it, they form and decay; with it, they
form and persist.

Two sub-mechanisms, both density-weighted from the cylindrical grid:

    lock = BOND_GAMMA * clamp(rho / rho_avg - 1, 0, 5)

**Frequency locking:**

    omega_orb += lock * sin(phase) * dt

Particles in arms adjust their angular velocity toward the pattern's
local rate. This is Kuramoto-style synchronization using the analytic
phase field as the reference oscillator — no need to measure neighbors
when the pattern is known analytically.

**Radial cohesion:**

    vel_r *= (1 - lock * |sin(phase)|)

Particles at arm centers (where |sin(phase)| is large) have their
radial velocity damped. Particles between arms are untouched. This
sharpens arm edges and prevents radial dispersion from dissolving
the structure.

`BOND_GAMMA = 0.003`. Overdensity clamped at 5.0 (max 1.5% extra
damping per frame). The clamp represents finite interaction bandwidth —
synchronization saturation, not a numerical hack.

## Cylindrical Density Grid

### Architecture
Aligned with disk geometry — eliminates Cartesian 4-fold artifacts:

    Nr = 64 radial shells
    Nphi = 96 azimuthal sectors (covers 0 to 2*pi)
    Ny = 32 vertical layers

    Total: 196,608 cells (3.0 MB)
    R_max = 200, Y_half = 100

### Pipeline
    cyl_scatter: particles -> (r, phi, y) bins via atomic adds
    cyl_stencil: central-difference density gradients in (r, phi, y)
                 azimuthal gradient wraps periodically
                 outputs pressure_r, pressure_phi, pressure_y

The grid serves two roles:
1. **Bonding source** — provides local overdensity for phase locking
2. **Diagnostic tool** — feeds the harmonic probe for m-mode analysis

The stencil pressure gradients are computed but currently used only
for diagnostics. The force feedback experiments (attractive self-gravity)
were abandoned due to fundamental heating: non-conservative forces
without a potential well accumulate kinetic energy indefinitely. The
V8 analytic approach (geometry-driven structure) replaced grid-force
feedback for arm formation.

## The Pump (coherence-driven emission)

`pump_history` repurposed as a coherence memory integrator:

    alignment = |vel_r| / |v|
    H = H * (1 - 0.01) + alignment * 0.04
    H = min(H, 4.0)

Ejection fires when `H > 1.2` at `r < 48`:

    muzzle = H * JET_SPEED
    vel_y += muzzle * collimation
    vel_r += muzzle * (1 - collimation)
    H *= 0.3  (partial reset)

Collimation: inner particles get more vertical kick, outer particles
get more radial scatter.

The pump state machine (8 states: IDLE -> PRIMED -> UC/UH -> EXPAND ->
DOWNSTROKE -> VALVE_OPEN -> RECIRCULATE) cycles the entire population
in lockstep — stimulated emission, not stochastic decay. Population
inversion followed by coherent discharge.

## What Evolves

From the Viviani geometry + analytic modulation + phase-locked bonding:

1. **Permanent m=3 spiral arms** — inner shell stabilizes at 35%
   density modulation from frame 4000 onward. No decline, no
   oscillation. Self-sustaining structure maintained by Kuramoto
   phase locking to the analytic pattern field.

2. **Trailing spiral geometry** — the `-r * 0.04` winding term in
   m3_phase creates arms that tilt with radius, matching the natural
   winding from differential rotation.

3. **Limit-cycle oscillator** — at critical feedback gain, the Viviani
   breathing cycle amplifies into periodic density wave pulses. Each
   pulse propagates outward. Period ~13K frames.

4. **Coherent three-fold jets** — ejection from arm regions where
   coherence memory accumulates. Phase-locked to pump cycle.

5. **Vajra topology** — the three-fold rotational symmetry with
   central void and coherent emission channels produces a structure
   matching the vajra form of Hindu/Buddhist iconography — crossed
   transport streams meeting at a central point with m=3 arms and
   axial jets.

## Energy Balance

The system maintains approximate energy neutrality:

1. **Multiplicative transport modulation** — sin(m3_phase) averages to
   zero over phi. No net energy injection from the Viviani modulation.

2. **Phase locking** — redistributes angular momentum within overdense
   cells toward the pattern frequency. Asymptotically neutral (once
   locked, no further adjustment needed).

3. **Anisotropic damping** — removes radial kinetic energy at 2%/frame
   (baseline) + density-dependent additional damping at arm centers.

4. **Coherent ejection** — removes high-alignment particles from the
   disk, carrying structured phase information outward.

Remaining energy drift: the omega perturbation injects small amounts
of angular momentum, causing slow disk expansion (|p|_mean 94.6 → 166.7
over 20K frames, stabilizing). The system approaches thermal equilibrium
as |v|_mean declines from initial transient toward steady state.

## Key Experimental Results (what was tried and rejected)

### Rejected: Cartesian grid feedback
The 64^3 Cartesian density grid imprinted 4-fold symmetry artifacts
(visible squares) on the particle distribution at any feedback gain.
Replaced by cylindrical grid.

### Rejected: Attractive density feedback (self-gravity analog)
Non-conservative force (-grad(rho)/rho, negated) heats the system
without a potential energy sink. Particles gain kinetic energy falling
into density wells but never give it back. All gains tested (0.03 to
3.0) produced runaway heating. Energy-conserving feedback requires a
Poisson solve — expensive and unnecessary given the V8 analytic approach.

### Rejected: Lab-frame transport gates
Static cos(3*phi) modulation of the Viviani transport is a zero
operator under differential rotation — particles orbit through the
modulation and the effect time-averages to nothing. Only rotating
patterns (with omega_p * t) or the comoving frame work.

### Rejected: Symmetric density wave
cos(3*phi) radial kick produces 6 blades (standing wave), not 3 arms.
The rarefactive and compressive phases are equally visible. Rectification
(max(cos, 0)) fixes this but introduces DC radial drift.

### Validated: Subcritical bifurcation
At DENSITY_GAIN < 3.0 on the Cartesian grid, the Viviani breathing
oscillation (~1.5% m3 amplitude) cannot spontaneously grow into
macroscopic structure. At gain >= 3.0, the breathing builds up over
~14K frames until feedback catches an upswing and drives exponential
growth. The system has two stable states (featureless vs structured)
with a finite perturbation threshold between them.
