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
normal. It is the asymmetric coupling that converts radial motion into
tangential and back. The 12-fold closure of the Viviani curve (it
returns to origin after theta traverses [0, 2*pi]) creates a
topologically protected transport loop.

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

## The Feedback Loop (what closes the system)

    particles -> scatter (density grid 64^3)
              -> stencil (grad(rho) / rho)
              -> siphon (force on particles)
              -> new positions
              -> scatter ...

The scatter pass bins 20M particles into a 64^3 cell grid via atomic
adds. The stencil pass computes central-difference density gradients
per cell: `-k * grad(rho) / rho` (dispersive pressure, pointing from
high density toward low density).

The siphon **negates** the stencil output, creating an attractive
self-gravity analog: particles are pulled toward overdensities.

    F_radial     = -DENSITY_GAIN * dp_radial
    F_tangential = -DENSITY_GAIN * dp_tangential / r
    F_vertical   = -DENSITY_GAIN * dp_y

`DENSITY_GAIN = 0.3`. The Cartesian pressure gradient is projected into
cylindrical components using the particle's orbital phase (cos phi,
sin phi).

## The Seed (temporary symmetry breaker)

Rectified m=3 density wave applied as a radial velocity perturbation:

    vel_r += epsilon * seed_strength * max(cos(3*(phi - omega_p * t)), 0) * dt

- `epsilon = 0.02` (weak — 40% of the value needed for visible arms alone)
- `omega_p = 0.004` (pattern speed, corotation at r ~ 40)
- `seed_strength` fades linearly from 1.0 to 0.0 between frame 5000-10000

The rectification (max with zero) ensures only the compressive (outward)
phase creates density enhancement. Without rectification, symmetric
cos(3*phi) produces 6 blades (standing wave) instead of 3 arms.

After frame 10000, seed_strength = 0 and the system runs on pure
density feedback. The seed's only purpose is to select m=3 over the
m=4 fossil from initial conditions.

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
in lockstep. This is stimulated emission, not stochastic decay:
population inversion followed by coherent discharge.

## What Evolves

From these boundary conditions, the following structures emerge:

1. **m=3 spiral arms** -- density wave seeded by the temporary
   perturbation, then self-sustained by the density feedback loop.
   Differential rotation winds overdensities into trailing spirals.
   Inner shell m3 reaches 0.08 (8% density modulation) and accelerates
   after the seed turns off.

2. **Coherent three-fold jets** -- ejection from arm tips where
   coherence memory accumulates. m3 = 0.88 in ejected particle
   population. Phase-locked to pump cycle (stimulated emission).

3. **Corotation resonance** -- frozen three-lobed core structure where
   the density wave pattern speed matches the local orbital speed.
   Particles stream through the pattern but it persists.

4. **Outward density wave propagation** -- m3 grows monotonically from
   inner to mid to outer shells over ~15K frames, carried by the
   density feedback amplification.
