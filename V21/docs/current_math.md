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

## Arm Formation: Viviani m=3 Analytic Modulation

### Transport Modulation (V8 approach)
The Viviani curve's inherent 3-fold symmetry modulates the transport
strength as a function of orbital phase and radius:

    d_vel_r     *= (1 + DENSITY_GAIN * sin(m3_phase))
    d_omega_orb *= (1 + DENSITY_GAIN * sin(m3_phase))
    d_vel_y     *= (1 + DENSITY_GAIN * sin(m3_phase))

where:

    m3_phase = 3 * phi - r * 0.04 - 3 * omega_p * t

- `DENSITY_GAIN = 3.0` (modulation amplitude)
- `omega_p = 0.004` (pattern speed, corotation at r ~ 40)
- `-r * 0.04` creates trailing spiral winding

This is multiplicative on the Viviani transport — it does not inject
energy. Over a full azimuthal orbit, <sin> = 0, so there is no net
radial bias or heating. Perfectly energy-neutral: |p|_mean drifts
less than 0.01% per 1000 frames.

### Azimuthal Density Perturbation
Direct omega modulation creates density bunching:

    omega_orb += 3.0 * 0.00001 * sin(m3_phase) * dt

Particles in the "fast" phase advance azimuthally (creating a gap);
particles in the "slow" phase fall behind (creating a pile-up).
Differential rotation winds this into trailing spirals.

## Cylindrical Density Grid

### Architecture
Replaces the Cartesian 64^3 grid. Aligned with disk geometry:

    Nr = 64 radial shells
    Nphi = 96 azimuthal sectors (covers 0 to 2*pi)
    Ny = 32 vertical layers

    Total: 196,608 cells (3.0 MB)
    R_max = 200, Y_half = 100

No 4-fold symmetry artifacts because the grid matches the flow.

### Pipeline
    cyl_scatter: particles -> (r, phi, y) bins via atomic adds
    cyl_stencil: central-difference density gradients in (r, phi, y)
                 azimuthal gradient wraps periodically
                 outputs pressure_r, pressure_phi, pressure_y

## Neighbor Bonding (density-dependent radial damping)

Particles in overdense cells (arms) have their radial velocity
dispersion reduced, keeping the arm coherent against shear:

    rho = cyl_density[cell]
    rho_avg = N / total_cells
    overdensity = clamp(rho / rho_avg - 1, 0, 5)
    vel_r *= (1 - BOND_GAMMA * overdensity)

- `BOND_GAMMA = 0.03`
- Only active in overdense regions (overdensity > 0)
- Clamped at 5.0 to prevent velocity reversal (max 15% extra damping)
- Uses the cylindrical grid — no new scatter passes needed

This is not a force — it is phase synchronization. Particles in dense
regions orbit more circularly (less radial scatter), which keeps them
in the arm longer. Equivalent to density-dependent anisotropic viscosity.

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

From the Viviani geometry + density feedback + bonding:

1. **m=3 spiral arms** — form spontaneously from the Viviani transport's
   inherent 3-fold symmetry. No external seed required. Arms peak at
   m3 = 0.54 (54% density modulation), then stabilize at ~0.25 (inner)
   / 0.31 (mid) through density-dependent bonding.

2. **Limit-cycle oscillator** — at critical gain (DENSITY_GAIN = 3.0),
   the Viviani breathing cycle amplifies into periodic density wave
   pulses. Each pulse stronger than the last, propagating outward.
   Period ~13K frames.

3. **Coherent three-fold jets** — ejection from arm regions where
   coherence memory accumulates. Phase-locked to pump cycle
   (stimulated emission, not stochastic decay).

4. **Self-sustaining structure** — the bonding mechanism (density-dependent
   radial damping) prevents arms from shearing apart. After the initial
   density wave passes, the arms recover and stabilize instead of
   dissolving. The system reaches equilibrium without external forcing.

## Energy Balance

The system maintains energy neutrality through three mechanisms:

1. **Multiplicative transport modulation** — sin(m3_phase) averages to
   zero over phi. No net energy injection.

2. **Anisotropic damping** — removes radial kinetic energy at 2%/frame
   (baseline) + up to 15% in overdense regions (bonding).

3. **Coherent ejection** — removes high-alignment particles from the
   disk, carrying structured phase information outward.

Measured stability: |p|_mean drifts 0.6% over 20K frames. |v|_mean
peaks during arm formation, then declines toward equilibrium.
No runaway heating.
