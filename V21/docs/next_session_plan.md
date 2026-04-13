# Next Session Plan: Research-Grade Analytics + Universal Constants Mapping

## Current State (what we have)

### Architecture
- 20M particle Viviani siphon at ~50 FPS on RTX 2060
- Grade-separated cylindrical state (10 SoA SSBOs)
- Packed meta bitfield (4 fields → 1 uint32)
- Cylindrical density grid (64r × 96phi × 32y, 3MB)
- Phase-locked bonding (Kuramoto sync to analytic field)
- Coherence-driven pump emission (stimulated, not stochastic)
- Harmonic probe (azimuthal Fourier decomposition, m=0-4)

### Physics
- Viviani m=3 transport modulation (energy-neutral, DENSITY_GAIN=3.0)
- Direct omega perturbation (amplitude 3e-5)
- Density-dependent phase locking (BOND_GAMMA=0.003)
- Permanent 35% m=3 arms from frame 4000+
- Vajra topology: 3-fold symmetry, central void, axial jets

### Known Issues
- Slow disk expansion (|p|_mean 94.6 → 166.7 over 20K frames, stabilizing)
- |v|_mean elevated (0.66 at equilibrium vs 0.114 baseline)
- Ejection rate ~3% in oracle window — needs characterization
- Mid shell m3 weaker than inner (~0.10 vs 0.35)

---

## Goal: Research-Grade Analytics

Make the simulation genuinely useful for researchers by adding
measurement infrastructure that maps simulation observables to
physical quantities.

### 1. Rotation Curve Measurement
Measure v_circular(r) = r * omega_orb averaged over azimuthal bins.
Plot against Keplerian prediction v = sqrt(BH_MASS/r).
Deviation = effective "dark matter" contribution from the Viviani
transport + density feedback.

### 2. Toomre Q Parameter (disk stability)
    Q = (kappa * sigma_r) / (pi * G * Sigma)
where:
- kappa = epicyclic frequency (from rotation curve)
- sigma_r = radial velocity dispersion per annulus
- Sigma = surface density per annulus
Q < 1 → unstable (arms form). Q > 1 → stable.
Measure Q(r) to verify the system is marginally unstable in the
arm-forming region and stable elsewhere.

### 3. Pattern Speed Measurement
The arms rotate at omega_p. Measure this independently by tracking
the m=3 phase angle over time in each radial shell.
Compare to the imposed omega_p = 0.004 and to the natural
Viviani breathing frequency.

### 4. Spiral Pitch Angle
The angle between the arm and the tangent to a circle at that radius.
Measure from the azimuthal phase gradient of the m=3 mode:
    tan(pitch) = d(phi_arm)/d(ln r)
Compare to observed galactic pitch angles (10-40 degrees).

### 5. Arm Contrast Ratio
Already measured as m3 amplitude. Convert to standard astronomical
measure: peak density / inter-arm density per annulus.
Typical galaxies: 2:1 to 5:1 contrast.

### 6. Angular Momentum Transport
Measure the Reynolds stress: <vel_r * vel_phi> per annulus.
This quantifies how much angular momentum the arms transport
outward (the fundamental function of spiral structure).

### 7. Energy Budget
Track per-frame:
- Total kinetic energy (sum of 0.5 * m * v^2)
- "Potential" energy (sum of density * grid cell volume)
- Energy injected by omega perturbation
- Energy removed by damping + ejection
Verify the budget closes (energy in ≈ energy out at equilibrium).

---

## The 26 Universal Constants: Mapping to V21 Parameters

### The Standard Model's 26 Free Parameters

#### Particle Masses (Yukawa couplings to Higgs) — 15 parameters
    1.  m_up          6.  m_bottom
    2.  m_down        7.  m_electron
    3.  m_charm       8.  m_muon
    4.  m_strange     9.  m_tau
    5.  m_top         10. m_nu_e
                      11. m_nu_mu
                      12. m_nu_tau
                      13. m_W
                      14. m_Z
                      15. m_Higgs

#### Quark Mixing (CKM matrix) — 4 parameters
    16. theta_12 (Cabibbo angle)
    17. theta_13
    18. theta_23
    19. delta_CP (CP-violating phase)

#### Neutrino Mixing (PMNS matrix) — 4 parameters
    20. theta_12 (solar angle)
    21. theta_13 (reactor angle)
    22. theta_23 (atmospheric angle)
    23. delta_CP (leptonic CP phase)

#### Gauge Couplings (force strengths) — 3 parameters
    24. alpha_EM ≈ 1/137 (electromagnetic)
    25. alpha_s (strong coupling)
    26. theta_QCD (strong CP angle, ≈ 0)

#### Often added:
    27. Lambda (cosmological constant / dark energy)
    28. G (gravitational constant)
    29. v_Higgs (Higgs vacuum expectation value)

### V21 Effective Constants (our "free parameters")

The simulation has its own set of free parameters that govern
emergent behavior. The question: can we map them to (or derive
them from) combinations of the 26?

#### Transport geometry (maps to gauge couplings)
    FIELD_STRENGTH = 0.01     → alpha_EM analog (coupling strength)
    omega_nat = 0.377/r       → mass-radius relationship
    BH_MASS = 1.0             → gravitational coupling (G * M)

#### Damping (maps to dissipative processes / weak force)
    COHERENCE_GAMMA = 0.02    → weak decay rate analog
    orbital damping = f(r)    → radiative cooling

#### Arm formation (maps to symmetry breaking)
    DENSITY_GAIN = 3.0        → Higgs VEV analog (symmetry breaking scale)
    omega_p = 0.004           → pattern speed (frame-dragging analog)
    r_winding = 0.04          → pitch angle (spiral geometry)
    omega_perturbation = 3e-5 → perturbation amplitude

#### Bonding (maps to strong force / confinement)
    BOND_GAMMA = 0.003        → alpha_s analog (coupling between neighbors)
    overdensity_clamp = 5.0   → confinement saturation
    
#### Emission (maps to radioactive/stimulated decay)
    COH_MEM_BUILD = 0.04      → excitation rate
    COH_MEM_DECAY = 0.01      → spontaneous decay rate
    COH_EJECT_THRESHOLD = 1.2 → activation energy
    JET_SPEED = 0.4           → emission velocity
    PUMP_RESET = 0.3          → partial reset (stimulated emission)

#### Grid (maps to spacetime discretization)
    CYL_NR = 64              → radial Planck divisions
    CYL_NPHI = 96            → azimuthal Planck divisions
    CYL_NY = 32              → vertical Planck divisions
    CYL_R_MAX = 200          → horizon scale

### Total V21 free parameters: ~20

---

## Tuning Strategy

### Phase 1: Measure What We Have
Add the analytics (rotation curve, Toomre Q, pattern speed, pitch
angle, contrast ratio, angular momentum transport, energy budget)
and characterize the current parameter set.

### Phase 2: Map to Physical Ratios
Express V21 parameters as dimensionless ratios:
- FIELD_STRENGTH / BH_MASS = transport-to-gravity ratio
- BOND_GAMMA / COHERENCE_GAMMA = bonding-to-damping ratio
- omega_p / omega_orb(r_corot) = pattern-to-orbital ratio
- DENSITY_GAIN * FIELD_STRENGTH = effective coupling

Compare these ratios to known physical ratios (alpha_EM, alpha_s,
Weinberg angle, etc.) to see if the simulation naturally lives
near physical values.

### Phase 3: Sensitivity Analysis
Vary each parameter independently by ±10%, ±50%, ×2, ×0.5.
Measure which observables change and by how much.
Build a Jacobian matrix: d(observable) / d(parameter).
This identifies which parameters are "stiff" (small changes →
large effects) and which are "soft" (insensitive).

### Phase 4: Constraint Satisfaction
Given target observables (e.g., pitch angle = 20°, contrast = 3:1,
Q ≈ 1.5), solve for the parameter set that satisfies all constraints
simultaneously. This is the "tuning" — finding the parameter set
where V21 reproduces observed galactic structure.

### Phase 5: Predict
Once tuned, the simulation makes predictions for observables that
weren't used in the tuning. If those predictions match real galaxies,
the parameter mapping is validated.

---

## Implementation Priority (next session)

1. Rotation curve measurement (simplest, most diagnostic)
2. Toomre Q profile (validates arm formation physics)
3. Energy budget tracking (closes the energy balance question)
4. Pattern speed measurement (validates omega_p)
5. Pitch angle measurement (connects to observations)
6. Parameter sensitivity sweep (systematic, can run overnight)
