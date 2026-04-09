# Squaragon → Sanity: O(1) Primitive Learnings

## Context

Squaragon (830 lines) is the O(1) successor to Resonance — a hardcoded cuboctahedral primitive with CPU, GPU (CUDA), GLSL, and SIMD (SSE/NEON) implementations. It's included in `blackhole_v20.cu` and `siphon_pump.h` but **barely used** — the actual siphon pump was reimplemented in `siphon_pump.cuh` as CUDA device code without referencing the squaragon API.

## What squaragon provides

### 1. N=12 cuboctahedral seed (O(1), zero trig)
`SQ_SEED[12]` — hardcoded unit-sphere vertices. `sq_init()` scales them. No runtime trig, no emergence — the closed primitive is precomputed.

**Sanity equivalent:** None. The simulation generates structure emergently from the Viviani field. The cuboctahedral geometry is implicit in the field's harmonic structure (N=12 from cos(3θ)·cos(4θ), LCM=12), but never instantiated as an explicit polyhedron.

**Useful for:** Reference frame for the N=12 envelope. If we ever need to compute the envelope's zero crossings or node positions explicitly, the seed vertices ARE those positions.

### 2. 16-state closure with precomputed quaternions
`SQ_CLOSURE_STATES[16]` — 12 cuboctahedral + 4 cube-diagonal quaternions. `sq_close_gate()` copies them in O(1).

**Sanity equivalent:** The hopfion algebra has 81 signed states (3^4). The 16-state closure is the UNSIGNED projection ({0,1}^4 = 16 states) that the hopfion spec explicitly calls "a starting approximation but missing the orientation degree of freedom."

**Useful for:** Fast rotation lookups. The quaternion table could replace runtime rotation computation in the Viviani field force or the ejection kick (Hopf fiber direction calculation in siphon_pump.cuh:250-304).

### 3. Viviani scatter LUT
`SQ_SCATTER_LUT[32]` — precomputed O(1) warp distribution table. Maps thread/warp ID to a scatter target using the Viviani curve's z-component.

**Sanity equivalent:** The scatter kernels (scatterParticlesToCells, scatterTopoToCells) use raw atomicAdd with no contention awareness. Multiple particles in the same cell serialize on the same memory address.

**Useful for:** Warp-level contention reduction. Instead of all threads in a warp potentially hitting the same cell, the scatter LUT distributes them across different targets. The Resonance learnings document notes this as a 12-15% throughput gain.

### 4. Seam phase shift
`sq_seam_forward_shift()` / `sq_seam_inverse_shift()` — 2-bit rotation in 64-bit invariants, representing the topological defect at the seam protofilament.

**Sanity equivalent:** The hopfion phason flip operator (`hopfion_phason_flip()` in hopfion.cuh) does sign negation on one axis. The seam phase shift is a more subtle operation — a fractional rotation rather than a full negation.

**Useful for:** Sub-axis resolution in the hopfion algebra. Currently topo_state axes are {-1, 0, +1}. The seam shift suggests a finer encoding where axes can carry fractional phase — relevant if N increases beyond 4.

### 5. SIMD implementations (SSE + NEON)
`sq_simd_triple_xor_residual_sse()` and `sq_simd_triple_xor_residual_neon()` — vectorized CPU implementations processing 4 vertices at a time with hardware SIMD.

**Sanity equivalent:** None. All physics runs on CUDA. No CPU fallback exists.

**Useful for:** The CPU backend migration. When replacing `<<<blocks, threads>>>` with OpenMP loops, the inner-loop math (Viviani curve evaluation, topo_dim computation, Q lookup) can use these SIMD patterns. The SSE version processes the 12 cuboctahedral vertices in 3 batches of 4 — same pattern would apply to processing 4 particles at a time on CPU.

### 6. O(1) direct-mapped index
`sq_index_t` — bidirectional vertex↔state lookup via modulo hash table (16 slots). O(1) insert and lookup.

**Sanity equivalent:** The hopfion Q LUT (`d_Q_lut[256]`) is a one-directional state→Q lookup. There's no reverse (Q→states that produce it).

**Useful for:** If we ever need "find all states with Q=2" queries, a reverse index following the squaragon pattern would be O(1) instead of scanning all 81 states.

### 7. Network serialization
`sq_serialize_compact()` — 24 bytes (scale + bias + orientation quaternion). Receiver reconstructs full gate from seed.

**Sanity equivalent:** None. No save/load or network protocol exists.

**Useful for:** Distributed simulation, checkpointing, or multi-GPU partitioning. A particle's full state (pos, vel, pump, topo) could be compressed similarly — transmit the minimum reconstruction data and let the receiver regenerate derived fields.

### 8. GLSL shader version
Complete GLSL implementation of the cuboctahedral primitive, ready to paste into fragment shaders.

**Sanity equivalent:** The Vulkan shaders (particle.vert, particle.frag) compute color from temperature/residual but don't use cuboctahedral geometry directly.

**Useful for:** If the photonic accumulators ever need to evaluate resonance in the shader rather than the compute kernel, the GLSL squaragon functions are ready.

## What Sanity improved upon

| Aspect | Squaragon | Sanity | Improvement |
|--------|-----------|--------|-------------|
| State space | 16 unsigned states | 81 signed states (3^4) | Orientation degree of freedom |
| Pump state machine | 12 states in siphon_pump.h | 8 states + coherence filter in siphon_pump.cuh | CUDA-native with velocity access |
| Viviani evaluation | z-component only (sq_viviani_z) | Full 3D tangent vector (forces.cuh) | Complete field geometry |
| Topology | Triple XOR residual (scalar) | Hopf invariant Q + 6 operators | Full reaction algebra |
| Scale hierarchy | 27/16 and φ scaling | Emergent from field harmonics | No hardcoded ratios in dynamics |

## Integration priorities (when ready)

| Priority | What | From | Effort |
|----------|------|------|--------|
| **HIGH** | SIMD CPU kernels | SSE/NEON sections | CPU backend migration |
| **MEDIUM** | Scatter LUT for contention | SQ_SCATTER_LUT[32] | Scatter kernel optimization |
| **MEDIUM** | Quaternion rotation table | SQ_CLOSURE_STATES[16] | Ejection kick optimization |
| **LOW** | Network serialization | sq_serialize_compact | Distributed/checkpoint feature |
| **LOW** | Reverse index for Q queries | sq_index_t pattern | Diagnostic tooling |

## Key files

| File | Lines | What it provides |
|------|-------|-----------------|
| `/home/zaiken/sanity/squaragon.h` | 830 | Full primitive: CPU + CUDA + GLSL + SSE + NEON |
| `/home/zaiken/sanity/Sanity/siphon_pump.h` | ~320 | Original pump context using sq_gate_t (not used by simulation) |
| `/home/zaiken/sanity/Sanity/siphon_pump.cuh` | 326 | Active pump: CUDA reimplementation without squaragon |
