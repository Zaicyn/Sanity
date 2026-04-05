# Blackhole V20 — Realtime Hopfion Shell Lattice Simulation

Realtime N-body simulation of a black hole accretion disk with emergent Hopfion topological structure. Particles self-organize into quantized orbital shells through siphon pump dynamics, forming a stable lattice with measurable Hopf invariant Q.

12 million particles at 200+ FPS on an RTX 2060.

## What Is This?

Each particle operates as an independent 12-to-16 dimensional siphon pump. The pump cycle couples radial stress to angular momentum, creating a natural mechanism for:

- **Shell quantization**: Particles settle into discrete orbital radii (like atomic orbitals)
- **Topological protection**: The Hopf invariant Q measures the "knottedness" of the velocity field
- **Coherent ejection**: Pump overload triggers jets along preferred angular sectors

The simulation uses a sparse tile-based grid for pressure/vorticity coupling, with an adaptive diagnostic system that monitors topology in real time.

## Building

Requires: CUDA toolkit, Vulkan SDK, GLFW, glslc (shader compiler)

```bash
make            # Compiles shaders + builds CUDA/Vulkan binary
make clean      # Remove binary and compiled shaders
```

GPU architecture (default sm_75 for RTX 20xx):
```bash
make CUDA_ARCH=sm_86   # RTX 30xx
make CUDA_ARCH=sm_89   # RTX 40xx
```

## Running

```bash
./blackhole_vulkan                        # Default: 3.5M particles
./blackhole_vulkan -n 10000000            # 10M particles
./blackhole_vulkan -n 20000000 --no-arms  # 20M, no spiral arms
./blackhole_vulkan --headless -n 10000000 --frames 500  # Benchmark mode
```

## Controls

| Key | Action |
|-----|--------|
| Left drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause/resume |
| R | Reset camera |
| C | Cycle color modes |
| A | Toggle attractor density rendering |
| P | Toggle phase-primary view (flattened disk, shows shell rings) |
| H | Toggle radial topology mode |
| L | Toggle hybrid LOD |
| E | Inject entropy cluster |
| V | Adjust shell brightness |
| 1/2/3/4 | Set seam coupling bits |
| \[ / \] | Adjust pump bias |
| T | Turbo bias |

## Architecture

### Physics Pipeline (per frame)

```
siphonDiskKernel        Particle integration (gravity, pump dynamics, ejection)
    |
scatterWithTileFlags    Scatter particles to sparse grid (active tiles only)
    |
decayAndComputePressure Pressure gradients + vorticity from grid
    |
gatherCellForces        Apply grid forces back to particles
```

### Optimization Stack

| Optimization | Mechanism | Impact |
|---|---|---|
| Sparse tile grid | Only compute 3-4% of grid where particles live | Eliminated dense grid overhead |
| Active particle compaction | "Bake" static shell mass, only scatter/gather ~7% active particles | -93% scatter/gather work |
| Half-rate gather | Skip gather on alternate frames when locked | -50% gather bandwidth |
| Lock-aware topology gating | Run diagnostic topology every 8th frame when stable | Reclaimed 25% GPU time |
| Mip-tree coherence | Hierarchical phase coupling replaces Morton-sorted octree | O(1) rebuild |

### Performance (RTX 2060, headless mode, locked state)

| Particles | FPS |
|-----------|-----|
| 1M | ~1900 |
| 5M | ~450 |
| 10M | ~220 |
| 20M | ~125 |

Rendering adds overhead depending on window size and shader mode. Expect ~60-80% of headless FPS with the Vulkan renderer active.

### Key Files

| File | Description |
|------|-------------|
| `blackhole_v20.cu` | Main simulation loop, grid physics, rendering |
| `physics.cu` | siphonDiskKernel (particle integration) |
| `disk.cuh` | GPUDisk struct, constants, derived field compute |
| `harmonic.cuh` | Heartbeat function (cos theta cos 3theta), coherence |
| `forces.cuh` | Viviani field, angular momentum, ion kick |
| `siphon_pump.cuh` | 8-state pump machine, ejection logic |
| `topology_recorder.cuh` | Hopf invariant Q computation, crystal detection |
| `mip_tree.cuh` | Hierarchical coherence tree |
| `octree.cuh` | Spatial octree for render compaction |
| `cell_grid.cuh` | Sparse tile grid definitions |
| `cuda_lut.cuh` | Fast trig lookup tables (constant memory) |
| `cuda_primitives.cuh` | Native GPU scan/reduce (no CUB/thrust) |
| `render_fill.cuh` | Vulkan vertex buffer fill kernels |
| `sun_trace.cuh` | Phase-primary particle representation |
| `aizawa.cuh` | Aizawa attractor for jet dynamics |
| `topology.cuh` | Spiral arm structure |
| `squaragon.h` | O(1) cuboctahedral gate primitive |
| `siphon_pump.h` | 12-to-16 dimensional siphon state machine |
| `vulkan/` | Vulkan renderer (interop, shaders, pipelines) |

## Topology

The simulation tracks the Hopf invariant Q in real time:

1. Particle velocities are scattered onto a 64x64x64 grid to build a unit vector field m(x)
2. The topological charge density B = m * (dm/dx x dm/dy) is integrated
3. Q = integral of B / (4 pi^2) gives the Hopf invariant

When Q stabilizes near an integer and shell structure is uniform, the system has formed a topological soliton (Hopfion). The topology recorder maintains a 128-frame ring buffer and auto-dumps on crystal detection.

## License

Public domain / CC0
