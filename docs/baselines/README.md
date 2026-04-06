# QR-corr reproducibility baselines

Captured at HEAD `f6d0b70` (pre–Tree Architecture Step 2) with:

```
./blackhole_vulkan -n 1000000 --no-spawn --frames 2200 --rng-seed 42 --headless --qr-corr
```

`qr_baseline.csv` is the first run, `qr_baseline_run2.csv` is a second
identical run used to measure run-to-run determinism. The two runs were
compared line-by-line.

## Determinism envelope (what is and isn't stable)

Columns produced by `[blackhole_v20.cu:6264](../../blackhole_v20.cu#L6264)`
are:

```
[QR-corr] frame R_global R_recon n_peaks peak_frac Q num_shells N r_inner r_mid active_frac
```

Step 4 added the `active_frac` column (11th, 0-indexed: 10): fraction of
alive particles classified as active (siphon path) by
`computeInActiveRegionMask`. Values range from 0.0 (all passive) to 1.0
(all active). At the default threshold of 0.15 with standard initialization,
`active_frac ≈ 0.0005` from frame 0 onwards (only boundary particles).

## Step 4 threshold sweep results

Sweep run via `sweep_threshold.sh --fast` (500 frames, 1M particles,
seed=42, `--no-spawn`):

| threshold | fps  | active_frac | num_shells | R_global |
|-----------|------|-------------|------------|----------|
| 0.01      | 1424 | 0.0005      | 8          | 0.001337 |
| 0.05      | 1390 | 0.0005      | 8          | 0.001337 |
| 0.10      | 1425 | 0.0005      | 8          | 0.001337 |
| 0.15      | 1390 | 0.0005      | 8          | 0.001337 |
| 0.20      | 1390 | 0.0005      | 8          | 0.001337 |
| 0.30      | 1382 | 0.0005      | 8          | 0.001337 |
| 0.50      | 1465 | 0.0005      | 8          | 0.001337 |
| 1.00      | 1445 | 0.0005      | 8          | 0.001337 |

**Finding: the threshold is currently irrelevant.** All values produce
identical active fraction (0.05%) and physics (8 shells, same R_global).
This is because:

1. `pump_history` is initialized to 1.0 (above the 0.7 forced-active
   threshold), so the warmup catch doesn't fire.
2. `pump_residual` is initialized to 0.0, below any threshold.
3. Passive particles never go through siphon, so their residual never
   increases — the feedback loop is in a stable "all passive" state.
4. Only the boundary conditions (`r_cyl < 3.0 || r_cyl > 200.0`) force
   particles active, producing the constant ~500 active particles.

## Step 5 sweep results (with shell-deviation injection)

Step 5a added shell-deviation residual injection: particles off-shell
accumulate `effective_residual = max(|pump_residual|, deviation * 0.1)`
in the mask kernel. The threshold is now meaningful.

Sweep via `sweep_threshold.sh --fast` (500 frames, 1M particles):

| threshold | fps  | active_frac | num_shells | R_global |
|-----------|------|-------------|------------|----------|
| 0.01      | 1024 | 0.8267      | 8          | 0.001304 |
| 0.05      | 1014 | 0.8138      | 8          | 0.001044 |
| 0.10      | 1009 | 0.7975      | 8          | 0.000960 |
| 0.15      | 1021 | 0.7813      | 8          | 0.000669 |
| 0.20      | 1013 | 0.7652      | 8          | 0.000586 |
| 0.30      | 978  | 0.7338      | 8          | 0.000673 |
| 0.50      | 1003 | 0.6734      | 8          | 0.001441 |
| 1.00      | 1030 | 0.5374      | 8          | 0.000511 |

**Key findings:**
- Active fraction ranges from 53.7% to 82.7% — the threshold IS relevant
- All 8 shells remain stable across the entire threshold range
- Active fraction decreases over time as siphon pulls particles onto
  shells (at 500 frames, settling is still in progress)
- FPS is ~1000 across the board (most particles still active due to
  off-shell initialization; longer runs will see more passivation and
  higher FPS as the system settles)

Run-to-run behavior across 26 sample frames (frames 0, 90, 180, ..., 2250):

| Column      | Max Δ across runs | Status       |
| ----------- | ----------------- | ------------ |
| `frame`     | 0                 | deterministic |
| `R_global`  | 0.000000          | deterministic |
| `R_recon`   | 0.000000          | deterministic |
| `n_peaks`   | 0                 | deterministic |
| `peak_frac` | 0.000000          | deterministic |
| `Q`         | **0.8789**        | **noisy**    |
| `num_shells`| 0                 | deterministic |
| `N`         | 0                 | deterministic |
| `r_inner`   | 0.0000            | deterministic |
| `r_mid`     | 0.0000            | deterministic |

`Q` is the only column that drifts run-to-run. Most frames drift by
≤ 0.0001, but three frames drifted by ~0.06 and one frame (2070) drifted
by 0.88. This is consistent with `git log` commit `b6600d9
REPRODUCIBILITY CORRECTION: Q drift IS reproducible` — Q's qualitative
trajectory is reproducible but its point values are dominated by
atomic-float non-determinism in the Kuramoto reduction.

Other drifting log lines (not in `[QR-corr]`): VRAM free bytes, Vulkan
buffer pointers, FPS/timing, shell radii at `%.1f` rounding boundaries
(±0.1 when a true value sits on a half-digit), `M_eff` in `[lensing]`
lines at the 3rd decimal. None affect the physics invariants captured
in `[QR-corr]` deterministic columns.

## Verification rule for Tree Architecture Step 2 commits

Commits 2b / 2c / 2d must leave the deterministic columns byte-identical
to `qr_baseline.csv`. `Q` is advisory only — its value is dominated by
order-dependent float atomic reductions and is expected to drift.

**Observed `Q` envelopes:**

- Pre-refactor self-determinism (run1 vs run2 at HEAD `f6d0b70`):
  max dQ = 0.8789 at frame 2070.
- Commit 2b (kernel compiled, not launched): max dQ = 0.4749.
- Commit 2c (buffer allocated, guard off): max dQ = 0.0568.
- Commit 2d (guard on, all-encompassing bootstrap): **max dQ = 7.5232
  at frame 900.** Larger than baseline run1-vs-run2, but verified
  to be pure scheduling noise — a diagnostic build with
  `advectPassiveParticles` stubbed to `return;` as the first statement
  produces the byte-identical 7.5232 drift at the byte-identical
  frame 900. The extra kernel launches (`computeInActiveRegionMask`
  + `advectPassiveParticles`) perturb warp scheduling enough to
  deterministically shift downstream atomic-float reductions, but
  every other observable (R_global, R_recon, peak_frac, n_peaks,
  num_shells, N, r_inner, r_mid) is byte-identical.
- Commit 2d self-determinism (run1 vs run2 of the 2d binary): max
  dQ = 0.0071 — the 2d binary is *more* deterministic than the
  baseline, because adding the two extra kernel launches acts as
  an implicit barrier that reduces subsequent launch-to-launch
  scheduling variance.

Concretely, commit verification runs:

```
./blackhole_vulkan -n 1000000 --no-spawn --frames 2200 --rng-seed 42 --headless --qr-corr > /tmp/run.log 2>&1
grep '\[QR-corr\]' /tmp/run.log > /tmp/qr_run.csv
# Compare all columns except Q (field 7):
awk '{$7=""; print}' Sanity/docs/baselines/qr_baseline.csv > /tmp/qr_baseline_noQ.csv
awk '{$7=""; print}' /tmp/qr_run.csv > /tmp/qr_run_noQ.csv
diff /tmp/qr_baseline_noQ.csv /tmp/qr_run_noQ.csv   # must be empty
# Q envelope advisory:
paste Sanity/docs/baselines/qr_baseline.csv /tmp/qr_run.csv | awk '{d=$7-$18; if (d<0) d=-d; if (d>0.88) print "Q DRIFT:", $2, d}'
```
