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
[QR-corr] frame R_global R_recon n_peaks peak_frac Q num_shells N r_inner r_mid
```

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
to `qr_baseline.csv`. `Q` is advisory only — a drift in `Q` beyond the
0.88 envelope is a red flag, but a drift within envelope is expected
noise.

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
