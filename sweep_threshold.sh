#!/bin/bash
# Tree Architecture Step 4: CORNER_THRESHOLD sweep
# Runs headless simulations at varying thresholds and collects QR-corr data.
#
# Usage: ./sweep_threshold.sh [--fast]
#   --fast  Use fewer frames (500 instead of 2200) for quick iteration
#
# Output: docs/baselines/sweep/thresh_<value>.log for each threshold,
# plus a summary line per threshold with FPS and active fraction.

set -e

BINARY=./blackhole_vulkan
N=1000000
FRAMES=2200
SEED=42
OUTDIR=docs/baselines/sweep

if [ "$1" = "--fast" ]; then
    FRAMES=500
    echo "Fast mode: $FRAMES frames"
fi

mkdir -p "$OUTDIR"

echo "=== Tree Architecture Step 4: CORNER_THRESHOLD sweep ==="
echo "Binary: $BINARY"
echo "Particles: $N, Frames: $FRAMES, Seed: $SEED"
echo ""

# Column header for summary
printf "%-12s %-8s %-12s %-12s %-10s %-10s\n" \
    "threshold" "fps" "active_frac" "num_shells" "R_global" "Q"
echo "----------------------------------------------------------------------"

for THRESH in 0.01 0.05 0.10 0.15 0.20 0.30 0.50 1.00; do
    LOGFILE="$OUTDIR/thresh_${THRESH}.log"

    $BINARY -n $N --no-spawn --frames $FRAMES --rng-seed $SEED \
        --headless --qr-corr --corner-threshold $THRESH \
        > "$LOGFILE" 2>&1

    # Extract metrics from last QR-corr line
    LAST_QR=$(grep '\[QR-corr\]' "$LOGFILE" | tail -1)
    R_GLOBAL=$(echo "$LAST_QR" | awk '{print $3}')
    Q_VAL=$(echo "$LAST_QR" | awk '{print $7}')
    SHELLS=$(echo "$LAST_QR" | awk '{print $8}')
    ACTIVE_FRAC=$(echo "$LAST_QR" | awk '{print $12}')

    # Extract steady-state FPS (average of last 5 FPS readings)
    FPS=$(grep 'fps=' "$LOGFILE" | tail -5 | \
        grep -oP 'fps=\K[0-9]+' | awk '{s+=$1; n++} END{printf "%d", s/n}')

    printf "%-12s %-8s %-12s %-12s %-10s %-10s\n" \
        "$THRESH" "$FPS" "$ACTIVE_FRAC" "$SHELLS" "$R_GLOBAL" "$Q_VAL"
done

echo ""
echo "Sweep complete. Detailed logs in $OUTDIR/"
echo "To extract all QR-corr data:"
echo "  for f in $OUTDIR/thresh_*.log; do echo \"=== \$f ===\"; grep '\\[QR-corr\\]' \"\$f\"; done"
