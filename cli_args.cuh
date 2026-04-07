// cli_args.cuh — Command Line Argument Parsing
// =============================================
// Parses all --flag and -n arguments. Returns the requested particle count.
// Requires sim_globals.h to be included first (all g_* globals).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Parse command-line arguments and set global state.
// Returns requested particle count (caller should clamp to VRAM cap).
// Returns 0 if --help was printed (caller should exit).
inline int parseCLI(int argc, char** argv) {
    int num_particles = 3500000;  // 3.5M particles for full resolution

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            num_particles = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--no-topology") == 0 || strcmp(argv[i], "--smooth") == 0) {

            g_use_hopfion_topology = false;
        }
        else if (strcmp(argv[i], "--topology") == 0 || strcmp(argv[i], "--hopfion") == 0) {

            g_use_hopfion_topology = true;
        }
        else if (strcmp(argv[i], "--discrete-arms") == 0 || strcmp(argv[i], "--arm-topology") == 0) {


            g_use_arm_topology = true;
            g_enable_arms = true;
        }
        else if (strcmp(argv[i], "--smooth-arms") == 0 || strcmp(argv[i], "--no-arm-topology") == 0) {


            g_use_arm_topology = false;
            g_enable_arms = true;
        }
        else if (strcmp(argv[i], "--no-arms") == 0) {

            g_enable_arms = false;
        }
        else if (strcmp(argv[i], "--shear-k") == 0 && i+1 < argc) {

            g_shear_k = (float)atof(argv[++i]);
            printf("[shear] Phase-misalignment shear coefficient: %.4f\n", g_shear_k);
        }
        else if (strcmp(argv[i], "--kuramoto-k") == 0 && i+1 < argc) {

            g_kuramoto_k = (float)atof(argv[++i]);
            printf("[kuramoto] Coupling strength K: %.4f\n", g_kuramoto_k);
        }
        else if (strcmp(argv[i], "--omega-base") == 0 && i+1 < argc) {

            g_omega_base = (float)atof(argv[++i]);
            printf("[kuramoto] Natural frequency ω₀: %.4f\n", g_omega_base);
        }
        else if (strcmp(argv[i], "--omega-spread") == 0 && i+1 < argc) {

            g_omega_spread = (float)atof(argv[++i]);
            printf("[kuramoto] Frequency spread σ: %.4f\n", g_omega_spread);
        }
        else if (strcmp(argv[i], "--no-n12") == 0) {

            g_n12_envelope = false;
            printf("[kuramoto] N12 envelope DISABLED (constant K)\n");
        }
        else if (strcmp(argv[i], "--envelope-scale") == 0 && i+1 < argc) {

            g_envelope_scale = (float)atof(argv[++i]);
            printf("[kuramoto] Envelope harmonic scale: %.3f (period = 2π/%.3f)\n",
                   g_envelope_scale, g_envelope_scale);
        }
        else if (strcmp(argv[i], "--corner-threshold") == 0 && i+1 < argc) {

            g_corner_threshold = (float)atof(argv[++i]);
            printf("[passive] Corner threshold: %.4f\n", g_corner_threshold);
        }
        else if (strcmp(argv[i], "--passive-tau") == 0 && i+1 < argc) {

            g_passive_residual_tau = (float)atof(argv[++i]);
            printf("[passive] Residual decay tau: %.2f\n", g_passive_residual_tau);
        }
        else if (strcmp(argv[i], "--shell-init") == 0) {

            g_shell_init = true;
            printf("[init] Shell-aware initialization: particles ON resonance shells\n");
        }
        else if (strcmp(argv[i], "--shell-lensing") == 0) {

            g_shell_lensing = true;
            printf("[render] Shell lensing ENABLED\n");
        }
        else if (strcmp(argv[i], "--ghost") == 0) {
            g_ghost_projection = true;
            printf("[render] Ghost projection ENABLED (transport channel visible)\n");
        }
        else if (strcmp(argv[i], "--no-ghost") == 0) {
            g_ghost_projection = false;
            printf("[render] Ghost projection DISABLED\n");
        }
        else if (strcmp(argv[i], "--no-spawn") == 0) {

            g_spawn_enabled = false;
            printf("[spawn] Natural growth DISABLED — particle count locked\n");
        }
        else if (strcmp(argv[i], "--qr-corr") == 0) {

            g_qr_corr_log = true;
            printf("[qr-corr] Kuramoto × topology correlation dump ENABLED\n");
        }
        else if (strcmp(argv[i], "--retrograde") == 0) {

            g_retrograde_init = true;
            printf("[init] Retrograde initial rotation (counterclockwise → clockwise)\n");
        }
        else if (strcmp(argv[i], "--r-export-interval") == 0 && i+1 < argc) {

            g_r_export_interval = atoi(argv[++i]);
            printf("[kuramoto] Per-cell R export every %d frames\n", g_r_export_interval);
        }
        else if (strcmp(argv[i], "--r-log-interval") == 0 && i+1 < argc) {

            g_r_log_interval = atoi(argv[++i]);
            printf("[kuramoto] Dense R(t) logging every %d frames\n", g_r_log_interval);
        }
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc) {

            g_rng_seed = (unsigned int)atoi(argv[++i]);
            printf("[rng] Initial-condition seed: %u\n", g_rng_seed);
        }
        else if (strcmp(argv[i], "--headless") == 0) {

            g_headless = true;
        }
        else if (strcmp(argv[i], "--target-radius") == 0 && i+1 < argc) {


            g_terminate_on_radius = true;
            g_target_ring_radius = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc) {

            g_target_frames = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--test-residence") == 0) {

            g_test_residence_time = true;
        }
        else if (strcmp(argv[i], "--matched-amplitude") == 0) {


            g_matched_amplitude = true;
            g_arm_boost_override = 1.25f;  // Match smooth amplitude
        }
        else if (strcmp(argv[i], "--hybrid") == 0) {

            g_hybrid_lod = true;
        }
        else if (strcmp(argv[i], "--octree-render") == 0) {

            g_octree_render = true;
        }
        else if (strcmp(argv[i], "--octree-physics") == 0) {

            g_octree_physics = true;
        }
        else if (strcmp(argv[i], "--no-octree-physics") == 0) {

            g_octree_physics = false;
        }
        else if (strcmp(argv[i], "--no-octree-phase") == 0) {

            g_octree_phase = false;
            printf("[config] Octree phase evolution DISABLED (mip-tree only)\n");
        }
        else if (strcmp(argv[i], "--octree-phase") == 0) {

            g_octree_phase = true;
        }
        else if (strcmp(argv[i], "--octree-rebuild") == 0) {

            g_octree_rebuild = true;
            printf("[config] Octree rebuild ENABLED (Morton sort + stochastic tree)\n");
        }
        else if (strcmp(argv[i], "--no-octree-rebuild") == 0) {

            g_octree_rebuild = false;
        }
        else if (strcmp(argv[i], "--predictive-lock") == 0) {

            g_predictive_locking = true;
            printf("[config] Predictive locking ENABLED (skip mip-tree when shells locked)\n");
        }
        else if (strcmp(argv[i], "--no-predictive-lock") == 0) {

            g_predictive_locking = false;
            printf("[config] Predictive locking DISABLED (always rebuild mip-tree)\n");
        }
        else if (strcmp(argv[i], "--active-compact") == 0) {

            g_active_compaction = true;
            printf("[config] Active particle compaction ENABLED (skip static shell mass)\n");
        }
        else if (strcmp(argv[i], "--no-active-compact") == 0) {

            g_active_compaction = false;
            printf("[config] Active particle compaction DISABLED (scatter all particles)\n");
        }
        else if (strcmp(argv[i], "--grid-physics") == 0) {


            g_grid_physics = true;
            g_octree_physics = false;  // Grid replaces octree physics
        }
        else if (strcmp(argv[i], "--grid-flags") == 0) {



            g_grid_physics = true;
            g_grid_flags = true;
            g_octree_physics = false;  // Grid replaces octree physics
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <num>           Number of particles (default: 3500000)\n");
            printf("  --topology         Start with hopfion topology (default)\n");
            printf("  --hopfion          Alias for --topology\n");
            printf("  --no-topology      Start with smooth gradient (no discrete shells)\n");
            printf("  --smooth           Alias for --no-topology\n");
            printf("  --discrete-arms    Start with discrete arm boundaries (default)\n");
            printf("  --arm-topology     Alias for --discrete-arms\n");
            printf("  --smooth-arms      Start with smooth arm density waves\n");
            printf("  --no-arm-topology  Alias for --smooth-arms\n");
            printf("  --no-arms          Disable spiral arm structure entirely\n");
            printf("  --shear-k <k>      Frictional shear: density × sin(2φ) hybrid (default 0, try 2-10; scale-invariant)\n");
            printf("  --kuramoto-k <K>   Kuramoto phase coupling strength (default 0, sweep 0-2 to find K_c)\n");
            printf("  --omega-base <ω₀>  Mean natural frequency (default 1.0)\n");
            printf("  --omega-spread <σ> Gaussian σ for ω distribution (default 0.05; K_c ≈ 2σ/π)\n");
            printf("  --no-n12           Disable N12 mixer envelope on Kuramoto coupling\n");
            printf("  --r-export-interval <N>  Dump per-cell R grid to r_export/frame_NNNNN.bin every N frames (0=off)\n");
            printf("  --r-log-interval <N>     Print dense R(t) samples every N frames for time-series analysis (0=off)\n");
            printf("  --corner-threshold <f>   Passive/active pump_residual threshold (default 0.15)\n");
            printf("  --passive-tau <f>        Passive residual decay tau in sim-time units (default 5.0)\n");
            printf("  --shell-init             Initialize particles ON resonance shells (skip settling transient)\n");
            printf("  --shell-lensing          Enable gravitational lensing distortion at shell boundaries\n");
            printf("  --no-spawn               Disable natural growth — particle count locked for clean measurements\n");
            printf("  --qr-corr                Dump [QR-corr] CSV rows each stats frame: R, Rrec, peaks, mass_frac, Q, shells, ..., active_frac\n");
            printf("  --headless         Disable rendering (physics + logging only, 10-20x speedup)\n");
            printf("  --hybrid           Enable hybrid LOD rendering (experimental)\n");
            printf("  --octree-render    Use octree traversal for render compaction\n");
            printf("  --octree-physics   Enable XOR neighbor stress physics (default: on)\n");
            printf("  --no-octree-physics Disable octree neighbor physics\n");
            printf("  --no-octree-phase  Disable octree phase evolution (mip-tree only)\n");
            printf("  --octree-rebuild   Enable Morton sort + octree rebuild (default: OFF)\n");
            printf("  --predictive-lock  Enable predictive locking (skip mip-tree when locked, default: ON)\n");
            printf("  --no-predictive-lock Disable predictive locking (always rebuild mip-tree)\n");
            printf("  --active-compact   Enable active particle compaction (skip static shell mass, default: ON)\n");
            printf("  --no-active-compact Disable active particle compaction (scatter all particles)\n");
            printf("  --grid-physics     Use streaming cell grid (DNA/RNA 30-frame cadence)\n");
            printf("  --grid-flags       Use sparse flags (optimal: no lists, no sort, no dedup)\n");
            printf("  --target-radius <R> Terminate when photon ring reaches radius R (instead of frame limit)\n");
            printf("  --frames <N>       Terminate after N frames (default: 50000)\n");
            printf("\nTest Suite (The Final Trilogy):\n");
            printf("  --test-residence   Test A: Track residence time (arm vs gap accumulation)\n");
            printf("  --matched-amplitude Test C: Set discrete boost=1.25× (isolate topology from amplitude)\n");
            printf("  --help, -h         Show this help message\n");
            printf("\nControls:\n");
            printf("  H key              Toggle radial topology mode at runtime\n");
            printf("  A key              Toggle arm topology mode at runtime\n");
            printf("  L key              Toggle hybrid LOD culling (requires --hybrid)\n");
            printf("  E key              Inject entropy cluster\n");
            printf("  C key              Cycle color modes\n");
            printf("  Space              Pause/unpause\n");
            printf("  R key              Reset camera\n");
            return 0;
        }
    }

    return num_particles;
}

// End of cli_args.cuh
