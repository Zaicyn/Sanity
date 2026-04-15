/*
 * V21 PHYSICS DIAGNOSTICS — Research-Grade Observables
 * =====================================================
 *
 * Computes 6 astrophysical diagnostics from GPU readback data:
 *   1. Rotation curve v_circ(r) vs Keplerian prediction
 *   2. Toomre Q stability parameter
 *   3. Energy budget (KE total + per-shell)
 *   4. Pattern speed (m=3 phase tracking)
 *   5. Spiral pitch angle
 *   6. Reynolds stress (angular momentum transport)
 *
 * Called every 500 frames from the existing readback path.
 * Pure C, header-only, no heap allocation.
 *
 * License: Public domain / CC0
 */

#ifndef V21_PHYSICS_DIAG_H
#define V21_PHYSICS_DIAG_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * CONFIGURATION
 * ======================================================================== */

#define DIAG_N_RBINS      32
#define DIAG_R_MIN        2.0
#define DIAG_R_MAX        200.0
#define DIAG_BH_MASS      1.0
#define DIAG_PI           3.14159265358979
#define DIAG_MIN_COUNT    50      /* Minimum particles per bin for valid stats */

/* Shell edges for energy budget (match harmonic probe) */
#define DIAG_N_SHELLS     4
static const double diag_shell_edges[DIAG_N_SHELLS + 1] = {0.0, 15.0, 50.0, 120.0, 250.0};
static const char*  diag_shell_names[DIAG_N_SHELLS] = {"core", "inner", "mid", "outer"};

/* ========================================================================
 * STATE (persistent across readback intervals)
 * ======================================================================== */

typedef struct {
    int    initialized;
    int    frames_computed;

    /* Previous m=3 phase per shell (for pattern speed measurement) */
    double prev_phi3[DIAG_N_SHELLS];
    int    prev_frame;

    /* Previous total KE (for energy drift tracking) */
    double prev_KE_total;
} v21_physics_diag_t;

static inline void v21_physics_diag_init(v21_physics_diag_t* d) {
    d->initialized = 0;
    d->frames_computed = 0;
    d->prev_frame = 0;
    d->prev_KE_total = 0.0;
    for (int i = 0; i < DIAG_N_SHELLS; i++) d->prev_phi3[i] = 0.0;
}

/* ========================================================================
 * RADIAL BINNING — log-spaced, r=2 to r=200
 * ======================================================================== */

static inline int diag_radial_bin(double r) {
    if (r < DIAG_R_MIN || r >= DIAG_R_MAX) return -1;
    double log_ratio = log(r / DIAG_R_MIN) / log(DIAG_R_MAX / DIAG_R_MIN);
    int bin = (int)(log_ratio * DIAG_N_RBINS);
    if (bin >= DIAG_N_RBINS) bin = DIAG_N_RBINS - 1;
    if (bin < 0) bin = 0;
    return bin;
}

static inline double diag_bin_r_inner(int bin) {
    double t = (double)bin / (double)DIAG_N_RBINS;
    return DIAG_R_MIN * pow(DIAG_R_MAX / DIAG_R_MIN, t);
}

static inline double diag_bin_r_outer(int bin) {
    double t = (double)(bin + 1) / (double)DIAG_N_RBINS;
    return DIAG_R_MIN * pow(DIAG_R_MAX / DIAG_R_MIN, t);
}

static inline double diag_bin_r_center(int bin) {
    return sqrt(diag_bin_r_inner(bin) * diag_bin_r_outer(bin));
}

/* ========================================================================
 * MAIN DIAGNOSTIC COMPUTATION
 * ======================================================================== */

static inline void v21_physics_diag_compute(
    v21_physics_diag_t* diag,
    const float* r,             /* graded binding 0 */
    const float* vel_r,         /* graded binding 3 */
    const float* phi,           /* graded binding 5 */
    const float* omega_orb,     /* graded binding 6 */
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* vel_x, const float* vel_y, const float* vel_z,
    const uint8_t* flags,
    int N, int frame)
{
    /* ---- Per-bin accumulators (stack-allocated) ---- */
    int    bin_count[DIAG_N_RBINS] = {};
    double bin_sum_omega[DIAG_N_RBINS] = {};
    double bin_sum_omega2[DIAG_N_RBINS] = {};
    double bin_sum_vr[DIAG_N_RBINS] = {};
    double bin_sum_vr2[DIAG_N_RBINS] = {};
    double bin_sum_vr_vphi[DIAG_N_RBINS] = {};  /* vel_r * r * omega_orb */
    double bin_sum_v2[DIAG_N_RBINS] = {};        /* |v|^2 for KE */

    /* Per-bin m=3 Fourier for pitch angle */
    double bin_fc3[DIAG_N_RBINS] = {};
    double bin_fs3[DIAG_N_RBINS] = {};

    /* Per-shell KE and m=3 Fourier for pattern speed */
    double shell_KE[DIAG_N_SHELLS] = {};
    int    shell_count[DIAG_N_SHELLS] = {};
    double shell_fc3[DIAG_N_SHELLS] = {};
    double shell_fs3[DIAG_N_SHELLS] = {};
    double shell_sum_damp[DIAG_N_SHELLS] = {};

    double KE_total = 0.0;
    double damp_total = 0.0;

    /* ---- Single pass over all particles ---- */
    for (int i = 0; i < N; i++) {
        if (flags[i] & 0x02) continue;  /* skip ejected */

        double ri     = (double)r[i];
        double vri    = (double)vel_r[i];
        double phi_i  = (double)phi[i];
        double omg_i  = (double)omega_orb[i];
        double vx     = (double)vel_x[i];
        double vy     = (double)vel_y[i];
        double vz     = (double)vel_z[i];
        double v2     = vx*vx + vy*vy + vz*vz;
        double vphi_i = ri * omg_i;

        /* KE */
        KE_total += 0.5 * v2;

        /* Radial damping energy estimate (2%/frame of radial KE) */
        double damp_i = 0.02 * vri * vri;
        damp_total += damp_i;

        /* Radial bin */
        int b = diag_radial_bin(ri);
        if (b >= 0) {
            bin_count[b]++;
            bin_sum_omega[b]    += omg_i;
            bin_sum_omega2[b]   += omg_i * omg_i;
            bin_sum_vr[b]       += vri;
            bin_sum_vr2[b]      += vri * vri;
            bin_sum_vr_vphi[b]  += vri * vphi_i;
            bin_sum_v2[b]       += v2;

            /* m=3 Fourier per bin (for pitch angle) */
            double angle3 = 3.0 * phi_i;
            bin_fc3[b] += cos(angle3);
            bin_fs3[b] += sin(angle3);
        }

        /* Shell assignments (for energy budget + pattern speed) */
        for (int s = 0; s < DIAG_N_SHELLS; s++) {
            if (ri >= diag_shell_edges[s] && ri < diag_shell_edges[s+1]) {
                shell_KE[s] += 0.5 * v2;
                shell_count[s]++;
                shell_sum_damp[s] += damp_i;

                double angle3 = 3.0 * phi_i;
                shell_fc3[s] += cos(angle3);
                shell_fs3[s] += sin(angle3);
                break;
            }
        }
    }

    /* ================================================================
     * DIAGNOSTIC 1: Rotation Curve
     * ================================================================ */
    printf("[rot-curve] frame=%d  (32 log-spaced bins, r=%.0f..%.0f)\n",
           frame, DIAG_R_MIN, DIAG_R_MAX);
    double mean_omega[DIAG_N_RBINS] = {};
    for (int b = 0; b < DIAG_N_RBINS; b++) {
        if (bin_count[b] < DIAG_MIN_COUNT) continue;
        double rc = diag_bin_r_center(b);
        double inv_n = 1.0 / bin_count[b];
        mean_omega[b] = bin_sum_omega[b] * inv_n;
        double v_circ = rc * mean_omega[b];
        double v_kep  = sqrt(DIAG_BH_MASS / rc);
        double dev    = (v_kep > 1e-12) ? (v_circ / v_kep - 1.0) : 0.0;
        printf("[rot-curve]   bin=%2d r=%6.1f N=%5d v_circ=%7.4f v_kep=%7.4f dev=%+.4f\n",
               b, rc, bin_count[b], v_circ, v_kep, dev);
    }

    /* ================================================================
     * DIAGNOSTIC 2: Toomre Q(r)
     * ================================================================ */
    printf("[toomre-Q]  frame=%d\n", frame);
    for (int b = 0; b < DIAG_N_RBINS; b++) {
        if (bin_count[b] < DIAG_MIN_COUNT) continue;
        double rc    = diag_bin_r_center(b);
        double r_in  = diag_bin_r_inner(b);
        double r_out = diag_bin_r_outer(b);
        double dr    = r_out - r_in;
        double inv_n = 1.0 / bin_count[b];

        /* Surface density: count / annulus area */
        double Sigma = bin_count[b] / (2.0 * DIAG_PI * rc * dr);

        /* Radial velocity dispersion */
        double mean_vr = bin_sum_vr[b] * inv_n;
        double var_vr  = bin_sum_vr2[b] * inv_n - mean_vr * mean_vr;
        if (var_vr < 0.0) var_vr = 0.0;
        double sigma_r = sqrt(var_vr);

        /* Epicyclic frequency from rotation curve: kappa^2 = 4*omega^2 + 2*omega*r*(d_omega/dr)
         * Use centered finite difference on mean_omega[] */
        double omega_c = mean_omega[b];
        double d_omega_dr = 0.0;
        if (b > 0 && b < DIAG_N_RBINS - 1 &&
            bin_count[b-1] >= DIAG_MIN_COUNT && bin_count[b+1] >= DIAG_MIN_COUNT) {
            double r_prev = diag_bin_r_center(b - 1);
            double r_next = diag_bin_r_center(b + 1);
            d_omega_dr = (mean_omega[b+1] - mean_omega[b-1]) / (r_next - r_prev);
        } else if (b > 0 && bin_count[b-1] >= DIAG_MIN_COUNT) {
            double r_prev = diag_bin_r_center(b - 1);
            d_omega_dr = (mean_omega[b] - mean_omega[b-1]) / (rc - r_prev);
        } else if (b < DIAG_N_RBINS - 1 && bin_count[b+1] >= DIAG_MIN_COUNT) {
            double r_next = diag_bin_r_center(b + 1);
            d_omega_dr = (mean_omega[b+1] - mean_omega[b]) / (r_next - rc);
        }

        double kappa2 = 4.0 * omega_c * omega_c + 2.0 * omega_c * rc * d_omega_dr;
        if (kappa2 < 0.0) kappa2 = 0.0;
        double kappa = sqrt(kappa2);

        /* G_eff for point-mass potential */
        double G_eff_Sigma = DIAG_BH_MASS / (rc * rc) * Sigma;
        double Q = (G_eff_Sigma > 1e-12) ? kappa * sigma_r / (DIAG_PI * G_eff_Sigma) : 999.0;

        printf("[toomre-Q]    bin=%2d r=%6.1f Sigma=%8.1f sigma_r=%7.4f kappa=%7.4f Q=%6.2f\n",
               b, rc, Sigma, sigma_r, kappa, Q);
    }

    /* ================================================================
     * DIAGNOSTIC 3: Energy Budget
     * ================================================================ */
    double dKE = (diag->initialized) ? (KE_total - diag->prev_KE_total) : 0.0;
    int d_frames = (diag->initialized) ? (frame - diag->prev_frame) : 1;
    double dKE_per_frame = (d_frames > 0) ? dKE / d_frames : 0.0;

    printf("[energy]    frame=%d  KE_total=%.4e", frame, KE_total);
    for (int s = 0; s < DIAG_N_SHELLS; s++) {
        printf("  %s=%.3e", diag_shell_names[s], shell_KE[s]);
    }
    printf("\n");
    printf("[energy]    frame=%d  damp_est=%.4e  dKE/frame=%+.4e  dKE_total=%+.4e (%d frames)\n",
           frame, damp_total, dKE_per_frame, dKE, d_frames);

    /* ================================================================
     * DIAGNOSTIC 4: Pattern Speed
     * ================================================================ */
    if (diag->initialized) {
        printf("[pattern-speed] frame=%d  (d_frame=%d, input omega_p=0.004)\n",
               frame, d_frames);
        for (int s = 0; s < DIAG_N_SHELLS; s++) {
            if (shell_count[s] < DIAG_MIN_COUNT) continue;

            double phi3_now = atan2(shell_fs3[s], shell_fc3[s]);
            double phi3_prev = diag->prev_phi3[s];

            /* Phase unwrap */
            double dphi = phi3_now - phi3_prev;
            while (dphi >  DIAG_PI) dphi -= 2.0 * DIAG_PI;
            while (dphi < -DIAG_PI) dphi += 2.0 * DIAG_PI;

            /* omega_p = dphi / (3 * d_frame) because m=3 mode phase = 3*omega_p*t */
            double omega_p_meas = dphi / (3.0 * d_frames);

            printf("[pattern-speed]   %5s  omega_p=%+.6f  dphi3=%+.4f\n",
                   diag_shell_names[s], omega_p_meas, dphi);
        }
    } else {
        printf("[pattern-speed] frame=%d  (first readback, no previous phase — skipped)\n", frame);
    }

    /* ================================================================
     * DIAGNOSTIC 5: Spiral Pitch Angle
     * ================================================================ */
    printf("[pitch]     frame=%d\n", frame);
    double bin_phi3[DIAG_N_RBINS];
    int    bin_phi3_valid[DIAG_N_RBINS];
    for (int b = 0; b < DIAG_N_RBINS; b++) {
        if (bin_count[b] >= DIAG_MIN_COUNT) {
            bin_phi3[b] = atan2(bin_fs3[b], bin_fc3[b]);
            bin_phi3_valid[b] = 1;
        } else {
            bin_phi3[b] = 0.0;
            bin_phi3_valid[b] = 0;
        }
    }

    /* Compute pitch angle from d(phi_3)/d(ln r) using centered finite differences */
    for (int b = 1; b < DIAG_N_RBINS - 1; b++) {
        if (!bin_phi3_valid[b-1] || !bin_phi3_valid[b] || !bin_phi3_valid[b+1]) continue;

        double ln_r_prev = log(diag_bin_r_center(b - 1));
        double ln_r_next = log(diag_bin_r_center(b + 1));

        /* Unwrap phase difference */
        double dphi = bin_phi3[b+1] - bin_phi3[b-1];
        while (dphi >  DIAG_PI) dphi -= 2.0 * DIAG_PI;
        while (dphi < -DIAG_PI) dphi += 2.0 * DIAG_PI;

        double d_ln_r = ln_r_next - ln_r_prev;
        double tan_pitch = dphi / d_ln_r;
        double pitch_deg = atan(tan_pitch) * 180.0 / DIAG_PI;

        double rc = diag_bin_r_center(b);
        printf("[pitch]       bin=%2d r=%6.1f pitch=%+6.1f deg  (tan=%.3f)\n",
               b, rc, pitch_deg, tan_pitch);
    }

    /* ================================================================
     * DIAGNOSTIC 6: Reynolds Stress (Angular Momentum Transport)
     * ================================================================ */
    printf("[reynolds]  frame=%d\n", frame);
    for (int b = 0; b < DIAG_N_RBINS; b++) {
        if (bin_count[b] < DIAG_MIN_COUNT) continue;
        double rc = diag_bin_r_center(b);
        double inv_n = 1.0 / bin_count[b];

        /* <vel_r * v_phi> - <vel_r>*<v_phi> (fluctuation correlation) */
        double mean_vr_vphi = bin_sum_vr_vphi[b] * inv_n;
        double mean_vr      = bin_sum_vr[b] * inv_n;
        double mean_vphi    = rc * mean_omega[b];
        double stress = mean_vr_vphi - mean_vr * mean_vphi;

        printf("[reynolds]    bin=%2d r=%6.1f stress=%+.6f  %s\n",
               b, rc, stress, (stress > 0.0) ? "(outward)" : "(inward)");
    }

    /* ---- Update persistent state ---- */
    for (int s = 0; s < DIAG_N_SHELLS; s++) {
        if (shell_count[s] >= DIAG_MIN_COUNT) {
            diag->prev_phi3[s] = atan2(shell_fs3[s], shell_fc3[s]);
        }
    }
    diag->prev_KE_total = KE_total;
    diag->prev_frame = frame;
    diag->initialized = 1;
    diag->frames_computed++;
}

/* ========================================================================
 * SUMMARY — call at shutdown
 * ======================================================================== */

static inline void v21_physics_diag_summary(const v21_physics_diag_t* d) {
    printf("[phys-diag] Summary: %d diagnostic snapshots computed\n",
           d->frames_computed);
}

#ifdef __cplusplus
}
#endif

#endif /* V21_PHYSICS_DIAG_H */
