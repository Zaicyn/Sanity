#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Allocator constants */
#define SQ2_TORUS_SHELLS    2
#define SQ2_TORUS_BINS      8
#define SQ2_TORUS_RING_SIZE 32
#define SQ2_TORUS_TOTAL     (SQ2_TORUS_SHELLS * SQ2_TORUS_BINS * SQ2_TORUS_RING_SIZE) /* 512 */

/* Cartesian grid */
#define GRID_DIM       64
#define GRID_CELLS     (GRID_DIM * GRID_DIM * GRID_DIM) /* 262144 */
#define GRID_HALF_SIZE 250.0f
#define GRID_CELL_SIZE (2.0f * GRID_HALF_SIZE / GRID_DIM)

/* Sim params */
#define N_PARTICLES    100000
#define DISK_OUTER_R   1200.0f
#define BH_MASS        100.0f

/* Squaragon scatter LUT */
static const uint8_t SQ2_SCATTER_LUT[32] = {
    6, 4, 6, 2, 7, 4, 3, 0, 2, 0, 3, 4, 7, 2, 6, 4,
    6, 4, 6, 2, 7, 4, 3, 0, 2, 0, 3, 4, 7, 2, 6, 4
};

typedef struct {
    float x, y, z;
    float theta;
    uint32_t cartesian_cell;
    uint32_t torus_id;
} Particle;

uint32_t cartesian_cell(float px, float py, float pz) {
    float inv_h = 1.0f / GRID_CELL_SIZE;
    uint32_t cx = (uint32_t)fminf(fmaxf((px + GRID_HALF_SIZE) * inv_h, 0.0f), GRID_DIM - 1);
    uint32_t cy = (uint32_t)fminf(fmaxf((py + GRID_HALF_SIZE) * inv_h, 0.0f), GRID_DIM - 1);
    uint32_t cz = (uint32_t)fminf(fmaxf((pz + GRID_HALF_SIZE) * inv_h, 0.0f), GRID_DIM - 1);
    return cx + cy * GRID_DIM + cz * GRID_DIM * GRID_DIM;
}

uint32_t torus_bucket(float px, float py, float pz, float theta) {
    /* Shell: inner (r < median_r) vs outer */
    float r = sqrtf(px*px + pz*pz);
    uint32_t shell = (r > 200.0f) ? 1 : 0;  /* rough median split */

    /* Bin: angular sector from Viviani scatter */
    float angle = atan2f(pz, px) + 3.14159265f; /* [0, 2π) */
    uint32_t angle_idx = (uint32_t)(angle * (32.0f / 6.28318f)) & 31;
    uint32_t bin = SQ2_SCATTER_LUT[angle_idx];

    /* Ring: fiber phase position */
    float wrapped = fmodf(theta, 6.28318f);
    if (wrapped < 0) wrapped += 6.28318f;
    uint32_t ring = (uint32_t)(wrapped * (32.0f / 6.28318f)) & 31;

    return (shell * SQ2_TORUS_BINS * SQ2_TORUS_RING_SIZE)
         + (bin * SQ2_TORUS_RING_SIZE)
         + ring;
}

int main(void) {
    srand(42);

    Particle *p = calloc(N_PARTICLES, sizeof(Particle));

    /* Generate particles like the sim does */
    for (int i = 0; i < N_PARTICLES; i++) {
        float x = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float y = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R * 0.3f;
        float z = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * DISK_OUTER_R;
        float r = sqrtf(x*x + y*y + z*z);
        if (r < 1.0f) { float s = 1.0f/r; x*=s; y*=s; z*=s; }

        p[i].x = x;
        p[i].y = y;
        p[i].z = z;
        p[i].theta = atan2f(z, x) + ((float)rand()/RAND_MAX) * 0.2f;
        p[i].cartesian_cell = cartesian_cell(x, y, z);
        p[i].torus_id = torus_bucket(x, y, z, p[i].theta);
    }

    /* === Stats for Cartesian cells === */
    uint32_t *cart_count = calloc(GRID_CELLS, sizeof(uint32_t));
    double   *cart_r_sum = calloc(GRID_CELLS, sizeof(double));
    double   *cart_r_sq  = calloc(GRID_CELLS, sizeof(double));

    for (int i = 0; i < N_PARTICLES; i++) {
        uint32_t c = p[i].cartesian_cell;
        if (c < GRID_CELLS) {
            float r = sqrtf(p[i].x*p[i].x + p[i].z*p[i].z);
            cart_count[c]++;
            cart_r_sum[c] += r;
            cart_r_sq[c]  += r * r;
        }
    }

    int cart_occupied = 0;
    double cart_total_spread = 0.0;
    int cart_max_count = 0;
    for (uint32_t c = 0; c < GRID_CELLS; c++) {
        if (cart_count[c] > 0) {
            cart_occupied++;
            if ((int)cart_count[c] > cart_max_count) cart_max_count = cart_count[c];
            double mean_r = cart_r_sum[c] / cart_count[c];
            double var_r = cart_r_sq[c] / cart_count[c] - mean_r * mean_r;
            if (var_r > 0) cart_total_spread += sqrtf(var_r);
        }
    }

    /* === Stats for Torus buckets === */
    uint32_t *torus_count = calloc(SQ2_TORUS_TOTAL, sizeof(uint32_t));
    double   *torus_r_sum = calloc(SQ2_TORUS_TOTAL, sizeof(double));
    double   *torus_r_sq  = calloc(SQ2_TORUS_TOTAL, sizeof(double));
    /* Also track spatial spread (how many unique Cartesian cells per torus) */
    uint32_t **torus_cells = calloc(SQ2_TORUS_TOTAL, sizeof(uint32_t*));
    uint32_t *torus_ncells = calloc(SQ2_TORUS_TOTAL, sizeof(uint32_t));

    for (int i = 0; i < N_PARTICLES; i++) {
        uint32_t t = p[i].torus_id;
        if (t < SQ2_TORUS_TOTAL) {
            float r = sqrtf(p[i].x*p[i].x + p[i].z*p[i].z);
            torus_count[t]++;
            torus_r_sum[t] += r;
            torus_r_sq[t]  += r * r;
        }
    }

    /* Count unique Cartesian cells per torus */
    for (int i = 0; i < N_PARTICLES; i++) {
        uint32_t t = p[i].torus_id;
        uint32_t c = p[i].cartesian_cell;
        if (t >= SQ2_TORUS_TOTAL) continue;

        /* Simple linear scan (OK for 100K) */
        int found = 0;
        for (uint32_t j = 0; j < torus_ncells[t]; j++) {
            if (torus_cells[t][j] == c) { found = 1; break; }
        }
        if (!found) {
            torus_cells[t] = realloc(torus_cells[t], (torus_ncells[t]+1) * sizeof(uint32_t));
            torus_cells[t][torus_ncells[t]++] = c;
        }
    }

    int torus_occupied = 0;
    double torus_total_spread = 0.0;
    int torus_max_count = 0;
    double torus_avg_cells = 0.0;
    int torus_max_cells = 0;
    for (uint32_t t = 0; t < SQ2_TORUS_TOTAL; t++) {
        if (torus_count[t] > 0) {
            torus_occupied++;
            if ((int)torus_count[t] > torus_max_count) torus_max_count = torus_count[t];
            double mean_r = torus_r_sum[t] / torus_count[t];
            double var_r = torus_r_sq[t] / torus_count[t] - mean_r * mean_r;
            if (var_r > 0) torus_total_spread += sqrtf(var_r);
            torus_avg_cells += torus_ncells[t];
            if ((int)torus_ncells[t] > torus_max_cells) torus_max_cells = torus_ncells[t];
        }
    }
    torus_avg_cells /= (torus_occupied > 0 ? torus_occupied : 1);

    printf("=== CARTESIAN GRID (64^3 = %d cells) ===\n", GRID_CELLS);
    printf("  Occupied cells:      %d / %d (%.1f%%)\n",
           cart_occupied, GRID_CELLS, 100.0*cart_occupied/GRID_CELLS);
    printf("  Particles/cell:      avg=%.1f  max=%d\n",
           (float)N_PARTICLES / cart_occupied, cart_max_count);
    printf("  Radial spread/cell:  avg=%.2f units\n",
           cart_total_spread / cart_occupied);
    printf("\n");

    printf("=== TORUS BUCKETS (%d buckets = %d×%d×%d) ===\n",
           SQ2_TORUS_TOTAL, SQ2_TORUS_SHELLS, SQ2_TORUS_BINS, SQ2_TORUS_RING_SIZE);
    printf("  Occupied buckets:    %d / %d (%.1f%%)\n",
           torus_occupied, SQ2_TORUS_TOTAL, 100.0*torus_occupied/SQ2_TORUS_TOTAL);
    printf("  Particles/bucket:    avg=%.1f  max=%d\n",
           (float)N_PARTICLES / torus_occupied, torus_max_count);
    printf("  Radial spread/bucket: avg=%.2f units\n",
           torus_total_spread / torus_occupied);
    printf("  Cartesian cells/bucket: avg=%.1f  max=%d\n",
           torus_avg_cells, torus_max_cells);
    printf("\n");

    printf("=== COMPARISON ===\n");
    printf("  Grid resolution:     %d cells vs %d buckets (%.0fx ratio)\n",
           GRID_CELLS, SQ2_TORUS_TOTAL, (float)GRID_CELLS / SQ2_TORUS_TOTAL);
    printf("  Stencil work:        %d cells vs %d buckets (%.0fx less work)\n",
           GRID_CELLS, SQ2_TORUS_TOTAL, (float)GRID_CELLS / SQ2_TORUS_TOTAL);
    printf("  Scatter atomics:     %d targets vs %d targets\n",
           cart_occupied, torus_occupied);

    /* Cleanup */
    for (uint32_t t = 0; t < SQ2_TORUS_TOTAL; t++) free(torus_cells[t]);
    free(torus_cells); free(torus_ncells);
    free(cart_count); free(cart_r_sum); free(cart_r_sq);
    free(torus_count); free(torus_r_sum); free(torus_r_sq);
    free(p);

    return 0;
}
