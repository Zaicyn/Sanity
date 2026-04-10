/*
 * BLACKHOLE V21 — Vendor-Independent Galaxy Simulator
 * ====================================================
 *
 * Pure C main loop. Zero CUDA, zero vendor lock-in.
 * Physics runs via SPIRV compute shaders on any Vulkan GPU.
 * CPU fallback via V21 allocator + OpenMP (future).
 *
 * Build:
 *   cd V21/build && cmake .. && make blackhole_v21
 *
 * Run:
 *   ./blackhole_v21 -n 1000 --frames 5000
 *   ./blackhole_v21 -n 3 --frames 50000    # 3 seeds → galaxy
 *
 * Architecture:
 *   GLSL source → SPIRV binary → Vulkan compute dispatch
 *   V21 backend interface routes alloc/free/launch
 *   SimulationContext bundles all state (void* pointers)
 *
 * License: Public domain / CC0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core/v21_types.h"
#include "core/v21_mem.h"
#include "core/v21_alloc_cpu.h"
#include "core/v21_geometry.h"
#include "core/v21_backend.h"
#include "core/v21_spirv.h"
#include "core/v21_accumulate.h"

/* ========================================================================
 * SIMULATION PARAMETERS
 * ======================================================================== */

#define DEFAULT_PARTICLES    1000
#define DEFAULT_FRAMES       5000
#define DEFAULT_SEED         42
#define DEFAULT_DT           (1.0f / 60.0f)

#define BH_MASS              100.0f
#define ISCO_R               6.0f
#define DISK_OUTER_R         120.0f

/* ========================================================================
 * PARTICLE STATE — Structure of Arrays (SoA) on the host
 * ======================================================================== */

typedef struct {
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* pump_scale;
    float* pump_residual;
    float* pump_history;
    int*   pump_state;
    float* theta;
    float* omega_nat;
    uint8_t* flags;
    uint8_t* topo_state;
    int    N;
    int    capacity;
} ParticleState;

/* ========================================================================
 * INITIALIZATION
 * ======================================================================== */

static void init_particles(ParticleState* ps, int N, unsigned int seed) {
    ps->N = N;
    ps->capacity = N * 100;  /* Room for growth */
    int cap = ps->capacity;

    ps->pos_x        = (float*)calloc(cap, sizeof(float));
    ps->pos_y        = (float*)calloc(cap, sizeof(float));
    ps->pos_z        = (float*)calloc(cap, sizeof(float));
    ps->vel_x        = (float*)calloc(cap, sizeof(float));
    ps->vel_y        = (float*)calloc(cap, sizeof(float));
    ps->vel_z        = (float*)calloc(cap, sizeof(float));
    ps->pump_scale   = (float*)calloc(cap, sizeof(float));
    ps->pump_residual= (float*)calloc(cap, sizeof(float));
    ps->pump_history = (float*)calloc(cap, sizeof(float));
    ps->pump_state   = (int*)  calloc(cap, sizeof(int));
    ps->theta        = (float*)calloc(cap, sizeof(float));
    ps->omega_nat    = (float*)calloc(cap, sizeof(float));
    ps->flags        = (uint8_t*)calloc(cap, sizeof(uint8_t));
    ps->topo_state   = (uint8_t*)calloc(cap, sizeof(uint8_t));

    srand(seed);

    float box_half = DISK_OUTER_R;
    float box_height = box_half * 0.3f;

    for (int i = 0; i < N; i++) {
        /* Uniform box initialization */
        float x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * box_half;
        float y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * box_height;
        float z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * box_half;

        float r = sqrtf(x*x + y*y + z*z);
        if (r < 6.0f) { float s = 6.0f / r; x *= s; y *= s; z *= s; }

        ps->pos_x[i] = x;
        ps->pos_y[i] = y;
        ps->pos_z[i] = z;

        /* Full Keplerian tangential velocity */
        float r_xz = sqrtf(x*x + z*z);
        if (r_xz > 0.1f) {
            float v_rot = sqrtf(BH_MASS / fmaxf(r_xz, ISCO_R));
            ps->vel_x[i] = -v_rot * (z / r_xz);
            ps->vel_y[i] = 0.0f;
            ps->vel_z[i] =  v_rot * (x / r_xz);
        }

        ps->pump_scale[i] = 1.0f;
        ps->pump_history[i] = 0.0f;  /* Analog nullable: 0 = uninitialized */
        ps->flags[i] = 0x01;  /* PFLAG_ACTIVE */
        ps->theta[i] = (float)rand() / RAND_MAX * V21_TWO_PI;
        ps->omega_nat[i] = 0.1f + ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

        /* Hopfion: random single-axis state */
        int axis = rand() % 4;
        int sign = (rand() % 2) ? 1 : -1;
        uint8_t topo = 0;
        uint8_t bits = (sign > 0) ? 1 : 2;
        topo |= (bits << (axis * 2));
        ps->topo_state[i] = topo;
    }

    printf("[v21-init] %d particles, box %.0fx%.0fx%.0f, full Keplerian\n",
           N, box_half*2, box_height*2, box_half*2);
}

static void free_particles(ParticleState* ps) {
    free(ps->pos_x); free(ps->pos_y); free(ps->pos_z);
    free(ps->vel_x); free(ps->vel_y); free(ps->vel_z);
    free(ps->pump_scale); free(ps->pump_residual); free(ps->pump_history);
    free(ps->pump_state); free(ps->theta); free(ps->omega_nat);
    free(ps->flags); free(ps->topo_state);
}

/* ========================================================================
 * VIVIANI FIELD FORCE — the one primitive (CPU version)
 * ======================================================================== */

static void apply_viviani_field(
    float px, float py, float pz,
    float r3d, float inv_r3d,
    float* ax, float* ay, float* az)
{
    float r_safe = fmaxf(r3d, 1.0f);
    float inv_r = (r3d >= 1.0f) ? inv_r3d : (1.0f / r_safe);

    float theta = atan2f(pz, px);

    /* Viviani tangent at this phase */
    float c1 = cosf(theta), s1 = sinf(theta);
    float c3 = cosf(3.0f * theta), s3 = sinf(3.0f * theta);

    float fx = c1 - 1.5f * c3;
    float fy = s1 - 1.5f * s3;
    float fz = -s1 * c3 - 3.0f * c1 * s3;

    float inv_f = 1.0f / sqrtf(fx*fx + fy*fy + fz*fz + 1e-8f);
    fx *= inv_f; fy *= inv_f; fz *= inv_f;

    float weight = 1.0f / (1.0f + r_safe * r_safe / 100.0f);

    float rx = px * inv_r, ry = py * inv_r, rz = pz * inv_r;

    float f_dot_r = fx*rx + fy*ry + fz*rz;
    float tx = fx - f_dot_r * rx;
    float ty = fy - f_dot_r * ry;
    float tz = fz - f_dot_r * rz;

    /* Centripetal */
    *ax += -rx * weight;
    *ay += -ry * weight;
    *az += -rz * weight;

    /* Tangential */
    *ax += tx * weight * 0.5f;
    *ay += ty * weight * 0.5f;
    *az += tz * weight * 0.5f;
}

/* ========================================================================
 * PHYSICS STEP — one frame of simulation (CPU reference implementation)
 * ======================================================================== */

static void physics_step(ParticleState* ps, float dt, float time) {
    int N = ps->N;

    for (int i = 0; i < N; i++) {
        if (!(ps->flags[i] & 0x01)) continue;  /* Skip inactive */

        float px = ps->pos_x[i], py = ps->pos_y[i], pz = ps->pos_z[i];
        float vx = ps->vel_x[i], vy = ps->vel_y[i], vz = ps->vel_z[i];

        float r3d_sq = px*px + py*py + pz*pz;
        float r3d = sqrtf(r3d_sq);
        float inv_r3d = 1.0f / (r3d + 1e-8f);

        /* Viviani field force */
        float ax = 0, ay = 0, az = 0;
        apply_viviani_field(px, py, pz, r3d, inv_r3d, &ax, &ay, &az);

        /* Orbital-plane damping */
        float Lx = py*vz - pz*vy, Ly = pz*vx - px*vz, Lz = px*vy - py*vx;
        float inv_L = 1.0f / sqrtf(Lx*Lx + Ly*Ly + Lz*Lz + 1e-8f);
        float lx = Lx*inv_L, ly = Ly*inv_L, lz = Lz*inv_L;

        float v_sq = vx*vx + vy*vy + vz*vz;
        if (v_sq > 0.001f) {
            float v_normal = vx*lx + vy*ly + vz*lz;
            float r3 = r3d * r3d * r3d;
            float omega_z = sqrtf(BH_MASS / fmaxf(r3, 1.0f));
            float damping = fminf(2.0f * omega_z, 0.50f);
            vx -= damping * v_normal * lx;
            vy -= damping * v_normal * ly;
            vz -= damping * v_normal * lz;
        }

        /* Integrate */
        vx += ax * dt;
        vy += ay * dt;
        vz += az * dt;

        /* NaN guard */
        if (isfinite(vx) && isfinite(vy) && isfinite(vz)) {
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;
        }

        ps->pos_x[i] = px; ps->pos_y[i] = py; ps->pos_z[i] = pz;
        ps->vel_x[i] = vx; ps->vel_y[i] = vy; ps->vel_z[i] = vz;

        /* Pump history update */
        float scale = ps->pump_scale[i];
        float history = ps->pump_history[i];
        ps->pump_history[i] = history * 0.98f + scale * 0.02f;

        /* Kuramoto phase advance */
        float th = ps->theta[i] + ps->omega_nat[i] * dt;
        if (th >= V21_TWO_PI) th -= V21_TWO_PI;
        if (th < 0.0f) th += V21_TWO_PI;
        ps->theta[i] = th;
    }
}

/* ========================================================================
 * FRAME RENDERER — PPM image output (no libraries needed)
 *
 * Projects particles to 2D with orthographic projection.
 * Color from topo_dim (same classification as V20 rendering).
 * Additive blending for density — brighter = more particles.
 *
 * Output: PPM files → convert to video with:
 *   ffmpeg -framerate 30 -i "frames/frame_%05d.ppm" -c:v libx264 output.mp4
 * ======================================================================== */

#define RENDER_WIDTH   1024
#define RENDER_HEIGHT  1024

/* Blackbody-ish color from topo_dim */
static void dim_to_rgb(int dim, uint8_t* r, uint8_t* g, uint8_t* b) {
    switch (dim) {
        case 0: *r = 80;  *g = 20;  *b = 20;  break;  /* deep red */
        case 1: *r = 255; *g = 140; *b = 40;  break;  /* orange */
        case 2: *r = 255; *g = 240; *b = 180; break;  /* yellow-white */
        case 3: *r = 160; *g = 200; *b = 255; break;  /* blue-white */
        case 4: *r = 100; *g = 150; *b = 255; break;  /* bright blue */
        default:*r = 255; *g = 255; *b = 255; break;
    }
}

static int topo_dim_simple(uint8_t state) {
    int d = 0;
    for (int a = 0; a < 4; a++) {
        uint8_t bits = (state >> (a * 2)) & 0x03;
        if (bits == 1 || bits == 2) d++;
    }
    return d;
}

static void render_frame_ppm(const char* filename, ParticleState* ps,
                              float cam_dist, float cam_yaw, float cam_pitch) {
    /* HDR framebuffer */
    float* fb_r = (float*)calloc(RENDER_WIDTH * RENDER_HEIGHT, sizeof(float));
    float* fb_g = (float*)calloc(RENDER_WIDTH * RENDER_HEIGHT, sizeof(float));
    float* fb_b = (float*)calloc(RENDER_WIDTH * RENDER_HEIGHT, sizeof(float));

    float cy = cosf(cam_yaw), sy = sinf(cam_yaw);
    float cp = cosf(cam_pitch), sp = sinf(cam_pitch);
    float proj_scale = (float)RENDER_WIDTH / (cam_dist * 2.0f);

    /* Single pass: one small Gaussian per particle. Additive blend.
     * No grid. No accumulator. No resonance gate. No multi-pass.
     * The wave cancellations happen through superposition (addition). */
    for (int i = 0; i < ps->N; i++) {
        if (!(ps->flags[i] & 0x01)) continue;

        float px = ps->pos_x[i], py = ps->pos_y[i], pz = ps->pos_z[i];

        /* Camera transform */
        float rx = px * cy + pz * sy;
        float rz = -px * sy + pz * cy;
        float ry = py;
        float ry2 = ry * cp - rz * sp;

        /* Screen position */
        float sx_f = rx * proj_scale + RENDER_WIDTH * 0.5f;
        float sy_f = -ry2 * proj_scale + RENDER_HEIGHT * 0.5f;

        /* Color from eigenstate */
        uint8_t cr, cg, cb;
        dim_to_rgb(topo_dim_simple(ps->topo_state[i]), &cr, &cg, &cb);

        /* Brightness from pump_scale (activity) */
        float brightness = ps->pump_scale[i] * 0.3f;

        /* σ: small, fixed per particle. Dense regions brighten through
         * additive overlap, not through larger Gaussians. */
        float sigma = 3.0f;

        float fr = (cr / 255.0f) * brightness;
        float fg = (cg / 255.0f) * brightness;
        float fb_val = (cb / 255.0f) * brightness;

        /* Splat: 3σ radius */
        int rad = (int)(sigma * 3.0f) + 1;
        int x0 = (int)sx_f - rad, x1 = (int)sx_f + rad;
        int y0 = (int)sy_f - rad, y1 = (int)sy_f + rad;
        if (x0 < 0) x0 = 0; if (x1 >= RENDER_WIDTH) x1 = RENDER_WIDTH - 1;
        if (y0 < 0) y0 = 0; if (y1 >= RENDER_HEIGHT) y1 = RENDER_HEIGHT - 1;

        float inv_2s2 = 1.0f / (2.0f * sigma * sigma);

        for (int py2 = y0; py2 <= y1; py2++) {
            float dy = py2 - sy_f;
            for (int px2 = x0; px2 <= x1; px2++) {
                float dx = px2 - sx_f;
                float w = expf(-(dx*dx + dy*dy) * inv_2s2);
                int idx = py2 * RENDER_WIDTH + px2;
                fb_r[idx] += fr * w;
                fb_g[idx] += fg * w;
                fb_b[idx] += fb_val * w;
            }
        }
    }

    /* Tonemap + write PPM */
    float peak = 0.001f;
    for (int i = 0; i < RENDER_WIDTH * RENDER_HEIGHT; i++) {
        float lum = fb_r[i] * 0.299f + fb_g[i] * 0.587f + fb_b[i] * 0.114f;
        if (lum > peak) peak = lum;
    }
    float exposure = 1.0f / (peak * 0.5f);

    FILE* f = fopen(filename, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", RENDER_WIDTH, RENDER_HEIGHT);
        for (int i = 0; i < RENDER_WIDTH * RENDER_HEIGHT; i++) {
            float r = fb_r[i] * exposure;
            float g = fb_g[i] * exposure;
            float b = fb_b[i] * exposure;
            r = r / (1.0f + r);  g = g / (1.0f + g);  b = b / (1.0f + b);
            r = powf(r, 1.0f/2.2f); g = powf(g, 1.0f/2.2f); b = powf(b, 1.0f/2.2f);
            uint8_t pixel[3] = {
                (uint8_t)(fminf(r, 1.0f) * 255),
                (uint8_t)(fminf(g, 1.0f) * 255),
                (uint8_t)(fminf(b, 1.0f) * 255)
            };
            fwrite(pixel, 3, 1, f);
        }
        fclose(f);
    }

    free(fb_r); free(fb_g); free(fb_b);
}

/* ========================================================================
 * DIAGNOSTICS — minimal output
 * ======================================================================== */

static void print_diagnostics(ParticleState* ps, int frame, float time) {
    float avg_vel = 0;
    int active = 0;
    for (int i = 0; i < ps->N; i++) {
        if (ps->flags[i] & 0x01) {
            float v = sqrtf(ps->vel_x[i]*ps->vel_x[i] +
                           ps->vel_y[i]*ps->vel_y[i] +
                           ps->vel_z[i]*ps->vel_z[i]);
            avg_vel += v;
            active++;
        }
    }
    if (active > 0) avg_vel /= active;
    printf("[frame %6d] particles=%d active=%d avg_vel=%.4f t=%.2f\n",
           frame, ps->N, active, avg_vel, time);
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(int argc, char** argv) {
    int num_particles = DEFAULT_PARTICLES;
    int num_frames = DEFAULT_FRAMES;
    unsigned int seed = DEFAULT_SEED;
    float dt = DEFAULT_DT;
    int stats_interval = 90;
    int render_interval = 0;   /* 0 = no rendering; >0 = write PPM every N frames */
    float cam_dist = 250.0f;
    float cam_yaw = 0.3f;
    float cam_pitch = 0.5f;

    /* Simple CLI parsing */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            num_particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc)
            num_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rng-seed") == 0 && i+1 < argc)
            seed = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--render-ppm") == 0 && i+1 < argc)
            render_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cam-dist") == 0 && i+1 < argc)
            cam_dist = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: blackhole_v21 [-n particles] [--frames N] [--rng-seed S]\n");
            printf("       [--render-ppm interval] [--cam-dist D]\n");
            printf("  --render-ppm N  Write PPM image every N frames to frames/\n");
            printf("  --cam-dist D    Camera distance (default 250)\n");
            return 0;
        }
    }

    printf("================================================================\n");
    printf("  BLACKHOLE V21 — Vendor-Independent Galaxy Simulator\n");
    printf("  SPIRV First. No CUDA. No Vendor Lock-In.\n");
    printf("================================================================\n\n");
    printf("[config] Particles: %d, Frames: %d, Seed: %u\n", num_particles, num_frames, seed);
    printf("[config] Backend: CPU (pure C reference implementation)\n");
    printf("[config] Physics: Viviani field + orbital damping + Kuramoto\n");
    if (render_interval > 0)
        printf("[config] Rendering: PPM every %d frames (cam_dist=%.0f)\n", render_interval, cam_dist);
    printf("\n");

    /* Initialize particles */
    ParticleState particles;
    init_particles(&particles, num_particles, seed);

    /* Main simulation loop */
    float sim_time = 0.0f;
    clock_t start = clock();

    /* Create frames directory if rendering */
    if (render_interval > 0) {
        #ifdef _WIN32
        system("mkdir frames 2>nul");
        #else
        system("mkdir -p frames");
        #endif
    }

    for (int frame = 0; frame < num_frames; frame++) {
        physics_step(&particles, dt * 2.0f, sim_time);
        sim_time += dt;

        /* Slowly rotate camera for visual interest */
        float yaw = cam_yaw + (float)frame * 0.001f;

        if (frame % stats_interval == 0) {
            print_diagnostics(&particles, frame, sim_time);
        }

        if (render_interval > 0 && frame % render_interval == 0) {
            static int render_count = 0;
            char filename[256];
            snprintf(filename, sizeof(filename), "frames/frame_%05d.ppm", render_count);
            render_frame_ppm(filename, &particles, cam_dist, yaw, cam_pitch);
            if (render_count % 50 == 0)
                printf("[render] Wrote %s (sim frame %d)\n", filename, frame);
            render_count++;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n[done] %d frames in %.2f seconds (%.1f fps)\n",
           num_frames, elapsed, num_frames / elapsed);

    /* Final diagnostics */
    print_diagnostics(&particles, num_frames, sim_time);

    /* Cleanup */
    free_particles(&particles);
    printf("[shutdown] Complete.\n");

    return 0;
}
