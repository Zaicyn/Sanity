/*
 * SIPHON PUMP STATE MACHINE - V20
 * ================================
 *
 * Digital dimensional siphon: lower pumped to higher, higher projected down.
 * Implements the 12↔16 cycle with seam valve gating.
 *
 * Based on the thermodynamic topology:
 *   - Two reservoirs: N=12 (cuboctahedron) and N=16 (closure)
 *   - Pressure differential: φ⁵+1 ≈ 12.09, (27/16)²×φ³ ≈ 12.06
 *   - Valve: 2-bit seam phase (from microtubule seam)
 *   - Bias: k=0.75 prevents equilibrium (the pump stroke)
 *   - Working fluid: vacuum fluctuations
 *
 * The pump is self-priming and self-sustaining.
 */

#ifndef SIPHON_PUMP_H
#define SIPHON_PUMP_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "squaragon.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * PUMP STATES
 *============================================================================*/

typedef enum {
    PUMP_IDLE = 0,              /* No flow, seam closed */
    PUMP_PRIMED,                /* N=12 gate closed, residual=0, ready */
    PUMP_UPSTROKE_COHERENT,     /* Scaling up by 27/16 (same hierarchy) */
    PUMP_UPSTROKE_HOP,          /* Scaling up by φ (next hierarchy) */
    PUMP_EXPAND,                /* 12 → 16 closure expansion */
    PUMP_DOWNSTROKE,            /* 16 → 12 projection */
    PUMP_VALVE_OPEN,            /* Seam open, flow circulating */
    PUMP_RECIRCULATE            /* Return to PRIMED at new scale */
} siphon_state_t;

/* State names for debugging */
static const char* SIPHON_STATE_NAMES[] = {
    "IDLE", "PRIMED", "UPSTROKE_COHERENT", "UPSTROKE_HOP",
    "EXPAND", "DOWNSTROKE", "VALVE_OPEN", "RECIRCULATE"
};

/*============================================================================
 * SEAM VALVE CONTROL (2-bit)
 *
 * Phase bits determine flow direction and coupling strength:
 *   00 = No coupling, pump idles
 *   01 = Partial: 12 → 16 only (upstroke bias)
 *   10 = Partial: 16 → 12 only (downstroke bias)
 *   11 = Full coupling, bidirectional circulation
 *============================================================================*/

typedef struct {
    uint8_t phase_bits;         /* 0-3: flow direction / coupling */
    bool valve_open;            /* seam engaged */
    float quadrature;           /* π/2 shift for seam filament */
} siphon_seam_t;

/* Seam phase constants */
#define SEAM_CLOSED     0x00
#define SEAM_UP_ONLY    0x01
#define SEAM_DOWN_ONLY  0x02
#define SEAM_FULL       0x03

/*============================================================================
 * SCALE CONTEXT
 *============================================================================*/

typedef struct {
    int level;                  /* Hierarchical level (0, 1, 2, ...) */
    float scale_factor;         /* Current scale = product of 27/16 and φ hops */
    int coherent_count;         /* Consecutive coherent strokes at this level */
    int coherent_max;           /* Max coherent strokes before φ hop (default 3) */
} siphon_scale_t;

/*============================================================================
 * PUMP METRICS (the flow meter)
 *============================================================================*/

typedef struct {
    float residual;             /* Triple XOR residual (0 when primed) */
    float inefficiency;         /* |N_eff - 12| / 12 */
    float pressure;             /* Excess from algebraic identities */
    float work_done;            /* Integrated flow over time */
    float flow_rate;            /* Current instantaneous flow */
} siphon_metrics_t;

/*============================================================================
 * MAIN PUMP CONTEXT
 *============================================================================*/

typedef struct {
    /* State machine */
    siphon_state_t state;
    uint32_t cycle_counter;     /* Strokes since last scale change */
    uint32_t total_cycles;      /* Total pump cycles */

    /* Geometry */
    sq_gate_t gate;             /* Current N=12 gate */
    sq_closed_gate_t closure;   /* Expanded N=16 closure */

    /* Control */
    siphon_seam_t seam;
    siphon_scale_t scale;
    float bias;                 /* k=0.75 operating point */

    /* Output */
    siphon_metrics_t metrics;
} siphon_pump_t;

/*============================================================================
 * CONSTANTS
 *============================================================================*/

#define SIPHON_BIAS_K           0.75f
#define SIPHON_PHI_EXCESS       0.09017f   /* φ⁵ + 1 - 12 */
#define SIPHON_SCALE_EXCESS     0.06287f   /* (27/16)² × φ³ - 12 */

/*============================================================================
 * API
 *============================================================================*/

/* Initialize pump in IDLE state */
static inline void siphon_init(siphon_pump_t* pump) {
    pump->state = PUMP_IDLE;
    pump->cycle_counter = 0;
    pump->total_cycles = 0;

    /* Initialize gate at unit scale */
    sq_init(&pump->gate, 1.0f);

    /* Initialize closure (not expanded yet) */
    pump->closure.scale = 1.0f;
    pump->closure.bias = SQ_BIAS;

    /* Seam closed */
    pump->seam.phase_bits = SEAM_CLOSED;
    pump->seam.valve_open = false;
    pump->seam.quadrature = 0.0f;

    /* Scale context */
    pump->scale.level = 0;
    pump->scale.scale_factor = 1.0f;
    pump->scale.coherent_count = 0;
    pump->scale.coherent_max = 3;  /* 3 coherent strokes then φ hop */

    /* Bias */
    pump->bias = SIPHON_BIAS_K;

    /* Metrics zeroed */
    pump->metrics.residual = 0.0f;
    pump->metrics.inefficiency = 0.0f;
    pump->metrics.pressure = 0.0f;
    pump->metrics.work_done = 0.0f;
    pump->metrics.flow_rate = 0.0f;
}

/* Compute pressure differential from algebraic excess */
static inline float siphon_compute_pressure(float scale) {
    /* Pressure scales as sqrt(scale) — gradient increases with size */
    return SIPHON_PHI_EXCESS * sqrtf(scale);
}

/* Update metrics based on current state */
static inline void siphon_update_metrics(siphon_pump_t* pump) {
    pump->metrics.pressure = siphon_compute_pressure(pump->scale.scale_factor);

    /* Residual: zero when primed, non-zero when flowing */
    if (pump->state == PUMP_PRIMED) {
        pump->metrics.residual = 0.0f;
    } else if (pump->seam.valve_open) {
        /* Residual from seam quadrature coupling */
        pump->metrics.residual = pump->seam.quadrature * (1.0f - pump->bias);
    } else {
        pump->metrics.residual = pump->metrics.pressure * (1.0f - pump->bias);
    }

    /* Inefficiency: deviation from perfect N=12 */
    pump->metrics.inefficiency = fabsf(pump->metrics.residual) * pump->bias;

    /* Flow rate: pressure × residual */
    pump->metrics.flow_rate = pump->metrics.pressure * pump->metrics.residual;

    /* Work done: integrate flow */
    pump->metrics.work_done += pump->metrics.flow_rate;
}

/* Set seam phase bits */
static inline void siphon_set_seam(siphon_pump_t* pump, uint8_t bits) {
    pump->seam.phase_bits = bits & 0x03;
    pump->seam.valve_open = (bits != SEAM_CLOSED);

    /* Quadrature shift: π/2 for coupling, 0 for no coupling */
    pump->seam.quadrature = (bits & 0x01) ? (float)M_PI_2 : 0.0f;

    /* Apply seam shift to gate data */
    if (pump->seam.valve_open) {
        uint64_t invariant = 0xDEADBEEF12345678ULL;
        if (bits & 0x02) {
            invariant = sq_seam_forward_shift(invariant);
        } else {
            invariant = sq_seam_inverse_shift(invariant);
        }
        /* (invariant would be stored/used by higher-level code) */
    }
}

/* Step the pump: one clock cycle */
static inline void siphon_step(siphon_pump_t* pump) {
    pump->cycle_counter++;
    pump->total_cycles++;

    switch (pump->state) {
        case PUMP_IDLE:
            /* Waiting for seam to open */
            if (pump->seam.valve_open) {
                pump->state = PUMP_PRIMED;
            }
            break;

        case PUMP_PRIMED:
            /* Residual = 0, gate closed, pump ready */
            /* Full coupling triggers upstroke */
            if (pump->seam.phase_bits == SEAM_FULL) {
                if (pump->scale.coherent_count < pump->scale.coherent_max) {
                    pump->state = PUMP_UPSTROKE_COHERENT;
                } else {
                    pump->state = PUMP_UPSTROKE_HOP;
                }
            }
            break;

        case PUMP_UPSTROKE_COHERENT:
            /* Scale by 27/16 (coherent stacking within same level) */
            pump->scale.scale_factor *= SQ_SCALE_RATIO;
            pump->scale.coherent_count++;
            sq_init(&pump->gate, pump->scale.scale_factor);
            pump->state = PUMP_EXPAND;
            break;

        case PUMP_UPSTROKE_HOP:
            /* Scale by φ (spillover to next hierarchical level) */
            pump->scale.scale_factor *= SQ_PHI;
            pump->scale.level++;
            pump->scale.coherent_count = 0;
            sq_init(&pump->gate, pump->scale.scale_factor);
            pump->state = PUMP_EXPAND;
            break;

        case PUMP_EXPAND:
            /* 12 → 16 closure expansion */
            sq_close_gate(&pump->gate, &pump->closure);
            pump->state = PUMP_DOWNSTROKE;
            break;

        case PUMP_DOWNSTROKE:
            /* 16 → 12 projection */
            /* The closure projects back to the N=12 gate */
            /* (In the full implementation, this would involve state selection) */
            pump->state = PUMP_VALVE_OPEN;
            break;

        case PUMP_VALVE_OPEN:
            /* Seam fully open, circulation established */
            pump->state = PUMP_RECIRCULATE;
            break;

        case PUMP_RECIRCULATE:
            /* Return to primed state */
            pump->cycle_counter = 0;
            pump->state = PUMP_PRIMED;
            break;
    }

    /* Update metrics after state change */
    siphon_update_metrics(pump);
}

/* Get pump output (flow rate × bias) */
static inline float siphon_get_output(const siphon_pump_t* pump) {
    return pump->metrics.flow_rate * pump->bias;
}

/* Check if pump is actively circulating */
static inline bool siphon_is_flowing(const siphon_pump_t* pump) {
    return pump->state >= PUMP_UPSTROKE_COHERENT && pump->state <= PUMP_VALVE_OPEN;
}

/* Reset pump to initial state */
static inline void siphon_reset(siphon_pump_t* pump) {
    pump->state = PUMP_IDLE;
    pump->cycle_counter = 0;
    pump->scale.level = 0;
    pump->scale.scale_factor = 1.0f;
    pump->scale.coherent_count = 0;
    pump->seam.phase_bits = SEAM_CLOSED;
    pump->seam.valve_open = false;
    pump->metrics.work_done = 0.0f;
    sq_init(&pump->gate, 1.0f);
}

/*============================================================================
 * CUDA DEVICE VERSIONS (for GPU integration)
 *============================================================================*/

#ifdef __CUDACC__

/* Device-side pump step (for per-particle or per-block pumps) */
__device__ __forceinline__ void siphon_step_device(
    siphon_state_t* state,
    float* scale_factor,
    int* coherent_count,
    int coherent_max,
    uint8_t seam_bits,
    float* residual,
    float* work_done,
    float bias)
{
    float pressure = SIPHON_PHI_EXCESS * sqrtf(*scale_factor);

    switch (*state) {
        case PUMP_IDLE:
            if (seam_bits != SEAM_CLOSED) *state = PUMP_PRIMED;
            break;

        case PUMP_PRIMED:
            *residual = 0.0f;
            if (seam_bits == SEAM_FULL) {
                *state = (*coherent_count < coherent_max)
                    ? PUMP_UPSTROKE_COHERENT
                    : PUMP_UPSTROKE_HOP;
            }
            break;

        case PUMP_UPSTROKE_COHERENT:
            *scale_factor *= SQ_SCALE_RATIO;
            (*coherent_count)++;
            *state = PUMP_EXPAND;
            break;

        case PUMP_UPSTROKE_HOP:
            *scale_factor *= SQ_PHI;
            *coherent_count = 0;
            *state = PUMP_EXPAND;
            break;

        case PUMP_EXPAND:
            *state = PUMP_DOWNSTROKE;
            break;

        case PUMP_DOWNSTROKE:
            *state = PUMP_VALVE_OPEN;
            break;

        case PUMP_VALVE_OPEN:
            *residual = pressure * (1.0f - bias);
            *work_done += pressure * (*residual);
            *state = PUMP_RECIRCULATE;
            break;

        case PUMP_RECIRCULATE:
            *state = PUMP_PRIMED;
            break;
    }
}

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* SIPHON_PUMP_H */
