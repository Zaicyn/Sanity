#version 450

// ============================================================
// ANALYTIC RAY-SHELL INTERSECTION RENDERER
// No marching. No texture lookups. Pure closed-form evaluation.
//
// 8 shells × 2 intersections = 16 density evaluations MAX per pixel
// raySphereAnalytic: one quadratic solve per shell, predicated
// ============================================================

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float time;
    float avgScale;
    float avgResidual;
    float heartbeat;
    float pump_phase;    // actual pump phase from CUDA
} ubo;

layout(set = 0, binding = 1) uniform VolumeUBO {
    mat4 invViewProj;
    vec3 volumeMin;
    float volumeScale;
    vec3 volumeMax;
    float seam_angle;       // locked seam orientation in radians
    float shellBrightness;  // Shell opacity multiplier (V key cycles 1.0→0.5→0.25→0)
    vec3 padding;
} vol;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// ============================================================
// PHYSICAL CONSTANTS — all derived from three parameters
// ============================================================
const float ISCO        = 6.0;
const float LAMBDA      = 4.0 * ISCO;          // 24.0 — shell spacing
const float OMEGA_PUMP  = 0.125;               // pump frequency
const float M_ARM       = 3.0;                 // cuboctahedral m-mode
const float ARM_AMP     = 0.3;                 // arm modulation depth
const float N_EXCESS    = 0.12;                // n_avg - 1.0, stable value
const int   N_SHELLS    = 8;
const float HYBRID_R    = 30.0;                // inner/outer boundary — particles handle r < 30

// Shell peak radii: λ/2, 3λ/2, 5λ/2, ...
float shellRadius(int k) {
    return LAMBDA * (float(k) + 0.5);
    // k=0 → 12, k=1 → 36, k=2 → 60, k=3 → 84 ...
}

// ============================================================
// BLACKBODY COLOR — physically derived from radius
// Shakura-Sunyaev: T ∝ r^(-3/4)
// ============================================================
vec3 blackbody(float r) {
    float t = clamp((r - HYBRID_R) / 200.0, 0.0, 1.0);
    vec3 hot  = vec3(1.0,  0.95, 0.7);   // yellow-white (inner)
    vec3 mid  = vec3(1.0,  0.4,  0.5);   // pink-red
    vec3 cool = vec3(0.55, 0.5,  0.9);   // lavender (outer)
    vec3 c = mix(hot, mid, smoothstep(0.0, 0.4, t));
         c = mix(c,   cool, smoothstep(0.4, 1.0, t));
    return c;
}

// ============================================================
// RAY-SPHERE INTERSECTION — returns (t_near, t_far), both < 0 = miss
// Predicated: no branch on hit/miss, caller masks result
// ============================================================
vec2 raySphereAnalytic(vec3 ro, vec3 rd, float R) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float disc = b * b - c;

    // Predicated: disc < 0 returns sentinel, no branch
    float sqrtDisc = sqrt(max(disc, 0.0));
    float mask = step(0.0, disc);   // 1.0 if hit, 0.0 if miss

    return vec2(-b - sqrtDisc, -b + sqrtDisc) * mask + vec2(-1.0) * (1.0 - mask);
}

// ============================================================
// MAIN — PURE ANALYTIC: O(N_SHELLS) per ray, no marching
// ============================================================
void main() {
    // --------------------------------------------------------
    // Ray setup from inverse view-projection
    // --------------------------------------------------------
    vec4 ndcNear = vec4(fragUV * 2.0 - 1.0, -1.0, 1.0);
    vec4 ndcFar  = vec4(fragUV * 2.0 - 1.0,  1.0, 1.0);

    vec4 worldNear = vol.invViewProj * ndcNear;
    vec4 worldFar  = vol.invViewProj * ndcFar;
    worldNear /= worldNear.w;
    worldFar  /= worldFar.w;

    vec3 ro = ubo.cameraPos;
    vec3 rd = normalize(worldFar.xyz - worldNear.xyz);

    // --------------------------------------------------------
    // Per-ray constants — computed ONCE, reused across all shells
    // --------------------------------------------------------

    // Shell brightness control (V key toggle)
    // 0 = shells invisible, 1 = full brightness
    float brightness = vol.shellBrightness;

    // Early out if shells are disabled
    if (brightness < 0.01) {
        outColor = vec4(0.04, 0.02, 0.08, 0.0);  // Background only
        return;
    }

    // Pump modulation — global, constant per frame
    // At low brightness, reduce pump effect for more stable viewing
    float pumpRange = mix(0.1, 0.5, brightness);  // Less modulation at low brightness
    float pumpFactor = 0.5 + pumpRange * sin(ubo.pump_phase);

    // --------------------------------------------------------
    // ANALYTIC SHELL LOOP — O(N_SHELLS) per pixel, NO MARCHING
    // Use ray-sphere intersection to find exact hit points
    // --------------------------------------------------------
    float accumAlpha = 0.0;
    vec3  accumColor = vec3(0.0);

    for (int k = 0; k < N_SHELLS; k++) {
        float R = shellRadius(k);

        // Skip shells inside hybrid boundary (particles handle those)
        if (R < HYBRID_R) continue;

        // Precompute shell color from radius — once per shell, not per hit
        vec3 shellColor = blackbody(R);

        // Analytic ray-sphere intersection — O(1) per shell
        vec2 tHits = raySphereAnalytic(ro, rd, R);

        // Skip if ray misses this shell
        if (tHits.x < 0.0 && tHits.y < 0.0) continue;

        // Process both intersection points (entry and exit)
        for (int i = 0; i < 2; i++) {
            float t = (i == 0) ? tHits.x : tHits.y;
            if (t < 0.0) continue;  // Behind camera

            vec3 P = ro + rd * t;

            // Disk-plane confinement: fade shells away from y=0
            float diskHeight = abs(P.y);
            float diskThickness = 15.0;
            float diskFade = exp(-0.5 * (diskHeight/diskThickness) * (diskHeight/diskThickness));

            // Skip if too far from disk plane
            if (diskFade < 0.05) continue;

            // Azimuthal m=3 modulation at hit point
            float theta = atan(P.z, P.x);
            float armFactor = 1.0 + ARM_AMP * cos(M_ARM * theta + vol.seam_angle);

            // Shell contribution
            float density = diskFade * armFactor * pumpFactor;

            // Alpha from density, scaled by shell brightness
            float alpha = density * 0.3 * brightness;

            // Front-to-back compositing
            accumColor += shellColor * alpha * (1.0 - accumAlpha);
            accumAlpha += alpha * (1.0 - accumAlpha);
        }
    }

    // --------------------------------------------------------
    // Background and tonemapping
    // --------------------------------------------------------
    vec3 background = vec3(0.04, 0.02, 0.08);  // dark purple

    vec3 final = background * (1.0 - accumAlpha) + accumColor;

    // Reinhard tonemapping
    final = final / (final + vec3(1.0));

    // Gamma correction: sqrt ≈ gamma 2.0, cheap approximation
    final = sqrt(final);

    outColor = vec4(final, 1.0);
}
