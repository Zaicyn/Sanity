#version 450

// Analytic volumetric renderer for self-consistent geometric pump model
// Instead of sampling a density texture, we reconstruct the field from
// the known spectral structure: λ=4×ISCO, m=3, ω=0.125
//
// The field is fully determined by three parameters - no sampling needed.

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float time;
    float avgScale;
    float avgResidual;
    float heartbeat;
    float padding;
} ubo;

layout(set = 0, binding = 1) uniform VolumeUBO {
    mat4 invViewProj;
    vec3 volumeMin;
    float volumeScale;
    vec3 volumeMax;
    float stepSize;
} vol;

layout(location = 0) in vec3 fragRayOrigin;
layout(location = 1) in vec3 fragRayDir;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// ============================================================================
// SELF-CONSISTENT GEOMETRY CONSTANTS (from SELF_CONSISTENT_GEOMETRY.md)
// ============================================================================

const float ISCO = 6.0;
const float WAVELENGTH = 4.0 * ISCO;          // λ = 24 units (shell spacing)
const int ARM_MODE = 3;                        // m = 3 (cuboctahedral symmetry)
const float OMEGA_PUMP = 0.125;                // ω_pump (pump frequency)
const float SHELL_FORMATION_RADIUS = 4.0 * ISCO;  // Where shells form (~24 units)

// Disk geometry
const float DISK_INNER = 2.0 * ISCO;          // ~12 units (inside ISCO, thin)
const float DISK_OUTER = 20.0 * ISCO;         // ~120 units
const float DISK_THICKNESS = 0.8;             // Scale height

// Rendering parameters
const float ARM_AMPLITUDE = 0.3;              // Spiral arm strength
const float SHELL_CONTRAST = 0.5;             // Hopfion shell visibility

// ============================================================================
// ANALYTIC FIELD RECONSTRUCTION
// ============================================================================

// Compute density from the self-consistent model
float analyticDensity(vec3 pos, float time) {
    // Cylindrical coordinates
    float r = length(pos.xz);
    float theta = atan(pos.z, pos.x);
    float y = pos.y;

    // Radial profile: exponential disk with inner cutoff
    float radialProfile = 0.0;
    if (r > DISK_INNER && r < DISK_OUTER) {
        // Shakura-Sunyaev-ish: ρ ∝ r^(-3/4)
        radialProfile = pow(DISK_INNER / r, 0.75);

        // Inner edge softening
        float innerFade = smoothstep(DISK_INNER, DISK_INNER * 1.5, r);
        radialProfile *= innerFade;

        // Outer edge fade
        float outerFade = 1.0 - smoothstep(DISK_OUTER * 0.7, DISK_OUTER, r);
        radialProfile *= outerFade;
    }

    // Vertical profile: Gaussian disk
    float scaleHeight = DISK_THICKNESS * sqrt(r / ISCO);  // Flaring
    float verticalProfile = exp(-0.5 * (y * y) / (scaleHeight * scaleHeight));

    // === HOPFION SHELLS (radial harmonic) ===
    // Shells propagate outward at v = ω × 4×ISCO
    float wavePhase = (r - SHELL_FORMATION_RADIUS) / WAVELENGTH - OMEGA_PUMP * time;
    float shellModulation = 1.0 + SHELL_CONTRAST * sin(2.0 * 3.14159 * wavePhase);

    // === SPIRAL ARMS (angular harmonic m=3) ===
    // Arms rotate with pattern speed (prograde)
    float patternAngle = theta - 0.01 * time;  // Slow pattern rotation
    // Logarithmic spiral: φ = A * ln(r/r0)
    float spiralPhase = float(ARM_MODE) * (patternAngle - 0.3 * log(max(r, DISK_INNER) / DISK_INNER));
    float armModulation = 1.0 + ARM_AMPLITUDE * cos(spiralPhase);

    // === PUMP BREATHING (temporal harmonic) ===
    // Global pump cycle oscillation
    float pumpPhase = OMEGA_PUMP * time * 2.0 * 3.14159;
    float pumpModulation = 1.0 + 0.1 * sin(pumpPhase);

    // Combine all modulations
    float density = radialProfile * verticalProfile * shellModulation * armModulation * pumpModulation;

    return max(density, 0.0);
}

// Temperature from Shakura-Sunyaev (T ∝ r^(-3/4))
float analyticTemperature(float r) {
    if (r < DISK_INNER) return 12.0;  // Hot inner edge
    float T_norm = pow(DISK_INNER / r, 0.75);
    return 2.0 + 10.0 * T_norm;  // Map to 2-12 range for blackbody
}

// Blackbody temperature to RGB
vec3 temperatureToRGB(float temp) {
    vec3 color;

    if (temp <= 6.6) {
        color.r = 1.0;
    } else {
        color.r = 1.29293 * pow(temp - 6.0, -0.1332);
    }

    if (temp <= 6.6) {
        color.g = 0.39008 * log(temp) - 0.63184;
    } else {
        color.g = 1.12989 * pow(temp - 6.0, -0.0755);
    }

    if (temp >= 6.6) {
        color.b = 1.0;
    } else if (temp <= 1.9) {
        color.b = 0.0;
    } else {
        color.b = 0.54320 * log(temp - 1.0) - 1.19625;
    }

    return clamp(color, 0.0, 1.0);
}

// Ray-AABB intersection
bool rayBoxIntersect(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    tNear = max(max(tmin.x, tmin.y), tmin.z);
    tFar = min(min(tmax.x, tmax.y), tmax.z);
    return tNear < tFar && tFar > 0.0;
}

void main() {
    // Reconstruct world-space ray direction
    vec4 clipPos = vec4(fragUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 worldPos = vol.invViewProj * clipPos;
    worldPos /= worldPos.w;

    vec3 rayOrigin = ubo.cameraPos;
    vec3 rayDir = normalize(worldPos.xyz - rayOrigin);

    // Intersect ray with volume AABB
    float tNear, tFar;
    if (!rayBoxIntersect(rayOrigin, rayDir, vol.volumeMin, vol.volumeMax, tNear, tFar)) {
        outColor = vec4(0.0);
        return;
    }

    tNear = max(tNear, 0.0);

    // Raymarch through volume
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    float t = tNear;
    float dt = vol.stepSize;
    int maxSteps = 256;

    for (int step = 0; step < maxSteps && t < tFar && accumAlpha < 0.99; step++) {
        vec3 samplePos = rayOrigin + rayDir * t;

        // Analytic density - no texture lookup needed!
        float density = analyticDensity(samplePos, ubo.time);

        if (density > 0.001) {
            // Temperature from radial position
            float r = length(samplePos.xz);
            float temp = analyticTemperature(r);

            // Blackbody color
            vec3 color = temperatureToRGB(temp);

            // Luminosity scales with density and pump heartbeat
            float luminosity = 0.5 + 0.5 * ubo.heartbeat;
            color *= luminosity;

            // Opacity from density
            float alpha = density * (dt / 8.0);
            alpha = clamp(alpha, 0.0, 0.2);

            // Front-to-back compositing
            accumColor += (1.0 - accumAlpha) * alpha * color;
            accumAlpha += (1.0 - accumAlpha) * alpha;
        }

        t += dt;
    }

    outColor = vec4(accumColor, accumAlpha);
}
