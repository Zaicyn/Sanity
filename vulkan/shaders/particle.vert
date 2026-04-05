#version 450

// Per-instance particle data
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inScale;
layout(location = 2) in float inResidual;
layout(location = 3) in float inTemp;
layout(location = 4) in vec3 inVelocity;
layout(location = 5) in float inElongation;

// Global uniforms
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float time;
    float avgScale;
    float avgResidual;
    float heartbeat;
    float padding;
} ubo;

// Output to fragment shader
layout(location = 0) out vec4 fragColor;
layout(location = 1) out float fragResidual;
layout(location = 2) out float fragElongation;

// Blackbody radiation approximation (temperature in K / 1000)
vec3 temperatureToRGB(float temp) {
    // Approximation for temperatures 1000K - 40000K
    // Input temp is already scaled (e.g., 5.0 = 5000K)
    vec3 color;

    // Red channel
    if (temp <= 6.6) {
        color.r = 1.0;
    } else {
        color.r = 1.29293 * pow(temp - 6.0, -0.1332);
    }

    // Green channel
    if (temp <= 6.6) {
        color.g = 0.39008 * log(temp) - 0.63184;
    } else {
        color.g = 1.12989 * pow(temp - 6.0, -0.0755);
    }

    // Blue channel
    if (temp >= 6.6) {
        color.b = 1.0;
    } else if (temp <= 1.9) {
        color.b = 0.0;
    } else {
        color.b = 0.54320 * log(temp - 1.0) - 1.19625;
    }

    return clamp(color, 0.0, 1.0);
}

void main() {
    // Transform position to clip space
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);

    // Point size: simple distance-based attenuation
    float dist = length(inPosition - ubo.cameraPos);

    // Base size of 1-2 pixels (reduced for 3.5M particles)
    float baseSize = 1.0 + inScale * 0.1;

    // Distance attenuation (reduced for better separation)
    float distAtten = 100.0 / max(dist, 50.0);

    gl_PointSize = baseSize * distAtten;
    gl_PointSize = clamp(gl_PointSize, 1.0, 4.0);  // Max 4 pixels

    // === LOD WEIGHT EXTRACTION ===
    // In hybrid LOD mode, elongation encodes: lodWeight * (1 + vel_mag * 0.01)
    // Extract LOD weight (0 = FAR/volume only, 1 = NEAR/full point)
    // For backwards compatibility, treat elongation > 0.5 as having LOD weight
    float lodWeight = 1.0;
    float actualElongation = inElongation;
    if (inElongation > 0.0 && inElongation < 10.0) {
        // LOD weight is encoded - extract it
        // Elongation = lodWeight * (1 + vel_mag * 0.01)
        // Approximate: lodWeight ≈ elongation (since vel_mag * 0.01 is small)
        lodWeight = clamp(inElongation, 0.0, 1.0);
        actualElongation = max(inElongation, 1.0);  // Restore for other uses
    }

    // Temperature to color (blackbody)
    // temp is in simulation units, assume ~5-20 range maps to 2000K-10000K
    float tempK = inTemp * 0.5 + 2.0;  // Map to ~2-12 range for temperatureToRGB
    vec3 tempColor = temperatureToRGB(tempK);

    // Boost color by pump scale (higher D = brighter)
    float luminosity = 0.5 + inScale * 0.5;

    // High residual particles (near dissolution) glow orange/red
    vec3 stressGlow = vec3(0.0);
    if (inResidual > 0.7) {
        float stressIntensity = (inResidual - 0.7) / 0.3;  // 0-1 over range 0.7-1.0
        stressGlow = vec3(1.0, 0.4, 0.1) * stressIntensity * 2.0;
    }

    // Alpha fades as particle approaches dissolution
    float dissolutionAlpha = 1.0 - smoothstep(0.85, 1.0, inResidual);

    // === HYBRID LOD ALPHA ===
    // Combine dissolution alpha with LOD weight for smooth point-to-volume transition
    float alpha = dissolutionAlpha * lodWeight;

    // Final color with temperature-based coloring
    fragColor = vec4(tempColor * luminosity + stressGlow, alpha);
    fragResidual = inResidual;
    fragElongation = actualElongation;
}
