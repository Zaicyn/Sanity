#version 450

// Volumetric raymarching fragment shader for far-field density rendering
// Samples a 128³ density texture and renders accumulated particle data
// as smooth, glowing hopfion shells and spiral arms

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 viewProj;
    vec3 cameraPos;
    float time;
    float avgScale;
    float avgResidual;
    float heartbeat;
    float padding;
} ubo;

// Volume configuration (push constants would be better, but this works)
layout(set = 0, binding = 1) uniform VolumeUBO {
    mat4 invViewProj;     // Inverse view-projection for ray unprojection
    vec3 volumeMin;       // World-space AABB min (e.g., -150, -150, -150)
    float volumeScale;    // Total extent (e.g., 300)
    vec3 volumeMax;       // World-space AABB max (e.g., 150, 150, 150)
    float stepSize;       // Raymarch step size (e.g., 2.0)
} vol;

// 3D density texture (128³, RGBA32F: scale_sum, temp_sum, count, coherence)
layout(set = 0, binding = 2) uniform sampler3D densityVolume;

layout(location = 0) in vec3 fragRayOrigin;
layout(location = 1) in vec3 fragRayDir;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Blackbody temperature to RGB (approximation for 2000K-12000K range)
vec3 temperatureToRGB(float temp) {
    // temp is in scaled units (2-12 range = 2000K-12000K)
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
    // Reconstruct world-space ray direction from clip coordinates
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

    // Clamp tNear to start at camera (not behind)
    tNear = max(tNear, 0.0);

    // Raymarch through volume
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;

    float t = tNear;
    float dt = vol.stepSize;
    int maxSteps = 256;

    for (int step = 0; step < maxSteps && t < tFar && accumAlpha < 0.99; step++) {
        vec3 samplePos = rayOrigin + rayDir * t;

        // Convert world position to texture coordinates [0, 1]
        vec3 texCoord = (samplePos - vol.volumeMin) / vol.volumeScale;

        // Sample density volume (RGBA: scale_sum, temp_sum, count, coherence)
        vec4 voxel = texture(densityVolume, texCoord);
        float scaleSum = voxel.r;
        float tempSum = voxel.g;
        float count = voxel.b;
        float coherence = voxel.a;

        if (count > 0.1) {
            // Compute average values for this voxel
            float avgScale = scaleSum / count;
            float avgTemp = tempSum / count;
            float avgCoherence = coherence / count;

            // Density-based opacity (more particles = more opaque)
            // Normalize by expected max particles per voxel (~100-1000)
            float density = count / 100.0;
            density = clamp(density, 0.0, 1.0);

            // Color from temperature (blackbody)
            float tempK = avgTemp * 0.5 + 2.0;  // Map to ~2-12 range
            vec3 tempColor = temperatureToRGB(tempK);

            // Luminosity boost from pump scale
            float luminosity = 0.3 + avgScale * 0.1;
            luminosity = clamp(luminosity, 0.1, 2.0);

            // Coherence tint: locked particles glow blue-white, chaotic glow orange
            vec3 coherenceTint = mix(vec3(1.0, 0.7, 0.4), vec3(0.8, 0.9, 1.0), avgCoherence);

            // Final sample color
            vec3 sampleColor = tempColor * luminosity * coherenceTint;

            // Opacity based on density and step size
            float sampleAlpha = density * (dt / 4.0);  // Normalize for step size
            sampleAlpha = clamp(sampleAlpha, 0.0, 0.3);  // Cap per-sample opacity

            // Front-to-back compositing
            accumColor += (1.0 - accumAlpha) * sampleAlpha * sampleColor;
            accumAlpha += (1.0 - accumAlpha) * sampleAlpha;
        }

        t += dt;
    }

    // Add subtle glow at edges (rim lighting effect)
    float edgeFactor = 1.0 - accumAlpha;
    accumColor += edgeFactor * vec3(0.02, 0.01, 0.03) * ubo.heartbeat * 0.1;

    // Output with premultiplied alpha for additive blending with particle pass
    outColor = vec4(accumColor, accumAlpha);
}
