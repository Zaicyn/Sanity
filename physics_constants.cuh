// physics_constants.cuh — Device-side __constant__ physics parameters
// ==================================================================
// Defines all __constant__ variables used by physics kernels.
// Guarded by PHYSICS_CONSTANTS_DEFINED so physics.cu doesn't redefine.
#pragma once

#include <cuda_runtime.h>
#include "disk.cuh"  // BH_MASS, SCHW_R, etc.

#ifndef PHYSICS_CONSTANTS_DEFINED
#define PHYSICS_CONSTANTS_DEFINED

__device__ __constant__ float d_PI = 3.14159265358979f;
__device__ __constant__ float d_TWO_PI = 6.28318530717959f;
__device__ __constant__ float d_ISCO = 6.0f;
__device__ __constant__ float d_BH_MASS = BH_MASS;
__device__ __constant__ float d_SCHW_R = 2.0f;
__device__ __constant__ float d_DISK_THICKNESS = 0.8f;
__device__ __constant__ float d_PHI = 1.6180339887498948f;
__device__ __constant__ float d_SCALE_RATIO = 1.6875f;        // 27/16
__device__ __constant__ float d_BIAS = 0.75f;
__device__ __constant__ float d_PHI_EXCESS = 0.09017f;

// Spiral arm constants (used by topology.cuh)
__device__ __constant__ int d_NUM_ARMS = 3;
__device__ __constant__ float d_ARM_WIDTH_DEG = 45.0f;
__device__ __constant__ float d_ARM_TRAP_STRENGTH = 0.15f;
__device__ __constant__ bool d_USE_ARM_TOPOLOGY = true;
__device__ __constant__ float d_ARM_BOOST_OVERRIDE = 0.0f;

// Atomic counters for spawn tracking
__device__ unsigned int d_current_particle_count = 0;
__device__ unsigned int d_spawn_count = 0;

#endif // PHYSICS_CONSTANTS_DEFINED
