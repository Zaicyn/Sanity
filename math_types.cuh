// math_types.cuh — Minimal Vec3 / Mat4 (no GLM dependency)
// ========================================================
#pragma once

#include <cstring>  // memset
#include <cmath>    // sqrtf, tanf

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float a=0, float b=0, float c=0): x(a), y(b), z(c) {}
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    __host__ __device__ float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    __host__ __device__ float len() const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ Vec3 norm() const { float l = len(); return l > 1e-7f ? *this * (1/l) : Vec3(0,0,1); }
};

struct Mat4 {
    float m[16];
    __host__ __device__ Mat4() { memset(m, 0, sizeof(m)); }

    static Mat4 identity() {
        Mat4 r; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r;
    }

    static Mat4 perspective(float fovY, float aspect, float near_, float far_) {
        Mat4 r;
        float f = 1.0f / tanf(fovY * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (far_ + near_) / (near_ - far_);
        r.m[11] = -1;
        r.m[14] = 2 * far_ * near_ / (near_ - far_);
        return r;
    }

    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f = (center - eye).norm();
        Vec3 r = f.cross(up).norm();
        Vec3 u = r.cross(f);
        Mat4 res;
        res.m[0]=r.x;  res.m[4]=r.y;  res.m[8]=r.z;   res.m[12]=-(r.x*eye.x+r.y*eye.y+r.z*eye.z);
        res.m[1]=u.x;  res.m[5]=u.y;  res.m[9]=u.z;   res.m[13]=-(u.x*eye.x+u.y*eye.y+u.z*eye.z);
        res.m[2]=-f.x; res.m[6]=-f.y; res.m[10]=-f.z; res.m[14]= (f.x*eye.x+f.y*eye.y+f.z*eye.z);
        res.m[15]=1;
        return res;
    }

    static Mat4 mul(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int c = 0; c < 4; c++) for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[k*4+rr] * b.m[c*4+k];
            r.m[c*4+rr] = s;
        }
        return r;
    }
};
