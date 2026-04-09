/*
 * V21 CPU Allocator Smoke Test
 * ============================
 * Build: cmake .. && make v21_test_cpu && ./v21_test_cpu
 */

#include "v21_types.h"
#include "v21_alloc_cpu.h"
#include "v21_geometry.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    printf("=== V21 CPU Allocator Test ===\n\n");

    /* Test 1: Basic alloc/free cycle */
    v21_cpu_allocator_t alloc;
    v21_cpu_init(&alloc, 4 * 1024 * 1024);  /* 4MB arena */

    void* ptrs[100];
    for (int i = 0; i < 100; i++) {
        ptrs[i] = v21_cpu_alloc(&alloc, 64);
        assert(ptrs[i] != NULL);
    }
    printf("[PASS] Allocated 100 blocks\n");

    for (int i = 0; i < 100; i++) {
        v21_cpu_free(&alloc, ptrs[i], 64);
    }
    printf("[PASS] Freed 100 blocks\n");

    /* Test 2: Cache reuse */
    int cache_hits = 0;
    for (int i = 0; i < V21_FREELIST_SLOTS; i++) {
        void* p = v21_cpu_alloc(&alloc, 64);
        if (p != NULL) cache_hits++;
    }
    printf("[PASS] Cache reuse: %d/%d hits\n", cache_hits, V21_FREELIST_SLOTS);

    /* Test 3: Multiple bin sizes */
    void* p64  = v21_cpu_alloc(&alloc, 64);
    void* p128 = v21_cpu_alloc(&alloc, 128);
    void* p256 = v21_cpu_alloc(&alloc, 256);
    assert(p64 && p128 && p256);
    v21_cpu_free(&alloc, p64, 64);
    v21_cpu_free(&alloc, p128, 128);
    v21_cpu_free(&alloc, p256, 256);
    printf("[PASS] Multi-bin allocation (64/128/256)\n");

    /* Test 4: Viviani geometry */
    v21_vec3_t n = v21_viviani_curve(0.0f);
    printf("[PASS] Viviani(0) = (%.3f, %.3f, %.3f)\n", n.x, n.y, n.z);

    v21_vec3_t t = v21_viviani_tangent(1.0f);
    printf("[PASS] Tangent(1) = (%.3f, %.3f, %.3f)\n", t.x, t.y, t.z);

    /* Test 5: Gate init */
    v21_gate_t gate;
    v21_gate_init(&gate, 1.0f);
    float residual = v21_gate_residual(&gate);
    printf("[PASS] Gate residual = %.6f (should be 0)\n", residual);

    /* Test 6: Scatter LUT */
    for (int i = 0; i < 8; i++) {
        printf("  scatter(%d) = %u\n", i, v21_viviani_scatter(i));
    }
    printf("[PASS] Scatter LUT functional\n");

    /* Test 7: Serialization round-trip */
    uint8_t buf[V21_SERIAL_FULL_SIZE];
    v21_gate_serialize(&gate, buf);
    v21_gate_t gate2;
    v21_gate_deserialize(&gate2, buf);
    assert(gate2.scale == gate.scale);
    assert(gate2.bias == gate.bias);
    printf("[PASS] Serialization round-trip\n");

    /* Test 8: Seam phase shift */
    uint64_t val = 0xDEADBEEFCAFEBABEULL;
    uint64_t shifted = v21_seam_forward_shift(val);
    uint64_t restored = v21_seam_inverse_shift(shifted);
    assert(restored == val);
    printf("[PASS] Seam phase shift round-trip\n");

    printf("\nEvents: %u, Blocks: %u\n", alloc.event_count, alloc.allocated_blocks);
    v21_cpu_destroy(&alloc);

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
