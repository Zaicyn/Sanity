/*
 * V21 NVIDIA Backend — SPIRV → PTX via NVVM (stub)
 * =================================================
 * Compile with: cmake .. -DENABLE_NVIDIA=ON
 *
 * This is the NVIDIA extension. It compiles SPIRV to PTX
 * and dispatches via cuLaunchKernel. The SPIRV core runs
 * without this — it's an optional performance path.
 */

#ifdef WITH_NVIDIA

#include <cstdio>

// Placeholder: full implementation will use NVVM JIT
// to compile SPIRV → PTX and cuLaunchKernel for dispatch.

void v21_nvidia_init(void) {
    printf("[v21-nvidia] NVIDIA backend initialized (stub)\n");
}

void v21_nvidia_shutdown(void) {
    printf("[v21-nvidia] NVIDIA backend shut down\n");
}

#endif /* WITH_NVIDIA */
