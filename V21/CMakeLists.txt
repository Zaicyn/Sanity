cmake_minimum_required(VERSION 3.16)
project(Sanity_V21 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ==============================================================================
# OPTIONS — vendors are extensions, not the core
# ==============================================================================

option(ENABLE_NVIDIA "Build NVIDIA/CUDA backend (optional)" OFF)
option(ENABLE_AMD    "Build AMD/ROCm backend (future)" OFF)
option(ENABLE_INTEL  "Build Intel/oneAPI backend (future)" OFF)

# ==============================================================================
# FIND DEPENDENCIES
# ==============================================================================

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found — install Vulkan SDK or set VULKAN_SDK")
endif()

# ==============================================================================
# SPIRV SHADER COMPILATION — always happens (this is the core)
# ==============================================================================

set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/kernels)

file(MAKE_DIRECTORY ${SPV_DIR})

set(KERNEL_SOURCES
    ${KERNEL_DIR}/siphon.comp
    ${KERNEL_DIR}/passive.comp
    ${KERNEL_DIR}/hopfion.comp
    ${KERNEL_DIR}/scatter.comp
    ${KERNEL_DIR}/gather.comp
    ${KERNEL_DIR}/spawn.comp
    ${KERNEL_DIR}/region.comp
    ${KERNEL_DIR}/accumulator.comp
)

set(SPV_OUTPUTS)
foreach(SHADER ${KERNEL_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_FILE ${SPV_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv"
    )
    list(APPEND SPV_OUTPUTS ${SPV_FILE})
endforeach()

add_custom_target(v21_shaders ALL DEPENDS ${SPV_OUTPUTS})

# ==============================================================================
# V21 CORE LIBRARY — vendor-neutral, always built
# ==============================================================================

add_library(v21_core INTERFACE)
target_include_directories(v21_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)
add_dependencies(v21_core v21_shaders)

# ==============================================================================
# V21 TEST — CPU allocator smoke test
# ==============================================================================

add_executable(v21_test_cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_alloc.c
)
target_link_libraries(v21_test_cpu PRIVATE v21_core m)

# ==============================================================================
# NVIDIA BACKEND (optional)
# ==============================================================================

if(ENABLE_NVIDIA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "NVIDIA backend ENABLED")

    add_library(v21_nvidia STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/vendors/nvidia_backend.cpp
    )
    target_link_libraries(v21_nvidia
        PUBLIC  v21_core
        PRIVATE CUDA::cudart
    )
    target_compile_definitions(v21_nvidia PUBLIC WITH_NVIDIA)
else()
    message(STATUS "NVIDIA backend disabled (default)")
endif()

# ==============================================================================
# AMD BACKEND (future)
# ==============================================================================

if(ENABLE_AMD)
    message(STATUS "AMD backend ENABLED (placeholder)")
    # find_package(amdgcn REQUIRED)
    # add_library(v21_amd STATIC vendors/amd_backend.cpp)
    # target_link_libraries(v21_amd PUBLIC v21_core PRIVATE amdgcn::compiler)
endif()

# ==============================================================================
# INTEL BACKEND (future)
# ==============================================================================

if(ENABLE_INTEL)
    message(STATUS "Intel backend ENABLED (placeholder)")
    # find_package(oneAPI REQUIRED)
    # add_library(v21_intel STATIC vendors/intel_backend.cpp)
    # target_link_libraries(v21_intel PUBLIC v21_core PRIVATE ze_loader)
endif()

# ==============================================================================
# INSTALL — ship SPIRV binaries + headers
# ==============================================================================

install(DIRECTORY core/ DESTINATION include/v21
    FILES_MATCHING PATTERN "*.h"
)
install(FILES ${SPV_OUTPUTS} DESTINATION share/v21/kernels)

# ==============================================================================
# STATUS
# ==============================================================================

message(STATUS "")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "  V21 Compute Backend — SPIRV First")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "  Core headers: ${CMAKE_CURRENT_SOURCE_DIR}/core/")
message(STATUS "  GLSL kernels: ${KERNEL_DIR}/ (${CMAKE_CURRENT_LIST_DIR})")
message(STATUS "  SPIRV output: ${SPV_DIR}/")
message(STATUS "  Vulkan SDK:   ${Vulkan_INCLUDE_DIRS}")
message(STATUS "  glslc:        ${GLSLC}")
message(STATUS "  NVIDIA:       ${ENABLE_NVIDIA}")
message(STATUS "  AMD:          ${ENABLE_AMD}")
message(STATUS "  Intel:        ${ENABLE_INTEL}")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "")
