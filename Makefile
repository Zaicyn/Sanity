# Squaragon V20 Makefile
# ======================

NVCC = nvcc
NVFLAGS = -O3 -std=c++17

# Detect GPU architecture (default sm_75 for RTX 20xx)
CUDA_ARCH ?= sm_75

# Vulkan sources
VK_SOURCES = vulkan/vk_instance.cpp vulkan/vk_device.cpp vulkan/vk_swapchain.cpp \
             vulkan/vk_pipeline.cpp vulkan/vk_buffer.cpp vulkan/vk_cuda_interop.cpp \
             vulkan/vk_attractor.cpp
VK_LDFLAGS = -lvulkan -lglfw

# All CUDA headers (for proper dependency tracking)
CUDA_HEADERS := $(shell ls *.cuh 2>/dev/null) $(shell ls validator/*.cuh 2>/dev/null) siphon_pump.h squaragon.h

SHADER_DIR = vulkan/shaders

# Standard shaders: <name>.<stage>.spv (loaded by vk_pipeline.cpp)
SHADER_SRCS = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag $(SHADER_DIR)/*.comp)
SHADER_SPVS = $(addsuffix .spv,$(SHADER_SRCS))

# Alias shaders: vk_attractor.cpp uses different naming conventions
# harmonic_sample.comp -> harmonic_sample.spv (not .comp.spv)
# harmonic_phase.comp  -> harmonic_phase.spv
# harmonic_attractor.comp -> harmonic_attractor.spv
# tone_map.vert -> tone_map_vert.spv (underscore, not dot)
# tone_map.frag -> tone_map_frag.spv
SHADER_ALIASES = $(SHADER_DIR)/harmonic_sample.spv \
                 $(SHADER_DIR)/harmonic_phase.spv \
                 $(SHADER_DIR)/harmonic_attractor.spv \
                 $(SHADER_DIR)/tone_map_vert.spv \
                 $(SHADER_DIR)/tone_map_frag.spv

.PHONY: all clean vulkan shaders help

all: vulkan

# Compile GLSL shaders to SPIR-V (standard naming)
$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	glslc $< -o $@

# Alias rules for attractor shaders (strip .comp extension)
$(SHADER_DIR)/harmonic_sample.spv: $(SHADER_DIR)/harmonic_sample.comp
	glslc $< -o $@

$(SHADER_DIR)/harmonic_phase.spv: $(SHADER_DIR)/harmonic_phase.comp
	glslc $< -o $@

$(SHADER_DIR)/harmonic_attractor.spv: $(SHADER_DIR)/harmonic_attractor.comp
	glslc $< -o $@

# Alias rules for tone_map shaders (underscore naming)
$(SHADER_DIR)/tone_map_vert.spv: $(SHADER_DIR)/tone_map.vert
	glslc $< -o $@

$(SHADER_DIR)/tone_map_frag.spv: $(SHADER_DIR)/tone_map.frag
	glslc $< -o $@

shaders: $(SHADER_SPVS) $(SHADER_ALIASES)
	@echo "Shaders compiled to SPIR-V"

# Combined CUDA + Vulkan build (true interop, single app)
# Output to current dir (not build/) so shaders are found at runtime
blackhole_vulkan: blackhole_v20.cu physics.cu $(VK_SOURCES) vulkan/*.h $(CUDA_HEADERS) $(SHADER_SPVS) $(SHADER_ALIASES)
	$(NVCC) $(NVFLAGS) -DVULKAN_INTEROP -arch=$(CUDA_ARCH) \
		-I/usr/include \
		blackhole_v20.cu $(VK_SOURCES) \
		-o $@ $(VK_LDFLAGS)

vulkan: blackhole_vulkan
	@echo "Vulkan interop build complete (run with ./blackhole_vulkan)"

clean:
	rm -f blackhole_vulkan $(SHADER_DIR)/*.spv

help:
	@echo "Squaragon V20 Build System"
	@echo "=========================="
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build CUDA + Vulkan interop (default)"
	@echo "  make vulkan    - Build CUDA + Vulkan interop (single app, zero-copy)"
	@echo "  make shaders   - Compile GLSL shaders to SPIR-V"
	@echo "  make clean     - Remove binary and compiled shaders"
	@echo ""
	@echo "CUDA architecture (default sm_75):"
	@echo "  make CUDA_ARCH=sm_75   # RTX 20xx"
	@echo "  make CUDA_ARCH=sm_86   # RTX 30xx"
	@echo "  make CUDA_ARCH=sm_89   # RTX 40xx"
