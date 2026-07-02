CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security -Isrc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCHS ?= auto
CUDA_ARCHS_FALLBACK ?= 75

# Auto-generate header dependencies for correct incremental builds.
# Without this, changing headers can silently produce ABI mismatches (e.g., sizeof changes)
# between stale *.o files and newly compiled translation units.
CXXFLAGS += -MMD -MP

# Optional debug toggles (default OFF)
DLLAMA_CONTROL_LOG ?= 0
CXXFLAGS += -DDLLAMA_CONTROL_LOG=$(DLLAMA_CONTROL_LOG)

DEBUG_OP_INPUT_OUTPUT ?= 0
CXXFLAGS += -DDEBUG_OP_INPUT_OUTPUT=$(DEBUG_OP_INPUT_OUTPUT)

# Compile-time gate for attention debug code blocks in nn-cpu/nn-cpu-ops.
DLLAMA_DEBUG_ATTN ?= 0
CXXFLAGS += -DDLLAMA_DEBUG_ATTN=$(DLLAMA_DEBUG_ATTN)

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifdef DLLAMA_VULKAN
ifeq ($(OS),Windows_NT)
	CGLSLC ?= glslc
else
	CGLSLC ?= $(shell command -v glslc 2>/dev/null || command -v glslangValidator 2>/dev/null)
endif

CGLSLC_NAME := $(notdir $(firstword $(CGLSLC)))

ifeq ($(strip $(CGLSLC)),)
$(error No GLSL compiler found. Please install glslc or glslang-tools, or pass CGLSLC=/path/to/glslc)
endif

$(info Vulkan shader compiler: $(CGLSLC_NAME) [$(CGLSLC)])

ifeq ($(OS),Windows_NT)
	LIBS += -L$(VK_SDK_PATH)\lib -lvulkan-1
	CXXFLAGS += -DDLLAMA_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DDLLAMA_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifdef DLLAMA_CUDA
CUDA_ARCHS_DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[^0-9.]//g; s/\.//g' | awk 'NF' | sort -u | tr '\n' ' ')
ifneq ($(filter auto native,$(CUDA_ARCHS)),)
ifneq ($(strip $(CUDA_ARCHS_DETECTED)),)
CUDA_ARCHS_EFFECTIVE := $(strip $(CUDA_ARCHS_DETECTED))
else
CUDA_ARCHS_EFFECTIVE := $(CUDA_ARCHS_FALLBACK)
$(warning CUDA_ARCHS=$(CUDA_ARCHS) requested but nvidia-smi did not report compute capability; falling back to $(CUDA_ARCHS_EFFECTIVE). Set CUDA_ARCHS manually for cross-compilation.)
endif
else
CUDA_ARCHS_EFFECTIVE := $(CUDA_ARCHS)
endif
$(info CUDA_ARCHS=$(CUDA_ARCHS_EFFECTIVE) requested=$(CUDA_ARCHS))
	CXXFLAGS += -DDLLAMA_CUDA -I$(CUDA_HOME)/include
	LIBS += -L$(CUDA_HOME)/lib64 -lcudart -lcublas
	NVCCFLAGS += -std=c++11 -Isrc -DDLLAMA_CUDA
	NVCCFLAGS += $(foreach arch,$(CUDA_ARCHS_EFFECTIVE),-gencode arch=compute_$(arch),code=sm_$(arch))
	DEPS += nn-cuda.o
endif

ifeq ($(TOPK), 1)
    CXXFLAGS += -DDLLAMA_DEBUG_TOPK_LOGITS=1
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

DEPS += io-profile.o

.PHONY: clean dllama

clean:
	$(DELETE_CMD) *.o *.d dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe src/nn/vulkan/*.spv

# nn
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
io-profile.o: src/nn/io-profile.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-network-local.o: src/nn/nn-network-local.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-test-runtime-stubs.o: src/nn/nn-test-runtime-stubs.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o -o $@ $(LIBS)
nn-slice-test: src/nn/nn-slice-test.cpp nn-quants.o nn-core.o
	$(CXX) $(CXXFLAGS) src/nn/nn-slice-test.cpp nn-quants.o nn-core.o -o $@ $(LIBS)
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cuda.o: src/nn/nn-cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ifdef DLLAMA_VULKAN
VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
DEPS += $(VULKAN_SHADER_BINS)

ifeq ($(CGLSLC_NAME),glslangValidator)
GLSL_COMPILE = $(CGLSLC) -V --target-env vulkan1.2 -S comp -Isrc/nn/vulkan $< -o $@
else
GLSL_COMPILE = $(CGLSLC) -c --target-env=vulkan1.2 -Isrc/nn/vulkan $< -o $@
endif

%.spv: %.comp
	$(GLSL_COMPILE)

nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-vulkan.o nn-test-runtime-stubs.o plan-command.o ablation.o ${DEPS}
	$(CXX) $(CXXFLAGS) src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-vulkan.o nn-test-runtime-stubs.o plan-command.o ablation.o -o $@ $(LIBS)
endif

ifdef DLLAMA_CUDA
nn-cuda-test: src/nn/nn-cuda-test.cpp nn-cuda.o nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o io-profile.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cuda-test.cpp nn-cuda.o nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o io-profile.o -o $@ $(LIBS)
endif

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
ablation.o: src/ablation.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
dynamic-layer.o: src/dynamic/dynamic_layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
tpot-algorithm.o: src/dynamic/tpot_algorithm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
dynamic-tpot.o: src/dynamic/dynamic_tpot.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
kv-collector.o: src/dynamic/kv_collector.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
plan-command.o: src/plan-command.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
plan-controller.o: src/plan-controller.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-cpu-ops.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

dynamic-tpot-test: src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o
	$(CXX) $(CXXFLAGS) $(filter-out %.hpp %.h, $^) -o $@ $(LIBS)

# --- (您的新测试目标) ---
uneven-slice-test: src/test/test_UnevenSlice.cpp nn-quants.o nn-core.o 
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

worker-load-test: src/test/test_LoadWeightLoacl.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-network-local.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

Hybie-plan-test: src/test/test_pp_tp.cpp nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o ablation.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

pointer-swizzling-microbench: src/test/pointer_swizzling_microbench.cpp nn-quants.o nn-core.o nn-executor.o nn-cpu.o nn-cpu-ops.o llamafile-sgemm.o plan-command.o ablation.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

dllama: src/dllama.cpp nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

dllama-api: src/dllama-api.cpp nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

uneven-llm-build-test: src/test/test_UnevenLlmBuild.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

# Include auto-generated dependency files (created by -MMD).
-include $(wildcard *.d)