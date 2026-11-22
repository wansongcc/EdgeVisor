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
EDGEVISOR_CONTROL_LOG ?= 0
CXXFLAGS += -DEDGEVISOR_CONTROL_LOG=$(EDGEVISOR_CONTROL_LOG)

DEBUG_OP_INPUT_OUTPUT ?= 0
CXXFLAGS += -DDEBUG_OP_INPUT_OUTPUT=$(DEBUG_OP_INPUT_OUTPUT)

# Compile-time gate for attention debug code blocks in nn-cpu/nn-cpu-ops.
EDGEVISOR_DEBUG_ATTN ?= 0
CXXFLAGS += -DEDGEVISOR_DEBUG_ATTN=$(EDGEVISOR_DEBUG_ATTN)

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

# Bare `make` builds the binary. `make clean` still cleans.
.DEFAULT_GOAL := edgevisor

# 0 or 1 from the command line or environment wins. Otherwise detect.
ifeq ($(strip $(EDGEVISOR_CUDA)),)
  ifneq ($(shell if [ -x "$(NVCC)" ] || command -v "$(NVCC)" >/dev/null 2>&1; then echo yes; fi),)
    EDGEVISOR_CUDA := 1
  else
    EDGEVISOR_CUDA := 0
  endif
endif
ifeq ($(strip $(EDGEVISOR_VULKAN)),)
  ifneq ($(shell command -v glslc >/dev/null 2>&1 && echo yes || command -v glslangValidator >/dev/null 2>&1 && echo yes),)
    EDGEVISOR_VULKAN := 1
  else
    EDGEVISOR_VULKAN := 0
  endif
endif
$(info edgevisor backends: cuda=$(EDGEVISOR_CUDA) vulkan=$(EDGEVISOR_VULKAN))

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifeq ($(EDGEVISOR_VULKAN),1)
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
	CXXFLAGS += -DEDGEVISOR_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DEDGEVISOR_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifeq ($(EDGEVISOR_CUDA),1)
CUDA_ARCHS_DETECTED := $(shell \
	detected="`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[^0-9.]//g; s/\.//g' | awk 'NF' | sort -u | tr '\n' ' '`"; \
	if [ -z "$$detected" ] && [ -r /proc/device-tree/compatible ]; then \
		compatible="`tr '\000' '\n' < /proc/device-tree/compatible 2>/dev/null || true`"; \
		if printf '%s' "$$compatible" | grep -q 'tegra234'; then detected="87"; \
		elif printf '%s' "$$compatible" | grep -q 'tegra194'; then detected="72"; \
		elif printf '%s' "$$compatible" | grep -q 'tegra186'; then detected="62"; \
		elif printf '%s' "$$compatible" | grep -q 'tegra210'; then detected="53"; fi; \
	fi; \
	printf '%s' "$$detected")
ifneq ($(filter auto native,$(CUDA_ARCHS)),)
ifneq ($(strip $(CUDA_ARCHS_DETECTED)),)
CUDA_ARCHS_EFFECTIVE := $(strip $(CUDA_ARCHS_DETECTED))
else
CUDA_ARCHS_EFFECTIVE := $(CUDA_ARCHS_FALLBACK)
$(warning CUDA_ARCHS=$(CUDA_ARCHS) requested but GPU architecture could not be detected; falling back to $(CUDA_ARCHS_EFFECTIVE). Set CUDA_ARCHS manually for cross-compilation.)
endif
else
CUDA_ARCHS_EFFECTIVE := $(CUDA_ARCHS)
endif
$(info CUDA_ARCHS=$(CUDA_ARCHS_EFFECTIVE) requested=$(CUDA_ARCHS))
	CXXFLAGS += -DEDGEVISOR_CUDA -I$(CUDA_HOME)/include
	LIBS += -L$(CUDA_HOME)/lib64 -lcudart -lcublas
	NVCCFLAGS += -std=c++11 -Isrc -DEDGEVISOR_CUDA
	# Device-code optimization. -DNDEBUG is deliberately NOT added globally so
	# the unit tests keep their assert() checks. --use_fast_math is deliberately
	# NOT added either: it flips rounding in the RMSNorm/softmax/SiLU device
	# kernels and changes the generated tokens (verified on 4xT4: the final
	# segment diverges while -O3 alone reproduces the baseline text exactly),
	# and it measured no faster than plain -O3.
	NVCCFLAGS += -O3
	# nvcc's ARM device front-end cannot parse GCC's arm_neon.h built-ins.
	# The CUDA translation unit does not execute the CPU NEON paths; keep NEON
	# enabled for normal C++ objects, but hide it while compiling .cu files.
	NVCCFLAGS += -U__ARM_NEON
	NVCCFLAGS += $(foreach arch,$(CUDA_ARCHS_EFFECTIVE),-gencode arch=compute_$(arch),code=sm_$(arch))
	DEPS += nn-cuda.o
endif

ifeq ($(TOPK), 1)
    CXXFLAGS += -DEDGEVISOR_DEBUG_TOPK_LOGITS=1
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

DEPS += io-profile.o product-log.o device-profile.o sustained-profile.o

.PHONY: clean edgevisor

clean:
	$(DELETE_CMD) *.o *.d edgevisor edgevisor-* socket-benchmark mmap-buffer-* *-test *.exe src/nn/vulkan/*.spv

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
product-log.o: src/product_log.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
device-profile.o: src/device_profile.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
sustained-profile.o: src/sustained_profile.cpp
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
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o product-log.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o product-log.o -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o product-log.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o product-log.o -o $@ $(LIBS)
nn-slice-test: src/nn/nn-slice-test.cpp nn-quants.o nn-core.o product-log.o
	$(CXX) $(CXXFLAGS) src/nn/nn-slice-test.cpp nn-quants.o nn-core.o product-log.o -o $@ $(LIBS)
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cuda.o: src/nn/nn-cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ifeq ($(EDGEVISOR_VULKAN),1)
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
	$(CXX) $(CXXFLAGS) src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-vulkan.o nn-test-runtime-stubs.o plan-command.o ablation.o io-profile.o product-log.o -o $@ $(LIBS)
endif

ifeq ($(EDGEVISOR_CUDA),1)
nn-cuda-test: src/nn/nn-cuda-test.cpp nn-cuda.o nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o io-profile.o product-log.o
	$(CXX) $(CXXFLAGS) src/nn/nn-cuda-test.cpp nn-cuda.o nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o nn-test-runtime-stubs.o plan-command.o ablation.o io-profile.o product-log.o -o $@ $(LIBS)
endif

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
token-timing.o: src/token_timing.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
terminal-ui.o: src/terminal_ui.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
ui-preview: src/terminal_ui_preview.cpp terminal-ui.o
	$(CXX) $(CXXFLAGS) src/terminal_ui_preview.cpp terminal-ui.o -o $@ $(LIBS)
ablation.o: src/ablation.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
dynamic-layer.o: src/dynamic/dynamic_layer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
tpot-algorithm.o: src/dynamic/tpot_algorithm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
algo2.o: src/dynamic/algo2.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
dynamic-tpot.o: src/dynamic/dynamic_tpot.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
kv-collector.o: src/dynamic/kv_collector.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
plan-command.o: src/plan-command.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
plan-controller.o: src/plan-controller.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-cpu-ops.o tokenizer.o product-log.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

dynamic-tpot-test: src/test/test_dynamic_tpot_algorithm.cpp tpot-algorithm.o
	$(CXX) $(CXXFLAGS) $(filter-out %.hpp %.h, $^) -o $@ $(LIBS)

algo2-test: src/test/test_algo2.cpp algo2.o tpot-algorithm.o
	$(CXX) $(CXXFLAGS) $(filter-out %.hpp %.h, $^) -o $@ $(LIBS)

failover-gate-test: src/test/test_failover_gate.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

device-profile-test: src/test/test_device_profile.cpp
	$(CXX) $(CXXFLAGS) src/test/test_device_profile.cpp -o $@ $(LIBS)

token-timing-test: src/test/test_token_timing.cpp token-timing.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

mmap-range-test: src/test/test_mmap_range.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

weight-chunking-test: src/test/test_weight_chunking.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

app-cli-test: src/test/test_app_cli.cpp terminal-ui.o token-timing.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

# --- (您的新测试目标) ---
uneven-slice-test: src/test/test_UnevenSlice.cpp nn-quants.o nn-core.o 
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

worker-load-test: src/test/test_LoadWeightLoacl.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-network-local.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

Hybie-plan-test: src/test/test_pp_tp.cpp nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o ablation.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

pointer-swizzling-microbench: src/test/pointer_swizzling_microbench.cpp nn-quants.o nn-core.o nn-executor.o nn-cpu.o nn-cpu-ops.o llamafile-sgemm.o plan-command.o ablation.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

edgevisor: src/edgevisor.cpp terminal-ui.o token-timing.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

edgevisor-api: src/edgevisor-api.cpp terminal-ui.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

uneven-llm-build-test: src/test/test_UnevenLlmBuild.cpp terminal-ui.o nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

# Include auto-generated dependency files (created by -MMD).
-include $(wildcard *.d)

pp-failover-network-test: src/test/test_pp_failover_network.cpp terminal-ui.o token-timing.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter %.cpp %.o,$^) -o $@ $(LIBS)

# Optional model-backed test; controller synchronization is confined to this
# executable. Build in CI, run on devices with user-prepared model files.
reserved-join-test: src/test/test_reserved_join.cpp src/edgevisor.cpp terminal-ui.o token-timing.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) src/test/test_reserved_join.cpp $(filter %.o,$^) -o $@ $(LIBS)

delayed-sample-worker-test: src/test/test_delayed_sample.cpp src/edgevisor.cpp terminal-ui.o token-timing.o nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o plan-command.o plan-controller.o tokenizer.o llm.o app.o ablation.o dynamic-layer.o tpot-algorithm.o algo2.o dynamic-tpot.o kv-collector.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter %.cpp %.o,$^) -Wl,--wrap=send -o $@ $(LIBS)

# Model-free Linux acceptance gate. GPU builds run the same host-side tests.
CHECK_TESTS := nn-cpu-test nn-cpu-ops-test tokenizer-test app-cli-test \
 dynamic-tpot-test algo2-test failover-gate-test token-timing-test device-profile-test \
 pp-failover-network-test
.PHONY: check
check: $(CHECK_TESTS) reserved-join-test delayed-sample-worker-test
	@set -e; for test in $(CHECK_TESTS); do ./$$test; done
