# Cross-Compile and Run on ARM (Jetson CPU-only + Raspberry Pi)

本文面向以下目标：

1. 在 x86_64 服务器上完成 `dllama` / `dllama-api` 交叉编译。
2. 将二进制直接分发到实验设备（Jetson、树莓派）运行。
3. 明确后端算子是否依赖特定 CPU 指令优化，以及编译时如何开启。

> 说明：本文重点是 **Jetson 仅 CPU 运行**。如果你需要 Jetson GPU（Vulkan），请参考 `docs/HOW_TO_RUN_GPU.md`。

---

## 1. 结论先看（务实版）

- 代码里存在大量 ARM SIMD 优化路径（NEON），并且部分量化 matmul 路径有 `DOTPROD` 分支。
- 这些优化是 **编译期** 选择，不是运行时参数。
- 目前 `Makefile` 默认有 `-march=native -mtune=native`，交叉编译时不应使用该默认行为。
- 推荐两套构建策略：
  - **兼容优先**：构建可在更多 ARM64 设备上运行（不强依赖 dotprod）。
  - **性能优先（Jetson 定向）**：为特定 Jetson CPU 目标开启更激进 ISA（例如 dotprod）。

---

## 2. 代码级检查结果：哪些算子有架构优化

以下结论来自当前仓库代码：

### 2.1 `src/nn/nn-cpu-ops.cpp`

- 文件头根据宏切换：
  - `__ARM_NEON` -> `#include <arm_neon.h>`
  - `__AVX2__` / `__AVX512F__` -> x86 SIMD
- 多个核心算子存在 ARM NEON 快路径，例如：
  - `invRms_F32`
  - `rmsNorm_F32`
  - `matmul_F32_F32_F32`
  - `matmul_Q80_Q40_F32`
  - `silu_F32`
  - `softmax_F32`
  - `add_Q80_F32`
- 在 `matmul_Q80_Q40_F32` 等路径中，存在 `#if defined(__ARM_FEATURE_DOTPROD)` 分支，表示可使用 ARM dot-product 指令。

### 2.2 `src/nn/nn-quants.cpp`

- `quantizeF32toQ80` 同时提供：
  - `__ARM_NEON` 路径
  - `__AVX2__` 路径
  - 标量回退路径

### 2.3 `src/nn/llamafile/sgemm.cpp`

- `llamafile_sgemm` 在 ARM 路径中也使用 `__ARM_FEATURE_DOTPROD` 条件分支。
- 说明不仅算子本体，矩阵乘法 fast path 也受 dotprod 宏影响。

### 2.4 运行时如何确认编译出的指令集

程序启动时会打印 CPU 指令集（`printCpuInstructionSet()`），如：

```text
🧠 CPU: neon dotprod fp16
```

如果只看到 `neon` 而没有 `dotprod`，通常说明编译目标未启用 dotprod ISA。

---

## 3. 交叉编译前准备（服务器）

以下示例以 Ubuntu 服务器为例。

### 3.1 安装交叉工具链

```bash
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu rsync file
```

### 3.2 准备目标设备 sysroot（强烈建议）

建议分别从 Jetson 与树莓派同步 sysroot：

```bash
# Jetson
mkdir -p /opt/sysroots/jetson
rsync -a --delete nvidia@<jetson-ip>:/lib/ /opt/sysroots/jetson/lib/
rsync -a --delete nvidia@<jetson-ip>:/usr/ /opt/sysroots/jetson/usr/

# Raspberry Pi
mkdir -p /opt/sysroots/rpi
rsync -a --delete pi@<rpi-ip>:/lib/ /opt/sysroots/rpi/lib/
rsync -a --delete pi@<rpi-ip>:/usr/ /opt/sysroots/rpi/usr/
```

这样可减少目标机 glibc / libstdc++ / 依赖版本不匹配问题。

---

## 4. 服务器交叉编译命令

先进入仓库根目录：

```bash
cd distributed-llama
```

> 关键点：当前 `Makefile` 在 `ifndef TERMUX_VERSION` 中会加 `-march=native -mtune=native`。交叉编译时可通过 `TERMUX_VERSION=1` 避免该行为。

### 4.1 Jetson CPU-only（兼容优先）

```bash
make clean
make TERMUX_VERSION=1 \
  CXX="aarch64-linux-gnu-g++ --sysroot=/opt/sysroots/jetson -O3" \
  dllama dllama-api
```

说明：
- 使用目标机 sysroot。
- 不强制 dotprod，兼容性更高。

### 4.2 Jetson CPU-only（性能优先，定向机型）

```bash
make clean
make TERMUX_VERSION=1 \
  CXX="aarch64-linux-gnu-g++ --sysroot=/opt/sysroots/jetson -O3 -mcpu=<your-jetson-cpu>" \
  dllama dllama-api
```

可选：如果你确定目标 CPU 与工具链支持，也可用 `-march=armv8.2-a+dotprod` 之类参数。

说明：
- 该模式可触发更多 ISA 宏（例如 `__ARM_FEATURE_DOTPROD`），使量化 matmul 与 sgemm 走更快路径。
- 产物可移植性下降，不保证可直接在不同 ARM 设备间通用。

### 4.3 Raspberry Pi（CPU）

```bash
make clean
make TERMUX_VERSION=1 \
  CXX="aarch64-linux-gnu-g++ --sysroot=/opt/sysroots/rpi -O3" \
  dllama dllama-api
```

> 若树莓派系统不是 64 位，请不要使用上述 aarch64 目标。

---

## 5. 打包与分发二进制

### 5.1 仅 CPU 包（Jetson CPU-only / RPi）

```bash
mkdir -p out/jetson-cpu
cp dllama dllama-api out/jetson-cpu/
tar czf out/jetson-cpu.tgz -C out/jetson-cpu .

scp out/jetson-cpu.tgz nvidia@<jetson-ip>:/opt/dllama/
```

树莓派同理，只需替换目标地址。

### 5.2 目标机解包并验证

```bash
mkdir -p /opt/dllama && cd /opt/dllama
tar xzf jetson-cpu.tgz

file ./dllama
ldd ./dllama
```

确认：
- `file` 显示 `aarch64`。
- `ldd` 无 `not found`。

---

## 6. 在 Jetson（CPU-only）和树莓派运行

以下命令两者通用（仅 CPU）：

### 6.1 启动 worker

```bash
./dllama worker --port 9999 --nthreads 4
```

### 6.2 root 节点做推理

```bash
./dllama inference \
  --prompt "Hello world" \
  --steps 32 \
  --model models/<your-model>/dllama_model_<your-model>.m \
  --tokenizer models/<your-model>/dllama_tokenizer_<your-model>.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers <worker1-ip>:9999 <worker2-ip>:9999
```

### 6.3 启动 API

```bash
./dllama-api \
  --port 9999 \
  --model models/<your-model>/dllama_model_<your-model>.m \
  --tokenizer models/<your-model>/dllama_tokenizer_<your-model>.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers <worker1-ip>:9999 <worker2-ip>:9999
```

---

## 7. 你最关心的问题：要不要“额外开启”CPU 架构优化？

### 7.1 NEON

- AArch64 目标下通常会自动可用（编译器定义 `__ARM_NEON`）。
- 一般不需要额外手动开关。

### 7.2 DOTPROD（重点）

- 代码中确实有 dotprod 优化分支（算子与 sgemm 都有）。
- 仅当编译目标启用了对应 ISA 特性时，编译器才会定义 `__ARM_FEATURE_DOTPROD`。
- 若你希望稳妥跨设备运行，可不强制 dotprod；若你追求某一类 Jetson 上的性能，可按设备能力定向 `-mcpu` 或 `-march=...+dotprod`。

### 7.3 实际建议

- **实验室多机混跑、重视可用性**：先用兼容优先版本。
- **固定 Jetson 型号、追求性能**：再单独出一份性能优先版本。
- 运行时观察启动日志中的 `🧠 CPU:`，验证是否命中预期 ISA。

---

## 8. 常见问题排查

1. **设备上 `./dllama: not found`**
   - 可能是动态链接器/依赖不匹配，不一定是文件不存在。
   - 用 `file` + `ldd` 检查架构和依赖。

2. **启动后没有 `dotprod` 标记**
   - 编译时未启用相应 ISA，或工具链未支持该扩展。
   - 调整 `-mcpu/-march` 后重新构建。

3. **同一二进制在另一台 ARM 板卡崩溃/非法指令**
   - 常见于“性能优先”构建过于激进。
   - 改用“兼容优先”构建。

---

## 9. 推荐产物组织方式

建议在 CI 或脚本中同时产出：

- `dllama-jetson-cpu-compatible.tgz`
- `dllama-jetson-cpu-optimized.tgz`
- `dllama-rpi-cpu-compatible.tgz`

并在包名中标注：

- toolchain 版本
- sysroot 来源
- 是否包含 `dotprod` 优化

这样最便于实验设备批量部署和问题回溯。
