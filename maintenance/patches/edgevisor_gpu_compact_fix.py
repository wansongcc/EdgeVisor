from pathlib import Path

root = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor")
p = root / "src/nn/nn-vulkan.cpp"
s = p.read_text()

old = """            } else if (opConfig->code == OP_MATMUL) {
            } else if (opConfig->code == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opConfig->config;
"""
new = """            } else if (opConfig->code == OP_MATMUL) {
                auto *cfg = (NnMatmulOpConfig *)opConfig->config;
"""
if old not in s:
    raise SystemExit("duplicate OP_MATMUL pattern not found")
s = s.replace(old, new, 1)

marker = """            } else if (opConfig->code == OP_MULTIHEAD_ATT) {
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
"""
replacement = """            } else if (opConfig->code == OP_MULTIHEAD_ATT) {
                // GPU dynamic migration is compact-contiguous: starts/lengths are
                // refreshed from NnDimSplit. Arbitrary sparse head sets require
                // a separate head-id map or weight/KV repacking path.
                auto *cfg = (NnMultiHeadAttOpConfig *)opConfig->config;
"""
if marker in s and replacement not in s:
    s = s.replace(marker, replacement, 1)

p.write_text(s)
print("fixed Vulkan MATMUL refresh branch and documented compact-contiguous GPU migration")
