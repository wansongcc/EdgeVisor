# Pure VG Absorption Ablation

- Run: `vg_absorb_20260527_080453`
- Output: `/home/cc/yhbian/B01_Copy_API/vg_boundary_results/vg_absorb_20260527_080453`
- Model: `3B`
- Steps: `24`

## Summary

| scenario | with VG wall | no VG wall | with recover | with post ms | no VG post ms | with tok/s | no VG tok/s | delta % | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| compute_skew | 11264.732 | 14347.787 | 1213.319 | 511.25 | 609.812 | 1.95599 | 1.63985 | 19.279 | VG absorption better |
| bandwidth_skew | 49448.71 | 12750.537 | 8368.103 | 2894.375 | 505.375 | 0.3455 | 1.97873 | -82.539 | No-VG better/mixed |
| compute_and_bandwidth_skew | 58026.037 | 17404.634 | 5615.602 | 3432.5 | 802.375 | 0.29133 | 1.2463 | -76.624 | No-VG better/mixed |

## Interpretation

with_vg_absorb uses local Stage1 head migration to absorb Node4's wave inside the VG.
no_vg_pp_no_absorb has no VG; Node4 remains its own PP stage, so the same wave appears as slower post-wave token latency.
