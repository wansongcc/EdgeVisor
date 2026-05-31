# VG Ablation Skew Matrix

- Run: `vg_ablation_skew_20260527_080133`
- Output: `/home/cc/yhbian/B01_Copy_API/vg_boundary_results/vg_ablation_skew_20260527_080133`
- Model: `3B`
- Steps: `8`
- Comparison: `3/3/2 VG` vs `PP fallback` with weak node assigned only one layer.

## Summary

| scenario | VG wall ms | PP wall ms | VG-PP wall ms | VG tok/s | PP tok/s | delta % | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_skew_reference | 3156.22 | 4569.269 | -1413.049 | 5.68182 | 2.5047 | 126.847 | VG better |
| compute_skew | 7986.606 | 5801.677 | 2184.93 | 1.207 | 1.96657 | -38.624 | PP better |
| bandwidth_skew | 38576.477 | 6960.601 | 31615.876 | 0.21513 | 1.46252 | -85.29 | PP better |
| compute_and_bandwidth_skew | 41618.272 | 7623.724 | 33994.548 | 0.20628 | 1.38313 | -85.086 | PP better |

## Notes

- Compute skew is injected by limiting Node4 to 0.30 CPU.
- Bandwidth skew is injected with per-destination tc: traffic to/from Node4 is shaped to 2 Mbps + 80 ms delay.
- This is a validation sweep for VG boundary, not a final exhaustive paper experiment.
