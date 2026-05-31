# VG Boundary Validation Experiment

- Run: `vg_boundary_20260527_072745`
- Output: `/home/cc/yhbian/B01_Copy_API/vg_boundary_results/vg_boundary_20260527_072745`
- Model: `3B`
- Steps: `8`
- Question: when should devices form a VG for intra-stage TP, and when is inter-VG PP better?

## Summary

| scenario | bandwidth Mbps | VG wall ms | PP wall ms | VG-PP wall ms | VG tok/s | PP tok/s | tok/s delta % | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| high_bw_low_skew | 1000 | 2935.344 | 4475.739 | -1540.395 | 6.9808 | 2.81889 | 147.644 | VG better/competitive |
| low_bw_low_skew | 10 | 3957.82 | 6240.996 | -2283.175 | 3.91389 | 1.99203 | 96.477 | VG better/competitive |
| high_bw_high_skew | 1000 | 6842.809 | 5823.439 | 1019.37 | 1.36799 | 1.96367 | -30.335 | PP better |

## Interpretation Guide

- If VG wins at high bandwidth/low skew, that supports VG as useful inside a bandwidth/compute boundary.
- If PP wins at low bandwidth or high compute skew, that supports the fallback-to-PP boundary condition.
- This is a validation sweep, not a final exhaustive paper experiment.
