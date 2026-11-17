# Native baseline concurrency comparison

# Native BF16 streaming baseline
| Concurrency | Valid audio | Wall time (s) | Audio seconds/s | Peak GPU (GiB) | Evidence |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1/1 | 0.70 | 15.53 | 4.83 | [20261117161834374360-9fc8577b](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117161834374360-9fc8577b/run.json) |
| 4 | 4/4 | 1.06 | 42.46 | 4.79 | [20261117160518967887-a44525a6](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160518967887-a44525a6/run.json) |
| 16 | 16/16 | 3.60 | 47.81 | 5.14 | [20261117161834374360-9fc8577b](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117161834374360-9fc8577b/run.json) |
| 64 | 64/64 | 10.45 | 68.12 | 6.49 | [20261117160519883604-b75f11cd](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160519883604-b75f11cd/run.json) |
| 256 | 256/256 | 163.58 | 17.35 | 7.48 | [20261117160602268099-ed904351](https://github.com/kreasof-ai/akas/blob/68ec4fd465c04f4552f8979a2204b601228f780b/benchmarks/runs/20261117160602268099-ed904351/run.json) |
