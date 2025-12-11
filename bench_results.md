ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1308.08 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        146.93 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: depth run 1/1
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |   pp512 @ d8192 |       1038.30 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: depth run 1/1 (cached)
llama-bench: benchmark 2/2: generation run 1/1
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |   tg128 @ d8192 |        135.38 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1515.38 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        109.62 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: depth run 1/1
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |   pp512 @ d8192 |        742.52 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: depth run 1/1 (cached)
llama-bench: benchmark 2/2: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |   tg128 @ d8192 |         85.43 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       2154.90 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        141.47 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: depth run 1/1
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   pp512 @ d8192 |       1028.79 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: depth run 1/1 (cached)
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   tg128 @ d8192 |        108.90 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_1                  |   2.41 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       2139.70 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q4_1                  |   2.41 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        144.77 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: depth run 1/1
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_1                  |   2.41 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   pp512 @ d8192 |       1076.72 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: depth run 1/1 (cached)
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q4_1                  |   2.41 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   tg128 @ d8192 |        104.18 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1383.90 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        117.62 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: depth run 1/1
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   pp512 @ d8192 |        834.13 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: depth run 1/1 (cached)
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      12 |   q8_0 |  1 |   tg128 @ d8192 |         93.83 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3next 80B.A3B Q2_K - Medium |  27.16 GiB |    79.67 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |        103.51 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3next 80B.A3B Q2_K - Medium |  27.16 GiB |    79.67 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |         17.84 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1522.11 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        109.46 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1667.58 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        110.62 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1666.03 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3vlmoe 30B.A3B Q4_1        |  17.87 GiB |    30.53 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        110.44 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1383.83 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        156.03 ± 0.00 |

build: 3190c82fd (7387)
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1417.11 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        158.05 ± 0.00 |

build: 3190c82fd (7387)
