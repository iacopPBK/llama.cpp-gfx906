ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI50/MI60, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
Flash Attention kernel: TILE_Q8 (dot4)
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | type_k | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           pp512 |       1250.02 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | ROCm       |  99 |      12 |   q8_0 |  1 |           tg128 |        141.82 ± 0.00 |

build: e98fff432 (7333)
