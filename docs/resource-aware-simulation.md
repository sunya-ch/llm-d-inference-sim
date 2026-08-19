# Resource-Aware Simulation

This document describes how to configure the simulator to reflect the characteristics of the GPU resource it is assigned to, enabling more realistic simulation of shared or partitioned GPU environments.

In a real vLLM deployment, resource properties are derived at runtime from the CUDA context: the amount of GPU memory available drives KV cache capacity, and the fraction of SMs (Streaming Multiprocessors) allocated to the process influences compute throughput and therefore latency. The simulator cannot query a CUDA context, but it can read the same information from environment variables — the same variables that a DRA driver injects into the container.

---

## Background: DRA and GPU partitioning

The [dra-example-driver](https://github.com/kubernetes-sigs/dra-example-driver/tree/main/demo/examples/gpu-allow-multiple-allocations-partitionable) shows how Dynamic Resource Allocation (DRA) partitions a GPU and exposes the resulting resource slice to a container via environment variables. When a pod is granted a partition of GPU 0, the driver injects variables such as:

```sh
GPU_DEVICE_0_MEMORY=20Gi    # memory bytes visible to this partition
GPU_DEVICE_0_COMPUTE=40     # percentage of SMs assigned to this partition (0–100)
```

The simulator reads these variables (and their equivalents for other device indices) to derive two things:

1. **KV cache capacity** — derived from the reported memory, mirroring the calculation vLLM performs at startup.
2. **Compute latency factor** — derived from the reported compute share, applied as a multiplier to all GPU-bound latency parameters.

---

## KV cache capacity from GPU memory

### How vLLM calculates KV cache size

At startup, vLLM computes the number of KV cache blocks that fit in the GPU memory left over after loading model weights. The key formula is:

```
available_memory = total_gpu_memory × gpu_memory_utilization - model_weight_bytes
kv_cache_blocks  = floor(available_memory / bytes_per_block)
```

where `gpu_memory_utilization` defaults to `0.90` and `bytes_per_block` depends on the model's number of layers, KV heads, head dimension, and the block size (tokens per block):

```
bytes_per_block = 2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_elem × block_size
```

### Simulator behaviour

When `GPU_DEVICE_<N>_MEMORY` is set (and `--kv-cache-size` is **not** explicitly provided), the simulator derives `kv-cache-size` using the same formula. The architecture parameters (`num_layers`, `num_kv_heads`, `head_dim`) are looked up automatically from the `--model` name — the same well-known models already tabulated in [Latency Reference Tables and Profiles](latency-profiles.md#kv-cache-transfer) are recognised. For any model not in the built-in table, the parameters can be overridden manually (see [Configuration reference](#configuration-reference)).

Built-in model architectures:

| Model | Layers | KV heads | head_dim | Weights (FP16) | KV bytes/token (FP16) |
|-------|--------|----------|----------|----------------|-----------------------|
| Llama-3 / 3.1 8B | 32 | 8 | 128 | ~16 GB | ~128 KB |
| Llama-3 / 3.1 70B | 80 | 8 | 128 | ~140 GB | ~320 KB |
| Llama-3.1 405B | 126 | 16 | 128 | ~810 GB | ~1 MB |
| Mistral 7B (v0.3) | 32 | 8 | 128 | ~14 GB | ~128 KB |
| Mixtral 8×7B (MoE) | 32 | 8 | 128 | ~90 GB | ~128 KB |
| Qwen2.5 7B | 28 | 4 | 128 | ~15 GB | ~56 KB |
| Qwen2.5 72B | 80 | 8 | 128 | ~145 GB | ~320 KB |

When `model-weights-size-gb` is not set, the simulator uses the **Weights (FP16)** value from this table as the default. If the model is not in the table, the HuggingFace lookup (described below) retrieves it from the Hub; if neither is available the weight size defaults to `0`, which over-estimates the available memory and therefore the number of KV cache blocks — set `model-weights-size-gb` explicitly in that case.

For architecture parameters (`num-hidden-layers`, `num-kv-heads`, `head-dim`) not in this table, supply them directly (available in the model's HuggingFace `config.json`), or rely on the HuggingFace lookup described below.

The `gpu-memory-utilization` and `kv-cache-dtype` parameters mirror the matching vLLM flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu-memory-utilization` | `0.90` | Fraction of the reported GPU memory reserved for the KV cache pool |
| `model-weights-size-gb` | auto | GiB occupied by model weights; looked up from the built-in table or HuggingFace, falls back to `0` if unknown |
| `kv-cache-dtype` | `float16` | Data type for KV cache elements: `float16`, `bfloat16`, or `float8` |

The derived `kv_cache_blocks` value is identical to what vLLM would report in its startup logs under `# GPU blocks`. Both `gpu_memory_utilization` and `kv-cache-dtype` are also exported as labels on the `vllm:cache_config_info` Prometheus metric, matching vLLM's behaviour.

If `GPU_DEVICE_<N>_MEMORY` is not set, the simulator falls back to the explicit `--kv-cache-size` value (default `1024`).

### HuggingFace lookup for unlisted models

When `--model` is a real HuggingFace model ID (e.g. `google/gemma-2-9b`) and the model is not in the built-in table, the simulator fetches `config.json` from the HuggingFace Hub at startup and reads `num_hidden_layers`, `num_key_value_heads`, and `head_dim` directly. This is the same mechanism the tokenizer already uses (see [Tokenization](tokenization.md)). The HuggingFace render sidecar must be available for this path; set `--render-url` if it is not on the default `http://localhost:8082`.

### Environment variable format

The device index `N` matches the GPU device ordinal. The DRA container runtime injects these variables directly into the container's environment — they do not need to be declared in the pod spec. For a pod with a single partitioned GPU device the runtime sets:

```
GPU_DEVICE_0_MEMORY=20Gi   # memory visible to this partition, as a Kubernetes quantity
GPU_DEVICE_0_COMPUTE=40    # percentage of SMs assigned to this partition
```

Values for `MEMORY` are parsed as Kubernetes quantity strings (`20Gi`, `40960Mi`, `21474836480`, etc.).

If the DRA driver you use injects memory and compute under different variable names, point the simulator at them with `gpu-memory-env-var` and `gpu-compute-env-var`:

```yaml
gpu-memory-env-var: "MY_GPU_MEM"     # default: GPU_DEVICE_0_MEMORY
gpu-compute-env-var: "MY_GPU_COMPUTE" # default: GPU_DEVICE_0_COMPUTE
```

### Example: deriving KV cache size for Llama-3.1-8B

A pod is granted 20 GiB of GPU 0 memory (injected by the runtime as `GPU_DEVICE_0_MEMORY=20Gi`). The `--model` flag is `meta-llama/Llama-3.1-8B-Instruct`, which is in the built-in table (32 layers, 8 KV heads, head_dim 128, ~16 GB weights). With `float16` KV cache and block size 16:

```
bytes_per_block = 2 × 32 × 8 × 128 × 2 × 16 = 2,097,152 bytes = 2 MiB

available_memory = 20 GiB × 0.90 - 16 GiB = 2 GiB = 2,147,483,648 bytes

kv_cache_blocks = floor(2,147,483,648 / 2,097,152) = 1024
```

No architecture, weight, or environment configuration is needed — the model name and the runtime-injected env var are sufficient:

```yaml
# simulator YAML config
model: "meta-llama/Llama-3.1-8B-Instruct"
enable-kvcache: true
block-size: 16
gpu-memory-utilization: 0.90
kv-cache-dtype: float16
# kv-cache-size is intentionally omitted; derived from GPU_DEVICE_0_MEMORY
# num-hidden-layers / num-kv-heads / head-dim are not needed; looked up from model name
```

---

## Compute latency factor from SM allocation

### How SM share affects inference latency

A GPU processes all workloads through Streaming Multiprocessors. When a partition receives only a fraction of the available SMs, the throughput of memory-bandwidth-bound and compute-bound kernels decreases proportionally. For a partition that holds `p %` of SMs on a GPU where the full-GPU latency is `L`, the expected latency scales approximately as:

```
latency_partitioned ≈ L × (100 / p)
```

This is an approximation — real workloads may see super-linear slowdown due to cache effects and scheduling overhead — but it is a useful first-order model.

### Simulator behaviour

When `GPU_DEVICE_<N>_COMPUTE` is set, the simulator computes a **SM compute factor** and applies it **once at startup** as a static multiplier to all GPU-bound latency parameters:

- `time-to-first-token` / `prefill-overhead` / `prefill-time-per-token`
- `inter-token-latency`

The factor is:

```
sm_factor = 100 / GPU_DEVICE_<N>_COMPUTE
```

For a partition with 40 % of SMs, `sm_factor = 2.5` — all GPU-bound latencies are 2.5× longer than they would be on the full GPU.

The SM factor is **always applied** to all four GPU-bound latency parameters, regardless of whether they were explicitly set on the command line. The values you configure (via flags, YAML, or defaults) are treated as full-GPU baselines; the SM factor scales them to reflect the actual SM allocation.

> **KV-cache transfer latencies are not scaled.** Transfer latency (`kv-cache-transfer-latency`, `kv-cache-transfer-time-per-token`) is network-bound, not GPU-bound, and is unaffected by the SM partition.

> **`time-factor-under-load` is independent.** The SM compute factor is a static baseline multiplier applied once at startup to the configured latency values. The `time-factor-under-load` is a separate, dynamic per-request multiplier applied on top when the request queue approaches saturation. Both effects are multiplicative and model different phenomena: SM fraction (hardware capacity) vs. queue contention (scheduling pressure).

### Environment variable format

Like `GPU_DEVICE_0_MEMORY`, the compute variable is injected by the container runtime and does not need to be declared in the pod spec:

```
GPU_DEVICE_0_COMPUTE=40   # 40% of SMs allocated to this container
```

Values are integers in the range `[1, 100]`. A value of `100` means the full GPU — the factor becomes `1.0` and latency is unchanged.

### Example: shared GPU simulation

Two pods share GPU 0 equally, each receiving 50 % of compute. The runtime injects `GPU_DEVICE_0_COMPUTE=50`. Pod A runs a 7B model on H100 with nominal (full-GPU) latency values:

```
time-to-first-token: 100ms
inter-token-latency: 12ms
```

With `GPU_DEVICE_0_COMPUTE=50`:

```
sm_factor = 100 / 50 = 2.0
effective time-to-first-token ≈ 200ms
effective inter-token-latency ≈ 24ms
```

The simulator config only needs the baseline (full-GPU) latency values — the SM factor is applied automatically:

```yaml
# simulator YAML config (base latencies for a full GPU)
model: "meta-llama/Llama-3.1-8B-Instruct"
latency-calculator: constant
time-to-first-token: 100ms
inter-token-latency: 12ms
# sm compute factor applied automatically from GPU_DEVICE_0_COMPUTE (injected by runtime)
```

---

## Using both features together (DRA partitioned GPU)

The DRA container runtime injects `GPU_DEVICE_0_MEMORY` and `GPU_DEVICE_0_COMPUTE` directly into the container environment. The only declaration needed in the pod spec is `POD_IP` (required for KV cache ZMQ topics):

```yaml
# Kubernetes pod spec (relevant sections)
env:
  - name: POD_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
  # GPU_DEVICE_0_MEMORY and GPU_DEVICE_0_COMPUTE are injected by the DRA runtime
```

```yaml
# simulator config.yaml
model: "meta-llama/Llama-3.1-8B-Instruct"
enable-kvcache: true
block-size: 16
gpu-memory-utilization: 0.90
kv-cache-dtype: float16
# kv-cache-size omitted — derived from GPU_DEVICE_0_MEMORY (runtime-injected)
# architecture params (layers/kv-heads/head-dim) omitted — looked up from model name

latency-calculator: per-token

inter-token-latency: 12ms
inter-token-latency-std-dev: 2ms

prefill-overhead: 30ms
prefill-time-per-token: 250us
prefill-time-std-dev: 5ms
```

Assuming the runtime injects `GPU_DEVICE_0_MEMORY=40Gi` and `GPU_DEVICE_0_COMPUTE=50`, at startup the simulator will:

1. Read `GPU_DEVICE_0_MEMORY=40Gi` → `gpuBytes = 42,949,672,960`. Look up Llama-3.1-8B from the built-in table (16 GB weights, 32 layers, 8 KV heads, head_dim 128, block size 16, `bpe=2`):

   ```text
   bytesPerBlock  = 2 × 32 × 8 × 128 × 2 × 16 = 2,097,152 (2 MiB)
   availableBytes = 40 GiB × 0.9 − 16 GiB = 20 GiB = 21,474,836,480
   kv_cache_blocks = floor(21,474,836,480 / 2,097,152) = 10,240
   ```

   Sets `kv-cache-size: 10240`.

2. Read `GPU_DEVICE_0_COMPUTE=50` → `sm_factor = 100 / 50 = 2.0`. Scales all GPU-bound latencies by `2.0×`:
   - `prefill-overhead`: 30 ms → **60 ms**
   - `prefill-time-per-token`: 250 µs → **500 µs**
   - `inter-token-latency`: 12 ms → **24 ms**

Startup log output:

> SM compute factor applied to GPU-bound latency parameters" smFactor=2 time-to-first-token="0s" prefill-overhead="60ms" prefill-time-per-token="500µs" inter-token-latency="24ms"

> KV cache auto-derivation succeeded" envVar="GPU_DEVICE_0_MEMORY" gpuBytes=42949672960 gpu-memory-utilization=0.9 model-weights-size-gb=16 num-hidden-layers=32 num-kv-heads=8 head-dim=128 bpe=2 block-size=16 bytesPerBlock=2097152 availableBytes=21474836480 kv-cache-blocks=10240

---

## Configuration reference

| Environment variable | Format | Description |
|----------------------|--------|-------------|
| `GPU_DEVICE_<N>_MEMORY` | Kubernetes quantity (`20Gi`, `21474836480`, …) | GPU memory for this partition, injected by the container runtime. Drives KV cache size derivation when `--kv-cache-size` is not set. Override the variable name with `gpu-memory-env-var`. |
| `GPU_DEVICE_<N>_COMPUTE` | Integer `1–100` | SM percentage for this partition, injected by the container runtime. A static `sm_factor = 100/compute%` is applied unconditionally at startup to all GPU-bound latency parameters (`time-to-first-token`, `prefill-overhead`, `prefill-time-per-token`, `inter-token-latency`). Override the variable name with `gpu-compute-env-var`. |

| Config parameter | Type | Default | Description |
|------------------|------|---------|-------------|
| `gpu-memory-env-var` | string | `GPU_DEVICE_0_MEMORY` | Name of the environment variable to read GPU memory from |
| `gpu-compute-env-var` | string | `GPU_DEVICE_0_COMPUTE` | Name of the environment variable to read GPU SM compute percentage from |
| `gpu-memory-utilization` | float | `0.90` | Fraction of the reported GPU memory reserved for the KV cache pool |
| `model-weights-size-gb` | float | auto | GiB occupied by model weights; looked up from the built-in table or HuggingFace, falls back to `0` if unknown |
| `kv-cache-dtype` | string | `float16` | Data type for KV cache elements; valid values: `float16`, `bfloat16`, `float8` |
| `num-hidden-layers` | int | auto | Number of transformer layers; only needed for models not in the built-in table and not on HuggingFace |
| `num-kv-heads` | int | auto | Number of KV attention heads per layer; only needed for models not in the built-in table and not on HuggingFace |
| `head-dim` | int | auto | Dimension of each KV attention head; only needed for models not in the built-in table and not on HuggingFace |

See [configuration.md](configuration.md) for all other simulator parameters, and [kv-cache.md](kv-cache.md) for detailed KV cache documentation.

---

## Troubleshooting

### Missing CDI Environment Variables in Kubernetes DRA Workloads

When the GPU memory environment variable is not injected into the container, startup logs will contain:

```text
"KV cache auto-derivation skipped"  reason="GPU memory env var not set" 
```

#### Symptom

- Target variables are completely missing from `crictl inspect <container-id> | jq '...env'`.
- The `.json`/`.yaml` spec file inside `/var/run/cdi/` correctly contains the `containerEdits` block.

#### Potential Root Cause

The container manifest omits standard CPU/Memory `requests`, placing the pod into the **BestEffort** QoS class. The Kubernetes DRA manager and container runtime (`containerd`/`cri-o`) require a baseline host footprint to calculate device scheduling tokens. Without it, the runtime treats the hardware claim as unbacked and silently skips the CDI injection phase.

❌ Broken Manifest Configuration

```yaml
resources:
  claims:
    - name: gpu-claim
```

✅ Fixed Manifest Configuration

```yaml
resources:
  requests:
    cpu: 100m      # Elevates QoS class to trigger DRA injection
    memory: 128Mi
  claims:
    - name: gpu-claim
```
