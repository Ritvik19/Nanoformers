# Parallelism Learning Guide

A practical guide to the five main parallelism strategies used for scaling transformer training, written in the context of the Nanoformers project.

Running example used throughout:

> A **12-layer transformer**, hidden size **4096**, batch size **64**, sequence length **8192**, running on **8 GPUs**.

---

## Table of Contents

1. [Data Parallelism (DP / DDP / FSDP)](#1-data-parallelism-dp--ddp--fsdp)
2. [Tensor Parallelism (TP)](#2-tensor-parallelism-tp--intra-layer)
3. [Pipeline Parallelism (PP)](#3-pipeline-parallelism-pp--inter-layer)
4. [Context / Sequence Parallelism (CP / SP)](#4-context--sequence-parallelism-cp--sp)
5. [Expert Parallelism (EP)](#5-expert-parallelism-ep--for-moe-models)
6. [Combining them — nD Parallelism](#6-combining-them--nd-parallelism)
7. [Decision Guide](#7-decision-guide)
8. [Where this fits in Nanoformers](#8-where-this-fits-in-nanoformers)
9. [Further Reading](#9-further-reading)

---

## 1. Data Parallelism (DP / DDP / FSDP)

**Split the batch across GPUs. Each GPU keeps a full copy of the model.**

```
GPU 0: model_copy  ← batch[0:8]
GPU 1: model_copy  ← batch[8:16]
...
GPU 7: model_copy  ← batch[56:64]
```

### Flow per step

1. Each GPU does a forward + backward on its mini-batch → produces local gradients.
2. `all-reduce` gradients across all GPUs (everyone now has the averaged gradient).
3. Each GPU updates its own weights identically.

### Variants


| Variant                             | Memory per GPU                            | Notes                                                   |
| ----------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| **DP** (single-process, multi-GPU)  | Full model replica                        | Legacy, avoid.                                          |
| **DDP** (Distributed Data Parallel) | Full model replica                        | Standard baseline. Fast, simple.                        |
| **ZeRO-1**                          | Shards optimizer states                   | Adam states are ~2× model size — big win.               |
| **ZeRO-2**                          | Shards optimizer + gradients              |                                                         |
| **ZeRO-3 / FSDP**                   | Shards optimizer + gradients + parameters | Parameters gathered on-the-fly during forward/backward. |


### Memory cost of a model (rough)

For a model with `P` parameters in fp16/bf16 mixed precision with Adam:


| What                | Bytes per param    |
| ------------------- | ------------------ |
| fp16 weights        | 2                  |
| fp16 gradients      | 2                  |
| fp32 master weights | 4                  |
| fp32 Adam `m`       | 4                  |
| fp32 Adam `v`       | 4                  |
| **Total**           | **16 bytes/param** |


So a 1B-param model needs ~16 GB just for states. FSDP shards this across N GPUs → 16/N GB/GPU.

### When to use

- Model fits on one GPU → **DDP**.
- Model barely fits or doesn't → **FSDP / ZeRO-3**.

### Communication

- One gradient `all-reduce` per step (DDP).
- FSDP additionally does `all-gather` of parameters during forward and backward.
- Bandwidth-friendly compared to TP.

---

## 2. Tensor Parallelism (TP) — *intra-layer*

**Split individual matrix multiplications across GPUs.** A single layer is sharded.

### Example: MLP

The MLP up-projection `Y = X @ W` where `W` is `[4096, 16384]`.

- Shard `W` column-wise across 2 GPUs:
  - GPU 0 gets `W[:, :8192]`
  - GPU 1 gets `W[:, 8192:]`
- Each GPU computes half of `Y` locally → no communication needed for this matmul.
- The next matmul (down-projection) is sharded row-wise, and you `all-reduce` at the end.

```
X  ──┬──►  X @ W_col0  ──►  Y_half0  ──►  Y_half0 @ W_row0  ──┐
     │                                                         ├──► all-reduce ──► Z
     └──►  X @ W_col1  ──►  Y_half1  ──►  Y_half1 @ W_row1  ──┘
```

### Example: Attention

Shard across the **heads** dimension. Each GPU holds a subset of heads (Q, K, V, O projections for those heads) and computes attention for them independently. One `all-reduce` at the output projection.

### When to use

- A single layer is too big to fit on one GPU (e.g., 70B+ models).
- Almost always used **within a node** because communication is heavy (two `all-reduce`s per transformer block).
- Requires high-bandwidth interconnect (NVLink / NVSwitch).

### Communication

- `all-reduce` **inside every transformer block** (once after attention, once after MLP).
- Very chatty — do not span this across nodes if you can help it.

**Reference:** Megatron-LM (Shoeybi et al., 2019).

---

## 3. Pipeline Parallelism (PP) — *inter-layer*

**Split the model by layers. Each GPU holds a contiguous stage.**

```
GPU 0: layers 0-2   →  GPU 1: layers 3-5   →  GPU 2: layers 6-8   →  GPU 3: layers 9-11
```

A batch flows through the pipeline: GPU 0 computes layers 0-2, sends activations to GPU 1, etc.

### The "bubble" problem

While GPU 1 works, GPU 0 is idle. With naive pipelining, most GPUs sit idle most of the time.

### Solution: micro-batches

Split the batch into micro-batches and pipeline them.

```
Time  ─────────────────────────────────────►
GPU0: μb0  μb1  μb2  μb3  ─── BWD0 BWD1 BWD2 BWD3
GPU1:      μb0  μb1  μb2  μb3  ─── BWD0 BWD1 BWD2 BWD3
GPU2:           μb0  μb1  μb2  μb3 ─── BWD0 BWD1 BWD2 BWD3
GPU3:                μb0  μb1  μb2  μb3 ─── BWD0 BWD1 BWD2 BWD3
        ↑                              ↑                    ↑
       warm-up bubble             steady state        cool-down bubble
```

### Scheduling strategies


| Schedule             | Idea                                          | Trade-off                                       |
| -------------------- | --------------------------------------------- | ----------------------------------------------- |
| **GPipe**            | All forwards, then all backwards              | Simple, but keeps all activations → high memory |
| **1F1B**             | Alternate forward and backward                | Lower activation memory                         |
| **Interleaved 1F1B** | Each GPU holds multiple non-contiguous stages | Smaller bubble, more communication              |


### Bubble fraction

Roughly: `bubble_fraction = (num_stages - 1) / (num_stages - 1 + num_micro_batches)`

With 4 stages and 16 micro-batches: `3 / 19 ≈ 16%` idle time.

### When to use

- Model too big for one GPU.
- Scaling **across nodes** (communication is point-to-point, lightweight — just activations between adjacent stages).
- Combined with TP inside each stage.

### Communication

- `send` / `recv` activations between adjacent stages only.
- Cheap compared to TP.

---

## 4. Context / Sequence Parallelism (CP / SP)

**Split along the sequence length dimension.** Each GPU holds all parameters but only a chunk of tokens.

```
Seq length 8192, 4 GPUs:
GPU 0: tokens[0:2048]
GPU 1: tokens[2048:4096]
GPU 2: tokens[4096:6144]
GPU 3: tokens[6144:8192]
```

### Why it's tricky

Attention is **not** embarrassingly parallel over sequence — token `i` needs to attend to tokens `0..i-1`, which may live on other GPUs.

### Solutions


| Method                         | Idea                                                                                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ring Attention**             | GPUs pass KV chunks around in a ring; each GPU computes partial attention against incoming KV, accumulates, then rotates. Overlaps communication with compute. |
| **DeepSpeed Ulysses**          | Uses `all-to-all` to rearrange data so attention is local to each GPU (sharded by heads temporarily), then `all-to-all` back.                                  |
| **Megatron Sequence Parallel** | Lightweight variant: only shards non-attention parts (LayerNorm, dropout) along the sequence dimension. Used together with TP to save activation memory.       |


### Why it matters

Activation memory scales with sequence length, not parameters. For a 32k / 128k / 1M context, this is the **only** way to fit training.

### When to use

- Very long contexts (32k+).
- Long-context fine-tuning of already-large models.

---

## 5. Expert Parallelism (EP) — *for MoE models*

For Mixture-of-Experts layers, place **different experts on different GPUs**.

```
Router  ──►  token A  ──►  Expert 3 (on GPU 1)
         ──►  token B  ──►  Expert 7 (on GPU 3)
         ──►  token C  ──►  Expert 1 (on GPU 0)
```

### Flow

1. Router picks top-k experts per token.
2. `all-to-all` sends each token to the GPU(s) owning its chosen expert(s).
3. Experts compute locally.
4. Another `all-to-all` sends outputs back.

### When to use

- MoE models (Mixtral, DeepSeek-V3, etc.).
- Composes with DP / TP / PP orthogonally.

---

## 6. Combining them — nD Parallelism

Real large-scale training stacks all of these. A typical recipe on 1024 GPUs:


| Dimension                | Size                       | Purpose                                    |
| ------------------------ | -------------------------- | ------------------------------------------ |
| **Tensor Parallel**      | 8                          | Intra-node, splits big matmuls.            |
| **Pipeline Parallel**    | 8                          | Splits layers across nodes.                |
| **Data Parallel (FSDP)** | 16                         | Scales throughput, shards remaining state. |
| **Total**                | **8 × 8 × 16 = 1024 GPUs** |                                            |


Add **Context Parallel** on top when sequences are long, and **Expert Parallel** for MoE models.

### Rule of thumb for placement

```
┌─────────────────── Node (8 GPUs, NVLink) ───────────────────┐
│                                                             │
│   TP group (fast, chatty)   ◄── must stay within a node    │
│                                                             │
│   CP group (send/recv)      ◄── can span within node       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
          │
          ├── PP group (point-to-point, across nodes OK)
          │
          └── DP / FSDP group (all-reduce, across nodes OK)
```

The general principle: **place the most communication-heavy dimensions on the fastest links**.

---

## 7. Decision Guide


| Situation                                                  | Use                                                              |
| ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Model fits on one GPU, want more throughput                | **DDP**                                                          |
| Model almost fits / doesn't fit, single node               | **FSDP (ZeRO-3)**                                                |
| Single layer doesn't fit on one GPU                        | **Tensor Parallel**                                              |
| Full model doesn't fit even with TP, scaling to many nodes | **+ Pipeline Parallel**                                          |
| Sequence length is the memory bottleneck                   | **+ Context / Sequence Parallel**                                |
| Model is MoE                                               | **+ Expert Parallel**                                            |
| You want a bigger effective batch without more GPUs        | **Gradient Accumulation** (not parallelism, but often conflated) |


---

## 8. Where this fits in Nanoformers

The models currently trained in this repo — `Qwen3-0.6B`, `DistilBERT`, `ViT-base`, `Flan-T5-base`, `RoBERTa-base` — all fit comfortably on a single modern GPU.

For this scale, the practically useful strategies, in order of value:

1. **DDP** — easy win, near-linear throughput scaling. ~10-line change using `torch.nn.parallel.DistributedDataParallel` + `torchrun`.
2. **FSDP** — useful once training a 3B+ model, or when using longer sequences.
3. **Gradient accumulation** — simulate a larger batch on one GPU; combines cleanly with DDP/FSDP.

TP / PP / CP only become necessary beyond ~7B params or extremely long contexts. They are still great to implement as educational exercises.

### Suggested implementation roadmap for this repo

- Add DDP support to all training CLIs (launch via `torchrun --nproc_per_node=N`).
- Add gradient accumulation as a config option.
- Add mixed-precision (`bf16` / `fp16`) as a config option.
- Add FSDP support behind a flag for larger models.
- (Educational) Toy TP implementation on a single layer.
- (Educational) Toy PP implementation with 1F1B scheduling.
- (Educational) Ring attention for long-context training.

---

## 9. Further Reading

### Papers

- **Megatron-LM** — Shoeybi et al., 2019. [arxiv:1909.08053](https://arxiv.org/abs/1909.08053) — Tensor parallelism.
- **GPipe** — Huang et al., 2018. [arxiv:1811.06965](https://arxiv.org/abs/1811.06965) — Pipeline parallelism with micro-batches.
- **PipeDream / 1F1B** — Narayanan et al., 2019. [arxiv:1806.03377](https://arxiv.org/abs/1806.03377).
- **ZeRO** — Rajbhandari et al., 2019. [arxiv:1910.02054](https://arxiv.org/abs/1910.02054).
- **Ring Attention** — Liu et al., 2023. [arxiv:2310.01889](https://arxiv.org/abs/2310.01889).
- **DeepSpeed Ulysses** — Jacobs et al., 2023. [arxiv:2309.14509](https://arxiv.org/abs/2309.14509).
- **Reducing Activation Recomputation** (Sequence Parallel) — Korthikanti et al., 2022. [arxiv:2205.05198](https://arxiv.org/abs/2205.05198).

### Libraries

- **PyTorch DDP / FSDP** — built-in.
- **DeepSpeed** — ZeRO, Ulysses, pipeline.
- **Megatron-LM / Megatron-Core** — TP, PP, SP, CP.
- **Accelerate** — high-level wrapper over DDP / FSDP / DeepSpeed.
- **TorchTitan** — reference implementation of nD parallelism in pure PyTorch.

