# LLM Architecture Gallery — Implementation Roadmap

This document tracks the plan to implement every architecture listed in Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) (54 models as of April 2026) inside Nanoformers, at nano scale, with a blog-style fact sheet per model.

---

## 1. Scope and design principles

- **Nano scale only.** Every model uses a small config (roughly `d_model=256`, 4–8 layers, `vocab≈32k`) so every variant can be trained on `roneneldan/TinyStories` or `openai/gsm8k` on a single GPU. No Hugging Face weight loading.
- **Hybrid layout.** Shared building blocks live under `src/models/components/`; each of the 54 listed variants gets its own thin `Model` class in `src/models/zoo/<slug>/model.py` that composes those blocks.
- **Blog-style docs.** Each model folder contains an `ARCHITECTURE.md` that mirrors the gallery fact sheet (Scale, Context, License, Date, Decoder type, Attention, Layer mix, Key detail, Related concepts) plus a short **"Nano recipe"** section documenting the scaled-down config actually used in Nanoformers.
- **Plug into existing CLM pipeline.** The existing `src/training/self_supervised_learning/causal_language_modeling` pipeline is reused; new models expose a Hugging Face–compatible `forward(input_ids, labels=...) -> ModelOutput(loss, logits)` signature so no training code changes are needed.
- **Testability.** A shared `tests/architectures/test_forward.py` instantiates every model at nano size and checks forward pass, loss, and generation; `test_components.py` unit-tests each primitive (RoPE, MLA, MoE routing, Mamba-2 recurrence, DeltaNet, etc.).
- **Correctness over speed.** Nano implementations of Mamba-2, DeltaNet, Lightning Attention, MLA, and DeepSeek Sparse Attention use straightforward PyTorch — no custom Triton/CUDA kernels.

---

## 2. Proposed directory structure

```
src/
  models/
    __init__.py                  # registry: slug -> (ModelCls, ConfigCls)
    components/
      __init__.py
      norms.py                   # RMSNorm, LayerNorm, QKNorm, SandwichNorm
      positional.py              # LearnedAbsPE, RoPE, YaRN, NoPE, PartialRoPE, per-layer embeds
      attention/
        mha.py                   # Multi-Head Attention
        mqa.py                   # Multi-Query Attention
        gqa.py                   # Grouped-Query Attention (+ sliding-window mask, gated variant)
        mla.py                   # Multi-head Latent Attention (+ KV LayerNorm, RoPE+NoPE)
        deepseek_sparse.py       # DeepSeek Sparse Attention
        delta_net.py             # Gated DeltaNet, Kimi Delta Attention
        lightning.py             # Lightning Attention
        attention_sink.py        # GPT-OSS attention sinks + bias
      ffn.py                     # SwiGLU, GeluMLP, Relu2FFN, ParallelMLP
      moe.py                     # TopKRouter, SparseMoE, SharedExpert, LatentMoE, CoarseMoE
      blocks.py                  # PreNorm, PostNorm (inside-residual), Sandwich, ParallelBlock
      mamba2.py                  # Mamba-2 block
      xlstm.py                   # mLSTM block with matrix memory
      mtp.py                     # Multi-token prediction heads (MTP-1/3, shared-weight MTP)
      embeddings.py              # TokenEmbed, TiedEmbed, PerLayerEmbed
      cache.py                   # KV cache / recurrent state helpers
      base_lm.py                 # Generic CausalLM wrapper (embed -> blocks -> lm_head -> loss)
    zoo/
      gpt2_xl/                   # 54 model folders total — see §5 for the full list
        __init__.py
        model.py
        config.py
        ARCHITECTURE.md
      llama3_8b/
      ...
configs/
  architectures/                 # one nano-training yaml per model, all point at the CLM pipeline
    gpt2_xl.yaml
    llama3_8b.yaml
    ...
docs/
  architectures/
    INDEX.md                     # gallery overview (grouped by family, links to every ARCHITECTURE.md)
tests/
  architectures/
    test_forward.py              # instantiate every model at nano size, check forward + loss
    test_components.py           # unit tests for RoPE, MLA, MoE, Mamba-2, DeltaNet, ...
src/cli/
  train_architecture.py          # thin CLI: --model <slug> --config configs/architectures/<slug>.yaml
```

---

## 3. Component library (Phase 0 foundation)

Maps the gallery's "Related concepts" tags to the component that implements them.

| Category | Components |
| :--- | :--- |
| **Norms** | `RMSNorm`, `LayerNorm`, `QKNorm`, `SandwichNorm` (Arcee Trinity) |
| **Positional** | `LearnedAbsPE` (GPT-2), `RoPE`, `PartialRoPE` (MiniMax M2, Gemma 4 global), `NoPE` (SmolLM3, Kimi Linear, Tiny Aya), `YaRN` (OLMo 3 global), `PerLayerEmbed` (Gemma 4 E2B/E4B) |
| **Attention** | `MHA`, `MQA`, `GQA`, `MLA`, `DeepSeekSparseAttention`, `GatedDeltaNet`, `KimiDelta`, `LightningAttention`, `AttentionSink` (GPT-OSS), `Mamba2`, `mLSTM` |
| **Attention masks** | `SlidingWindowMask` with configurable ratio (3:1, 4:1, 5:1) |
| **MLP** | `GeluMLP` (GPT-2, Gemma 4 E2B double-wide), `SwiGLU`, `Relu2FFN` (Nemotron 3 Nano 4B) |
| **MoE** | `TopKRouter`, `SparseMoE`, `SharedExpert`, `LatentMoE` (Nemotron 3 Super), `CoarseMoE` (Arcee Trinity) |
| **Blocks** | `PreNormBlock` (default), `PostNormBlock` (OLMo 2/3), `SandwichNormBlock` (Arcee Trinity), `ParallelBlock` (Tiny Aya), plus a dense-prefix + MoE wrapper |
| **MTP** | `MTPHead` (Step 3.5 Flash MTP-3, Xiaomi MiMo, Nemotron 3 Super shared-weight MTP) |
| **Base wrapper** | `CausalLM` — takes a block factory + config, handles embeddings, `lm_head` (tied or untied), loss, KV cache |

---

## 4. Architecture-to-family mapping

The 54 models decompose into ~13 architectural templates. Models in the same row reuse the same block factory and differ mainly in config numbers and a couple of flags.

- **GPT-2 classic** — GPT-2 XL
- **Llama dense** (GQA + RoPE + RMSNorm + SwiGLU + pre-norm) — Llama 3 8B, Llama 3.2 1B, Phi-4 14B, Nanbeige 4.1 3B
- **OLMo post-norm** — OLMo 2 7B (MHA), OLMo 3 7B (MHA + SWA + YaRN), OLMo 3 32B (GQA + SWA + YaRN)
- **Qwen3 dense** (GQA + QK-Norm) — Qwen3 4B / 8B / 32B
- **Mistral / SmolLM / Tiny Aya dense** — Mistral Small 3.1 24B, SmolLM3 3B (periodic NoPE), Tiny Aya 3.35B (parallel block + RoPE+NoPE + SWA)
- **Gemma local-global dense** — Gemma 3 27B, Gemma 3 270M (MQA), Gemma 4 31B, Gemma 4 E2B (MQA + per-layer embeds), Gemma 4 E4B
- **xLSTM recurrent** — xLSTM 7B
- **DeepSeek MLA + MoE** — DeepSeek V3, DeepSeek R1, Kimi K2, Kimi K2.5, Mistral Large 3, Sarvam 105B
- **MLA + DeepSeek Sparse Attention** — DeepSeek V3.2, GLM-5 744B, GLM-5.1 744B
- **GQA-based MoE** — Llama 4 Maverick (chunked + full), Qwen3 235B-A22B, Grok 2.5, GLM-4.5 / GLM-4.5-Air / INTELLECT-3, GLM-4.7, Qwen3 Coder Flash 30B-A3B, MiniMax M2, MiniMax-M2.5, Sarvam 30B
- **Sliding-window MoE** — GPT-OSS 20B/120B (attention sinks), Step 3.5 Flash (MTP-3), Xiaomi MiMo-V2-Flash, Gemma 4 26B-A4B, Arcee Trinity Large (sandwich norm + gated attn + coarse MoE)
- **Linear / hybrid attention** — Qwen3 Next 80B-A3B, Qwen3.5 397B, Kimi Linear 48B-A3B, Ling 2.5 1T
- **Mamba hybrid** — Nemotron 3 Nano 30B-A3B, Nemotron 3 Super 120B-A12B, Nemotron 3 Nano 4B

### Family diagram

```mermaid
flowchart LR
  Foundation[Shared components] --> Dense[Dense GQA/MHA]
  Foundation --> MLA[MLA family]
  Foundation --> SSM[Mamba-2 / xLSTM / DeltaNet]
  Dense --> LlamaLike[Llama / Phi / Nanbeige]
  Dense --> Qwen3Dense[Qwen3 4B/8B/32B]
  Dense --> GemmaLG[Gemma 3/4 local-global]
  Dense --> OLMo[OLMo 2/3 post-norm]
  MLA --> DeepSeekFam[DeepSeek V3/R1 / Kimi K2/K2.5 / Mistral Large 3 / Sarvam 105B]
  MLA --> SparseAttn[DeepSeek V3.2 / GLM-5 / GLM-5.1]
  Dense --> GQAMoE[Qwen3 MoE / GLM-4.5 / Llama 4 / Grok / MiniMax / GPT-OSS / Step3.5 / Arcee Trinity]
  SSM --> Nemotron[Nemotron 3 Nano/Super/Nano-4B]
  SSM --> Hybrid[Qwen3 Next / Qwen3.5 / Kimi Linear / Ling 2.5 / xLSTM]
```

---

## 5. Implementation roadmap

Each phase ends with all of its models trainable via `python -m src.cli.train_architecture --model <slug>` on TinyStories nano configs and passing `tests/architectures/test_forward.py`.

### Phase 0 — Foundation (week 1)
Build the full component library under `src/models/components/`, except Mamba-2, xLSTM, DeltaNet, Lightning Attention, and DeepSeek Sparse Attention (each added in its own phase). Also write `base_lm.CausalLM`, the model registry, the `train_architecture.py` CLI, and `test_components.py`.

### Phase 1 — Classic dense baselines (week 2)
`gpt2_xl`, `llama3_8b`, `llama3_2_1b`, `phi4_14b`, `nanbeige41_3b`.

### Phase 2 — QK-Norm and post-norm dense (week 2)
`olmo2_7b`, `olmo3_7b`, `olmo3_32b` (adds YaRN on global layers).

### Phase 3 — Qwen3 dense + Mistral / SmolLM / Tiny Aya (week 3)
`qwen3_4b`, `qwen3_8b`, `qwen3_32b`, `mistral_small_31_24b`, `smollm3_3b` (periodic NoPE), `tiny_aya_335b` (parallel block).

### Phase 4 — Gemma local/global dense (week 3)
`gemma3_27b`, `gemma3_270m` (MQA), `gemma4_31b` (unified K/V, p-RoPE global), `gemma4_e2b`, `gemma4_e4b` (per-layer embeddings).

### Phase 5 — xLSTM (week 4)
Implement `components/xlstm.py` (mLSTM block with matrix memory); model `xlstm_7b`.

### Phase 6 — MLA + MoE (DeepSeek family, week 4–5)
Implement `components/attention/mla.py` and the full MoE stack. Models: `deepseek_v3`, `deepseek_r1`, `kimi_k2`, `kimi_k25`, `mistral_large3_673b`, `sarvam_105b`.

### Phase 7 — GQA-based MoE (week 5)
`llama4_maverick_400b` (alt dense/MoE + chunked attention), `qwen3_235b_a22b`, `grok25_270b` (always-on SwiGLU), `glm45_355b`, `glm45_air_106b`, `intellect3_106b`, `glm47_355b`, `qwen3_coder_flash_30b_a3b`, `minimax_m2_230b`, `minimax_m25_230b`, `sarvam_30b`.

### Phase 8 — Sliding-window MoE (week 6)
Implement `attention_sink.py` and MTP heads. Models: `gpt_oss_20b`, `gpt_oss_120b`, `step35_flash_196b`, `xiaomi_mimo_v2_flash_309b`, `gemma4_26b_a4b`, `arcee_trinity_large_400b` (sandwich norm + gated attn + coarse MoE).

### Phase 9 — MLA + DeepSeek Sparse Attention (week 6)
Implement `components/attention/deepseek_sparse.py`. Models: `deepseek_v32_671b`, `glm5_744b`, `glm51_744b`.

### Phase 10 — Linear / hybrid attention (week 7)
Implement `delta_net.py` (Gated DeltaNet, Kimi Delta) and `lightning.py`. Models: `qwen3_next_80b_a3b`, `qwen35_397b`, `kimi_linear_48b_a3b`, `ling25_1t`.

### Phase 11 — Mamba hybrid (week 7–8)
Implement `components/mamba2.py` (Mamba-2 block with selective SSM) and `LatentMoE`. Models: `nemotron3_nano_30b_a3b`, `nemotron3_super_120b_a12b`, `nemotron3_nano_4b`.

### Phase 12 — Polish (week 8)
- `docs/architectures/INDEX.md` gallery overview (grouped by family, mirroring the blog).
- Final sweep: every model has a training run on TinyStories nano config with loss curve logged to `wandb`.
- Update the top-level `README.md` with an "Architectures" section linking into `docs/architectures/INDEX.md`.

---

## 6. Per-model deliverables

"Done" for each of the 54 models means:

- `src/models/zoo/<slug>/config.py` — nano `@dataclass` config with sane defaults and a `scale: str` field documenting the original size.
- `src/models/zoo/<slug>/model.py` — `class <Name>Model(CausalLM)` composing blocks from `components/`.
- `src/models/zoo/<slug>/ARCHITECTURE.md` — blog-style fact sheet with the gallery's sections (Scale, Context, License, Date, Decoder type, Attention, Layer mix, Key detail, Related concepts) plus a **"Nano recipe"** section listing the scaled-down config.
- `configs/architectures/<slug>.yaml` — training config pointing at the CLM pipeline.
- Entry added in `src/models/__init__.py` registry and in `docs/architectures/INDEX.md`.

---

## 7. Full model checklist

All 54 architectures from the gallery, grouped by phase.

### Phase 1 — Classic dense baselines
- [ ] GPT-2 XL (1.5B)
- [ ] Llama 3 (8B)
- [ ] Llama 3.2 (1B)
- [ ] Phi-4 (14B)
- [ ] Nanbeige 4.1 (3B)

### Phase 2 — OLMo post-norm
- [ ] OLMo 2 (7B)
- [ ] OLMo 3 (7B)
- [ ] OLMo 3 (32B)

### Phase 3 — Qwen3 dense + Mistral / SmolLM / Tiny Aya
- [ ] Qwen3 (4B)
- [ ] Qwen3 (8B)
- [ ] Qwen3 (32B)
- [ ] Mistral Small 3.1 (24B)
- [ ] SmolLM3 (3B)
- [ ] Tiny Aya (3.35B)

### Phase 4 — Gemma local/global dense
- [ ] Gemma 3 (27B)
- [ ] Gemma 3 (270M)
- [ ] Gemma 4 (31B)
- [ ] Gemma 4 (E2B)
- [ ] Gemma 4 (E4B)

### Phase 5 — xLSTM
- [ ] xLSTM (7B)

### Phase 6 — DeepSeek MLA + MoE
- [ ] DeepSeek V3 (671B)
- [ ] DeepSeek R1 (671B)
- [ ] Kimi K2 (1T)
- [ ] Kimi K2.5 (1T)
- [ ] Mistral Large 3 (673B)
- [ ] Sarvam (105B)

### Phase 7 — GQA-based MoE
- [ ] Llama 4 Maverick (400B)
- [ ] Qwen3 (235B-A22B)
- [ ] Grok 2.5 (270B)
- [ ] GLM-4.5 (355B)
- [ ] GLM-4.5-Air (106B)
- [ ] INTELLECT-3 (106B)
- [ ] GLM-4.7 (355B)
- [ ] Qwen3 Coder Flash (30B-A3B)
- [ ] MiniMax M2 (230B)
- [ ] MiniMax-M2.5 (230B)
- [ ] Sarvam (30B)

### Phase 8 — Sliding-window MoE
- [ ] GPT-OSS (20B)
- [ ] GPT-OSS (120B)
- [ ] Step 3.5 Flash (196B)
- [ ] Xiaomi MiMo-V2-Flash (309B)
- [ ] Gemma 4 (26B-A4B)
- [ ] Arcee AI Trinity Large (400B)

### Phase 9 — MLA + DeepSeek Sparse Attention
- [ ] DeepSeek V3.2 (671B)
- [ ] GLM-5 (744B)
- [ ] GLM-5.1 (744B)

### Phase 10 — Linear / hybrid attention
- [ ] Qwen3 Next (80B-A3B)
- [ ] Qwen3.5 (397B)
- [ ] Kimi Linear (48B-A3B)
- [ ] Ling 2.5 (1T)

### Phase 11 — Mamba hybrid
- [ ] Nemotron 3 Nano (30B-A3B)
- [ ] Nemotron 3 Super (120B-A12B)
- [ ] Nemotron 3 Nano (4B)

---

## 8. Open risks / items to revisit

- **Mamba-2, DeltaNet, Lightning Attention, MLA, DeepSeek Sparse Attention** all rely on nontrivial custom kernels in the original papers. Nano implementations use correct-but-unoptimized PyTorch — performance is not a goal; correctness is.
- **Exact block counts.** The blog's "Layer mix" numbers are for full-scale models. Nano configs preserve the *ratio* (e.g., Gemma's 5:1 local:global) but shrink the total layer count to 6–8.
- **Markdown sources.** The 54 `ARCHITECTURE.md` files are drafted directly from the gallery text so wording stays faithful to the blog, with each file linking back to the original figure.
