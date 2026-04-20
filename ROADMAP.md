# Nanoformers — Master Roadmap

A consolidated roadmap covering the five big tracks of work on Nanoformers. Tracks are ordered from **quickest wins** to **largest scope**, so that early phases unlock the later ones (e.g. having SFT and a base model before RL, having DDP + mixed precision before the largest architecture experiments).

Two tracks already have dedicated sub-roadmaps:

- **Parallelisation** → [`PARALLELISM_GUIDE.md`](PARALLELISM_GUIDE.md) (DP / DDP / FSDP / TP / PP / CP / EP)
- **Architectures** → [`ARCHITECTURE_GALLERY_ROADMAP.md`](ARCHITECTURE_GALLERY_ROADMAP.md) (54 LLM architectures from Sebastian Raschka's gallery)

This document is the index; those docs own the detail for their tracks.

---

## Track overview

| # | Track | Scope | Detail doc | Status |
| :-: | :--- | :--- | :--- | :---: |
| 1 | Self-Supervised Learning | MLM, Span Corruption | below | ⬜ |
| 2 | Reinforcement Learning | REINFORCE → PPO → GRPO → DAPO → Dr. GRPO | below | ⬜ |
| 3 | Parameter-Efficient Fine-Tuning | LoRA, QLoRA (+ optional DoRA, IA³) | below | ⬜ |
| 4 | Parallelisation | DDP, FSDP, TP, PP, CP, EP, grad-accum, mixed precision | [`PARALLELISM_GUIDE.md`](PARALLELISM_GUIDE.md) | ⬜ |
| 5 | LLM Architecture Gallery | 54 architectures, nano scale | [`ARCHITECTURE_GALLERY_ROADMAP.md`](ARCHITECTURE_GALLERY_ROADMAP.md) | ⬜ |

---

## Track 1 — Self-Supervised Learning

Fills the two remaining gaps in the **Self-Supervised** row of `README.md`. Both mirror the existing CLM pipeline layout under `src/training/self_supervised_learning/`.

### Goals

- Encoder-only pre-training loop with **Masked Language Modeling (MLM)**, BERT-style.
- Encoder-decoder pre-training loop with **Span Corruption**, T5-style.

### Phase 1.1 — Masked Language Modeling (encoder-only)

- [ ] `src/training/self_supervised_learning/masked_language_modeling/` — dataset, collator, trainer
- [ ] 80/10/10 masking collator (`[MASK]` / random / unchanged) with configurable mask ratio
- [ ] CLI: `src/cli/train_mlm.py`
- [ ] Config: `configs/mlm_distilbert_wikitext.yaml`
- [ ] Train run: `distilbert/distilbert-base-uncased` on a small WikiText / BookCorpus subset
- [ ] Update `README.md` row + training command

**Deliverable:** `python -m src.cli.train_mlm --config configs/mlm_distilbert_wikitext.yaml` trains to a decreasing MLM loss and logs masked-token accuracy.

### Phase 1.2 — Span Corruption (encoder-decoder)

- [ ] `src/training/self_supervised_learning/span_corruption/` — dataset, collator, trainer
- [ ] T5-style sentinel-token span corruption collator (mean span length 3, 15% mask ratio by default)
- [ ] CLI: `src/cli/train_span_corruption.py`
- [ ] Config: `configs/span_corruption_t5_small_c4.yaml`
- [ ] Train run: `google/t5-efficient-tiny` or `t5-small` on a C4 / TinyStories subset
- [ ] Update `README.md` row + training command

**Deliverable:** `python -m src.cli.train_span_corruption --config configs/span_corruption_t5_small_c4.yaml` trains to a decreasing loss with sensible generated infills.

### Shared tasks

- [ ] Factor a `MaskingCollator` utility under `src/training/common/` so MLM and Span Corruption reuse one masking primitive.
- [ ] Unit tests for collators in `tests/` (mask ratio statistics, sentinel token assignment, label vs input alignment).

---

## Track 2 — Reinforcement Learning

The RL track goes from the simplest on-policy estimator to the current state of practice, sharing as much infrastructure as possible. Every variant shares one rollout loop, one reward interface, and one advantage-computation layer — only the loss differs.

### Goals

- One rollout engine (vLLM or pure HF `generate`) shared across all RL methods.
- A pluggable reward interface: rule-based (GSM8K answer matching) first, learned reward models later.
- Six variants: **REINFORCE**, **REINFORCE + baseline**, **PPO**, **GRPO**, **DAPO**, **Dr. GRPO**.

### Phase 2.0 — Shared RL infrastructure

- [ ] `src/training/reinforcement_learning/common/` — rollout runner, reference-model snapshot, reward registry, KL-penalty utilities, advantage normalization helpers
- [ ] Reward functions for GSM8K (exact-match on final answer) and a format/length penalty
- [ ] Shared YAML schema: `policy_model`, `ref_model`, `rollouts_per_prompt`, `kl_coef`, `clip_range`, etc.
- [ ] Dataset plumbing that re-uses the on-policy GSM8K dataset already used for IFT/DPO

### Phase 2.1 — REINFORCE (vanilla)

- [ ] Loss: `-E[log π(a|s) · R]` with full-trajectory reward
- [ ] `src/training/reinforcement_learning/reinforce/` + `src/cli/train_reinforce.py`
- [ ] Config: `configs/reinforce_qwen_gsm8k.yaml`
- [ ] Train `Qwen/Qwen3-0.6B` on GSM8K, log reward / pass@1 / KL-to-ref

### Phase 2.2 — REINFORCE with baseline

- [ ] Add a value head (or use a running-mean baseline as a simpler first pass)
- [ ] Subtract baseline from reward to form advantage
- [ ] Config: `configs/reinforce_baseline_qwen_gsm8k.yaml`

### Phase 2.3 — Proximal Policy Optimization (PPO)

- [ ] Value head trained with MSE loss alongside the clipped policy loss
- [ ] GAE(λ) for advantages
- [ ] Ratio clipping `clip(π_new/π_old, 1-ε, 1+ε)` + KL penalty to the frozen reference model
- [ ] `src/cli/train_ppo.py` + `configs/ppo_qwen_gsm8k.yaml`
- [ ] Ablations: with vs without value head, with vs without KL penalty

### Phase 2.4 — Group Relative Policy Optimization (GRPO)

- [ ] No value head — use group-relative advantages instead
- [ ] For each prompt, sample G completions, advantage = `(r_i - mean(r)) / std(r)`
- [ ] Reuse PPO's clipped surrogate loss + KL-to-ref
- [ ] `src/cli/train_grpo.py` + `configs/grpo_qwen_gsm8k.yaml`
- [ ] Benchmark GRPO vs PPO on the same reward, same budget

### Phase 2.5 — DAPO (Decoupled clip + Dynamic sAmpling)

- [ ] Decoupled clipping: different `ε_low` / `ε_high` for positive vs negative advantage tokens
- [ ] Dynamic sampling: filter out prompt groups where all rollouts have identical reward (zero-advantage groups)
- [ ] Token-level loss (length-normalized across the whole group, not per-sequence)
- [ ] Overlong-response filtering / soft penalty
- [ ] `src/cli/train_dapo.py` + `configs/dapo_qwen_gsm8k.yaml`

### Phase 2.6 — Dr. GRPO

- [ ] Remove GRPO's length and std normalization biases
- [ ] Advantage = `r_i - mean(r)` (no `std` division)
- [ ] Loss aggregated over tokens without per-sequence length normalization
- [ ] `src/cli/train_dr_grpo.py` + `configs/dr_grpo_qwen_gsm8k.yaml`
- [ ] Head-to-head: GRPO vs Dr. GRPO on GSM8K, same rollout budget

### Phase 2.7 — Polish

- [ ] Comparison table in `README.md` of all six RL variants (pass@1, KL-to-ref, tokens/sec)
- [ ] A short `docs/RL_GUIDE.md` explaining objectives and trade-offs (analogous to `PARALLELISM_GUIDE.md`)

**Deliverable:** all six RL variants trainable with `python -m src.cli.train_<variant> --config configs/<variant>_qwen_gsm8k.yaml`, sharing one rollout/reward engine.

---

## Track 3 — Parameter-Efficient Fine-Tuning (PEFT)

PEFT composes with every existing supervised loop (IFT, DPO, Seq2Seq, classification, QA) **and** with the RL variants from Track 2 — once it lands, every Track 2 run can be re-done with a tiny trainable footprint.

### Goals

- Single reusable PEFT layer that wraps any `nn.Linear` in a frozen base model.
- LoRA first, then 4-bit QLoRA via `bitsandbytes`, then optional variants.
- Every existing training CLI gains a `peft:` block in its YAML config — no pipeline rewrites.

### Phase 3.1 — LoRA core

- [ ] `src/peft/lora.py` — `LoRALinear`, `apply_lora(model, target_modules, r, alpha, dropout)`, `mark_only_lora_as_trainable`, `merge_and_unload`
- [ ] `src/peft/__init__.py` — unified `apply_peft(model, config)` entrypoint
- [ ] `src/training/common/` hook so every trainer calls `apply_peft` when `peft:` is in the config
- [ ] Unit tests: parameter count check, merge-equivalence check (merged vs unmerged forward pass must match within fp tolerance)

### Phase 3.2 — Wire into existing pipelines

- [ ] IFT + LoRA: `configs/ift_qwen_gsm8k_lora.yaml` — train `Qwen/Qwen3-0.6B` and compare vs full-FT run
- [ ] DPO + LoRA: `configs/dpo_qwen_gsm8k_lora.yaml`
- [ ] Seq-classification + LoRA: `configs/seqclf_distilbert_emotion_lora.yaml`
- [ ] RL + LoRA: at least one of the Track 2 CLIs trained with LoRA on the policy (e.g. `grpo_qwen_gsm8k_lora.yaml`)
- [ ] Update `README.md` with a PEFT section and wandb links

### Phase 3.3 — QLoRA (4-bit)

- [ ] `bitsandbytes` dependency pinned in `requirements.txt`
- [ ] `src/peft/quantization.py` — helper to load a base model with nf4 quantization and bf16 compute dtype
- [ ] `peft: {type: qlora, bits: 4, ...}` config branch
- [ ] QLoRA IFT run on `Qwen/Qwen3-0.6B` with memory and throughput comparison vs LoRA

### Phase 3.4 — Optional variants (nice-to-have)

- [ ] **DoRA** (Weight-Decomposed LoRA) — magnitude/direction decomposition on top of LoRA
- [ ] **IA³** — learned rescaling vectors on K, V, FFN activations
- [ ] **Prefix / Prompt tuning** — trainable virtual tokens prepended to the input
- [ ] Toggle any of them with `peft: {type: <name>, ...}`

**Deliverable:** every existing `train_*.py` CLI (supervised and RL) accepts a `peft:` block and trains with trainable-param counts ~0.1–1% of full-FT while matching quality on a small eval.

---

## Track 4 — Parallelisation

Scaling existing training loops across multiple GPUs. **Detail doc:** [`PARALLELISM_GUIDE.md`](PARALLELISM_GUIDE.md) covers DP/DDP/FSDP, TP, PP, CP/SP, and EP with a running 12-layer / 8-GPU example.

### Sequencing for this repo

The guide's own "Suggested implementation roadmap" is the source of truth; summarised here for the master index:

### Phase 4.1 — Throughput basics (single-node, practical)

- [ ] **Gradient accumulation** as a config option across all training CLIs
- [ ] **Mixed precision** (`bf16` / `fp16`) config flag, with an `autocast` wrapper in `src/training/common/trainer.py`
- [ ] **DDP** launch support via `torchrun --nproc_per_node=N` in every CLI
- [ ] Smoke-test DDP on 2 GPUs for CLM, IFT, and one classification task

### Phase 4.2 — Sharding for bigger models

- [ ] **FSDP** (ZeRO-3) behind a `parallelism.type: fsdp` config flag
- [ ] Wrap-policy helpers (by size, by block type) under `src/training/common/parallelism.py`
- [ ] Benchmark FSDP vs DDP on a 3B+ model (e.g. one of the nano architectures scaled up)

### Phase 4.3 — Educational / advanced

- [ ] Toy **Tensor Parallel** implementation on a single MLP + attention layer (Megatron-style column/row sharding)
- [ ] Toy **Pipeline Parallel** with 1F1B scheduling across 2 stages
- [ ] Toy **Ring Attention** for long-context experiments
- [ ] Toy **Expert Parallel** wrapper for the MoE components from Track 5

### Phase 4.4 — Composition

- [ ] A small `nD` parallelism example combining DDP + FSDP + grad-accum on a nano-MoE model from Track 5

**Sequencing note:** Phase 4.1 is cheap and should land early (alongside Track 1) because every subsequent run benefits from mixed precision and grad accumulation. Phases 4.2–4.4 are best attacked after Tracks 1–3 produce more training loops to scale and once Track 5 supplies MoE / long-context architectures worth scaling.

---

## Track 5 — LLM Architecture Gallery

Nano-scale re-implementations of 54 modern LLM architectures from Sebastian Raschka's gallery, from GPT-2 to DeepSeek V3.2 / GLM-5 / Kimi Linear / Nemotron 3.

**This track has its own detailed roadmap.** See [`ARCHITECTURE_GALLERY_ROADMAP.md`](ARCHITECTURE_GALLERY_ROADMAP.md) for:

- The full component library plan (norms, positional, attention variants, MoE, Mamba-2, xLSTM, DeltaNet, Lightning Attention, DeepSeek Sparse Attention, MTP)
- The 12-phase implementation plan grouped by architectural family
- The per-model deliverables checklist (config, model, `ARCHITECTURE.md` fact sheet, training YAML)
- The full 54-model checklist

**Sequencing note:** Track 5 runs largely in parallel with Tracks 1–4. The architectures only need the CLM pipeline to exist (it already does), so the gallery can advance independently. Items that benefit from PEFT (Track 3), RL (Track 2), and FSDP / TP / EP (Track 4) come for free once those tracks land because every gallery model exposes the same HF-compatible `forward` signature.

---

## Suggested high-level timeline

Approximate, assuming one developer part-time. Tracks in the same row run in parallel.

| Slot | Primary track | Parallel work |
| :--- | :--- | :--- |
| **Weeks 1–2** | Track 1 — MLM + Span Corruption | Track 4.1 — grad-accum, mixed precision, DDP flag |
| **Weeks 3–6** | Track 2 — REINFORCE → PPO → GRPO → DAPO → Dr. GRPO | Track 5 — gallery Phases 0–3 (foundations, dense baselines) |
| **Weeks 7–8** | Track 3 — LoRA + QLoRA wired into supervised + RL | Track 5 — gallery Phases 4–6 |
| **Weeks 9–10** | Track 4.2 — FSDP rollout, scale up RL runs | Track 5 — gallery Phases 7–9 |
| **Weeks 11–12** | Track 4.3–4.4 — toy TP / PP / CP / EP | Track 5 — gallery Phases 10–12 |

---

## Conventions (apply to all tracks)

- Every new method gets: a `src/training/.../<method>/` package, a `src/cli/train_<method>.py` entrypoint, at least one `configs/<method>_<model>_<dataset>.yaml`, a row in `README.md` with a wandb link, and an Update Log entry.
- Shared primitives (collators, reward functions, parallelism wrappers, PEFT adapters) live under `src/training/common/` or a top-level `src/peft/` so any future training loop can reuse them.
- Every new CLI must run on a single consumer GPU with the provided default config, using a small enough model/dataset that a full smoke-test takes under ~15 minutes.
