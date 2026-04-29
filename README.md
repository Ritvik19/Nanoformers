# 🧠 Nanoformers

A minimal playground for building and training transformer models from scratch.  
It covers **self-supervised**, **supervised**, and **reinforcement learning** training loops, along with a **tiny transformer architecture** for research and experimentation.

---

## 🔍 Objectives

- Implement tiny transformer architectures from scratch  
- Build training loops for:

| Paradigm | Task | Architecture | Status |
| :--- | :--- | :--- | :---: |
| **Self-Supervised** | Causal Language Modeling | Decoder-only | ✅ |
| | Masked Language Modeling | Encoder-only | ✅ |
| | Span Corruption | Encoder-Decoder | ✅ |
| **Supervised** | Instruction Fine-Tuning | Decoder-only | ✅ |
| | Direct Preference Optimization | Decoder-only | ✅ |
| | Sequence Classification | Encoder-only | ✅ |
| | Token Classification | Encoder-only | ✅ |
| | Extractive Question Answering | Encoder-only | ✅ |
| | Sequence-to-Sequence Modeling | Encoder-Decoder | ✅ |
| **Reinforcement** | Reinforce <br> with the following addons: <br> - Baseline <br> - KL Penalty <br> - Length Normalization | Decoder-only | ✅ |
| | Proximal Policy Optimization | Decoder-only | ⬜️ |
| | Group Relative Policy Optimization | Decoder-only | ⬜️ |
| **Contrastive** | Contrastive Loss | Encoder-only | ✅ |
| | Triplet Loss | Encoder-only | ✅ |
| | InfoNCE Loss | Encoder-only | ✅ |
| | Image-Text Contrastive | Dual Encoder (Vision + Text) | ✅ |
| | Image-Text Sigmoid Contrastive | Dual Encoder (Vision + Text) | ✅ |

- Implement parallelization strategies for scaling training across multiple GPUs:

| Strategy | Scope | Status |
| :--- | :--- | :---: |
| **Gradient Accumulation** | Single GPU, larger effective batch | ✅ |
| **Mixed Precision (fp16 / bf16)** | Single GPU, memory & speed | ✅ |
| **Data Parallelism (DDP)** | Replicate model, shard batch | ⬜️ |
| **Fully Sharded Data Parallelism (FSDP / ZeRO-3)** | Shard params, grads, optimizer states | ⬜️ |
| **Tensor Parallelism (TP)** | Shard individual matmuls within a layer | ⬜️ |
| **Pipeline Parallelism (PP)** | Shard layers across GPUs with micro-batching | ⬜️ |
| **Context / Sequence Parallelism (CP / SP)** | Shard along sequence length | ⬜️ |
| **Expert Parallelism (EP)** | Shard MoE experts across GPUs | ⬜️ |

> See [PARALLELISM_GUIDE.md](PARALLELISM_GUIDE.md) for detailed explanations of each strategy.

- Implement parameter-efficient fine-tuning (PEFT) methods:

| Method | Description | Status |
| :--- | :--- | :---: |
| **LoRA** | Low-Rank Adaptation (`W + BA`), trainable low-rank adapters on a frozen base model | ⬜️ |
| **QLoRA** | LoRA on top of a quantized base model (4-bit weights, bf16 adapters) | ⬜️ |

---

## 🚀 Models Trained

| Model | Dataset | Task | Configuration | Logs |
|-------|----------|------|----------------|------|
| `Qwen/Qwen3-0.6B` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-ift` | Instruction Fine-Tuning | [ift_qwen_gsm8k.yaml](configs/ift_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/6njm3m9q?nw=nwuserritvik19) |
| `Qwen/Qwen3-0.6B` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-dpo` | Direct Preference Optimization | [dpo_qwen_gsm8k.yaml](configs/dpo_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/3hsxkyfp?nw=nwuserritvik19) |
| `Qwen/Qwen3-0.6B-Base` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-cpt` | Causal Language Modeling | [clm_qwen_gsm8k.yaml](configs/clm_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/twq18n69) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/dair-ai-emotion` | Sequence Classification | [seqclf_distilbert_emotion.yaml](configs/seqclf_distilbert_emotion.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/6wlqge7k?nw=nwuserritvik19) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/conll-2003-ner` | Token Classification | [tokclf_distilbert_conll2003.yaml](configs/tokclf_distilbert_conll2003.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/hrpxlrfe?nw=nwuserritvik19) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/squad-v2` | Extractive Question Answering | [qa_distilbert_squad.yaml](configs/qa_distilbert_squad.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/x00m7mfb?nw=nwuserritvik19) |
| `google/flan-t5-base` | `Ritvik19/gsm8k-seq2seq` | Sequence-to-Sequence Modeling | [seq2seq_flan_t5_base_gsm8k.yaml](configs/seq2seq_flan_t5_base_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/n5iq698c?nw=nwuserritvik19) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/qqp-contrastive` | Contrastive Loss | [contrastive_loss_distilbert_qqp.yaml](configs/contrastive_loss_distilbert_qqp.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/o8y891t1?nw=nwuserritvik19) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/qqp-triplet` | Triplet Loss | [triplet_loss_distilbert_qqp.yaml](configs/triplet_loss_distilbert_qqp.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/2s72jin4?nw=nwuserritvik19) |
| `distilbert/distilbert-base-uncased` | `Ritvik19/qqp-info_nce` | InfoNCE Loss | [info_nce_loss_distilbert_qqp.yaml](configs/info_nce_loss_distilbert_qqp.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/jpl7ndhk?nw=nwuserritvik19) |
| `FacebookAI/roberta-base` and `google/vit-base-patch16-224` | `Ritvik19/flickr30k` | Image-Text Contrastive | [image_text_contrastive_clip_flickr30k.yaml](configs/image_text_contrastive_clip_flickr30k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/1s18t8h4?nw=nwuserritvik19) |
| `FacebookAI/roberta-base` and `google/vit-base-patch16-224` | `Ritvik19/flickr30k` | Image-Text Sigmoid Contrastive | [image_text_sigmoid_contrastive_siglip_flickr30k.yaml](configs/image_text_sigmoid_contrastive_siglip_flickr30k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/6q89d7xe?nw=nwuserritvik19) |
| `bert-base-uncased` | `Ritvik19/open-web-text` | Masked Language Modeling | [mlm_bert_open_web_text.yaml](configs/mlm_bert_open_web_text.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/1jiqkkmp?nw=nwuserritvik19) |
| `t5-base` | `Ritvik19/open-web-text` | Span Corruption | [span_corruption_t5_base_open_web_text.yaml](configs/span_corruption_t5_base_open_web_text.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/tnotg2s3?nw=nwuserritvik19) |
| `Qwen/Qwen3-0.6B` | `Ritvik19/math-rl` | REINFORCE | [reinforce_qwen_math.yaml](configs/reinforce_qwen_math.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/a2ttdud6?nw=nwuserritvik19) |

### Causal Language Modeling (CLM)
- `text`: string

### Masked Language Modeling (MLM)
- `text`: string

### Span Corruption
- `text`: string

### Instruction Fine-Tuning (IFT)
- `messages`: list of dicts  
  Each dict typically contains:
  - `role`: "system" | "user" | "assistant"
  - `content`: string

### Direct Preference Optimization (DPO)
- `prompt`: list of dicts
- `chosen`: list of dicts
- `rejected`: list of dicts  
  *(same format as IFT: each dict has `role` and `content`)*

### Sequence Classification
- `text`: string
- `label`: integer or string

### Token Classification
- `tokens`: list of strings
- `labels`: list of integers or strings aligned with `tokens`

### Extractive Question Answering
- `question`: string
- `context`: string
- `answers`: dictionary containing `text` (list of strings) and `answer_start` (list of integers)

### Sequence-to-Sequence Modeling
- `source`: string
- `target`: string

### Contrastive Loss
- `text_a`: string
- `text_b`: string
- `label`: integer (1 = similar, 0 = dissimilar)

### Triplet Loss
- `anchor`: string
- `positive`: string
- `negative`: string

### InfoNCE Loss
- `text_a`: string (anchor)
- `text_b`: string (positive pair; negatives are sampled in-batch)

### Image-Text Contrastive
- `image`: PIL Image
- `text`: string

### Image-Text Sigmoid Contrastive
- `image`: PIL Image
- `text`: string

### REINFORCE / REINFORCE with baseline
- `prompt`: list of dicts (chat messages, same format as IFT)
- `answer`: string (ground-truth final answer; reward is `1.0` if the model's `\boxed{...}` matches via `math_verify`, else `0.0`)


## ⚡ Getting Started

### Cloning

```bash
git clone https://github.com/Ritvik19/nanoformers.git
cd nanoformers
```

### Installation
```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Usage

#### Causal Language Modeling (CLM)

```bash
python -m src.cli.train_clm --config configs/clm_qwen_gsm8k.yaml
```

#### Masked Language Modeling (MLM)

```bash
python -m src.cli.train_mlm --config configs/mlm_bert_open_web_text.yaml
```

#### Span Corruption

```bash
python -m src.cli.train_span_corruption --config configs/span_corruption_t5_base_open_web_text.yaml
```

#### Instruction Fine-Tuning (IFT)

```bash
python -m src.cli.train_ift --config configs/ift_qwen_gsm8k.yaml
```

#### Direct Preference Optimization (DPO)

```bash
python -m src.cli.train_dpo --config configs/dpo_qwen_gsm8k.yaml
```

#### Sequence Classification

```bash
python -m src.cli.train_sequence_classification --config configs/seqclf_distilbert_emotion.yaml
```

#### Token Classification

```bash
python -m src.cli.train_token_classification --config configs/tokclf_distilbert_conll2003.yaml
```

#### Extractive Question Answering

```bash
python -m src.cli.train_question_answering --config configs/qa_distilbert_squad.yaml
```

#### Sequence-to-Sequence Modeling

```bash
python -m src.cli.train_sequence_to_sequence --config configs/seq2seq_flan_t5_base_gsm8k.yaml
```

#### Contrastive Loss

```bash
python -m src.cli.train_contrastive --config configs/contrastive_loss_distilbert_qqp.yaml
```

#### Triplet Loss

```bash
python -m src.cli.train_triplet --config configs/triplet_loss_distilbert_qqp.yaml
```

#### InfoNCE Loss

```bash
python -m src.cli.train_info_nce --config configs/info_nce_loss_distilbert_qqp.yaml
```

#### Image-Text Contrastive

```bash
python -m src.cli.train_image_text_contrastive --config configs/image_text_contrastive_clip_flickr30k.yaml
```

#### Image-Text Sigmoid Contrastive

```bash
python -m src.cli.train_image_text_sigmoid_contrastive --config configs/image_text_sigmoid_contrastive_siglip_flickr30k.yaml
```

#### REINFORCE 

The RL training loop is split into two processes that you launch on separate
GPU pools using `CUDA_VISIBLE_DEVICES`:

1. **vLLM rollout server** — produces on-policy completions, exposes an
   OpenAI-compatible API plus a `/collective_rpc` admin endpoint that the
   trainer uses to hot-swap weights between optimizer steps. The launch
   script sets `VLLM_SERVER_DEV_MODE=1`, which is what gates `/collective_rpc`
   and `/reset_prefix_cache` in vLLM 0.10.x — without it those endpoints
   return 404 and weight sync silently fails.

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh
   ```

   Override defaults via env vars: `MODEL=Qwen/Qwen3-0.6B PORT=8000
   GPU_MEMORY_UTILIZATION=0.9 bash scripts/serve_vllm.sh`.

2. **REINFORCE trainer** — pulls rollouts from vLLM, scores them with
   `math_verify` against the boxed answer, runs the REINFORCE backward pass
   on the local policy.

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_reinforce.sh
```

   Or directly:

```bash
python -m src.cli.train_reinforce --config configs/reinforce_qwen_gsm8k.yaml
```

Toggle the variant via [configs/reinforce_qwen_math.yaml](configs/reinforce_qwen_math.yaml):

- `use_baseline: false` → vanilla REINFORCE (`L = -R * sum_t log pi(y_t)`).
- `use_baseline: true` → REINFORCE with batch-mean baseline
  (`L = -(R - mean(R)) * sum_t log pi(y_t)`).
- `length_normalize: false` → divide the per-sequence log-prob sum by the
  completion length before scaling by the advantage. The unbiased
  policy-gradient estimator uses the raw sum (`false`), but the mean form
  (`true`) prevents long rollouts from dominating the gradient by token count
  and matches what PPO / GRPO / RLOO use in practice.
- `kl_coeff: 0.0` → no reference model (lighter on training-side memory).
  Set `> 0` to load a frozen reference and add a KL-to-reference penalty.

---

## 📰 Update Log

### 2026-04-29
- Added a REINFORCE training module with vLLM rollout server, REINFORCE with baseline, and REINFORCE with KL penalty.
- Trained `Qwen/Qwen3-0.6B` on `rasbt/math_full_minus_math500` dataset resulting in 6% lift in pass@1 accuracy (average of 4) on `HuggingFaceH4/MATH-500`.
- Fixed gradient-accumulation logging across all training pipelines: each logged scalar now describes the full effective batch (`gradient_accumulation_steps` micro-batches) instead of just the last micro-batch.


### 2026-04-23
- Standardized automatic mixed precision in all training `forward_loss` implementations: `torch.cuda.amp.autocast` now passes `dtype=torch.bfloat16` (instead of the default float16) whenever CUDA is available. 

### 2026-04-22
- Trained `bert-base-uncased` on `open-web-text` for 1 epoch of MLM, reducing loss from 2.7277 (perplexity 15.30) for the pretrained `bert-base-uncased` to 1.4968 (perplexity 4.47).
- Trained `t5-base` on `open-web-text` for 1 epoch of Span Corruption, reducing loss from 1.7892 (perplexity 5.98) for the pretrained `t5-base` to 1.5889 (perplexity 4.90).

### 2026-04-21
- Added a T5-style span corruption training module with the standard `random_spans_noise_mask` algorithm (15% corruption, mean span length 3), online sentinel construction, manual seq2seq NLL.

### 2026-04-20
- Added a masked language modeling training module with dynamic 15% / 80-10-10 masking, manual MLM NLL computation.

### 2026-04-04
- Trained `distilbert/distilbert-base-uncased` on `qqp-contrastive` dataset resulting in 93.90% accuracy and 92.35% F1 score `glue/qqp` validation set.
- Trained `distilbert/distilbert-base-uncased` on `qqp-triplet` dataset resulting in 61.69% accuracy and 65.77% F1 score `glue/qqp` validation set.
- Trained `distilbert/distilbert-base-uncased` on `qqp-info_nce` dataset resulting in 76.37% accuracy and 75.64% F1 score `glue/qqp` validation set.
- Updated image-text contrastive learning modules to use separate text and image encoders.
- Trained `FacebookAI/roberta-base` and `google/vit-base-patch16-224` on `Ritvik19/flickr30k` dataset using image-text contrastive loss resulting in 90.1% image-to-text R@10 and 80.82% text-to-image R@10 on the Flickr30k test set.
- Trained `FacebookAI/roberta-base` and `google/vit-base-patch16-224` on `Ritvik19/flickr30k` dataset using image-text sigmoid contrastive loss resulting in 84.4% image-to-text R@10 and 75.14% text-to-image R@10 on the Flickr30k test set.

### 2026-04-03
- Added contrastive learning modules: contrastive loss, triplet loss, and InfoNCE loss for text-text representation learning with encoder-only models.
- Added image-text contrastive learning modules: softmax-based (image-text contrastive) and sigmoid-based (image-text sigmoid contrastive) for dual encoder vision-language training.

### 2026-04-01
- Added a sequence-to-sequence modeling training module with dataset tokenization, dynamic padding, perplexity evaluation, and a CLI/config example.
- Trained `google/flan-t5-base` on `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-seq2seq` dataset resulting in 15% pass@1 accuracy (average of 4).

### 2026-03-30
- Added a sequence classification training module with dataset preprocessing, evaluation accuracy, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `dair-ai/emotion` dataset resulting in 93.8% accuracy.
- Added a token classification training module with subword label alignment, masked token accuracy, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `conll-2003` dataset resulting in 90.0% accuracy and 90.0% F1 score.
- Added an extractive question answering training module with SQuAD dataset parsing, exact match boundary evaluation, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `squad-v2` dataset resulting in 62.29% exact match and 65.74% F1 score.

### 2026-03-29
- Implemented manual causal language modeling loss calculation for Instruction Fine-Tuning.
- Implemented manual causal language modeling loss computation in the Causal Language Modeling pipeline.
- Made BOS token prepending optional in Causal Language Modeling dataset processing based on `bos_token_id` configuration.
- Trained `Qwen/Qwen3-0.6B-Base` on `openai/gsm8k` dataset distilled from `Qwen/Qwen3-0.6B` resulting in 20% lift in zero shot pass@1 accuracy (average of 4).

### 2026-03-28
- Verified the DPO loss implementation step by step and added comprehensive documentation.
- Refined DPO training configuration with optimized parameters. Trained `Qwen/Qwen3-0.6B` on a custom onpolicy variant of `openai/gsm8k` dataset resulting in 10% lift in pass@1 accuracy (average of 4).

### 2026-03-27
- Updated all training pipelines to natively support datasets from the Hugging Face Hub (via `datasets.load_dataset`) alongside local files.

### 2026-03-26
- Trained `Qwen/Qwen3-0.6B` on a custom onpolicy variant of `openai/gsm8k` dataset resulting in 5% lift in pass@1 accuracy (average of 4).

### 2026-03-23
- Fixed 4 bugs in training scripts across CLM, IFT, and DPO
- Resolved critical `NameError` in the CLM dataset utilities (missing `for` clause in target token masking)
- Added `test_size` guards in the IFT and DPO pipelines to prevent `test_size=0` crashes
- Ensured `model.train()` is called after evaluation in all training loops

### 2025-10-24
- Removed stride parameter from `group_texts` function for consistency in Instruction Fine-Tuning

### 2025-10-22
- Added training scripts for Instruction Fine-Tuning  
- Trained `unsloth/gemma-3-270m-it` on `openai/gsm8k` dataset  
- Fixed loss masking for padding tokens in Causal Language Modeling

### 2025-10-11
- Added training scripts for Causal Language Modeling  
- Trained `google/gemma-3-270m` on `roneneldan/TinyStories` dataset
---
