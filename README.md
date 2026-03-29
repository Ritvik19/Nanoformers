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
| | Masked Language Modeling | Encoder-only | ⬜️ |
| | Span Corruption | Encoder-Decoder | ⬜️ |
| **Supervised** | Instruction Fine-Tuning | Decoder-only | ✅ |
| | Direct Preference Optimization | Decoder-only | ✅ |
| | Sequence Classification | Encoder-only | ⬜️ |
| | Token Classification | Encoder-only | ⬜️ |
| | Extractive Question Answering | Encoder-only | ⬜️ |
| | Sequence-to-Sequence Modeling | Encoder-Decoder | ⬜️ |
| **Reinforcement** | Proximal Policy Optimization | Decoder-only | ⬜️ |
| | Group Relative Policy Optimization | Decoder-only | ⬜️ |
| **Contrastive** | Contrastive Loss | Agnostic | ⬜️ |
| | Triplet Loss | Agnostic | ⬜️ |
| | InfoNCE Loss | Agnostic | ⬜️ |

---

## 🚀 Models Trained

| Model | Dataset | Task | Configuration | Logs |
|-------|----------|------|----------------|------|
| `Qwen/Qwen3-0.6B` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-ift` | Instruction Fine-Tuning | [ift_qwen_gsm8k.yaml](configs/ift_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/6njm3m9q?nw=nwuserritvik19) |
| `Qwen/Qwen3-0.6B` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-dpo-dedup` | Direct Preference Optimization | [dpo_qwen_gsm8k.yaml](configs/dpo_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/3hsxkyfp?nw=nwuserritvik19) |
| `Qwen/Qwen3-0.6B-Base` | `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-cpt` | Causal Language Modeling | [clm_qwen_gsm8k.yaml](configs/clm_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/twq18n69) |


## 🗂️ Dataset Schemas

### Causal Language Modeling (CLM)
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

#### Instruction Fine-Tuning (IFT)

```bash
python -m src.cli.train_ift --config configs/ift_qwen_gsm8k.yaml
```

#### Direct Preference Optimization (DPO)

```bash
python -m src.cli.train_dpo --config configs/dpo_qwen_gsm8k.yaml
```
--- 

## 📰 Update Log

### 2025-10-11
- Added training scripts for Causal Language Modeling  
- Trained `google/gemma-3-270m` on `roneneldan/TinyStories` dataset

### 2025-10-22
- Added training scripts for Instruction Fine-Tuning  
- Trained `unsloth/gemma-3-270m-it` on `openai/gsm8k` dataset  
- Fixed loss masking for padding tokens in Causal Language Modeling

### 2025-10-24
- Removed stride parameter from `group_texts` function for consistency in Instruction Fine-Tuning

### 2026-03-23
- Fixed 4 bugs in training scripts across CLM, IFT, and DPO
- Resolved critical `NameError` in the CLM dataset utilities (missing `for` clause in target token masking)
- Added `test_size` guards in the IFT and DPO pipelines to prevent `test_size=0` crashes
- Ensured `model.train()` is called after evaluation in all training loops

### 2026-03-26
- Trained `Qwen/Qwen3-0.6B` on a custom onpolicy variant of `openai/gsm8k` dataset resulting in 5% lift in pass@1 accuracy (average of 4).

### 2026-03-27
- Updated all training pipelines to natively support datasets from the Hugging Face Hub (via `datasets.load_dataset`) alongside local files.

### 2026-03-28
- Verified the DPO loss implementation step by step and added comprehensive documentation.
- Refined DPO training configuration with optimized parameters. Trained `Qwen/Qwen3-0.6B` on a custom onpolicy variant of `openai/gsm8k` dataset resulting in 10% lift in pass@1 accuracy (average of 4).

### 2026-03-29
- Implemented manual causal language modeling loss calculation for Instruction Fine-Tuning.
- Implemented manual causal language modeling loss computation in the Causal Language Modeling pipeline.
- Made BOS token prepending optional in Causal Language Modeling dataset processing based on `bos_token_id` configuration.
- Trained `Qwen/Qwen3-0.6B-Base` on `openai/gsm8k` dataset distilled from `Qwen/Qwen3-0.6B` resulting in 20% lift in zero shot pass@1 accuracy (average of 4).

---