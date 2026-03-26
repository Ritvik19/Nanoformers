# 🧠 Nanoformers

A minimal playground for building and training transformer models from scratch.  
It covers **self-supervised**, **supervised**, and **reinforcement learning** training loops, along with a **tiny transformer architecture** for research and experimentation.

---

## 🔍 Objectives

- Implement tiny transformer architectures from scratch  
- Build training loops for:
  - **Self-Supervised Learning**
    - [ ] Causal Language Modeling
  - **Supervised Learning**
    - [x] Instruction Fine-Tuning
    - [ ] Direct Preference Optimization
  - **Reinforcement Learning**
    - [ ] Proximal Policy Optimization
    - [ ] Group Relative Policy Optimization
  - **Contrastive Learning**
    - [ ] Contrastive Loss
    - [ ] Triplet Loss
    - [ ] InfoNCE Loss

---

## 📰 Updates

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

---

## 🚀 Models Trained

| Model | Dataset | Task | Configuration | Logs |
|-------|----------|------|----------------|------|
| `Qwen/Qwen3-0.6B` | `openai/gsm8k` | Instruction Fine-Tuning | [ift_gemma_gsm8k.yaml](configs/ift_qwen_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/6njm3m9q?nw=nwuserritvik19) |


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
python src/cli/train_clm.py --config configs/clm_gemma_tiny_stories.yaml
```

#### Instruction Fine-Tuning (IFT)

```bash
python src/cli/train_ift.py --config configs/ift_gemma_gsm8k.yaml
```

#### Direct Preference Optimization (DPO)

```bash
python src/cli/train_dpo.py --config configs/dpo_gemma_ultra_feedback.yaml
```
