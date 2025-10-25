# 🧠 Nanoformers

A minimal playground for building and training transformer models from scratch.  
It covers **self-supervised**, **supervised**, and **reinforcement learning** training loops, along with a **tiny transformer architecture** for research and experimentation.

---

## 🔍 Objectives

- Implement tiny transformer architectures from scratch  
- Build training loops for:
  - **Self-Supervised Learning**
    - [x] Causal Language Modeling
  - **Supervised Learning**
    - [x] Instruction Fine-Tuning
    - [x] Direct Preference Optimization
  - **Reinforcement Learning**
    - [ ] Proximal Policy Optimization
    - [ ] Group Relative Policy Optimization

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

### 2025-10-25
- Added training scripts for Direct Preference Optimization  
- Trained `unsloth/gemma-3-270m-it` on `argilla/ultrafeedback-binarized-preferences-cleaned` dataset

---

## 🚀 Models Trained

| Model | Dataset | Task | Configuration | Logs |
|-------|----------|------|----------------|------|
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Causal Language Modeling | [clm_gemma_tiny_stories.yaml](configs/clm_gemma_tiny_stories.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/1vy7mhf1?nw=nwuserritvik19) |
| [unsloth/gemma-3-270m-it](https://huggingface.co/unsloth/gemma-3-270m-it) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Instruction Fine-Tuning | [ift_gemma_gsm8k.yaml](configs/ift_gemma_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/klfnahkm?nw=nwuserritvik19) |
| [unsloth/gemma-3-270m-it](https://huggingface.co/unsloth/gemma-3-270m-it) | [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | Direct Preference Optimization | [dpo_gemma_ultrafeedback.yaml](configs/dpo_gemma_ultrafeedback.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/bd4dlvqf?nw=nwuserritvik19) |

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
python src/training/self_supervised_learning/causal_language_modeling.py \
  --config configs/clm_gemma_tiny_stories.yaml
```

#### Instruction Fine-Tuning (IFT)

```bash
python src/training/supervised_learning/instruction_fine_tuning.py \
  --config configs/ift_gemma_gsm8k.yaml
```

#### Direct Preference Optimization (DPO)

```bash
python src/training/supervised_learning/direct_preference_optimization.py \
  --config configs/dpo_gemma_ultra_feedback.yaml
```
