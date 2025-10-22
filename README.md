# Nanoformers

A minimal playground for building and training transformer models from scratch.
It covers self-supervised, supervised, and reinforcement learning training loops, along with a tiny transformer architecture for research and experimentation.

## 🔍 Objectives

* Implement tiny transformer architectures from scratch
* Build training loops:
    * Self-supervised learning 
        * [x] Causal Language Modeling
    * Supervised learning
        * [x] Instruction Fine-tuning
        * [ ] Direct Preference Optimization
    * Reinforcement learning
        * [ ] Proximal Policy Optimization
        * [ ] Group Relative Policy Optimization


## 📰 News

#### 2025-10-11
- Added training scripts for Causal Language Modeling. 
- Trained `google/gemma-3-270m` on `roneneldan/TinyStories` dataset.

#### 2025-10-22
- Added training scripts for Instruction Fine-tuning.
- Trained `unsloth/gemma-3-270m-it` on `openai/gsm8k` dataset.
- Fixed loss masking for padding tokens in Causal Language Modeling.

## 🚀 Models Trained 

| Model | Dataset | Task | Configuration | Logs |
|-------|---------|------|---------------|------|
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Causal Language Modeling | [clm_gemma_tiny_stories.yaml](configs/clm_gemma_tiny_stories.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/1vy7mhf1?nw=nwuserritvik19) |
| [unsloth/gemma-3-270m-it](https://huggingface.co/unsloth/gemma-3-270m-it) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Instruction Fine-tuning | [ift_gemma_gsm8k.yaml](configs/ift_gemma_gsm8k.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/klfnahkm?nw=nwuserritvik19) |