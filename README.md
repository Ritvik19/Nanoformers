# Nanoformers

A minimal playground for building and training transformer models from scratch.
It covers self-supervised, supervised, and reinforcement learning training loops, along with a tiny transformer architecture for research and experimentation.

## 🔍 Objectives

* Implement tiny transformer architectures from scratch
* Build training loops:
    * Self-supervised learning 
        * [x] Causal Language Modeling
    * Supervised learning
        * [ ] Instruction Fine-tuning
        * [ ] Direct Preference Optimization
    * Reinforcement learning
        * [ ] Proximal Policy Optimization
        * [ ] Group Relative Policy Optimization


## 📰 News

#### 2025-10-11
- Added training scripts for Causal Language Modeling. 
- Trained `google/gemma-3-270m` on `roneneldan/TinyStories` dataset.

## 🚀 Models Trained 

| Model | Dataset | Task | Configuration | Logs |
|-------|---------|------|---------------|------|
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Causal Language Modeling | [clm_gemma_tiny_stories.yaml](configs/clm_gemma_tiny_stories.yaml) | [wandb](https://wandb.ai/ritvik19/nanoformers/runs/1vy7mhf1?nw=nwuserritvik19) |