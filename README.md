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
| | Sequence Classification | Encoder-only | ✅ |
| | Token Classification | Encoder-only | ✅ |
| | Extractive Question Answering | Encoder-only | ✅ |
| | Sequence-to-Sequence Modeling | Encoder-Decoder | ✅ |
| **Reinforcement** | Reinforce | Decoder-only | ⬜️ |
| | Reinforce with baseline | Decoder-only | ⬜️ |
| | Proximal Policy Optimization | Decoder-only | ⬜️ |
| | Group Relative Policy Optimization | Decoder-only | ⬜️ |
| **Contrastive** | Contrastive Loss | Agnostic | ⬜️ |
| | Triplet Loss | Agnostic | ⬜️ |
| | InfoNCE Loss | Agnostic | ⬜️ |

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

### 2026-03-30
- Added a sequence classification training module with dataset preprocessing, evaluation accuracy, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `dair-ai/emotion` dataset resulting in 93.8% accuracy.
- Added a token classification training module with subword label alignment, masked token accuracy, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `conll-2003` dataset resulting in 90.0% accuracy and 90.0% F1 score.
- Added an extractive question answering training module with SQuAD dataset parsing, exact match boundary evaluation, and a CLI/config example.
- Trained `distilbert/distilbert-base-uncased` on `squad-v2` dataset resulting in 62.29% exact match and 65.74% F1 score.

### 2026-04-01
- Added a sequence-to-sequence modeling training module with dataset tokenization, dynamic padding, perplexity evaluation, and a CLI/config example.
- Trained `google/flan-t5-base` on `Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-seq2seq` dataset resulting in 15% pass@1 accuracy (average of 4).
---
