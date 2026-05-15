"""PEFT (LoRA / QLoRA) utilities shared across all training pipelines.

Usage in any pipeline
---------------------
    from src.training.common.peft import (
        build_quantization_config,
        apply_peft_to_model,
        peft_enabled,
        qlora_enabled,
    )

    qcfg = build_quantization_config(args)
    model = load_causal_lm_model(model_path, device, quantization_config=qcfg)
    model = apply_peft_to_model(model, args, task_type="CAUSAL_LM")

Config block (all fields optional, block itself is optional):
    peft:
      use_lora: true
      use_qlora: false
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: null         # auto-inferred when null/absent
      modules_to_save: null        # e.g. ["score"] for seqclf head
      bnb_4bit_quant_type: "nf4"
      bnb_4bit_compute_dtype: "bfloat16"
      bnb_4bit_use_double_quant: true
      save_mode: "adapter"         # "adapter" | "merged" | "both"
      use_disable_adapter_as_ref: true
"""

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def peft_enabled(args: dict) -> bool:
    cfg = args.get("peft") or {}
    return bool(cfg.get("use_lora", False))


def qlora_enabled(args: dict) -> bool:
    cfg = args.get("peft") or {}
    return peft_enabled(args) and bool(cfg.get("use_qlora", False))


def use_disable_adapter_as_ref(args: dict) -> bool:
    """Return True when the LoRA model should serve as its own reference by
    temporarily disabling the adapter instead of loading a separate frozen copy.
    Defaults to True so callers don't have to spell it out every time."""
    cfg = args.get("peft") or {}
    return peft_enabled(args) and bool(cfg.get("use_disable_adapter_as_ref", True))


# ---------------------------------------------------------------------------
# QLoRA quantization config
# ---------------------------------------------------------------------------

def build_quantization_config(args: dict) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for 4-bit NF4 loading, or None."""
    if not qlora_enabled(args):
        return None

    cfg = args.get("peft") or {}
    quant_type = cfg.get("bnb_4bit_quant_type", "nf4")
    compute_dtype_str = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str)
    double_quant = bool(cfg.get("bnb_4bit_use_double_quant", True))

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
    )


# ---------------------------------------------------------------------------
# Target-module auto-detection
# ---------------------------------------------------------------------------

# Architecture families → candidate linear module name sets (checked by substring
# matching against the named modules in the loaded model).
_ARCH_PATTERNS = {
    # Decoder-only (Qwen, LLaMA, Mistral, Phi, …)
    "causal": [
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        # Fallback: GPT-2-style names
        ["c_attn", "c_proj"],
    ],
    # Encoder-only (BERT, DistilBERT, RoBERTa, …)
    "masked": [
        ["query", "key", "value"],
        # Fallback: attention.self.query style, still matched as substrings
        ["q_lin", "k_lin", "v_lin"],
    ],
    # Encoder-decoder (T5, FLAN-T5, …)
    "seq2seq": [
        ["q", "k", "v", "o"],
    ],
    # Vision (ViT, …)
    "vision": [
        ["query", "key", "value"],
        ["to_q", "to_k", "to_v"],
    ],
}


def _module_names_in_model(model) -> set[str]:
    """Return the set of leaf module *base* names (not full path) present in model."""
    return {name.split(".")[-1] for name, _ in model.named_modules()}


def resolve_target_modules(model, args: dict) -> list[str] | None:
    """Return the LoRA target module list from config, or auto-infer from model.

    Returns None when the model family is unrecognised — peft will then default
    to its own heuristic (which is usually fine for standard HF models).
    """
    cfg = args.get("peft") or {}
    explicit = cfg.get("target_modules")
    if explicit:
        return list(explicit)

    leaf_names = _module_names_in_model(model)

    # Walk through candidates in priority order; return the first fully-present set.
    for family_candidates in _ARCH_PATTERNS.values():
        for candidate_set in family_candidates:
            if all(m in leaf_names for m in candidate_set):
                return candidate_set

    return None


# ---------------------------------------------------------------------------
# Apply PEFT to a loaded model
# ---------------------------------------------------------------------------

_TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "MASKED_LM": TaskType.TOKEN_CLS,   # peft uses TOKEN_CLS as a proxy for MLM
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "SEQ_CLS": TaskType.SEQ_CLS,
    "TOKEN_CLS": TaskType.TOKEN_CLS,
    "QUESTION_ANS": TaskType.QUESTION_ANS,
    "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
}


def apply_peft_to_dual_encoder(model, args: dict):
    """Wrap *model.text_encoder* and *model.image_encoder* each with LoRA.

    The projection heads (`text_projection`, `image_projection`) and the
    `logit_scale` / `logit_bias` scalars are not wrapped — they remain regular
    full-precision trainable parameters in the parent `DualEncoderModel`.

    When PEFT is disabled the model is returned unchanged.
    """
    if not peft_enabled(args):
        return model

    if qlora_enabled(args):
        model.text_encoder = prepare_model_for_kbit_training(
            model.text_encoder, use_gradient_checkpointing=True
        )
        model.image_encoder = prepare_model_for_kbit_training(
            model.image_encoder, use_gradient_checkpointing=True
        )

    cfg = args.get("peft") or {}
    # Do NOT forward modules_to_save to the sub-encoder wrapping — projections
    # live outside the sub-encoder PeftModels and are already trainable.
    sub_args = {**args, "peft": {**cfg, "modules_to_save": None}}

    model.text_encoder = apply_peft_to_model(
        model.text_encoder, sub_args, task_type="FEATURE_EXTRACTION"
    )
    model.image_encoder = apply_peft_to_model(
        model.image_encoder, sub_args, task_type="FEATURE_EXTRACTION"
    )
    return model


def apply_peft_to_model(model, args: dict, task_type: str = "CAUSAL_LM"):
    """Wrap *model* with a LoraConfig and return the PeftModel.

    When PEFT is disabled (peft block absent or use_lora=false) the model is
    returned unchanged.

    task_type must be one of the keys in _TASK_TYPE_MAP above.
    """
    if not peft_enabled(args):
        return model

    cfg = args.get("peft") or {}
    peft_task = _TASK_TYPE_MAP.get(task_type, TaskType.CAUSAL_LM)

    if qlora_enabled(args):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    target_modules = resolve_target_modules(model, args)
    modules_to_save = cfg.get("modules_to_save") or None
    if isinstance(modules_to_save, list) and len(modules_to_save) == 0:
        modules_to_save = None

    lora_config = LoraConfig(
        task_type=peft_task,
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
