from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def load_causal_lm_model(model_path, device, load_weights=True):
    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))
    model.to(device)
    return model


def load_sequence_classification_model(model_path, device, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


def load_token_classification_model(model_path, device, num_labels):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


def load_question_answering_model(model_path, device):
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


def load_reference_model(model_path, device):
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def load_sequence_to_sequence_model(model_path, device, load_weights=True):
    if load_weights:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(model_path))
    model.to(device)
    return model
