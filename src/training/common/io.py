from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.training.contrastive_learning.dual_encoder import DualEncoderModel


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def load_causal_lm_model(model_path, device, load_weights=True):
    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))
    model.to(device)
    return model


def load_masked_lm_model(model_path, device, load_weights=True):
    if load_weights:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    else:
        model = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(model_path))
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


def load_encoder_model(model_path, device):
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    return model


def load_image_text_contrastive_model(text_model_path, image_model_path, projection_dim, device):
    text_encoder = AutoModel.from_pretrained(text_model_path)
    image_encoder = AutoModel.from_pretrained(image_model_path)
    model = DualEncoderModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        text_hidden_size=text_encoder.config.hidden_size,
        image_hidden_size=image_encoder.config.hidden_size,
        projection_dim=projection_dim,
        use_logit_bias=False,
    )
    model.to(device)
    return model


def load_image_text_sigmoid_contrastive_model(text_model_path, image_model_path, projection_dim, device):
    text_encoder = AutoModel.from_pretrained(text_model_path)
    image_encoder = AutoModel.from_pretrained(image_model_path)
    model = DualEncoderModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        text_hidden_size=text_encoder.config.hidden_size,
        image_hidden_size=image_encoder.config.hidden_size,
        projection_dim=projection_dim,
        use_logit_bias=True,
    )
    model.to(device)
    return model


def load_image_text_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def load_image_text_processor(model_path):
    return AutoImageProcessor.from_pretrained(model_path)


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
