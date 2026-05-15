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


def load_causal_lm_model(
    model_path, device, load_weights=True, quantization_config=None, device_map=None
):
    kwargs = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_path), **kwargs
        )
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_masked_lm_model(
    model_path, device, load_weights=True, quantization_config=None, device_map=None
):
    kwargs = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_weights:
        model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(model_path), **kwargs
        )
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_sequence_classification_model(
    model_path, device, num_labels, quantization_config=None, device_map=None
):
    kwargs = dict(num_labels=num_labels, ignore_mismatched_sizes=True)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_token_classification_model(
    model_path, device, num_labels, quantization_config=None, device_map=None
):
    kwargs = dict(num_labels=num_labels, ignore_mismatched_sizes=True)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_question_answering_model(
    model_path, device, quantization_config=None, device_map=None
):
    kwargs = dict(ignore_mismatched_sizes=True)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, **kwargs)
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_reference_model(model_path, device):
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def load_encoder_model(
    model_path, device, quantization_config=None, device_map=None
):
    kwargs = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(model_path, **kwargs)
    if quantization_config is None and device_map is None:
        model.to(device)
    return model


def load_image_text_contrastive_model(
    text_model_path,
    image_model_path,
    projection_dim,
    device,
    text_quantization_config=None,
    image_quantization_config=None,
    device_map=None,
):
    text_kwargs = {}
    if text_quantization_config is not None:
        text_kwargs["quantization_config"] = text_quantization_config
    if device_map is not None:
        text_kwargs["device_map"] = device_map
    text_encoder = AutoModel.from_pretrained(text_model_path, **text_kwargs)

    image_kwargs = {}
    if image_quantization_config is not None:
        image_kwargs["quantization_config"] = image_quantization_config
    if device_map is not None:
        image_kwargs["device_map"] = device_map
    image_encoder = AutoModel.from_pretrained(image_model_path, **image_kwargs)

    model = DualEncoderModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        text_hidden_size=text_encoder.config.hidden_size,
        image_hidden_size=image_encoder.config.hidden_size,
        projection_dim=projection_dim,
        use_logit_bias=False,
    )
    if (
        text_quantization_config is None
        and image_quantization_config is None
        and device_map is None
    ):
        model.to(device)
    return model


def load_image_text_sigmoid_contrastive_model(
    text_model_path,
    image_model_path,
    projection_dim,
    device,
    text_quantization_config=None,
    image_quantization_config=None,
    device_map=None,
):
    text_kwargs = {}
    if text_quantization_config is not None:
        text_kwargs["quantization_config"] = text_quantization_config
    if device_map is not None:
        text_kwargs["device_map"] = device_map
    text_encoder = AutoModel.from_pretrained(text_model_path, **text_kwargs)

    image_kwargs = {}
    if image_quantization_config is not None:
        image_kwargs["quantization_config"] = image_quantization_config
    if device_map is not None:
        image_kwargs["device_map"] = device_map
    image_encoder = AutoModel.from_pretrained(image_model_path, **image_kwargs)

    model = DualEncoderModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        text_hidden_size=text_encoder.config.hidden_size,
        image_hidden_size=image_encoder.config.hidden_size,
        projection_dim=projection_dim,
        use_logit_bias=True,
    )
    if (
        text_quantization_config is None
        and image_quantization_config is None
        and device_map is None
    ):
        model.to(device)
    return model


def load_image_text_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def load_image_text_processor(model_path):
    return AutoImageProcessor.from_pretrained(model_path)


def load_sequence_to_sequence_model(
    model_path, device, load_weights=True, quantization_config=None, device_map=None
):
    kwargs = dict(ignore_mismatched_sizes=True)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_weights:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_config(
            AutoConfig.from_pretrained(model_path), **kwargs
        )
    if quantization_config is None and device_map is None:
        model.to(device)
    return model
