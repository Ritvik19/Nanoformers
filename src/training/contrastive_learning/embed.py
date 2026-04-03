import torch
import torch.nn.functional as F


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp_min(1)
    return summed / count


def encode(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
    return F.normalize(embeddings, p=2, dim=-1)
