import torch
import torch.nn.functional as F


def per_example_sum_logprob(log_probs, labels):
    sequence_length = log_probs.size(1)
    labels = labels[:, :sequence_length]

    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = 0
    token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * labels.ne(-100)
    return token_logp.sum(dim=1)


def forward_loss(model, ref_model, batch, beta):
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        chosen_logits = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            return_dict=True,
        ).logits
        rejected_logits = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            return_dict=True,
        ).logits

        with torch.no_grad():
            ref_chosen_logits = ref_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                return_dict=True,
            ).logits
            ref_rejected_logits = ref_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                return_dict=True,
            ).logits

        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
        ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

        pi_chosen = per_example_sum_logprob(chosen_log_probs, batch["chosen_target_ids"])
        pi_rejected = per_example_sum_logprob(
            rejected_log_probs, batch["rejected_target_ids"]
        )
        ref_pi_chosen = per_example_sum_logprob(
            ref_chosen_log_probs, batch["chosen_target_ids"]
        )
        ref_pi_rejected = per_example_sum_logprob(
            ref_rejected_log_probs, batch["rejected_target_ids"]
        )

        advantage = (pi_chosen - pi_rejected) - (ref_pi_chosen - ref_pi_rejected)
        loss = -F.logsigmoid(beta * advantage).mean()

    return loss, advantage, pi_chosen - pi_rejected
