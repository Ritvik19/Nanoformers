import torch
import torch.nn.functional as F


def per_example_sum_logprob(log_probs, labels):
    # Causal LM alignment:
    # logits/log_probs at position t predict the token at position t+1.
    # HF `model(..., labels=...)` handles this shift internally, but here we
    # compute token logprobs manually, so we must shift ourselves.
    log_probs = log_probs[:, :-1, :]
    labels = labels[:, 1:]

    sequence_length = log_probs.size(1)
    labels = labels[:, :sequence_length]

    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = 0
    token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * labels.ne(-100)
    return token_logp.sum(dim=1)


def forward_loss(model, ref_model, batch, beta):
    # Standard DPO objective for one preference pair (x, y_w, y_l):
    #   L_DPO = -log sigma(beta * [
    #       (log pi_theta(y_w | x) - log pi_theta(y_l | x))
    #       - (log pi_ref(y_w | x) - log pi_ref(y_l | x))
    #   ])
    #
    # In the code below:
    # - "chosen" is y_w (the preferred completion)
    # - "rejected" is y_l (the dispreferred completion)
    # - per_example_sum_logprob(...) gives the sequence log-probability term
    #   log pi(... | x) by summing token log-probs over the target completion.
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

        # Convert token logits into log-probabilities so we can recover the
        # sequence-level log pi(y | x) terms that appear in the DPO equation.
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
        ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

        # These are the four log-probability terms from the equation:
        #   pi_chosen       -> log pi_theta(y_w | x)
        #   pi_rejected     -> log pi_theta(y_l | x)
        #   ref_pi_chosen   -> log pi_ref(y_w | x)
        #   ref_pi_rejected -> log pi_ref(y_l | x)
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

        # This is the bracketed term inside the sigmoid:
        #   (log pi_theta(y_w | x) - log pi_theta(y_l | x))
        #   - (log pi_ref(y_w | x) - log pi_ref(y_l | x))
        # DPO encourages this value to be positive, meaning the policy prefers
        # the chosen response more strongly than the reference does.
        advantage = (pi_chosen - pi_rejected) - (ref_pi_chosen - ref_pi_rejected)

        # Final loss:
        #   -log sigma(beta * advantage)
        # beta controls how sharply we push the policy toward the preference.
        loss = -F.logsigmoid(beta * advantage).mean()

    return loss, advantage, pi_chosen - pi_rejected
