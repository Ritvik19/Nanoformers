"""On-policy RL prompt dataset.

Each example is `{"prompt": [chat messages], "answer": str}`. The dataset is
materialised once at startup: the chat template is applied with
`add_generation_prompt=True` so vLLM can generate the assistant turn directly.
"""

from torch.utils.data import Dataset


class RLPromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.prompts = []
        self.answers = []
        for item in hf_dataset:
            prompt_text = tokenizer.apply_chat_template(
                item["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            self.prompts.append(prompt_text)
            self.answers.append(str(item["answer"]))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "answer": self.answers[idx],
        }

    def __repr__(self):
        feature_list = ["prompt", "answer"]
        return (
            f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
        )
