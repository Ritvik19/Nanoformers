import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

def tokenize_function(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    input_ids = tokenizer(text, return_special_tokens_mask=False, add_special_tokens=False)['input_ids']
    prompt = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False)
    prompt_length = len(tokenizer(prompt, return_special_tokens_mask=False, add_special_tokens=False)['input_ids'])

    tokens = {
        'input_ids': tokenizer(text, return_special_tokens_mask=False, add_special_tokens=False)['input_ids'],
        'prompt_length': prompt_length,
    }
    return tokens

def group_texts(batch, block_size, tokenizer):
    input_ids = []
    target_ids = []
    for token_ids, prompt_length in zip(batch["input_ids"], batch["prompt_length"]):
        input_chunk = token_ids[:block_size]
        if len(input_chunk) != block_size :
            delta_input = block_size - len(input_chunk)
            input_chunk = input_chunk + [tokenizer.pad_token_id for _ in range(delta_input)]

        target_chunk = input_chunk.copy()
        target_chunk = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in target_chunk]
        target_chunk[:prompt_length] = [-100] * prompt_length
        input_ids.append(input_chunk)
        target_ids.append(target_chunk)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
    }

class IFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.ds = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(item["target_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": target_ids,
        }

    def __repr__(self): 
        feature_list = ['input_ids', 'labels']
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"

def collate_fn(batch, tokenizer):
    input_ids = [ex["input_ids"] for ex in batch]
    labels = [ex["labels"] for ex in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }   