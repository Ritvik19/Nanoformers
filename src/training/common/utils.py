from datasets import Dataset as HFDataset, load_dataset


def load_hf_dataset(dataset_path):
    if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        return HFDataset.from_json(dataset_path)
    return load_dataset(dataset_path)['train']


def compute_test_size(num_rows):
    test_size = num_rows % 100
    return test_size if test_size >= 50 else test_size + 100


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
