from datasets import Dataset as HFDataset


def load_json_dataset(dataset_path):
    return HFDataset.from_json(dataset_path)


def compute_test_size(num_rows):
    test_size = num_rows % 100
    return test_size if test_size >= 50 else test_size + 100


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
