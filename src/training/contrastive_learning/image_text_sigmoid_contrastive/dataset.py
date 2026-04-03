from torch.utils.data import Dataset


class ImageTextSigmoidContrastiveDataset(Dataset):
    """Image-text pair dataset for sigmoid-based contrastive training.

    Expects HF dataset columns: image (PIL Image), text (str).
    """

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return {"image": image, "text": item["text"]}

    def __repr__(self):
        feature_list = ["image", "text"]
        return f"Dataset({{\n    features: {feature_list},\n    num_rows: {len(self)}\n}})"
