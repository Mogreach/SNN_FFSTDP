from torch.utils.data import Dataset
import torch
class GroupedSortedMNIST(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
