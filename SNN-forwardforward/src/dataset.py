from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import torch
import random
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
    
class AugmentedMNIST(torchvision.datasets.MNIST):
    """带有增强的 MNIST 数据集。
    在 __getitem__ 中：若样本 label 是难类（5,8,9）则有概率用增强版本替换原图。
    注意：保持返回 (tensor, label) 格式，与原 MNIST 兼容。
    """
    def __init__(self, root, train=True, transform=None, aug_transform=None, aug_labels=(5,8,9), aug_prob=0.6, download=False):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.aug_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        self.aug_labels = set(aug_labels)
        self.aug_prob = aug_prob

    def __getitem__(self, index):
        img, label = super().__getitem__(index)  # img is PIL Image if transform is not None
        # 如果 transform 已经是 ToTensor， we re-create PIL from tensor to apply aug_transform
        # 为了兼容 pipeline，我们 apply aug_transform on PIL image before ToTensor
        if label in self.aug_labels and random.random() < self.aug_prob:
            # 原始数据 self.data 存的是 uint8 ndarray，直接用 PIL 来增强更方便
            pil_img = Image.fromarray(self.data[index].numpy(), mode='L')
            img = self.aug_transform(pil_img)
            return img, label
        else:
            # 使用基础 transform (ToTensor)
            # If base transform expects PIL, use Image.fromarray
            if isinstance(self.transform, transforms.ToTensor) or self.transform is None:
                pil_img = Image.fromarray(self.data[index].numpy(), mode='L')
                return transforms.ToTensor()(pil_img), label
            else:
                return super().__getitem__(index)