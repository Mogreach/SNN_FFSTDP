import torch
import torchvision
import torchvision.transforms as transforms
import os
from config import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
def reorder_and_save_mnist(train_dataset, save_path):
    # 创建标签字典：label -> list of (image, label)
    label_dict = {i: [] for i in range(10)}

    for img, label in train_dataset:
        label_dict[label].append((img, label))

    # 找到最小可以组成完整组的数量
    min_group_count = min(len(v) for v in label_dict.values())

    num_full_groups = min_group_count  # 每组一个 label，有多少个样本就有多少组

    reordered_images = []
    reordered_labels = []

    for g in range(num_full_groups):
        for label in range(10):
            img, lbl = label_dict[label][g]
            reordered_images.append(img)
            reordered_labels.append(lbl)

    # 转成 Tensor
    reordered_images_tensor = torch.stack(reordered_images)
    reordered_labels_tensor = torch.tensor(reordered_labels)

    # 保存
    torch.save(
        {"images": reordered_images_tensor, "labels": reordered_labels_tensor},
        os.path.join(save_path, "MNIST_train_grouped_sorted.pt")
    )
    print(f"[✓] 重排后的数据已保存到: {os.path.join(save_path, 'MNIST_train_grouped_sorted.pt')}")


# =============================
# 示例使用方式（嵌入主函数中）
# =============================
if __name__ == "__main__":
    config = ConfigParser()
    args = config.parse()
    # 假设 config.parse() 得到的是 args

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    reorder_and_save_mnist(train_dataset, args.data_dir)