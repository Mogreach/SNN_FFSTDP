"""
====================================================================
File          : visualization_debug.py
Description   : 记录调试ff_stdp的可视化代码片段
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-05-04
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

"""
Function Name: 可视化权重特征

参数:
param_name (type): param为 torch.Tensor 或 np.array，形状为 [500, 784]。

"""
def vis_weight_feature(param):
    # 假设 param 是 shape: [500, 784] 的 torch.Tensor 或 np.array
    if isinstance(param, torch.Tensor):
        images = param.view(-1, 28, 28).cpu().numpy()
    else:
        images = param.reshape(-1, 28, 28)

    # 可视化前 256 张图像
    num_imgs = min(256, images.shape[0])  # 最多显示256张
    rows = cols = int(np.sqrt(num_imgs))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i])
        ax.axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.show()
