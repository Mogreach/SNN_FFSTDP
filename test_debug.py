"""
====================================================================
File          : test_debug.py
Description   : SNN-FF测试代码
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""


import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import argparse
import sys
import torch
import torch.utils.data as data
import torchvision
from src.ff_snn_net import Net
import torch.nn.functional as F
from config import ConfigParser
def get_y_neg(y,device):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(10))
        # print("allowed_indices:", allowed_indices)
        # print("y_samp:", y_samp.item())
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)

def overlay_y_on_x(x, y,classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, 0, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        label = y[i].item()  # y[i]是该样本的标签
        # 确保标签在0到9之间（根据设置的 classes）
        if 0 <= label < classes:
            # 将第一通道前10个像素位置中对应标签的像素赋值为最大值
            x_[i, 0, label, 0] = x_.max()  # 将每个样本前10个像素中，对应标签类别序号赋为当前矩阵最大值
    return x_

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().sum(dim=0) #.reshape(1, 28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def plot_loss(loss_of_layer_list, save_path):
    # 获取层数和每层的损失数据
    num_layers = len(loss_of_layer_list)
    
    # 创建一个图形
    plt.figure(figsize=(10, 6))  # 设置图像大小
    
    # 绘制每一层的损失随 epoch 变化的曲线
    for layer_idx in range(num_layers):
        plt.plot(loss_of_layer_list[layer_idx], 'o-', label=f'Layer {layer_idx + 1}')
    
    # 设置图形的标签和标题
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')    # y轴标签
    plt.title('Loss vs Epoch for Each Layer')  # 图形标题
    
    # 显示图例和网格
    plt.legend()  # 显示图例，标识不同的层
    plt.grid(True)  # 显示网格
    
    # 保存图像到文件
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
def visualize_goodness(goodness_label_layer_goodness, save_path=None):
    # goodness_label_layer_goodness: [10, num_layers, batch_size]
    data = goodness_label_layer_goodness[:, :, 0].cpu().numpy()
    num_labels, num_layers = data.shape

    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Layer {i}" for i in range(num_layers)],
                yticklabels=[f"Label {i}" for i in range(num_labels)])
    plt.xlabel("Layer")
    plt.ylabel("Label")
    plt.title("Goodness Distribution per Label and Layer")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
def visualize_freq(freq_label_layer_freq, label_id, save_path=None):
    # freq_label_layer_freq: [10, num_layers, batch_size, 1000]
    freq_data = freq_label_layer_freq[label_id, :, 0, :].cpu().numpy()
    num_layers, num_neurons = freq_data.shape

    plt.figure(figsize=(12, 2 * num_layers))
    for i in range(num_layers):
        plt.subplot(num_layers, 1, i + 1)
        plt.plot(freq_data[i], color='blue')
        plt.title(f"Label {label_id} - Layer {i}")
        plt.xlabel("Neuron index")
        plt.ylabel("Firing frequency")
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_layer_weights(net):
    """
    在同一张图中绘制每层权重的分布直方图和二维热力图
    """
    num_layers = len(net.layers)
    fig, axes = plt.subplots(num_layers, 2, figsize=(10, 3 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, 2)  # 兼容单层情况

    for layer_idx, layer in enumerate(net.layers):
        # 遍历命名子模块，找到真正的可学习层（你模型中 name=="layer" 的模块）
        for name, module in layer.named_modules():
            if name == "layer":
                for p_name, param in module[1].named_parameters():
                    w = param.detach().cpu().numpy()

                    # ------------------------
                    # 1️⃣ 权重分布直方图
                    # ------------------------
                    axes[layer_idx, 0].hist(w.flatten(), bins=50, color='steelblue', alpha=0.7)
                    axes[layer_idx, 0].set_title(f"Layer {layer_idx} - {p_name} weight distribution")
                    axes[layer_idx, 0].set_xlabel("Weight value")
                    axes[layer_idx, 0].set_ylabel("Count")

                    # ------------------------
                    # 2️⃣ 二维权重热力图
                    # ------------------------
                    # 如果是高维卷积权重，取第一输出通道可视化
                    if w.ndim > 2:
                        w2d = w[0, 0, :, :]  # 只取第一组卷积核
                    else:
                        w2d = w

                    sns.heatmap(w2d, ax=axes[layer_idx, 1], cmap="coolwarm", cbar=True)
                    axes[layer_idx, 1].set_title(f"Layer {layer_idx} - {p_name} heatmap")

    plt.tight_layout()
    plt.show()
def visualize_goodness_mean(net, test_data_loader, device):
    """
    统计所有预测正确样本的各类别平均 goodness，并绘制热力图
    """
    num_labels = 10
    num_layers = len(net.layers)

    # 累计各类 goodness 总和与计数
    goodness_sum = torch.zeros(num_labels, num_layers).to(device)
    goodness_count = torch.zeros(num_labels).to(device)

    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            x_te, y_te = x_te.to(device), y_te.to(device)
            predict_result, goodness, _ = net.predict_analyze(x_te)  
            goodness = goodness.to(device)
            # goodness: [10, num_layers, batch_size]

            # 获取每个样本是否预测正确
            correct_mask = predict_result.eq(y_te)
            for idx in range(x_te.shape[0]):
                label = y_te[idx].item()
                if correct_mask[idx]:
                    # 取该样本下对应标签的goodness: [num_layers]
                    sample_goodness = goodness[label, :, idx]
                    goodness_sum[label] += sample_goodness
                    goodness_count[label] += 1

    # 避免除以0
    goodness_mean = torch.zeros_like(goodness_sum)
    for l in range(num_labels):
        if goodness_count[l] > 0:
            goodness_mean[l] = goodness_sum[l] / goodness_count[l]

    # 转CPU并numpy化
    goodness_mean = goodness_mean.cpu().numpy()

    # ===============================
    # 绘制热力图
    # ===============================
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        goodness_mean,
        annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[f"Layer {i}" for i in range(num_layers)],
        yticklabels=[f"Label {i}" for i in range(num_labels)]
    )
    plt.xlabel("Layer")
    plt.ylabel("Label")
    plt.title("Average Goodness per Label and Layer (Correct Predictions Only)")
    plt.tight_layout()
    plt.show()

    return goodness_mean
def main():
    config = ConfigParser()
    args = config.parse()
###########################################################################################
####################################前向学习的代码结构######################################
    # 初始化数据加载器
    # 加载训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    # 划分训练集和验证集
    train_size = int(0.95 * len(train_dataset))  # 80% 用于训练
    val_size = len(train_dataset) - train_size  # 20% 用于验证
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1000,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    device = torch.device("cuda")
    net = Net(dims=[784, 512,256, 10],tau=args.tau, epoch=args.epochs, T=8, lr=args.lr,
              v_threshold_pos=1.0,v_threshold_neg=-1.2, opt=args.opt, loss_threshold=0.5)
    net.load("logs/analyze/checkpoint_last_suprevised.pth")
    test_acc = 0
    test_samples = 0
    test_count = 0
    goodness_mean = visualize_goodness_mean(net, test_data_loader, device)
    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            test_samples += y_te.numel()
            test_count += 1
            x_te, y_te = x_te.to(device), y_te.to(device)
            predict_result , goodness,freq = net.predict_analyze(x_te)
            test_acc += predict_result.eq(y_te).cpu().float().mean().item()
            # if(x_te.shape[0] != args.b or test_samples >= args.b):
            #     break
        print("test Acc:", 100 * test_acc / test_count, "%")
    
    # visualize_layer_weights(net)
    # # 查看标签0-9的goodness分布情况
    # with torch.no_grad():
    #     for l in range(10):
    #         for x_te, y_te in test_data_loader:
    #             if (l != y_te):
    #                 continue
    #             else:
    #                 x_te, y_te = x_te.to(device), y_te.to(device)
    #                 predict_result , goodness,freq = net.predict_analyze(x_te)
    #                 test_acc = predict_result.eq(y_te).cpu().float().mean().item()
    #                 if test_acc == 1:
    #                     print("test Acc:", 100 * test_acc / test_count, "%")
    #                     visualize_goodness(goodness)
                        # break
    # with torch.no_grad():
    #     for l in range(10):
    #         for x_te, y_te in test_data_loader:
    #             if (l != y_te):
    #                 continue
    #             else:
    #                 x_te, y_te = x_te.to(device), y_te.to(device)
    #                 predict_result , goodness,freq = net.predict_analyze(x_te)
    #                 test_acc = predict_result.eq(y_te).cpu().float().mean().item()
    #                 if test_acc == 1:
    #                     print("test Acc:", 100 * test_acc / test_count, "%")
    #                     visualize_goodness(goodness)
    #                     break
    # with torch.no_grad():
    #     for x_te, y_te in test_data_loader:
    #         test_samples += y_te.numel()
    #         x_te, y_te = x_te.to(device), y_te.to(device)
    #         predict_result , goodness,freq = net.predict_analyze(x_te)
    #         test_acc = predict_result.eq(y_te).cpu().float().mean().item()
    #         if test_acc == 0:
    #             test_count += 1
    #         if (test_count == 10):
    #             break
    #     print("test Acc:", 100 * test_acc / test_count, "%")
    # visualize_goodness(goodness)
    # for label_id in range(10):
    #     visualize_freq(freq, label_id)
if __name__ == "__main__":
    main()

