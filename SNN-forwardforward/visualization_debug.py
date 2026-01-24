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
import sys
sys.path.append('D:\OneDrive\SNN_FFSTDP\SNN-ForwardForward')
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
from config import ConfigParser
from src.ff_snn_net import Net, spike_encoder
from src.generate_neg_sample import *

class SpikeHook:
    def __init__(self,num_neurons):
        self.records = []
        self.num_neurons = num_neurons

    def hook_fn(self, module, input, output):
        # output: [T,B,N] or [B,N] (step_mode='s')
        self.records.append(output.detach())

    def clear(self):
        self.records = []


def register_hook(net, type="SCFF"):
    hooks = []
    spike_hooks = []
    if type == "SCFF":
        layers = net.layers
    elif type == "embed_label_onehot":
        layers = net.layers  # 不hook最后一层
    for layer in layers:
        if layer == net.layers[-1]:
            continue
        h = SpikeHook(layer.out_features)
        handle = layer.layer[2].register_forward_hook(h.hook_fn)
        hooks.append(handle)
        spike_hooks.append(h)
    return hooks, spike_hooks
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
                    print(f"Layer {layer_idx} - {p_name} weight shape: {w.shape}")
                    print(f"Layer {layer_idx} - {p_name} weight stats: min={w.min()}, max={w.max()}, mean={w.mean():.4f}, std={w.std():.4f}")
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
def plot_firing_rate_neuron_all(firing_neuron_mean, num_layers):
    """
    绘制每个标签每层的 neuron-wise firing rate（柱状图）
    横轴：Neuron Index（按顺序排列）
    纵轴：Firing Rate（统一纵轴范围）
    firing_neuron_mean[label][layer] = np.ndarray [num_neurons]
    """

    num_labels = len(firing_neuron_mean)

    # 计算纵轴统一范围
    all_rates = np.concatenate([firing_neuron_mean[label][layer] 
                                for label in range(num_labels) 
                                for layer in range(num_layers)])
    y_max = all_rates.max() * 1.1  # 留一点空白

    fig, axes = plt.subplots(
        num_labels, num_layers,
        figsize=(3.5 * num_layers, 2.5 * num_labels),
        sharex=False, sharey=True,  # 纵轴统一
        gridspec_kw={'wspace':0.3, 'hspace':0.4}
    )

    if num_labels == 1:
        axes = axes[np.newaxis, :]
    if num_layers == 1:
        axes = axes[:, np.newaxis]

    for label in range(num_labels):
        for layer in range(num_layers):
            ax = axes[label, layer]
            fr = firing_neuron_mean[label][layer]
            # 横轴重新排列为排序顺序（从低到高发放率）
            neurons = np.argsort(fr)
            fr_sorted = fr[neurons]

            ax.bar(neurons, fr_sorted, color='skyblue', alpha=0.8, edgecolor='black')

            ax.set_ylim(0, y_max)
            ax.set_xlabel("Neuron Index (sorted)", fontsize=8)
            if layer == 0:
                ax.set_ylabel(f"Label {label}\nFiring Rate", fontsize=8)
            if label == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)

            # ax.tick_params(axis='both', labelsize=7)
            ax.tick_params(axis='y', labelleft=True)  # ⚠ 强制显示纵轴刻度
            ax.tick_params(axis='x', labelsize=7)

    fig.suptitle("Neuron-wise Firing Rate (All Labels & Layers, Sorted)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_firing_rate_hist_all(firing_neuron_mean, num_layers, bins=40):
    """
    firing_neuron_mean[label][layer] = np.ndarray [num_neurons]
    绘制每个 label × layer 的 neuron-wise firing rate 分布直方图
    横坐标：Firing rate
    纵坐标：Density
    """

    num_labels = len(firing_neuron_mean)

    fig, axes = plt.subplots(
        num_labels, num_layers,
        figsize=(3.2 * num_layers, 2.5 * num_labels),
        sharex=False, sharey=False,
        gridspec_kw={'wspace':0.3, 'hspace':0.4}
    )

    # 当只有一行或一列时，axes可能是一维，需要统一成二维索引
    if num_labels == 1:
        axes = axes[np.newaxis, :]
    if num_layers == 1:
        axes = axes[:, np.newaxis]

    for label in range(num_labels):
        for layer in range(num_layers):
            ax = axes[label, layer]
            fr = firing_neuron_mean.get(label, {}).get(layer, None)
            if fr is None:
                ax.axis("off")
                continue
            # fr = firing_neuron_mean[label][layer]
            ax.hist(fr, bins=bins, density=True, alpha=0.8, color='skyblue', edgecolor='black')

            ax.set_xlabel("Firing Rate", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)

            # 标题只在第一行显示
            if label == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
            # y 轴标签只在第一列显示
            if layer == 0:
                ax.set_ylabel(f"Label {label}", fontsize=8)

            # 设置字体大小
            ax.tick_params(axis='both', labelsize=7)

    fig.suptitle("Neuron-wise Firing Rate Distribution (All Labels & Layers)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出 suptitle 空间
    plt.show()


def plot_layerwise_label_compare(firing_neuron_mean, bins=40):
    num_labels = len(firing_neuron_mean)
    num_layers = len(next(iter(firing_neuron_mean.values())))

    fig, axes = plt.subplots(
        num_layers, 1,
        figsize=(6, 2.5 * num_layers),
        sharex=True
    )

    if num_layers == 1:
        axes = [axes]

    for layer in range(num_layers):
        ax = axes[layer]
        for label in range(num_labels):
            fr = firing_neuron_mean[label][layer]
            ax.hist(fr, bins=bins, density=True, alpha=0.4, label=f"L{label}")

        ax.set_title(f"Layer {layer}")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, ncol=5)

    axes[-1].set_xlabel("Firing Rate")
    fig.suptitle("Firing Rate Distribution per Layer (Label Comparison)", fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_hook(net, test_data_loader, device, type="SCFF"):
    """
    使用 forward hook 统计所有预测正确样本的各类别平均 goodness，并绘制热力图
    """
    # net.eval()

    hooks, spike_hooks = register_hook(net,type)

    num_labels = 10
    num_layers = len(spike_hooks)

    # ===============================
    # 统计准确率
    # ===============================
    per_class_correct = torch.zeros(num_labels)
    per_class_total   = torch.zeros(num_labels)

    # ===============================
    # Fire rate, goodness 累计
    # ===============================
    goodness_sum   = torch.zeros(num_labels, num_layers)
    firing_rates_sum = torch.zeros(num_labels, num_layers)
    sample_count = torch.zeros(num_labels)
    # ===============================
    # 每个 label × layer × neuron 的 firing 累计
    # ===============================
    firing_neuron_sum = {}   # dict[label][layer] = Tensor[num_neurons]

    test_acc = 0
    test_count = 0

    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            x_te = x_te.to(device)
            y_te = y_te.to(device)
            # forward（hook 会自动记录 spikes）
            if type == "SCFF":
                pred = net.predict_winner(x_te)
            elif type == "embed_label_onehot":
                pred = net.predict_multiple(x_te)
            test_acc += pred.eq(y_te).cpu().float().mean().item()
            test_count += 1
            # ----------------------------
            # 逐样本统计
            # ----------------------------
            for i in range(x_te.shape[0]):
                label = y_te[i].item()
                per_class_total[label] += 1
                # 仅统计预测正确/错误的样本
                if pred[i].item() == label:
                    per_class_correct[label] += 1

                    # 对该样本，逐层算 goodness
                    for layer_idx, h in enumerate(spike_hooks):
                        if len(h.records) == 0:
                            h.records = [torch.zeros((net.T, x_te.shape[0], h.num_neurons))]
                        # spikes: [T,B,N] 或 [B,N]
                        if len(h.records[0].shape) == 2:
                            spikes = torch.stack(h.records, dim=0)  # [T,B,N]
                        else:
                            spikes = torch.cat(h.records)

                        # 取该样本的脉冲
                        # [T,N]
                        s = spikes[:, i]
                        g = net.T * s.mean(0).pow(2).sum().item()
                        firing_rates_sum[label, layer_idx] += s.mean().item()
                        goodness_sum[label, layer_idx] += g
                        # neuron-wise firing rate for this sample
                        neuron_fr = s.mean(0).cpu()   # [N]
                        if label not in firing_neuron_sum:
                            firing_neuron_sum[label] = {}
                        if layer_idx not in firing_neuron_sum[label]:
                            firing_neuron_sum[label][layer_idx] = neuron_fr.clone()
                        else:
                            firing_neuron_sum[label][layer_idx] += neuron_fr

                    sample_count[label] += 1
            for h in spike_hooks:
                h.records.clear()

    # ===============================
    # 打印准确率
    # ===============================
    print("test Acc:", 100 * test_acc / test_count, "%")
    print("\n============================")
    print("   Per-Class Accuracy (%)")
    print("============================")
    for i in range(num_labels):
        if per_class_total[i] > 0:
            acc = 100 * per_class_correct[i] / per_class_total[i]
            print(f"Label {i}: {acc:.2f}%  ({int(per_class_correct[i])}/{int(per_class_total[i])})")

    # ===============================
    # 计算平均 goodness
    # ===============================
    goodness_mean = torch.zeros_like(goodness_sum)
    freq_mean = torch.zeros_like(firing_rates_sum)
    for l in range(num_labels):
        if sample_count[l] > 0:
            goodness_mean[l] = goodness_sum[l] / sample_count[l]
            freq_mean[l] = firing_rates_sum[l] / sample_count[l]

    goodness_mean = goodness_mean.numpy()
    freq_mean = freq_mean.numpy()
    # ===============================
    # neuron-wise firing rate mean
    # ===============================
    firing_neuron_mean = {}

    for label in firing_neuron_sum:
        firing_neuron_mean[label] = {}
        for layer_idx in firing_neuron_sum[label]:
            firing_neuron_mean[label][layer_idx] = (
                firing_neuron_sum[label][layer_idx] / sample_count[label]
            ).numpy()

    def plot_heatmap(data, title):
        # ===============================
        # 可视化
        # ===============================
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            data,
            annot=True, fmt=".2f", cmap="viridis",
            xticklabels=[f"Layer {i}" for i in range(num_layers)],
            yticklabels=[f"Label {i}" for i in range(num_labels)]
        )
        plt.xlabel("Layer")
        plt.ylabel("Label")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    plot_heatmap(goodness_mean, "Average Goodness per Label and Layer (Correct Predictions Only)")
    plot_heatmap(freq_mean, "Average Firing Rate per Label and Layer (Correct Predictions Only)")
    plot_firing_rate_hist_all(firing_neuron_mean, num_layers)
    plot_layerwise_label_compare(firing_neuron_mean)
    plot_firing_rate_neuron_all(firing_neuron_mean, num_layers)
    # 移除 hook
    for h in hooks:
        h.remove()
def main():
    config = ConfigParser()
    args = config.parse()
    # 初始化数据加载器
    # 加载训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
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
        batch_size=1,
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
    # ===============================
    # 权重分布
    # ===============================
    # # for name, module in net.named_modules():
    # #     if isinstance(module, torch.nn.Conv2d):
    # #         w = module.weight.data.clone()  # (Co, Ci, Kh, Kw)
    # goodness_mean = visualize_goodness_mean(net, test_data_loader, device)
    # visualize_layer_weights(net)
    # visualize_correct_sample_of_all_label_goodness(net,val_data_loader,device)
    # visualize_correct_sample_of_all_label_freq(net,val_data_loader,device)

    # ===============================
    # 单层多次推理可视化分析
    # ===============================
    # net = Net(dims=[784,256,10],tau=args.tau, epoch=args.epochs, T=8, lr=args.lr,
    #           v_threshold_pos=1.2,v_threshold_neg=-1.2, opt=args.opt, loss_threshold=0.5)
    # net.load("./SNN-forwardforward/logs/analyze/784-256-10.pth")
    # visualize_layer_weights(net)
    # visualize_hook(net, test_data_loader, device, type="embed_label_onehot")
    # # ===============================
    # # 多层多次推理可视化分析
    # # ===============================
    # net = Net(dims=[784,256, 128, 64,10],tau=args.tau, epoch=args.epochs, T=8, lr=args.lr,
    #           v_threshold_pos=1.3,v_threshold_neg=-1.2, opt=args.opt, loss_threshold=0.5)
    # net.load("./SNN-forwardforward/logs/analyze/784-256-128-64-10.pth")
    # visualize_layer_weights(net)
    # visualize_hook(net, test_data_loader, device, type="embed_label_onehot")
    # ===============================
    # 单次推理可视化分析
    # ===============================
    net = Net(dims=[784,512,512,10],tau=args.tau, epoch=args.epochs, T=16, lr=args.lr,
              v_threshold_pos=1.2,v_threshold_neg=-1.2, opt=args.opt, loss_threshold=0.25)
    # net.load("./SNN-forwardforward/logs/MNIST/T8_b1000_adam_lr0.015625/2026-01-13_15-42-17/checkpoint_last.pth")
    # 784-512-10 未tdLN："./SNN-forwardforward/logs/FashionMNIST/T8_b1000_adam_lr0.00390625/2026-01-22_22-25-19/checkpoint_last.pth"
    # net.load("./SNN-forwardforward/logs/FashionMNIST/T8_b1000_adam_lr0.00390625/2026-01-22_22-25-19/checkpoint_last.pth")
    # 784-512-10 tdLN: "./SNN-forwardforward/logs/FashionMNIST/T8_b1000_adam_lr0.00390625/2026-01-23_00-45-43/checkpoint_last.pth"
    # net.load("./SNN-forwardforward/logs/FashionMNIST/T8_b1000_adam_lr0.00390625/2026-01-23_00-45-43/checkpoint_last.pth")
    
    # 784-512-512-10 no tdLN + no IF："./SNN-forwardforward/logs/MNIST/T16_b1000_adam_lr0.00390625/2026-01-24_14-18-16/checkpoint_last.pth"
    # net.load("./SNN-forwardforward/logs/FashionMNIST/T16_b1000_adam_lr0.00390625/2026-01-24_14-18-16/checkpoint_last.pth")
    # 784-512-512-10 tdLN + no IF："./SNN-forwardforward/logs/MNIST/T16_b1000_adam_lr0.00390625/2026-01-23_00-51-57/checkpoint_last.pth"
    net.load("./SNN-forwardforward/logs/FashionMNIST/T16_b1000_adam_lr0.00390625/2026-01-23_00-51-57/checkpoint_last.pth")
    
    
    # 784-512-512-10 未tdLN："./SNN-forwardforward/logs/MNIST/T16_b1000_adam_lr0.00390625/2026-01-15_12-11-53/checkpoint_last.pth"
    # net.load("./SNN-forwardforward/logs/MNIST/T16_b1000_adam_lr0.00390625/2026-01-15_12-11-53/checkpoint_last.pth")
    # 784-512-10 tdLN："./SNN-forwardforward/logs/MNIST/T8_b1000_adam_lr0.00390625/2026-01-22_20-18-51/checkpoint_last.pth"
    # net.load("./SNN-forwardforward/logs/MNIST/T8_b1000_adam_lr0.00390625/2026-01-22_20-18-51/checkpoint_last.pth")
    visualize_layer_weights(net)
    visualize_hook(net, test_data_loader, device, type="SCFF")


if __name__ == "__main__":
    main()

