"""
====================================================================
File          : ff-snn.py
Description   : SNN-FF训练代码
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
import time
import argparse
import sys
import datetime
import json
import torch
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
import torchvision
import numpy as np
from tqdm import tqdm
from spikingjelly.activation_based import encoding, functional
import torch.nn.functional as F
from src.ff_snn_mlp import Net
from src.ff_snn_cnn import ConvNet
from config import ConfigParser
from src.dataset import GroupedSortedMNIST, AugmentedMNIST
import logging
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from src.generate_neg_sample import *


def visualize_sample(data, name="", idx=0):
    reshaped = data[idx].cpu().sum(dim=0)  # .reshape(1, 28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def normalize_frame(x):
    x = x.astype(np.float32)
    # N-MNIST frame 通常是 [T, C, H, W]，先沿时间维求和 -> [C, H, W]
    if x.ndim == 4:
        x = x.sum(axis=0)
    # 对整体 C/H/W 做统一归一化（全局 max）
    max_v = x.max()
    if max_v > 0:
        x = x / max_v
    return x
def plot_loss(loss_of_layer_list, save_path):
    # 获取层数和每层的损失数据
    num_layers = len(loss_of_layer_list)

    # 创建一个图形
    plt.figure(figsize=(10, 6))  # 设置图像大小

    # 绘制每一层的损失随 epoch 变化的曲线
    for layer_idx in range(num_layers):
        plt.plot(loss_of_layer_list[layer_idx], "o-", label=f"Layer {layer_idx + 1}")

    # 设置图形的标签和标题
    plt.xlabel("Epochs")  # x轴标签
    plt.ylabel("Loss")  # y轴标签
    plt.title("Loss vs Epoch for Each Layer")  # 图形标题

    # 显示图例和网格
    plt.legend()  # 显示图例，标识不同的层
    plt.grid(True)  # 显示网格

    # 保存图像到文件
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")

def plot_cos_sim(cos_sim_of_layer_list,is_pos, save_path):
    # 获取层数和每层的损失数据
    num_layers = len(cos_sim_of_layer_list)
    label = "Positive" if is_pos else "Negative"
    # 创建一个图形
    plt.figure(figsize=(10, 6))  # 设置图像大小

    # 绘制每一层的损失随 epoch 变化的曲线
    for layer_idx in range(num_layers):
        plt.plot(cos_sim_of_layer_list[layer_idx], "o-", label=f"Layer {layer_idx + 1}")

    # 设置图形的标签和标题
    plt.xlabel("Epochs")  # x轴标签
    plt.ylabel(f"{label} Cosine similarity")  # y轴标签
    plt.title(f"{label} Cosine similarity vs Epoch for Each Layer")  # 图形标题

    # 显示图例和网格
    plt.legend()  # 显示图例，标识不同的层
    plt.grid(True)  # 显示网格

    # 保存图像到文件
    plt.savefig(save_path)
    print(f"{label} Cosine similarity plot saved to {save_path}")
def plot_goodness(goodness_of_layer_list,is_pos, save_path):
    # 获取层数和每层的损失数据
    num_layers = len(goodness_of_layer_list)
    label = "Positive" if is_pos else "Negative"
    # 创建一个图形
    plt.figure(figsize=(10, 6))  # 设置图像大小

    # 绘制每一层的损失随 epoch 变化的曲线
    for layer_idx in range(num_layers):
        plt.plot(goodness_of_layer_list[layer_idx], "o-", label=f"Layer {layer_idx + 1}")

    # 设置图形的标签和标题
    plt.xlabel("Epochs")  # x轴标签
    plt.ylabel(f"{label} Goodness")  # y轴标签
    plt.title(f"{label} Goodness vs Epoch for Each Layer")  # 图形标题

    # 显示图例和网格
    plt.legend()  # 显示图例，标识不同的层
    plt.grid(True)  # 显示网格

    # 保存图像到文件
    plt.savefig(save_path)
    print(f"{label} Goodness plot saved to {save_path}")
def plot_firing_rate_pos_neg(pos_firing_rate,neg_firing_rate,save_path):

    num_layers = len(pos_firing_rate) 
    plt.figure(figsize=(10, 6))
    for layer_idx in range(num_layers):
        plt.plot(pos_firing_rate[layer_idx],"o-",label=f"Layer {layer_idx + 1} Positive")
        plt.plot(neg_firing_rate[layer_idx],"x--",label=f"Layer {layer_idx + 1} Negative")

    plt.xlabel("Epochs")
    plt.ylabel("Average Firing Rate")
    plt.title("Positive / Negative Firing Rate vs Epoch")

    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    print(f"Firing rate (pos/neg) plot saved to {save_path}")

def _mean_last_epoch(metric_per_layer_list):
    values = []
    for layer_metric in metric_per_layer_list:
        if not layer_metric:
            continue
        v = layer_metric[-1]
        if torch.is_tensor(v):
            if v.numel() == 1:
                v = v.detach().cpu().item()
            else:
                v = v.detach().cpu().mean().item()
        elif hasattr(v, "item"):
            v = v.item()
        values.append(float(v))
    if not values:
        return None
    return float(np.mean(values))

def _to_chw_tensor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    # Event frames are usually [T, C, H, W]; reduce over time for this pipeline.
    if x.ndim == 4:
        x = x.sum(dim=0)
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    # Some datasets return [H, W, C].
    if x.ndim == 3 and x.shape[0] not in (1, 2, 3):
        x = x.permute(2, 0, 1)
    max_v = x.max()
    if max_v > 0:
        x = x / max_v
    return x

def _bytes_to_mb(x):
    return float(x) / (1024.0 * 1024.0)


def main():
    config = ConfigParser()
    args = config.parse()
    ###########################################################################################
    ####################################前向学习的代码结构######################################
    # 初始化数据加载器
    # 加载训练集和测试集
    if args.dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    elif args.dataset in ("NMNIST", "N-MNIST"):
        try:
            train_dataset = NMNIST(
                root=(args.data_dir + '/NMNIST'),
                train=True,
                data_type='frame',
                frames_number=args.T,
                split_by='number',
                transform=normalize_frame
            )
            test_dataset = NMNIST(
                root=(args.data_dir + '/NMNIST'),
                train=False,
                data_type='frame',
                frames_number=args.T,
                split_by='number',
                transform=normalize_frame
            )
        except Exception as e:
            raise RuntimeError(
                "NMNIST does not support automatic download in this spikingjelly version. "
                "Please manually prepare N-MNIST data under data_dir, then rerun."
            ) from e
    elif args.dataset in ("DVS128Gesture", "DVS128-Gesture", "DVS 128 Gesture"):
        train_dataset = DVS128Gesture(
            root=(args.data_dir + '/DVSgesture'),
            train=True,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
            transform=normalize_frame
        )
        test_dataset = DVS128Gesture(
            root=(args.data_dir + '/DVSgesture'),
            train=False,
            data_type='frame',
            frames_number=args.T,
            split_by='number',
            transform=normalize_frame
        )
    elif args.dataset == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    elif args.dataset == "CIFAR10":

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    else:
        raise ValueError(
            "Unsupported dataset. Please choose from: MNIST, N-MNIST/NMNIST, FashionMNIST, CIFAR10, DVS128Gesture."
        )
   
    # 划分训练集和验证集
    train_size = int(0.95 * len(train_dataset))  # 80% 用于训练
    val_size = len(train_dataset) - train_size  # 20% 用于验证
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # 创建 DataLoader
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True,
    )

    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True,
    )

    x, y = next(iter(train_data_loader))
    
    num_classes = int(y.max().item() + 1)
    device = torch.device("cuda")
    use_cuda_mem_stat = torch.cuda.is_available() and device.type == "cuda"
    out_dir = os.path.join(
        args.out_dir,
        args.predict_type,
        args.dataset,
        args.model,   # 模型类型
        f"T{args.T}_b{args.b}_{args.opt}_lr{args.lr}",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
            
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Mkdir {out_dir}.")
    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    # 配置 logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # 记录超参数
    print(f"args: {args}")
    print(f"Saving at {out_dir}")
    if args.model == "MLP":
        net = Net(
            dims=args.dims,
            tau=args.tau,
            epoch=args.epochs,
            T=args.T,
            lr=args.lr,
            v_threshold=args.v_threshold,
            v_threshold_neg=args.v_threshold_neg,
            opt=args.opt,
            loss_threshold=args.loss_threshold,
            num_classes=num_classes
        )
    elif args.model == "CNN":
        _, __, H, W = x.shape
        net = ConvNet(
            conv_cfg=args.conv_cfg,
            T=args.T,
            epoch=args.epochs,                
            lr=args.lr,
            tau=args.tau,
            v_threshold=args.v_threshold,
            loss_threshold=args.loss_threshold,
            num_classes=num_classes,
            H=H,
            W=W,
        )
    # 初始化存储训练精度的列表
    epochs = args.epochs

    train_acc = 0
    train_acc_list = []

    max_tran_acc = 0
    loss_of_layer_list = [[] for _ in range(len(net.layers))]
    goodness_pos_of_layer_list = [[] for _ in range(len(net.layers))]
    goodness_neg_of_layer_list = [[] for _ in range(len(net.layers))]
    cos_pos_of_layer_list = [[] for _ in range(len(net.layers))]
    cos_neg_of_layer_list = [[] for _ in range(len(net.layers))]
    spike_out_pos_of_layer_list = [[] for _ in range(len(net.layers))]
    spike_out_neg_of_layer_list = [[] for _ in range(len(net.layers))]
    # GPU memory stats for training, with focus on train_ff_stdp (includes backward pass).
    train_mem_sample_count = 0
    train_alloc_sum = 0.0
    train_reserved_sum = 0.0
    bp_peak_alloc_sum = 0.0
    bp_peak_reserved_sum = 0.0
    bp_peak_alloc_max = 0.0
    bp_peak_reserved_max = 0.0
    bp_only_peak_alloc_sum = 0.0
    bp_only_peak_reserved_sum = 0.0
    bp_only_peak_alloc_max = 0.0
    bp_only_peak_reserved_max = 0.0
    bp_only_sample_count = 0
    manual_grad_peak_alloc_sum = 0.0
    manual_grad_peak_reserved_sum = 0.0
    manual_grad_peak_alloc_max = 0.0
    manual_grad_peak_reserved_max = 0.0
    manual_grad_time_sum_ms = 0.0
    manual_grad_sample_count = 0
    manual_grad_sample_total = 0
    manual_grad_ops_est_sum = 0.0
    autograd_cmp_peak_alloc_sum = 0.0
    autograd_cmp_peak_reserved_sum = 0.0
    autograd_cmp_peak_alloc_max = 0.0
    autograd_cmp_peak_reserved_max = 0.0
    autograd_cmp_time_sum_ms = 0.0
    autograd_cmp_sample_count = 0
    autograd_cmp_sample_total = 0
    # 定义输出文件路径
    log_file_path = os.path.join(out_dir, "output_log.txt")
    # 保存原始标准输出
    original_stdout = sys.stdout
    start_time = time.time()
    frozen = False
    with open(log_file_path, "w") as f:
        sys.stdout = f  # 替换标准输出
        for i in tqdm(range(epochs)):
            batch_samples = 0
            val_samples = 0
            loss = 0
            val_acc = 0
            goodness_pos_sum = 0
            goodness_neg_sum = 0
            cos_pos_sum = 0
            cos_neg_sum = 0
            spike_out_pos_sum = 0
            spike_out_neg_sum = 0
            if i > (0.8*epochs):
                frozen = True
            for x, y in train_data_loader:
                batch_samples += 1
                x, y = x.to(device), y.to(device)
                if use_cuda_mem_stat:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                goodness_pos, goodness_neg, cos_pos, cos_neg, spike_out_pos, spike_out_neg = net.train_ff_stdp(x, y, frozen)
                if use_cuda_mem_stat:
                    torch.cuda.synchronize(device)
                    cur_alloc = torch.cuda.memory_allocated(device)
                    cur_reserved = torch.cuda.memory_reserved(device)
                    bp_peak_alloc = torch.cuda.max_memory_allocated(device)
                    bp_peak_reserved = torch.cuda.max_memory_reserved(device)
                    train_alloc_sum += cur_alloc
                    train_reserved_sum += cur_reserved
                    bp_peak_alloc_sum += bp_peak_alloc
                    bp_peak_reserved_sum += bp_peak_reserved
                    bp_peak_alloc_max = max(bp_peak_alloc_max, float(bp_peak_alloc))
                    bp_peak_reserved_max = max(bp_peak_reserved_max, float(bp_peak_reserved))
                    train_mem_sample_count += 1
                bp_only_peak_alloc = getattr(net, "last_backward_peak_alloc_bytes", None)
                bp_only_peak_reserved = getattr(net, "last_backward_peak_reserved_bytes", None)
                if bp_only_peak_alloc is not None:
                    bp_only_peak_alloc_sum += float(bp_only_peak_alloc)
                    bp_only_peak_alloc_max = max(bp_only_peak_alloc_max, float(bp_only_peak_alloc))
                if bp_only_peak_reserved is not None:
                    bp_only_peak_reserved_sum += float(bp_only_peak_reserved)
                    bp_only_peak_reserved_max = max(bp_only_peak_reserved_max, float(bp_only_peak_reserved))
                if (bp_only_peak_alloc is not None) or (bp_only_peak_reserved is not None):
                    bp_only_sample_count += 1
                manual_grad_peak_alloc = getattr(net, "last_manual_grad_peak_alloc_bytes", None)
                manual_grad_peak_reserved = getattr(net, "last_manual_grad_peak_reserved_bytes", None)
                manual_grad_time_ms = getattr(net, "last_manual_grad_time_ms", None)
                if manual_grad_peak_alloc is not None:
                    manual_grad_peak_alloc_sum += float(manual_grad_peak_alloc)
                    manual_grad_peak_alloc_max = max(manual_grad_peak_alloc_max, float(manual_grad_peak_alloc))
                if manual_grad_peak_reserved is not None:
                    manual_grad_peak_reserved_sum += float(manual_grad_peak_reserved)
                    manual_grad_peak_reserved_max = max(manual_grad_peak_reserved_max, float(manual_grad_peak_reserved))
                if manual_grad_time_ms is not None:
                    manual_grad_time_sum_ms += float(manual_grad_time_ms)
                if (manual_grad_peak_alloc is not None) or (manual_grad_peak_reserved is not None) or (manual_grad_time_ms is not None):
                    manual_grad_sample_count += 1
                    manual_grad_sample_total += int(y.numel())
                manual_grad_ops_est = getattr(net, "last_manual_grad_ops_est", None)
                if manual_grad_ops_est is not None:
                    manual_grad_ops_est_sum += float(manual_grad_ops_est)
                autograd_cmp_peak_alloc = getattr(net, "last_backward_cmp_peak_alloc_bytes", None)
                autograd_cmp_peak_reserved = getattr(net, "last_backward_cmp_peak_reserved_bytes", None)
                autograd_cmp_time_ms = getattr(net, "last_backward_cmp_time_ms", None)
                if autograd_cmp_peak_alloc is not None:
                    autograd_cmp_peak_alloc_sum += float(autograd_cmp_peak_alloc)
                    autograd_cmp_peak_alloc_max = max(autograd_cmp_peak_alloc_max, float(autograd_cmp_peak_alloc))
                if autograd_cmp_peak_reserved is not None:
                    autograd_cmp_peak_reserved_sum += float(autograd_cmp_peak_reserved)
                    autograd_cmp_peak_reserved_max = max(autograd_cmp_peak_reserved_max, float(autograd_cmp_peak_reserved))
                if autograd_cmp_time_ms is not None:
                    autograd_cmp_time_sum_ms += float(autograd_cmp_time_ms)
                if (autograd_cmp_peak_alloc is not None) or (autograd_cmp_peak_reserved is not None) or (autograd_cmp_time_ms is not None):
                    autograd_cmp_sample_count += 1
                    autograd_cmp_sample_total += int(y.numel())
                # 单个batch获取所有层的平均余弦相似度以及优度值
                goodness_pos = torch.tensor(goodness_pos)
                goodness_neg = torch.tensor(goodness_neg)
                cos_pos = torch.tensor(cos_pos)
                cos_neg = torch.tensor(cos_neg)
                spike_out_pos = torch.tensor(spike_out_pos)
                spike_out_neg = torch.tensor(spike_out_neg)

                goodness_pos_sum += goodness_pos
                goodness_neg_sum += goodness_neg
                cos_pos_sum += cos_pos
                cos_neg_sum += cos_neg
                spike_out_pos_sum += spike_out_pos
                spike_out_neg_sum += spike_out_neg

            # 累加的goodness先求所有层的平均值，在求batch的平均值计算loss
            loss = (torch.log(1+ torch.exp(-goodness_pos_sum/batch_samples + args.loss_threshold)) + torch.log(1+ torch.exp(goodness_neg_sum/batch_samples - args.loss_threshold))) / 2
            # loss = torch.log(1+torch.exp(-args.loss_threshold * (goodness_pos_sum/batch_samples - goodness_neg_sum/batch_samples)))
            print(f"Epoch: {i+1}/{epochs}, Loss: {loss.mean():.4f}")
            for l in range(len(net.layers)-1):
                goodness_pos_of_layer_list[l].append(goodness_pos_sum[l]/batch_samples)
                goodness_neg_of_layer_list[l].append(goodness_neg_sum[l]/batch_samples)
                cos_pos_of_layer_list[l].append(cos_pos_sum[l]/batch_samples)
                cos_neg_of_layer_list[l].append(cos_neg_sum[l]/batch_samples)
                spike_out_pos_of_layer_list[l].append(spike_out_pos_sum[l]/batch_samples)
                spike_out_neg_of_layer_list[l].append(spike_out_neg_sum[l]/batch_samples)
                loss_of_layer_list[l].append(loss[l])

            with torch.no_grad():
                for x_val, y_val in val_data_loader:
                    val_samples += 1
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_acc += net.predict_multiple(x_val).eq(y_val).cpu().float().mean().item()
                train_acc = 100 * (val_acc / val_samples)
                train_acc_list.append(train_acc)
                print(f"Train Acc:  {train_acc:.2f}%")
                if train_acc >= max_tran_acc:
                    net.save(args, os.path.join(out_dir, "checkpoint_max.pth"))
                    max_tran_acc = train_acc
            logger.info(f"Epoch {i+1}: Train Loss = {loss.mean():.4f} Train Acc = {train_acc:.2f}%")
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"Training completed. Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        # 绘制训练精度曲线
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, len(train_acc_list) + 1),
            train_acc_list,
            marker="o",
            label="Train Accuracy",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Training Accuracy Curve when training last layer")
        plt.legend()
        plt.grid(True)
        # 保存曲线到本地
        plt.savefig(os.path.join(out_dir, "training_accuracy_curve.png"), dpi=300)
        plot_loss(loss_of_layer_list, os.path.join(out_dir, "loss_of_each_layer.png"))
        plot_cos_sim(cos_pos_of_layer_list, True, os.path.join(out_dir, "cosine_similarity_positive.png"))
        plot_cos_sim(cos_neg_of_layer_list, False, os.path.join(out_dir, "cosine_similarity_negative.png"))
        plot_goodness(goodness_pos_of_layer_list, True, os.path.join(out_dir, "goodness_positive.png"))
        plot_goodness(goodness_neg_of_layer_list, False, os.path.join(out_dir, "goodness_negative.png"))
        plot_firing_rate_pos_neg(spike_out_pos_of_layer_list, spike_out_neg_of_layer_list, os.path.join(out_dir, "spike_out_positive.png"))
        logger.info(f"Training completed in Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        test_acc = 0
        test_samples = 0
        test_count = 0
        start_time = time.time()
        with torch.no_grad():
            for x_te, y_te in test_data_loader:
                test_samples += y_te.numel()
                test_count += 1
                x_te, y_te = x_te.to(device), y_te.to(device)
                test_acc += net.predict_multiple(x_te).eq(y_te).cpu().float().mean().item()
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("test Acc:", 100 * test_acc / test_count, "%")
        print(
            f"Testing completed. Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        save = True
        if save or args.save_model:
            net.save(args, os.path.join(out_dir, "checkpoint_last.pth"))
        if train_mem_sample_count > 0:
            train_gpu_mem_alloc_mean_mb = _bytes_to_mb(train_alloc_sum / train_mem_sample_count)
            train_gpu_mem_reserved_mean_mb = _bytes_to_mb(train_reserved_sum / train_mem_sample_count)
            bp_gpu_mem_peak_alloc_mean_mb = _bytes_to_mb(bp_peak_alloc_sum / train_mem_sample_count)
            bp_gpu_mem_peak_reserved_mean_mb = _bytes_to_mb(bp_peak_reserved_sum / train_mem_sample_count)
            bp_gpu_mem_peak_alloc_max_mb = _bytes_to_mb(bp_peak_alloc_max)
            bp_gpu_mem_peak_reserved_max_mb = _bytes_to_mb(bp_peak_reserved_max)
        else:
            train_gpu_mem_alloc_mean_mb = None
            train_gpu_mem_reserved_mean_mb = None
            bp_gpu_mem_peak_alloc_mean_mb = None
            bp_gpu_mem_peak_reserved_mean_mb = None
            bp_gpu_mem_peak_alloc_max_mb = None
            bp_gpu_mem_peak_reserved_max_mb = None
        if bp_only_sample_count > 0:
            bp_only_gpu_mem_peak_alloc_mean_mb = _bytes_to_mb(bp_only_peak_alloc_sum / bp_only_sample_count)
            bp_only_gpu_mem_peak_reserved_mean_mb = _bytes_to_mb(bp_only_peak_reserved_sum / bp_only_sample_count)
            bp_only_gpu_mem_peak_alloc_max_mb = _bytes_to_mb(bp_only_peak_alloc_max)
            bp_only_gpu_mem_peak_reserved_max_mb = _bytes_to_mb(bp_only_peak_reserved_max)
        else:
            bp_only_gpu_mem_peak_alloc_mean_mb = None
            bp_only_gpu_mem_peak_reserved_mean_mb = None
            bp_only_gpu_mem_peak_alloc_max_mb = None
            bp_only_gpu_mem_peak_reserved_max_mb = None
        if manual_grad_sample_count > 0:
            manual_grad_peak_alloc_mean_mb = _bytes_to_mb(manual_grad_peak_alloc_sum / manual_grad_sample_count)
            manual_grad_peak_reserved_mean_mb = _bytes_to_mb(manual_grad_peak_reserved_sum / manual_grad_sample_count)
            manual_grad_peak_alloc_max_mb = _bytes_to_mb(manual_grad_peak_alloc_max)
            manual_grad_peak_reserved_max_mb = _bytes_to_mb(manual_grad_peak_reserved_max)
            manual_grad_time_mean_ms = manual_grad_time_sum_ms / manual_grad_sample_count
        else:
            manual_grad_peak_alloc_mean_mb = None
            manual_grad_peak_reserved_mean_mb = None
            manual_grad_peak_alloc_max_mb = None
            manual_grad_peak_reserved_max_mb = None
            manual_grad_time_mean_ms = None
        if manual_grad_sample_total > 0:
            manual_grad_peak_alloc_per_sample_kb = (manual_grad_peak_alloc_sum / manual_grad_sample_total) / 1024.0
            manual_grad_peak_reserved_per_sample_kb = (manual_grad_peak_reserved_sum / manual_grad_sample_total) / 1024.0
            manual_grad_time_per_sample_us = (manual_grad_time_sum_ms * 1000.0) / manual_grad_sample_total
            manual_grad_ops_est_per_sample = manual_grad_ops_est_sum / manual_grad_sample_total
        else:
            manual_grad_peak_alloc_per_sample_kb = None
            manual_grad_peak_reserved_per_sample_kb = None
            manual_grad_time_per_sample_us = None
            manual_grad_ops_est_per_sample = None
        if autograd_cmp_sample_count > 0:
            autograd_cmp_peak_alloc_mean_mb = _bytes_to_mb(autograd_cmp_peak_alloc_sum / autograd_cmp_sample_count)
            autograd_cmp_peak_reserved_mean_mb = _bytes_to_mb(autograd_cmp_peak_reserved_sum / autograd_cmp_sample_count)
            autograd_cmp_peak_alloc_max_mb = _bytes_to_mb(autograd_cmp_peak_alloc_max)
            autograd_cmp_peak_reserved_max_mb = _bytes_to_mb(autograd_cmp_peak_reserved_max)
            autograd_cmp_time_mean_ms = autograd_cmp_time_sum_ms / autograd_cmp_sample_count
        else:
            autograd_cmp_peak_alloc_mean_mb = None
            autograd_cmp_peak_reserved_mean_mb = None
            autograd_cmp_peak_alloc_max_mb = None
            autograd_cmp_peak_reserved_max_mb = None
            autograd_cmp_time_mean_ms = None
        if autograd_cmp_sample_total > 0:
            autograd_cmp_peak_alloc_per_sample_kb = (autograd_cmp_peak_alloc_sum / autograd_cmp_sample_total) / 1024.0
            autograd_cmp_peak_reserved_per_sample_kb = (autograd_cmp_peak_reserved_sum / autograd_cmp_sample_total) / 1024.0
            autograd_cmp_time_per_sample_us = (autograd_cmp_time_sum_ms * 1000.0) / autograd_cmp_sample_total
        else:
            autograd_cmp_peak_alloc_per_sample_kb = None
            autograd_cmp_peak_reserved_per_sample_kb = None
            autograd_cmp_time_per_sample_us = None
        if manual_grad_time_sum_ms > 0:
            manual_grad_samples_per_s = manual_grad_sample_total / (manual_grad_time_sum_ms / 1000.0)
        else:
            manual_grad_samples_per_s = None
        if autograd_cmp_time_sum_ms > 0:
            autograd_cmp_samples_per_s = autograd_cmp_sample_total / (autograd_cmp_time_sum_ms / 1000.0)
        else:
            autograd_cmp_samples_per_s = None
        if manual_grad_ops_est_sum > 0 and manual_grad_time_sum_ms > 0:
            manual_grad_ops_est_gops = manual_grad_ops_est_sum / 1e9
            manual_grad_ops_est_gops_per_s = manual_grad_ops_est_gops / (manual_grad_time_sum_ms / 1000.0)
        else:
            manual_grad_ops_est_gops = None
            manual_grad_ops_est_gops_per_s = None
        if (
            manual_grad_peak_alloc_mean_mb is not None
            and autograd_cmp_peak_alloc_mean_mb is not None
            and autograd_cmp_peak_alloc_mean_mb > 0
        ):
            manual_vs_autograd_alloc_reduction_pct = 100.0 * (1.0 - manual_grad_peak_alloc_mean_mb / autograd_cmp_peak_alloc_mean_mb)
        else:
            manual_vs_autograd_alloc_reduction_pct = None
        if (
            manual_grad_time_mean_ms is not None
            and autograd_cmp_time_mean_ms is not None
            and autograd_cmp_time_mean_ms > 0
        ):
            manual_vs_autograd_time_reduction_pct = 100.0 * (1.0 - manual_grad_time_mean_ms / autograd_cmp_time_mean_ms)
        else:
            manual_vs_autograd_time_reduction_pct = None
        if (
            manual_grad_samples_per_s is not None
            and autograd_cmp_samples_per_s is not None
            and autograd_cmp_samples_per_s > 0
        ):
            manual_vs_autograd_throughput_gain_pct = 100.0 * (manual_grad_samples_per_s / autograd_cmp_samples_per_s - 1.0)
        else:
            manual_vs_autograd_throughput_gain_pct = None
        if (
            manual_grad_peak_alloc_per_sample_kb is not None
            and autograd_cmp_peak_alloc_per_sample_kb is not None
            and autograd_cmp_peak_alloc_per_sample_kb > 0
        ):
            manual_vs_autograd_alloc_per_sample_reduction_pct = 100.0 * (
                1.0 - manual_grad_peak_alloc_per_sample_kb / autograd_cmp_peak_alloc_per_sample_kb
            )
        else:
            manual_vs_autograd_alloc_per_sample_reduction_pct = None
        if (
            manual_grad_time_per_sample_us is not None
            and autograd_cmp_time_per_sample_us is not None
            and autograd_cmp_time_per_sample_us > 0
        ):
            manual_vs_autograd_time_per_sample_reduction_pct = 100.0 * (
                1.0 - manual_grad_time_per_sample_us / autograd_cmp_time_per_sample_us
            )
        else:
            manual_vs_autograd_time_per_sample_reduction_pct = None
        metrics = {
            "test_acc": 100 * test_acc / test_count,
            "last_epoch_loss_mean": _mean_last_epoch(loss_of_layer_list),
            "last_epoch_goodness_pos_mean": _mean_last_epoch(goodness_pos_of_layer_list),
            "last_epoch_goodness_neg_mean": _mean_last_epoch(goodness_neg_of_layer_list),
            "last_epoch_firing_pos_mean": _mean_last_epoch(spike_out_pos_of_layer_list),
            "last_epoch_firing_neg_mean": _mean_last_epoch(spike_out_neg_of_layer_list),
            "train_gpu_mem_alloc_mean_mb": train_gpu_mem_alloc_mean_mb,
            "train_gpu_mem_reserved_mean_mb": train_gpu_mem_reserved_mean_mb,
            "bp_gpu_mem_peak_alloc_mean_mb": bp_gpu_mem_peak_alloc_mean_mb,
            "bp_gpu_mem_peak_reserved_mean_mb": bp_gpu_mem_peak_reserved_mean_mb,
            "bp_gpu_mem_peak_alloc_max_mb": bp_gpu_mem_peak_alloc_max_mb,
            "bp_gpu_mem_peak_reserved_max_mb": bp_gpu_mem_peak_reserved_max_mb,
            "bp_only_gpu_mem_peak_alloc_mean_mb": bp_only_gpu_mem_peak_alloc_mean_mb,
            "bp_only_gpu_mem_peak_reserved_mean_mb": bp_only_gpu_mem_peak_reserved_mean_mb,
            "bp_only_gpu_mem_peak_alloc_max_mb": bp_only_gpu_mem_peak_alloc_max_mb,
            "bp_only_gpu_mem_peak_reserved_max_mb": bp_only_gpu_mem_peak_reserved_max_mb,
            "manual_grad_peak_alloc_mean_mb": manual_grad_peak_alloc_mean_mb,
            "manual_grad_peak_reserved_mean_mb": manual_grad_peak_reserved_mean_mb,
            "manual_grad_peak_alloc_max_mb": manual_grad_peak_alloc_max_mb,
            "manual_grad_peak_reserved_max_mb": manual_grad_peak_reserved_max_mb,
            "manual_grad_time_mean_ms": manual_grad_time_mean_ms,
            "manual_grad_peak_alloc_per_sample_kb": manual_grad_peak_alloc_per_sample_kb,
            "manual_grad_peak_reserved_per_sample_kb": manual_grad_peak_reserved_per_sample_kb,
            "manual_grad_time_per_sample_us": manual_grad_time_per_sample_us,
            "manual_grad_ops_est_total": manual_grad_ops_est_sum if manual_grad_ops_est_sum > 0 else None,
            "manual_grad_ops_est_gops": manual_grad_ops_est_gops,
            "manual_grad_ops_est_per_sample": manual_grad_ops_est_per_sample,
            "manual_grad_ops_est_gops_per_s": manual_grad_ops_est_gops_per_s,
            "manual_grad_samples_per_s": manual_grad_samples_per_s,
            "autograd_cmp_peak_alloc_mean_mb": autograd_cmp_peak_alloc_mean_mb,
            "autograd_cmp_peak_reserved_mean_mb": autograd_cmp_peak_reserved_mean_mb,
            "autograd_cmp_peak_alloc_max_mb": autograd_cmp_peak_alloc_max_mb,
            "autograd_cmp_peak_reserved_max_mb": autograd_cmp_peak_reserved_max_mb,
            "autograd_cmp_time_mean_ms": autograd_cmp_time_mean_ms,
            "autograd_cmp_peak_alloc_per_sample_kb": autograd_cmp_peak_alloc_per_sample_kb,
            "autograd_cmp_peak_reserved_per_sample_kb": autograd_cmp_peak_reserved_per_sample_kb,
            "autograd_cmp_time_per_sample_us": autograd_cmp_time_per_sample_us,
            "autograd_cmp_samples_per_s": autograd_cmp_samples_per_s,
            "manual_vs_autograd_alloc_reduction_pct": manual_vs_autograd_alloc_reduction_pct,
            "manual_vs_autograd_time_reduction_pct": manual_vs_autograd_time_reduction_pct,
            "manual_vs_autograd_alloc_per_sample_reduction_pct": manual_vs_autograd_alloc_per_sample_reduction_pct,
            "manual_vs_autograd_time_per_sample_reduction_pct": manual_vs_autograd_time_per_sample_reduction_pct,
            "manual_vs_autograd_throughput_gain_pct": manual_vs_autograd_throughput_gain_pct,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as metrics_f:
            json.dump(metrics, metrics_f, ensure_ascii=False, indent=2)
    logger.info(f"Test Acc: {100 * test_acc / test_count}%")
    logging.basicConfig(
        filename="./logs/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='a' # 文件模式：'a'为追加模式，'w'为覆盖模式
    )
    # 恢复标准输出
    sys.stdout = original_stdout
    print("Back to console.")


if __name__ == "__main__":
    main()
