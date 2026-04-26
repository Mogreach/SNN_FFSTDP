"""
====================================================================
File          : bp-snn.py
Description   : 基于 BPTT 的 SNN 训练脚本（自动记录指标并绘图）
Author        : Codex
Version       : 1.0.0
Date          : 2026-04-01
License       : MIT
====================================================================
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from spikingjelly.activation_based import encoding, functional, layer, neuron, surrogate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from torch.cuda import amp

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(description="BPTT SNN training")
    parser.add_argument(
        "-dataset",
        default="CIFAR10",
        type=str,
        choices=["MNIST", "N-MNIST", "NMNIST", "FashionMNIST", "CIFAR10", "DVS128Gesture"],
        help="Train dataset",
    )
    parser.add_argument("-T", default=64, type=int, help="simulating time-steps")
    parser.add_argument(
        "-frames-number",
        default=None,
        type=int,
        help="frames number for event-frame datasets, default uses T",
    )
    parser.add_argument(
        "-dvs-crop-size",
        default=32,
        type=int,
        help="center crop size for DVS128Gesture frames; <=0 disables cropping",
    )
    parser.add_argument("-device", default="cuda:0", help="device")
    parser.add_argument("-b", default=1024, type=int, help="batch size")
    parser.add_argument("-epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", default=8, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("-data-dir", default="./SNN-forwardforward/data", type=str, help="dataset root dir")
    parser.add_argument(
        "-out-dir",
        type=str,
        default="./SNN-forwardforward/logs/bp",
        help="root dir for saving logs and checkpoint",
    )
    parser.add_argument("-resume", type=str, default=None, help="resume from checkpoint path")
    parser.add_argument("-amp", action="store_true", help="automatic mixed precision training")
    parser.add_argument("-opt", type=str, choices=["sgd", "adam"], default="adam", help="optimizer")
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument("-lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-tau", default=2.0, type=float, help="tau of LIF neuron")
    parser.add_argument("-v_threshold", default=1.0, type=float, help="v_threshold of LIF neuron")
    parser.add_argument(
        "-hidden-dims",
        default=[1024,1024,1024],
        type=int,
        nargs="+",
        help="hidden layer dimensions for MLP SNN",
    )
    parser.add_argument("-train-split", default=0.95, type=float, help="train split ratio")
    parser.add_argument("-seed", default=123, type=int, help="random seed")
    parser.add_argument("-save-model", action="store_true", help="always save last checkpoint")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_frame(x, crop_size=None):
    if np is not None:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 4:
            x = x.sum(axis=0)
        if crop_size is not None and crop_size > 0 and x.ndim >= 2:
            h, w = x.shape[-2], x.shape[-1]
            if h >= crop_size and w >= crop_size:
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
                x = x[..., top:top + crop_size, left:left + crop_size]
        max_v = x.max()
        if max_v > 0:
            x = x / max_v
        return x.astype(np.float32)

    x = torch.as_tensor(x).float()
    if x.ndim == 4:
        x = x.sum(dim=0)
    if crop_size is not None and crop_size > 0 and x.ndim >= 2:
        h, w = x.shape[-2], x.shape[-1]
        if h >= crop_size and w >= crop_size:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            x = x[..., top:top + crop_size, left:left + crop_size]
    max_v = x.max()
    if max_v > 0:
        x = x / max_v
    return x


def _ensure_chw_tensor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        x = x.sum(dim=0)
    if x.ndim == 3 and x.shape[0] not in (1, 2, 3):
        x = x.permute(2, 0, 1)
    return x


def _ensure_nchw_tensor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 5:
        x = x.sum(dim=1)
    if x.ndim == 4 and x.shape[1] not in (1, 2, 3):
        x = x.permute(0, 3, 1, 2)
    return x


def build_datasets(args):
    dataset_name = "NMNIST" if args.dataset == "N-MNIST" else args.dataset
    frames_number = args.frames_number if args.frames_number is not None else args.T

    if dataset_name == "MNIST":
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
    elif dataset_name == "NMNIST":
        train_dataset = NMNIST(
            root=os.path.join(args.data_dir, "NMNIST"),
            train=True,
            data_type="frame",
            frames_number=frames_number,
            split_by="number",
            transform=normalize_frame,
        )
        test_dataset = NMNIST(
            root=os.path.join(args.data_dir, "NMNIST"),
            train=False,
            data_type="frame",
            frames_number=frames_number,
            split_by="number",
            transform=normalize_frame,
        )
    elif dataset_name == "DVS128Gesture":
        dvs_transform = partial(normalize_frame, crop_size=args.dvs_crop_size)
        train_dataset = DVS128Gesture(
            root=os.path.join(args.data_dir, "DVSgesture"),
            train=True,
            data_type="frame",
            frames_number=frames_number,
            split_by="number",
            transform=dvs_transform,
        )
        test_dataset = DVS128Gesture(
            root=os.path.join(args.data_dir, "DVSgesture"),
            train=False,
            data_type="frame",
            frames_number=frames_number,
            split_by="number",
            transform=dvs_transform,
        )
    elif dataset_name == "FashionMNIST":
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
    elif dataset_name == "CIFAR10":
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
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset_name, train_dataset, test_dataset


def infer_num_classes(dataset):
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return int(len(classes))
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        targets = torch.as_tensor(targets)
        if targets.numel() > 0:
            return int(targets.max().item() + 1)
    labels = set()
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.add(int(y))
    return int(len(labels))


def split_train_val(train_dataset, train_split, seed):
    if len(train_dataset) < 2:
        raise RuntimeError("train dataset length must be >= 2 to split train/val.")
    train_size = int(train_split * len(train_dataset))
    train_size = max(1, min(train_size, len(train_dataset) - 1))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = data.random_split(train_dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


class BPTTSNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, tau, v_threshold):
        super().__init__()
        layers_list = [layer.Flatten()]

        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers_list.append(layer.Linear(in_dim, hidden_dim, bias=False))
            layers_list.append(
                neuron.LIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    surrogate_function=surrogate.ATan(),
                    step_mode="s",
                )
            )
            in_dim = hidden_dim

        layers_list.append(layer.Linear(in_dim, num_classes, bias=False))
        layers_list.append(
            neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            )
        )
        self.layer = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layer(x)


def create_logger(log_file):
    logger = logging.getLogger("bp_snn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def forward_rate(net, encoder, img, T):
    out_fr = None
    for _ in range(T):
        encoded_img = encoder(img)
        step_out = net(encoded_img)
        if out_fr is None:
            out_fr = step_out
        else:
            out_fr = out_fr + step_out
    return out_fr / T


def _bytes_to_mb(x):
    return float(x) / (1024.0 * 1024.0)


def run_train_epoch(net, loader, optimizer, scaler, encoder, args, device, num_classes, use_cuda_mem_stat):
    net.train()
    start_time = time.time()
    train_loss = 0.0
    train_correct = 0.0
    train_samples = 0

    mem_alloc_sum = 0.0
    mem_reserved_sum = 0.0
    mem_peak_alloc_sum = 0.0
    mem_peak_reserved_sum = 0.0
    mem_peak_alloc_max = 0.0
    mem_peak_reserved_max = 0.0
    mem_count = 0

    use_amp = scaler.is_enabled()
    for img, label in loader:
        optimizer.zero_grad(set_to_none=True)
        img = _ensure_nchw_tensor(img).to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        label_onehot = F.one_hot(label, num_classes).float()

        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        autocast_ctx = amp.autocast() if use_amp else nullcontext()
        with autocast_ctx:
            out_fr = forward_rate(net, encoder, img, args.T)
            loss = F.mse_loss(out_fr, label_onehot)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            cur_alloc = torch.cuda.memory_allocated(device)
            cur_reserved = torch.cuda.memory_reserved(device)
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)

            mem_alloc_sum += float(cur_alloc)
            mem_reserved_sum += float(cur_reserved)
            mem_peak_alloc_sum += float(peak_alloc)
            mem_peak_reserved_sum += float(peak_reserved)
            mem_peak_alloc_max = max(mem_peak_alloc_max, float(peak_alloc))
            mem_peak_reserved_max = max(mem_peak_reserved_max, float(peak_reserved))
            mem_count += 1

        train_samples += int(label.numel())
        train_loss += float(loss.item()) * int(label.numel())
        train_correct += float((out_fr.argmax(1) == label).float().sum().item())

        functional.reset_net(net)

    elapsed = time.time() - start_time
    train_loss = train_loss / max(train_samples, 1)
    train_acc = train_correct / max(train_samples, 1)
    train_speed = train_samples / max(elapsed, 1e-8)

    if mem_count > 0:
        mem_stat = {
            "train_gpu_mem_alloc_mean_mb": _bytes_to_mb(mem_alloc_sum / mem_count),
            "train_gpu_mem_reserved_mean_mb": _bytes_to_mb(mem_reserved_sum / mem_count),
            "train_gpu_mem_peak_alloc_mean_mb": _bytes_to_mb(mem_peak_alloc_sum / mem_count),
            "train_gpu_mem_peak_reserved_mean_mb": _bytes_to_mb(mem_peak_reserved_sum / mem_count),
            "train_gpu_mem_peak_alloc_max_mb": _bytes_to_mb(mem_peak_alloc_max),
            "train_gpu_mem_peak_reserved_max_mb": _bytes_to_mb(mem_peak_reserved_max),
        }
    else:
        mem_stat = {
            "train_gpu_mem_alloc_mean_mb": None,
            "train_gpu_mem_reserved_mean_mb": None,
            "train_gpu_mem_peak_alloc_mean_mb": None,
            "train_gpu_mem_peak_reserved_mean_mb": None,
            "train_gpu_mem_peak_alloc_max_mb": None,
            "train_gpu_mem_peak_reserved_max_mb": None,
        }

    return train_loss, train_acc, train_speed, elapsed, mem_stat


@torch.no_grad()
def run_eval_epoch(net, loader, encoder, args, device, num_classes):
    net.eval()
    start_time = time.time()
    eval_loss = 0.0
    eval_correct = 0.0
    eval_samples = 0

    for img, label in loader:
        img = _ensure_nchw_tensor(img).to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        label_onehot = F.one_hot(label, num_classes).float()

        out_fr = forward_rate(net, encoder, img, args.T)
        loss = F.mse_loss(out_fr, label_onehot)

        eval_samples += int(label.numel())
        eval_loss += float(loss.item()) * int(label.numel())
        eval_correct += float((out_fr.argmax(1) == label).float().sum().item())

        functional.reset_net(net)

    elapsed = time.time() - start_time
    eval_loss = eval_loss / max(eval_samples, 1)
    eval_acc = eval_correct / max(eval_samples, 1)
    eval_speed = eval_samples / max(elapsed, 1e-8)
    return eval_loss, eval_acc, eval_speed, elapsed


def save_epoch_csv(epoch_rows, save_path):
    if not epoch_rows:
        return
    fieldnames = list(epoch_rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)


def plot_curves(epoch_rows, out_dir):
    if not epoch_rows:
        return
    if plt is None:
        return

    epochs = [row["epoch"] for row in epoch_rows]
    train_loss = [row["train_loss"] for row in epoch_rows]
    val_loss = [row["val_loss"] for row in epoch_rows]
    train_acc = [row["train_acc"] * 100.0 for row in epoch_rows]
    val_acc = [row["val_acc"] * 100.0 for row in epoch_rows]
    train_speed = [row["train_speed_img_s"] for row in epoch_rows]
    val_speed = [row["val_speed_img_s"] for row in epoch_rows]

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(epochs, train_loss, "o-", label="Train Loss")
    ax1.plot(epochs, val_loss, "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("BP SNN Loss Curve")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epochs, train_acc, "o-", label="Train Acc (%)")
    ax2.plot(epochs, val_acc, "s-", label="Val Acc (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("BP SNN Accuracy Curve")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_acc_curve.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_speed, "o-", label="Train Speed (img/s)")
    ax.plot(epochs, val_speed, "s-", label="Val Speed (img/s)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Images / s")
    ax.set_title("BP SNN Speed Curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_curve.png"), dpi=300)
    plt.close(fig)


def build_optimizer(net, args):
    if args.opt == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    if args.opt == "adam":
        return torch.optim.Adam(net.parameters(), lr=args.lr)
    raise NotImplementedError(args.opt)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.T <= 0:
        raise ValueError("T must be > 0")
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("train-split must be in (0, 1)")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = torch.device("cpu")
    else:
        actual_device = torch.device(args.device)

    dataset_name, train_dataset_full, test_dataset = build_datasets(args)
    train_subset, val_subset = split_train_val(train_dataset_full, args.train_split, args.seed)

    num_classes = max(infer_num_classes(train_dataset_full), infer_num_classes(test_dataset))

    sample_x, _ = train_dataset_full[0]
    sample_x = _ensure_chw_tensor(sample_x)
    input_dim = int(sample_x.numel())

    net = BPTTSNN(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=num_classes,
        tau=args.tau,
        v_threshold=args.v_threshold,
    ).to(actual_device)
    optimizer = build_optimizer(net, args)

    use_amp = args.amp and actual_device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)
    encoder = encoding.PoissonEncoder()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"T{args.T}_b{args.b}_{args.opt}_lr{args.lr}"
    if use_amp:
        run_tag += "_amp"

    out_dir = os.path.join(args.out_dir, dataset_name, "MLP", run_tag, now)
    os.makedirs(out_dir, exist_ok=True)

    logger = create_logger(os.path.join(out_dir, "output_log.txt"))
    logger.info("args: %s", args)
    logger.info("device: %s", actual_device)
    logger.info("dataset: %s, train=%d, val=%d, test=%d", dataset_name, len(train_subset), len(val_subset), len(test_dataset))
    logger.info("model: input_dim=%d, hidden_dims=%s, num_classes=%d", input_dim, args.hidden_dims, num_classes)
    logger.info("save dir: %s", out_dir)

    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
        f.write(str(args))
        f.write("\n")
        f.write(" ".join(sys.argv))
        f.write("\n")
        f.write(f"resolved_device={actual_device}\n")
        f.write(f"num_classes={num_classes}\n")
        f.write(f"input_dim={input_dim}\n")

    pin_memory = actual_device.type == "cuda"
    train_loader = data.DataLoader(
        dataset=train_subset,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=pin_memory,
    )
    val_loader = data.DataLoader(
        dataset=val_subset,
        batch_size=min(512, args.b),
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=pin_memory,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=min(512, args.b),
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=pin_memory,
    )

    start_epoch = 0
    best_val_acc = -1.0
    best_epoch = -1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        logger.info("Resumed from %s, start_epoch=%d", args.resume, start_epoch)

    epoch_rows = []
    train_start_time = time.time()
    use_cuda_mem_stat = actual_device.type == "cuda" and torch.cuda.is_available()

    for epoch in range(start_epoch, args.epochs):
        epoch_begin = time.time()
        train_loss, train_acc, train_speed, train_elapsed, mem_stat = run_train_epoch(
            net=net,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            encoder=encoder,
            args=args,
            device=actual_device,
            num_classes=num_classes,
            use_cuda_mem_stat=use_cuda_mem_stat,
        )
        val_loss, val_acc, val_speed, val_elapsed = run_eval_epoch(
            net=net,
            loader=val_loader,
            encoder=encoder,
            args=args,
            device=actual_device,
            num_classes=num_classes,
        )

        save_max = False
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_max = True

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_speed_img_s": train_speed,
            "val_speed_img_s": val_speed,
            "train_time_s": train_elapsed,
            "val_time_s": val_elapsed,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.time() - epoch_begin,
            "train_gpu_mem_alloc_mean_mb": mem_stat["train_gpu_mem_alloc_mean_mb"],
            "train_gpu_mem_reserved_mean_mb": mem_stat["train_gpu_mem_reserved_mean_mb"],
            "train_gpu_mem_peak_alloc_mean_mb": mem_stat["train_gpu_mem_peak_alloc_mean_mb"],
            "train_gpu_mem_peak_reserved_mean_mb": mem_stat["train_gpu_mem_peak_reserved_mean_mb"],
            "train_gpu_mem_peak_alloc_max_mb": mem_stat["train_gpu_mem_peak_alloc_max_mb"],
            "train_gpu_mem_peak_reserved_max_mb": mem_stat["train_gpu_mem_peak_reserved_max_mb"],
        }
        epoch_rows.append(row)

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "args": vars(args),
            "num_classes": num_classes,
            "input_dim": input_dim,
        }
        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_latest.pth"))
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_max.pth"))

        eta_seconds = (time.time() - train_start_time) / max((epoch - start_epoch + 1), 1) * (args.epochs - epoch - 1)
        eta_text = (datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)).strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | best_val_acc=%.4f(epoch=%d) | train_speed=%.2f val_speed=%.2f img/s | ETA=%s",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_val_acc,
            best_epoch + 1,
            train_speed,
            val_speed,
            eta_text,
        )

    total_train_time = time.time() - train_start_time

    if os.path.exists(os.path.join(out_dir, "checkpoint_max.pth")):
        best_checkpoint = torch.load(os.path.join(out_dir, "checkpoint_max.pth"), map_location="cpu")
        net.load_state_dict(best_checkpoint["net"])
        logger.info("Loaded checkpoint_max.pth for final test.")

    test_loss, test_acc, test_speed, test_time = run_eval_epoch(
        net=net,
        loader=test_loader,
        encoder=encoder,
        args=args,
        device=actual_device,
        num_classes=num_classes,
    )
    logger.info(
        "Test | loss=%.4f acc=%.4f speed=%.2f img/s time=%.2fs",
        test_loss,
        test_acc,
        test_speed,
        test_time,
    )

    if args.save_model:
        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "args": vars(args),
            "num_classes": num_classes,
            "input_dim": input_dim,
        }
        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_last.pth"))

    save_epoch_csv(epoch_rows, os.path.join(out_dir, "epoch_metrics.csv"))
    if plt is None:
        logger.warning("matplotlib is not available, skip plotting curves.")
    plot_curves(epoch_rows, out_dir)

    metrics = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "epochs": args.epochs,
        "T": args.T,
        "optimizer": args.opt,
        "lr": args.lr,
        "tau": args.tau,
        "v_threshold": args.v_threshold,
        "batch_size": args.b,
        "best_val_acc": best_val_acc * 100.0,
        "best_val_epoch": best_epoch + 1,
        "test_loss": test_loss,
        "test_acc": test_acc * 100.0,
        "test_speed_img_s": test_speed,
        "test_time_s": test_time,
        "train_total_time_s": total_train_time,
        "avg_epoch_time_s": (sum([r["epoch_time_s"] for r in epoch_rows]) / len(epoch_rows)) if epoch_rows else None,
        "last_epoch_train_loss": epoch_rows[-1]["train_loss"] if epoch_rows else None,
        "last_epoch_train_acc": (epoch_rows[-1]["train_acc"] * 100.0) if epoch_rows else None,
        "last_epoch_val_loss": epoch_rows[-1]["val_loss"] if epoch_rows else None,
        "last_epoch_val_acc": (epoch_rows[-1]["val_acc"] * 100.0) if epoch_rows else None,
        "max_train_speed_img_s": max([r["train_speed_img_s"] for r in epoch_rows]) if epoch_rows else None,
        "max_val_speed_img_s": max([r["val_speed_img_s"] for r in epoch_rows]) if epoch_rows else None,
        "train_gpu_mem_peak_alloc_max_mb": max(
            [r["train_gpu_mem_peak_alloc_max_mb"] for r in epoch_rows if r["train_gpu_mem_peak_alloc_max_mb"] is not None],
            default=None,
        ),
        "train_gpu_mem_peak_reserved_max_mb": max(
            [r["train_gpu_mem_peak_reserved_max_mb"] for r in epoch_rows if r["train_gpu_mem_peak_reserved_max_mb"] is not None],
            default=None,
        ),
        "run_dir": out_dir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Training done. Artifacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
