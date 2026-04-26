from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time

import torch
import torch.utils.data as data
import torchvision
from tqdm import tqdm

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST

from src.experiment import ExperimentModeConfig, TrainMemorySnapshot
from src.ff_snn_cnn import ConvNet
from src.ff_snn_mlp import Net
from src.metrics_tracker import ExperimentMetricsTracker


def normalize_frame(x):
    x = x.astype("float32")
    if x.ndim == 4:
        x = x.sum(axis=0)
    max_v = x.max()
    if max_v > 0:
        x = x / max_v
    return x


def build_datasets(args):
    # Keep dataset wiring centralized so future supervised/unsupervised branches
    # can reuse the same input pipeline.
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
                root=(args.data_dir + "/NMNIST"),
                train=True,
                data_type="frame",
                frames_number=args.T,
                split_by="number",
                transform=normalize_frame,
            )
            test_dataset = NMNIST(
                root=(args.data_dir + "/NMNIST"),
                train=False,
                data_type="frame",
                frames_number=args.T,
                split_by="number",
                transform=normalize_frame,
            )
        except Exception as exc:
            raise RuntimeError(
                "NMNIST does not support automatic download in this spikingjelly version. "
                "Please manually prepare N-MNIST data under data_dir, then rerun."
            ) from exc
    elif args.dataset in ("DVS128Gesture", "DVS128-Gesture", "DVS 128 Gesture"):
        train_dataset = DVS128Gesture(
            root=(args.data_dir + "/DVSgesture"),
            train=True,
            data_type="frame",
            frames_number=args.T,
            split_by="number",
            transform=normalize_frame,
        )
        test_dataset = DVS128Gesture(
            root=(args.data_dir + "/DVSgesture"),
            train=False,
            data_type="frame",
            frames_number=args.T,
            split_by="number",
            transform=normalize_frame,
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
    return train_dataset, test_dataset


def build_data_loaders(args, train_dataset, test_dataset):
    train_size = int(0.95 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
    )
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
    return train_data_loader, val_data_loader, test_data_loader


def infer_num_classes(dataset):
    # Prefer dataset metadata so we do not trigger per-sample decoding or transforms.
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return len(classes)

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)

    if targets is not None:
        if torch.is_tensor(targets):
            return int(torch.unique(targets).numel())
        return len({int(target) for target in targets})

    labels = set()
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        labels.add(int(y))
    return len(labels)


def build_model(args, num_classes, mode_config, sample_batch=None):
    # Model construction stays independent from the training loop so the top-level
    # mode switch only needs to decide "what to run", not "how to build it".
    if args.model == "MLP":
        return Net(
            dims=args.dims,
            tau=args.tau,
            epoch=args.epochs,
            T=args.T,
            lr=args.lr,
            v_threshold=args.v_threshold,
            v_threshold_neg=args.v_threshold_neg,
            opt=args.opt,
            loss_threshold=args.loss_threshold,
            num_classes=num_classes,
            mode_config=mode_config,
        )
    if args.model == "CNN":
        if sample_batch is None:
            raise ValueError("CNN model construction requires a sample batch.")
        _, _, H, W = sample_batch.shape
        return ConvNet(
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
    raise ValueError(f"Unsupported model type: {args.model}")


def evaluate_accuracy(net, data_loader, device):
    acc_sum = 0.0
    batch_count = 0
    net.eval()
    with torch.no_grad():
        for x_val, y_val in data_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            acc_sum += net.predict_multiple(x_val).eq(y_val).cpu().float().mean().item()
            batch_count += 1
    return 100.0 * (acc_sum / batch_count)


def train_one_step(net, mode_config, x, y, frozen):
    if mode_config.is_unsupervised:
        return net.train_unsupervised(x, y, frozen)
    # Supervised mode is intentionally left here as the future extension seam.
    raise NotImplementedError(
        "Supervised mode is reserved as an extension point and is not implemented yet."
    )


def create_output_dir(args, mode_config):
    out_dir = os.path.join(
        args.out_dir,
        mode_config.learning_mode,
        args.dataset,
        args.model,
        mode_config.run_name,
        f"T{args.T}_b{args.b}_{args.opt}_lr{args.lr}",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_experiment(args):
    mode_config = ExperimentModeConfig.from_args(args)
    # The runner owns the orchestration only; model-specific math stays inside
    # the network modules and metrics formatting stays inside the tracker.
    train_dataset, test_dataset = build_datasets(args)
    train_loader, val_loader, test_loader = build_data_loaders(
        args,
        train_dataset,
        test_dataset,
    )
    num_classes = infer_num_classes(test_dataset)
    sample_batch = next(iter(train_loader))[0] if args.model == "CNN" else None

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    use_cuda_mem_stat = torch.cuda.is_available() and device.type == "cuda"
    net = build_model(args, num_classes, mode_config, sample_batch=sample_batch)
    out_dir = create_output_dir(args, mode_config)

    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(json.dumps(mode_config.to_dict(), ensure_ascii=False, indent=2))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    print(f"args: {args}")
    print(f"Saving at {out_dir}")

    tracker = ExperimentMetricsTracker(
        num_hidden_layers=max(0, len(net.layers) - 1),
        mode_config=mode_config,
    )
    log_file_path = os.path.join(out_dir, "output_log.txt")
    original_stdout = sys.stdout
    max_train_acc = 0.0
    training_start_time = time.time()

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            sys.stdout = log_file
            for epoch_idx in tqdm(range(args.epochs)):
                net.train()
                tracker.begin_epoch()
                # Preserve the original late-epoch freezing behavior.
                frozen = epoch_idx > (0.8 * args.epochs)
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if use_cuda_mem_stat:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    step_result = train_one_step(net, mode_config, x, y, frozen)
                    if use_cuda_mem_stat:
                        torch.cuda.synchronize(device)
                        train_memory_snapshot = TrainMemorySnapshot(
                            current_alloc_bytes=torch.cuda.memory_allocated(device),
                            current_reserved_bytes=torch.cuda.memory_reserved(device),
                            peak_alloc_bytes=torch.cuda.max_memory_allocated(device),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        )
                    else:
                        train_memory_snapshot = None
                    tracker.record_train_step(
                        step_result,
                        batch_size=int(y.numel()),
                        train_memory_snapshot=train_memory_snapshot,
                    )

                # Validation is kept outside the per-batch tracker so the tracker
                # remains focused on training-side measurements.
                train_acc = evaluate_accuracy(net, val_loader, device)
                loss = tracker.finalize_epoch(
                    loss_threshold=args.loss_threshold,
                    train_acc=train_acc,
                )
                print(f"Epoch: {epoch_idx + 1}/{args.epochs}, Loss: {loss.mean():.4f}")
                print(f"Train Acc:  {train_acc:.2f}%")
                if train_acc >= max_train_acc:
                    net.save(args, os.path.join(out_dir, "checkpoint_max.pth"))
                    max_train_acc = train_acc
                logger.info(
                    f"Epoch {epoch_idx + 1}: Train Loss = {loss.mean():.4f} Train Acc = {train_acc:.2f}%"
                )
    finally:
        sys.stdout = original_stdout

    training_total_time = time.time() - training_start_time
    train_h, train_rem = divmod(training_total_time, 3600)
    train_m, train_s = divmod(train_rem, 60)
    print(f"Training completed. Total time: {int(train_h)}h {int(train_m)}m {int(train_s)}s")

    tracker.save_plots(out_dir)

    test_acc = 0.0
    test_count = 0
    test_start_time = time.time()
    with torch.no_grad():
        for x_te, y_te in test_loader:
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_acc += net.predict_multiple(x_te).eq(y_te).cpu().float().mean().item()
            test_count += 1
    test_acc = 100.0 * test_acc / test_count
    test_total_time = time.time() - test_start_time
    test_h, test_rem = divmod(test_total_time, 3600)
    test_m, test_s = divmod(test_rem, 60)
    print(f"test Acc: {test_acc} %")
    print(f"Testing completed. Total time: {int(test_h)}h {int(test_m)}m {int(test_s)}s")

    if args.save_model or True:
        net.save(args, os.path.join(out_dir, "checkpoint_last.pth"))

    metrics = tracker.build_metrics(test_acc=test_acc)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as metrics_f:
        json.dump(metrics, metrics_f, ensure_ascii=False, indent=2)
    logger.info(f"Test Acc: {test_acc}%")
    print("Back to console.")
