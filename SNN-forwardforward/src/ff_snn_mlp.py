"""
====================================================================
File          : ff_snn_mlp.py
Description   : MLP-based FF-SNN model definition and training logic
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from spikingjelly.activation_based import (
    encoding,
    functional,
    layer,
    neuron,
    surrogate,
)

from src.experiment import (
    ExperimentModeConfig,
    GradientProfilingSnapshot,
    UnsupervisedStepResult,
    UNSUPERVISED_UPDATE_AUTOGRAD,
)
from src.generate_neg_sample import generate_pos_n_neg_sample
from src.loss import gradient_calculation_mlp


def spike_encoder(images: torch.Tensor, T: int) -> torch.Tensor:
    """
    Encode images into a spike train with T time-steps.
    """
    B, C, H, W = images.shape
    spike_train = torch.zeros((T, B, C, H, W), device=images.device)
    v_mem = torch.zeros((B, C, H, W), device=images.device)
    for t in range(T):
        v_mem += images
        spike = (v_mem >= 1.0).to(torch.float32)
        spike_train[t] = spike
        v_mem = v_mem * (1.0 - spike)
    return spike_train


class tdLayerNorm(nn.Module):
    def __init__(self, dim, v_threshold, eps=1e-5, alpha=1.0):
        super().__init__()
        self.v_threshold = v_threshold
        self.eps = eps
        self.alpha = alpha
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        ln_x = (self.v_threshold * (x - mean)) / torch.sqrt(var + self.eps)
        ln_x = self.gamma * ln_x + self.beta
        return ln_x


class Net(torch.nn.Module):
    def __init__(
        self,
        dims,
        tau,
        epoch,
        T,
        lr,
        v_threshold,
        v_threshold_neg,
        opt,
        loss_threshold,
        num_classes,
        mode_config: ExperimentModeConfig | None = None,
    ):
        super().__init__()
        self.T = T
        self.loss_threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.num_classes = num_classes
        self.mode_config = mode_config or ExperimentModeConfig()

        self.last_backward_peak_alloc_bytes = None
        self.last_backward_peak_reserved_bytes = None
        self.last_manual_grad_peak_alloc_bytes = None
        self.last_manual_grad_peak_reserved_bytes = None
        self.last_manual_grad_time_ms = None
        self.last_manual_grad_ops_est = None
        self.last_backward_cmp_peak_alloc_bytes = None
        self.last_backward_cmp_peak_reserved_bytes = None
        self.last_backward_cmp_time_ms = None

        self.layers = nn.ModuleList()
        for layer_idx in range(len(dims) - 1):
            # Hidden layers follow the FF-style local objective, while the last
            # layer stays as the classifier/readout head.
            if layer_idx == len(dims) - 2:
                module = OutputLayer(
                    in_features=sum(dims[1 : layer_idx + 1]),
                    out_features=dims[layer_idx + 1],
                    epoch=epoch,
                    T=T,
                    lr=lr,
                    v_threshold=v_threshold,
                    v_threshold_neg=v_threshold_neg,
                    tau=tau,
                    loss_threshold=loss_threshold,
                ).cuda()
            else:
                module = Layer(
                    in_features=dims[layer_idx],
                    out_features=dims[layer_idx + 1],
                    epoch=epoch,
                    T=T,
                    lr=lr,
                    v_threshold=v_threshold,
                    v_threshold_neg=v_threshold_neg,
                    tau=tau,
                    loss_threshold=loss_threshold,
                    mode_config=self.mode_config,
                ).cuda()
            self.layers.append(module)

    def _aggregate_runtime_stats(self) -> GradientProfilingSnapshot:
        # Each layer records its own local stats; the network exposes only the
        # max / summed view that the runner cares about.
        max_backward_peak_alloc = None
        max_backward_peak_reserved = None
        max_manual_peak_alloc = None
        max_manual_peak_reserved = None
        max_backward_cmp_peak_alloc = None
        max_backward_cmp_peak_reserved = None
        manual_time_total_ms = 0.0
        manual_ops_total_est = 0.0
        backward_cmp_time_total_ms = 0.0

        for layer_module in self.layers:
            layer_bp_alloc = getattr(layer_module, "last_backward_peak_alloc_bytes", None)
            layer_bp_reserved = getattr(
                layer_module,
                "last_backward_peak_reserved_bytes",
                None,
            )
            layer_manual_alloc = getattr(
                layer_module,
                "last_manual_grad_peak_alloc_bytes",
                None,
            )
            layer_manual_reserved = getattr(
                layer_module,
                "last_manual_grad_peak_reserved_bytes",
                None,
            )
            layer_manual_time = getattr(layer_module, "last_manual_grad_time_ms", None)
            layer_bp_cmp_alloc = getattr(
                layer_module,
                "last_backward_cmp_peak_alloc_bytes",
                None,
            )
            layer_bp_cmp_reserved = getattr(
                layer_module,
                "last_backward_cmp_peak_reserved_bytes",
                None,
            )
            layer_bp_cmp_time = getattr(
                layer_module,
                "last_backward_cmp_time_ms",
                None,
            )
            layer_manual_ops = getattr(layer_module, "last_manual_grad_ops_est", None)

            if layer_bp_alloc is not None:
                max_backward_peak_alloc = (
                    layer_bp_alloc
                    if max_backward_peak_alloc is None
                    else max(max_backward_peak_alloc, layer_bp_alloc)
                )
            if layer_bp_reserved is not None:
                max_backward_peak_reserved = (
                    layer_bp_reserved
                    if max_backward_peak_reserved is None
                    else max(max_backward_peak_reserved, layer_bp_reserved)
                )
            if layer_manual_alloc is not None:
                max_manual_peak_alloc = (
                    layer_manual_alloc
                    if max_manual_peak_alloc is None
                    else max(max_manual_peak_alloc, layer_manual_alloc)
                )
            if layer_manual_reserved is not None:
                max_manual_peak_reserved = (
                    layer_manual_reserved
                    if max_manual_peak_reserved is None
                    else max(max_manual_peak_reserved, layer_manual_reserved)
                )
            if layer_bp_cmp_alloc is not None:
                max_backward_cmp_peak_alloc = (
                    layer_bp_cmp_alloc
                    if max_backward_cmp_peak_alloc is None
                    else max(max_backward_cmp_peak_alloc, layer_bp_cmp_alloc)
                )
            if layer_bp_cmp_reserved is not None:
                max_backward_cmp_peak_reserved = (
                    layer_bp_cmp_reserved
                    if max_backward_cmp_peak_reserved is None
                    else max(max_backward_cmp_peak_reserved, layer_bp_cmp_reserved)
                )
            if layer_manual_time is not None:
                manual_time_total_ms += float(layer_manual_time)
            if layer_manual_ops is not None:
                manual_ops_total_est += float(layer_manual_ops)
            if layer_bp_cmp_time is not None:
                backward_cmp_time_total_ms += float(layer_bp_cmp_time)

        snapshot = GradientProfilingSnapshot(
            backward_peak_alloc_bytes=max_backward_peak_alloc,
            backward_peak_reserved_bytes=max_backward_peak_reserved,
            manual_grad_peak_alloc_bytes=max_manual_peak_alloc,
            manual_grad_peak_reserved_bytes=max_manual_peak_reserved,
            manual_grad_time_ms=manual_time_total_ms if manual_time_total_ms > 0 else None,
            manual_grad_ops_est=manual_ops_total_est if manual_ops_total_est > 0 else None,
            backward_cmp_peak_alloc_bytes=max_backward_cmp_peak_alloc,
            backward_cmp_peak_reserved_bytes=max_backward_cmp_peak_reserved,
            backward_cmp_time_ms=(
                backward_cmp_time_total_ms if backward_cmp_time_total_ms > 0 else None
            ),
        )
        self.last_backward_peak_alloc_bytes = snapshot.backward_peak_alloc_bytes
        self.last_backward_peak_reserved_bytes = snapshot.backward_peak_reserved_bytes
        self.last_manual_grad_peak_alloc_bytes = snapshot.manual_grad_peak_alloc_bytes
        self.last_manual_grad_peak_reserved_bytes = snapshot.manual_grad_peak_reserved_bytes
        self.last_manual_grad_time_ms = snapshot.manual_grad_time_ms
        self.last_manual_grad_ops_est = snapshot.manual_grad_ops_est
        self.last_backward_cmp_peak_alloc_bytes = (
            snapshot.backward_cmp_peak_alloc_bytes
        )
        self.last_backward_cmp_peak_reserved_bytes = (
            snapshot.backward_cmp_peak_reserved_bytes
        )
        self.last_backward_cmp_time_ms = snapshot.backward_cmp_time_ms
        return snapshot

    def predict_multiple(self, x):
        goodness_per_label = []
        for label_idx in range(self.num_classes):
            goodness = []
            # Prediction still follows the original FF idea: evaluate every label
            # hypothesis and choose the one with the largest total hidden goodness.
            label = torch.full(
                (x.shape[0],),
                label_idx,
                device=x.device,
                dtype=torch.long,
            )
            h, _ = generate_pos_n_neg_sample(
                x,
                label,
                num_classes=self.num_classes,
                type="embed_label_onehot",
            )
            h = spike_encoder(h, self.T)
            h = h.flatten(2)
            for layer_idx, layer_module in enumerate(self.layers):
                if layer_idx == len(self.layers) - 1:
                    break
                h = layer_module.predict(h)
                freq = h.mean(0)
                goodness.append(layer_module.cal_goodness(freq).sum(1))
            goodness_per_label.append(sum(goodness).unsqueeze(1))
        goodness_of_all_label = torch.cat(goodness_per_label, 1)
        return goodness_of_all_label.argmax(1)

    def predict_winner(self, x):
        label = torch.randint(0, self.num_classes, (x.shape[0],), device=x.device)
        h, _ = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
            type="SCFF",
        )
        h = spike_encoder(x, self.T)
        h = h.flatten(2)
        spike_in_of_output_layer = torch.empty(
            (h.shape[0], h.shape[1], 0),
            device=h.device,
        )
        for layer_idx, layer_module in enumerate(self.layers):
            if layer_idx == len(self.layers) - 1:
                spike_out = layer_module.predict(spike_in_of_output_layer)
            else:
                h = layer_module.predict(h)
                spike_in_of_output_layer = torch.cat(
                    (spike_in_of_output_layer, h),
                    dim=2,
                )
        spike_out_sum = spike_out.sum(0)
        return spike_out_sum.argmax(1)

    def train_layerwise(self, x_pos, x_neg, y, layer_idx):
        raise NotImplementedError(
            "Layer-wise training path is not part of the current refactor target."
        )

    def train_unsupervised(self, x, label, frozen) -> UnsupervisedStepResult:
        x_pos, x_neg = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
            type="embed_label_onehot",
        )
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        spike_input_pos = x_pos_encoded.flatten(2)
        spike_input_neg = x_neg_encoded.flatten(2)
        return self.train_unsupervised_step(
            spike_input_pos,
            spike_input_neg,
            label,
            frozen,
        )

    def train_ff_stdp(self, x, label, frozen):
        return self.train_unsupervised(x, label, frozen)

    def train_unsupervised_step(
        self,
        input_pos,
        input_neg,
        label,
        frozen,
    ) -> UnsupervisedStepResult:
        T, B, _ = input_pos.shape
        pos_goodness_per_layer = []
        neg_goodness_per_layer = []
        pos_cos_sim_per_layer = []
        neg_cos_sim_per_layer = []
        pos_spike_out_per_layer = []
        neg_spike_out_per_layer = []

        pos_spike_in_of_output_layer = torch.empty((T, B, 0), device=input_pos.device)
        neg_spike_in_of_output_layer = torch.empty((T, B, 0), device=input_neg.device)

        for layer_idx, layer_module in enumerate(self.layers):
            if layer_idx == len(self.layers) - 1:
                # The readout head is trained separately from the hidden FF layers.
                layer_module.train_bp_stdp(pos_spike_in_of_output_layer, label)
                continue

            (
                input_pos,
                pos_goodness,
                pos_cos_sim,
                input_neg,
                neg_goodness,
                neg_cos_sim,
            ) = layer_module.train_unsupervised(input_pos, input_neg, frozen)
            pos_goodness_per_layer.append(pos_goodness)
            neg_goodness_per_layer.append(neg_goodness)
            pos_cos_sim_per_layer.append(pos_cos_sim)
            neg_cos_sim_per_layer.append(neg_cos_sim)
            pos_spike_out_per_layer.append(input_pos.mean().detach().cpu().item())
            neg_spike_out_per_layer.append(input_neg.mean().detach().cpu().item())
            pos_spike_in_of_output_layer = torch.cat(
                (pos_spike_in_of_output_layer, input_pos),
                dim=2,
            )
            neg_spike_in_of_output_layer = torch.cat(
                (neg_spike_in_of_output_layer, input_neg),
                dim=2,
            )

        profiler_snapshot = self._aggregate_runtime_stats()
        return UnsupervisedStepResult(
            goodness_pos=pos_goodness_per_layer,
            goodness_neg=neg_goodness_per_layer,
            cos_pos=pos_cos_sim_per_layer,
            cos_neg=neg_cos_sim_per_layer,
            spike_out_pos=pos_spike_out_per_layer,
            spike_out_neg=neg_spike_out_per_layer,
            profiler=profiler_snapshot,
        )

    def save(self, args, path):
        check_point = {
            "net": {
                f"layer_{i}": layer_module.state_dict()
                for i, layer_module in enumerate(self.layers)
            },
            "args": args,
        }
        torch.save(check_point, path)

    def load(self, path):
        check_point = torch.load(path)
        for i, layer_module in enumerate(self.layers):
            key = f"layer_{i}"
            if key in check_point["net"]:
                layer_module.load_state_dict(check_point["net"][key])
                print(f"[OK] Loaded {key}")
            else:
                print(f"[Skip] {key} not found in checkpoint, skipped.")
        print(check_point["args"])


class Layer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        epoch,
        T,
        lr,
        v_threshold,
        v_threshold_neg,
        tau,
        loss_threshold,
        mode_config: ExperimentModeConfig | None = None,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            ),
        )
        self.lr = lr
        self.v_threshold = v_threshold
        self.spike_input_rate = 0
        self.in_features = in_features
        self.out_features = out_features
        self.num_epochs = epoch
        self.T = T
        self.threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.opt = Adam(self.parameters(), lr=lr)
        self.mode_config = mode_config or ExperimentModeConfig()

        self.last_backward_peak_alloc_bytes = None
        self.last_backward_peak_reserved_bytes = None
        self.last_manual_grad_peak_alloc_bytes = None
        self.last_manual_grad_peak_reserved_bytes = None
        self.last_manual_grad_time_ms = None
        self.last_manual_grad_ops_est = None
        self.last_backward_cmp_peak_alloc_bytes = None
        self.last_backward_cmp_peak_reserved_bytes = None
        self.last_backward_cmp_time_ms = None

        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, std=np.sqrt(2 / self.out_features))
                module.weight.data += 0.1

    def cal_goodness(self, freq):
        return self.T * freq.abs().pow(2) * freq.sign()

    def _get_linear_weight(self):
        for module in self.layer.modules():
            if isinstance(module, nn.Linear):
                return module.weight
        raise RuntimeError("Linear layer not found.")

    def _capture_cuda_peak_stats(self, fn):
        # Wrap a callable once so manual-grad profiling, comparison backward, and
        # real training backward all use the same measurement path.
        device = self._get_linear_weight().device
        use_cuda_mem_stat = device.type == "cuda"
        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            base_alloc = torch.cuda.memory_allocated(device)
            base_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            base_alloc = None
            base_reserved = None

        start_time = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            peak_alloc = max(
                0.0,
                float(torch.cuda.max_memory_allocated(device) - base_alloc),
            )
            peak_reserved = max(
                0.0,
                float(torch.cuda.max_memory_reserved(device) - base_reserved),
            )
        else:
            peak_alloc = None
            peak_reserved = None

        return result, peak_alloc, peak_reserved, elapsed_ms

    def _forward_spike_sequence(self, encoded):
        # This helper isolates the repeated "simulate over T, then summarize into
        # firing frequency and goodness" pattern shared by pos/neg samples.
        _, batch_size, _ = encoded.shape
        output_spike = torch.zeros(
            self.T,
            batch_size,
            self.out_features,
            device=encoded.device,
        )
        ln_mean = torch.zeros((batch_size,), device=encoded.device)
        ln_var = torch.zeros((batch_size,), device=encoded.device)
        for t in range(self.T):
            spike_out, ln_mean, ln_var = self.forward(encoded[t], ln_mean, ln_var)
            output_spike[t] += spike_out
        out_freq = output_spike.mean(0)
        goodness = self.cal_goodness(out_freq)
        return encoded.sum(0), output_spike, out_freq, goodness, ln_mean, ln_var

    def _manual_gradient(
        self,
        input_spike_sum,
        out_freq,
        goodness,
        ln_var,
        ln_mean,
        batch_size,
        is_pos,
    ):
        # Manual gradient is used both as an executable update path and as a
        # profiling reference when the real update still uses autograd.
        def _compute():
            with torch.no_grad():
                return gradient_calculation_mlp(
                    input_spike_sum,
                    out_freq,
                    goodness,
                    ln_var,
                    ln_mean,
                    self.threshold,
                    self.v_threshold,
                    batch_size,
                    is_pos,
                )[0]

        return self._capture_cuda_peak_stats(_compute)

    def _autograd_branch_comparison(
        self,
        input_spike_sum,
        out_freq,
        goodness,
        ln_var,
        ln_mean,
        batch_size,
        is_pos,
        *,
        retain_graph,
    ):
        # This branch exists only for gradient/memory/time comparison and should
        # not silently replace the actual update mode.
        _, loss = gradient_calculation_mlp(
            input_spike_sum,
            out_freq,
            goodness,
            ln_var,
            ln_mean,
            self.threshold,
            self.v_threshold,
            batch_size,
            is_pos,
        )

        def _backward():
            self.opt.zero_grad()
            loss.backward(retain_graph=retain_graph)
            return self._get_linear_weight().grad.detach().clone()

        grad, peak_alloc, peak_reserved, elapsed_ms = self._capture_cuda_peak_stats(
            _backward
        )
        self.opt.zero_grad()
        return grad, peak_alloc, peak_reserved, elapsed_ms

    def _apply_manual_update(self, weight_grad, frozen):
        if frozen:
            return
        with torch.no_grad():
            # Keep the original direct weight update rule for the manual branch.
            self._get_linear_weight().add_(self.lr * weight_grad)

    def _apply_autograd_update(self, pos_out_freq, neg_out_freq):
        def _backward_and_step():
            self.opt.zero_grad()
            p = self.T * pos_out_freq.pow(2).mean(1)
            n = self.T * neg_out_freq.pow(2).mean(1)
            loss = torch.log(
                1 + torch.exp(torch.cat([-p + self.threshold, n - self.threshold]))
            ).mean()
            loss.backward()
            self.opt.step()
            return loss

        _, peak_alloc, peak_reserved, _ = self._capture_cuda_peak_stats(
            _backward_and_step
        )
        self.last_backward_peak_alloc_bytes = peak_alloc
        self.last_backward_peak_reserved_bytes = peak_reserved

    def _reset_runtime_stats(self):
        self.last_backward_peak_alloc_bytes = None
        self.last_backward_peak_reserved_bytes = None
        self.last_manual_grad_peak_alloc_bytes = None
        self.last_manual_grad_peak_reserved_bytes = None
        self.last_manual_grad_time_ms = None
        self.last_manual_grad_ops_est = None
        self.last_backward_cmp_peak_alloc_bytes = None
        self.last_backward_cmp_peak_reserved_bytes = None
        self.last_backward_cmp_time_ms = None

    def forward(self, x, mean, var):
        x = self.layer[0](x)
        x = self.layer[1](x)
        x = self.layer[2](x)
        return x, mean, var

    def train_unsupervised(self, pos_encoded, neg_encoded, frozen):
        _, batch_size, _ = pos_encoded.shape
        self._reset_runtime_stats()

        (
            pos_input_spike_sum,
            pos_output_spike,
            pos_out_freq,
            pos_goodness,
            pos_ln_mean,
            pos_ln_var,
        ) = self._forward_spike_sequence(pos_encoded)
        functional.reset_net(self.layer)

        (
            neg_input_spike_sum,
            neg_output_spike,
            neg_out_freq,
            neg_goodness,
            neg_ln_mean,
            neg_ln_var,
        ) = self._forward_spike_sequence(neg_encoded)
        functional.reset_net(self.layer)

        needs_manual_grad = (
            self.mode_config.uses_manual_update
            or self.mode_config.profiling.capture_manual_grad_metrics
        )
        needs_autograd_cmp = self.mode_config.profiling.capture_autograd_comparison

        pos_weight_grad = None
        neg_weight_grad = None
        pos_cos_sim = None
        neg_cos_sim = None
        pos_manual_peak_alloc = None
        pos_manual_peak_reserved = None
        neg_manual_peak_alloc = None
        neg_manual_peak_reserved = None
        pos_bp_peak_alloc = None
        pos_bp_peak_reserved = None
        neg_bp_peak_alloc = None
        neg_bp_peak_reserved = None

        if needs_manual_grad:
            # Even in autograd mode we may still collect manual-gradient stats for
            # hardware-cost analysis and later comparison.
            (
                pos_weight_grad,
                pos_manual_peak_alloc,
                pos_manual_peak_reserved,
                pos_manual_time_ms,
            ) = self._manual_gradient(
                pos_input_spike_sum,
                pos_out_freq,
                pos_goodness,
                pos_ln_var,
                pos_ln_mean,
                batch_size,
                True,
            )
            (
                neg_weight_grad,
                neg_manual_peak_alloc,
                neg_manual_peak_reserved,
                neg_manual_time_ms,
            ) = self._manual_gradient(
                neg_input_spike_sum,
                neg_out_freq,
                neg_goodness,
                neg_ln_var,
                neg_ln_mean,
                batch_size,
                False,
            )
            alloc_peaks = [
                value
                for value in [pos_manual_peak_alloc, neg_manual_peak_alloc]
                if value is not None
            ]
            reserved_peaks = [
                value
                for value in [pos_manual_peak_reserved, neg_manual_peak_reserved]
                if value is not None
            ]
            self.last_manual_grad_peak_alloc_bytes = (
                max(alloc_peaks) if alloc_peaks else None
            )
            self.last_manual_grad_peak_reserved_bytes = (
                max(reserved_peaks) if reserved_peaks else None
            )
            self.last_manual_grad_time_ms = float(
                (pos_manual_time_ms or 0.0) + (neg_manual_time_ms or 0.0)
            )
            self.last_manual_grad_ops_est = float(
                4.0 * batch_size * self.out_features * self.in_features
            )

        if needs_autograd_cmp:
            # Comparison backward is optional because it adds extra graph work and
            # should be disable-able for pure throughput experiments.
            retain_graph_for_cmp = (
                self.mode_config.unsupervised_update_mode
                == UNSUPERVISED_UPDATE_AUTOGRAD
            )
            (
                pos_autograd_grad,
                pos_bp_peak_alloc,
                pos_bp_peak_reserved,
                pos_bp_time_ms,
            ) = self._autograd_branch_comparison(
                pos_input_spike_sum,
                pos_out_freq,
                pos_goodness,
                pos_ln_var,
                pos_ln_mean,
                batch_size,
                True,
                retain_graph=retain_graph_for_cmp,
            )
            (
                neg_autograd_grad,
                neg_bp_peak_alloc,
                neg_bp_peak_reserved,
                neg_bp_time_ms,
            ) = self._autograd_branch_comparison(
                neg_input_spike_sum,
                neg_out_freq,
                neg_goodness,
                neg_ln_var,
                neg_ln_mean,
                batch_size,
                False,
                retain_graph=retain_graph_for_cmp,
            )
            cmp_alloc_peaks = [
                value
                for value in [pos_bp_peak_alloc, neg_bp_peak_alloc]
                if value is not None
            ]
            cmp_reserved_peaks = [
                value
                for value in [pos_bp_peak_reserved, neg_bp_peak_reserved]
                if value is not None
            ]
            self.last_backward_cmp_peak_alloc_bytes = (
                max(cmp_alloc_peaks) if cmp_alloc_peaks else None
            )
            self.last_backward_cmp_peak_reserved_bytes = (
                max(cmp_reserved_peaks) if cmp_reserved_peaks else None
            )
            self.last_backward_cmp_time_ms = float(
                (pos_bp_time_ms or 0.0) + (neg_bp_time_ms or 0.0)
            )
            if pos_weight_grad is not None:
                pos_cos_sim = torch.cosine_similarity(
                    pos_autograd_grad.flatten(),
                    -pos_weight_grad.flatten(),
                    dim=0,
                ).detach().cpu().item()
            if neg_weight_grad is not None:
                neg_cos_sim = torch.cosine_similarity(
                    neg_autograd_grad.flatten(),
                    -neg_weight_grad.flatten(),
                    dim=0,
                ).detach().cpu().item()

        if self.mode_config.uses_manual_update:
            if pos_weight_grad is None or neg_weight_grad is None:
                raise RuntimeError("Manual update mode requires manual gradients.")
            self._apply_manual_update(pos_weight_grad + neg_weight_grad, frozen)
        else:
            # Default behavior remains the original autograd-based hidden-layer update.
            self._apply_autograd_update(pos_out_freq, neg_out_freq)

        functional.reset_net(self.layer)
        return (
            pos_output_spike.detach(),
            pos_goodness.detach().mean().cpu().item(),
            pos_cos_sim,
            neg_output_spike.detach(),
            neg_goodness.detach().mean().cpu().item(),
            neg_cos_sim,
        )

    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        return self.train_unsupervised(pos_encoded, neg_encoded, frozen)

    def predict(self, x):
        batch_size = x.shape[1]
        output = torch.zeros(
            self.T,
            batch_size,
            self.out_features,
            device=x.device,
        )
        ln_mean = torch.zeros((batch_size,), device=x.device)
        ln_var = torch.zeros((batch_size,), device=x.device)
        for t in range(self.T):
            spike_out, ln_mean, ln_var = self.forward(x[t], ln_mean, ln_var)
            output[t] += spike_out
        functional.reset_net(self.layer)
        return output


class OutputLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        epoch,
        T,
        lr,
        v_threshold,
        v_threshold_neg,
        tau,
        loss_threshold,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
        )
        self.lr = lr
        self.spike_input_rate = 0
        self.in_features = in_features
        self.out_features = out_features
        self.num_epochs = epoch
        self.T = T
        self.threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.opt = Adam(self.parameters(), lr=lr)
        self.last_backward_peak_alloc_bytes = None
        self.last_backward_peak_reserved_bytes = None
        self.last_manual_grad_peak_alloc_bytes = None
        self.last_manual_grad_peak_reserved_bytes = None
        self.last_manual_grad_time_ms = None
        self.last_manual_grad_ops_est = None
        self.last_backward_cmp_peak_alloc_bytes = None
        self.last_backward_cmp_peak_reserved_bytes = None
        self.last_backward_cmp_time_ms = None
        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)

    def forward(self, x):
        return self.layer(x)

    def train_bp_stdp(self, x_encoded, label):
        batch_size = x_encoded.shape[1]
        device = x_encoded.device
        use_cuda_mem_stat = device.type == "cuda"
        output_spike = torch.zeros(
            self.T,
            batch_size,
            self.out_features,
            device=device,
        )
        for t in range(self.T):
            output_spike[t] += self.forward(x_encoded[t])
        spike_freq = output_spike.mean(0)
        self.opt.zero_grad()
        loss = F.cross_entropy(spike_freq.view(-1, self.out_features), label.view(-1))
        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            base_alloc = torch.cuda.memory_allocated(device)
            base_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        loss.backward()
        if use_cuda_mem_stat:
            torch.cuda.synchronize(device)
            self.last_backward_peak_alloc_bytes = max(
                0.0,
                float(torch.cuda.max_memory_allocated(device) - base_alloc),
            )
            self.last_backward_peak_reserved_bytes = max(
                0.0,
                float(torch.cuda.max_memory_reserved(device) - base_reserved),
            )
        else:
            self.last_backward_peak_alloc_bytes = None
            self.last_backward_peak_reserved_bytes = None
        self.opt.step()
        # input_spike_sum = x_encoded.sum(0).cuda()
        # ksi_output = torch.zeros(N,self.out_features).cuda() 
        # spike_sums = output_spike.sum(0)  # 对时间维度求和，形状为 [N, out_features]
        #  # 创建一个布尔掩码，判断每个样本的每个输出神经元是否满足条件
        # neg_mask = (spike_sums >= 1) & (torch.arange(self.out_features).cuda() != label.unsqueeze(1))
        # pos_mask = (spike_sums <= (self.T/2)) & (torch.arange(self.out_features).cuda() == label.unsqueeze(1))
        # ksi_output[pos_mask] = 1
        # ksi_output[neg_mask] = -1
        # ksi_output = ksi_output.transpose(0,1)
        # self.opt.zero_grad()
        # weight_grad = ksi_output @ input_spike_sum / N
        # with torch.no_grad():
        #     for param in self.layer.parameters():
        #             # 使用优化器更新权重           
        #             param += self.lr * weight_grad
        functional.reset_net(self.layer)
        return output_spike.detach()

    def predict(self, x):
        output = torch.zeros(
            self.T,
            x.shape[1],
            self.out_features,
            device=x.device,
        )
        for t in range(self.T):
            spike_out = self.forward(x[t])
            output[t] += spike_out
        functional.reset_net(self.layer)
        return output
