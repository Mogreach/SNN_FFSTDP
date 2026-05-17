from __future__ import annotations

"""
====================================================================
File          : ff_snn_cnn_unsup.py
Description   : Unsupervised CNN-based FF-SNN model definition and training logic
Author        : Morgreach
Version       : 1.0.0
Date          : 2026-01-25
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""

import time

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
    HIDDEN_LAYER_UPDATE_AUTOGRAD,
    StepResult,
)
from src.cnn_models.common import (
    compute_conv_output_size,
    compute_pool_output_size,
    normalize_conv_layer_spec,
)
from src.generate_neg_sample import generate_pos_n_neg_sample
from src.loss import (
    ff_pairwise_goodness_loss,
    gradient_calculation_cnn,
)


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


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        conv_cfg,
        tau,
        epoch,
        T,
        lr,
        v_threshold,
        v_threshold_neg,
        opt,
        loss_threshold,
        num_classes=10,
        H=28,
        W=28,
        mode_config: ExperimentModeConfig | None = None,
        device=None,
    ):
        super().__init__()
        self.T = T
        self.loss_threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.num_classes = num_classes
        self.mode_config = mode_config or ExperimentModeConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
        input_feature_of_linear = 0
        for raw_layer_spec in conv_cfg:
            layer_spec = normalize_conv_layer_spec(raw_layer_spec)
            in_ch = layer_spec.in_channels
            out_ch = layer_spec.out_channels
            kernel_size = layer_spec.kernel_size
            stride = layer_spec.stride
            padding = layer_spec.padding

            H = compute_conv_output_size(H, kernel_size, stride, padding)
            W = compute_conv_output_size(W, kernel_size, stride, padding)
            Hp = compute_pool_output_size(
                H,
                layer_spec.pool_kernel_size,
                layer_spec.pool_stride,
            )
            Wp = compute_pool_output_size(
                W,
                layer_spec.pool_kernel_size,
                layer_spec.pool_stride,
            )
            input_feature_of_linear += out_ch * Hp * Wp
            module = ConvLayer(
                in_channels=in_ch,
                out_channels=out_ch,
                H=H,
                W=W,
                Hp=Hp,
                Wp=Wp,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pool_kernel_size=layer_spec.pool_kernel_size,
                pool_stride=layer_spec.pool_stride,
                epoch=epoch,
                T=T,
                lr=lr,
                v_threshold=v_threshold,
                v_threshold_neg=v_threshold_neg,
                tau=tau,
                loss_threshold=loss_threshold,
                mode_config=self.mode_config,
            ).to(self.device)
            self.layers.append(module)
            H = Hp
            W = Wp

        self.layers.append(
            OutputLayer(
                in_features=input_feature_of_linear,
                out_features=num_classes,
                epoch=epoch,
                T=T,
                lr=lr,
                v_threshold=v_threshold,
                v_threshold_neg=v_threshold_neg,
                tau=tau,
                loss_threshold=loss_threshold,
            ).to(self.device)
        )

    def _aggregate_runtime_stats(self) -> GradientProfilingSnapshot:
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
            layer_bp_cmp_time = getattr(layer_module, "last_backward_cmp_time_ms", None)
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
            # Prediction follows the original FF idea: evaluate every label
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
            for layer_idx, layer_module in enumerate(self.layers):
                if layer_idx == len(self.layers) - 1:
                    break
                h = layer_module.predict(h)
                freq = h.mean(0)
                goodness.append(layer_module.cal_goodness(freq).flatten(1).sum(1))

            if goodness:
                label_goodness = torch.stack(goodness, dim=1).sum(1)
            else:
                label_goodness = torch.zeros(x.shape[0], device=x.device)
            goodness_per_label.append(label_goodness.unsqueeze(1))

        goodness_of_all_label = torch.cat(goodness_per_label, 1)
        return goodness_of_all_label.argmax(1)

    def predict_winner(self, x):
        return self.predict_multiple(x)

    def _encode_training_samples(self, x, label):
        x_pos, x_neg = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
            type="embed_label_onehot",
        )
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        return x_pos_encoded, x_neg_encoded

    def train_unsupervised(self, x, label, frozen) -> StepResult:
        input_pos, input_neg = self._encode_training_samples(x, label)
        return self._train_step(
            input_pos,
            input_neg,
            label,
            frozen,
        )

    def train_ff_stdp(self, x, label, frozen):
        return self.train_unsupervised(x, label, frozen)

    def _train_step(
        self,
        input_pos,
        input_neg,
        label,
        frozen,
    ) -> StepResult:
        T, B, _, _, _ = input_pos.shape
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
                (pos_spike_in_of_output_layer, input_pos.flatten(2)),
                dim=2,
            )
            neg_spike_in_of_output_layer = torch.cat(
                (neg_spike_in_of_output_layer, input_neg.flatten(2)),
                dim=2,
            )

        profiler_snapshot = self._aggregate_runtime_stats()
        return StepResult(
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


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        H,
        W,
        Hp,
        Wp,
        kernel_size,
        stride,
        padding,
        pool_kernel_size,
        pool_stride,
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
        pool_layer = (
            layer.MaxPool2d(pool_kernel_size, pool_stride)
            if pool_kernel_size is not None and pool_stride is not None
            else nn.Identity()
        )
        self.layer = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            ),
            pool_layer,
        )

        self.lr = lr
        self.T = T
        self.threshold = loss_threshold
        self.v_threshold = v_threshold
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

        self.Cin = in_channels
        self.Cout = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.Hout = H
        self.Wout = W
        self.Hp = Hp
        self.Wp = Wp

    def cal_goodness(self, freq):
        return self.T * freq.pow(2)

    def _get_conv_weight(self):
        for module in self.layer.modules():
            if isinstance(module, nn.Conv2d):
                return module.weight
        raise RuntimeError("Conv2d layer not found.")

    def _begin_profile(self):
        device = self._get_conv_weight().device
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
        return device, use_cuda_mem_stat, base_alloc, base_reserved, start_time

    def _end_profile(self, profile_ctx):
        device, use_cuda_mem_stat, base_alloc, base_reserved, start_time = profile_ctx
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

        return peak_alloc, peak_reserved, elapsed_ms

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

    def _forward_spike_sequence(self, encoded):
        _, batch_size, _, _, _ = encoded.shape
        device = encoded.device
        patch = self.kernel_size * self.kernel_size * self.Cin
        output_spike = torch.zeros(
            self.T,
            batch_size,
            self.Cout,
            self.Hout,
            self.Wout,
            device=device,
        )
        pool_output_spike = torch.zeros(
            self.T,
            batch_size,
            self.Cout,
            self.Hp,
            self.Wp,
            device=device,
        )
        ln_mean = torch.zeros((batch_size,), device=device)
        ln_var = torch.zeros((batch_size,), device=device)

        for t in range(self.T):
            spike_out, ln_mean, ln_var = self.forward(encoded[t], ln_mean, ln_var)
            output_spike[t] = spike_out
            pool_output_spike[t] = self.layer[2](spike_out)

        input_spike_sum = encoded.sum(0)
        input_spike_sum_unfold = F.unfold(
            input_spike_sum,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.stride,
            padding=self.padding,
        )
        input_spike_sum_unfold = input_spike_sum_unfold.permute(1, 0, 2).reshape(
            patch,
            -1,
        )
        out_freq = output_spike.mean(0)
        goodness = self.cal_goodness(out_freq)
        return (
            input_spike_sum_unfold,
            pool_output_spike,
            out_freq,
            goodness,
            ln_mean,
            ln_var,
        )

    def _manual_gradient(
        self,
        input_spike_sum_unfold,
        out_freq,
        goodness,
        ln_var,
        ln_mean,
        batch_size,
        is_pos,
    ):
        profile_ctx = self._begin_profile()
        with torch.no_grad():
            weight_grad = gradient_calculation_cnn(
                input_spike_sum_unfold,
                out_freq,
                goodness,
                ln_var,
                ln_mean,
                self.threshold,
                self.v_threshold,
                batch_size,
                self.Cout,
                is_pos,
            )[0]
        weight_grad = weight_grad.view_as(self._get_conv_weight())
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        return weight_grad, peak_alloc, peak_reserved, elapsed_ms

    def _autograd_branch_comparison(
        self,
        input_spike_sum_unfold,
        out_freq,
        goodness,
        ln_var,
        ln_mean,
        batch_size,
        is_pos,
        *,
        retain_graph,
    ):
        _, loss = gradient_calculation_cnn(
            input_spike_sum_unfold,
            out_freq,
            goodness,
            ln_var,
            ln_mean,
            self.threshold,
            self.v_threshold,
            batch_size,
            self.Cout,
            is_pos,
        )

        profile_ctx = self._begin_profile()
        self.opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grad = self._get_conv_weight().grad.detach().clone()
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        self.opt.zero_grad()
        return grad, peak_alloc, peak_reserved, elapsed_ms

    def _apply_manual_update(self, weight_grad, frozen):
        if frozen:
            return
        with torch.no_grad():
            self._get_conv_weight().add_(self.lr * weight_grad)

    def _apply_autograd_update(self, pos_goodness, neg_goodness):
        profile_ctx = self._begin_profile()
        self.opt.zero_grad()
        loss = ff_pairwise_goodness_loss(
            pos_goodness,
            neg_goodness,
            self.threshold,
        )
        loss.backward()
        self.opt.step()
        peak_alloc, peak_reserved, _ = self._end_profile(profile_ctx)
        self.last_backward_peak_alloc_bytes = peak_alloc
        self.last_backward_peak_reserved_bytes = peak_reserved

    def forward(self, x, mean, var):
        x = self.layer[0](x)
        mean = (1 - 1 / self.T) * mean + (1 / self.T) * x.mean(dim=(1, 2, 3))
        var = (1 - 1 / self.T) * var + (1 / self.T) * x.var(dim=(1, 2, 3), unbiased=False)
        x = (
            self.v_threshold * (x - mean.view(-1, 1, 1, 1))
        ) / torch.sqrt(var.view(-1, 1, 1, 1) + 1e-5)
        x = self.layer[1](x)
        return x, mean, var

    def train_unsupervised(self, pos_encoded, neg_encoded, frozen):
        _, batch_size, _, _, _ = pos_encoded.shape
        self._reset_runtime_stats()
        needs_manual_grad = self.mode_config.uses_manual_update

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
            (
                pos_input_spike_sum_unfold,
                pos_pool_out,
                pos_out_freq,
                pos_goodness,
                pos_ln_mean,
                pos_ln_var,
            ) = self._forward_spike_sequence(pos_encoded)
            (
                pos_weight_grad,
                pos_manual_peak_alloc,
                pos_manual_peak_reserved,
                pos_manual_time_ms,
            ) = self._manual_gradient(
                pos_input_spike_sum_unfold,
                pos_out_freq,
                pos_goodness,
                pos_ln_var,
                pos_ln_mean,
                batch_size,
                True,
            )
            if self.mode_config.profiling.capture_autograd_comparison:
                # The comparison backward must run before the manual in-place
                # weight update, otherwise the saved graph sees a newer
                # parameter version and backward will fail.
                retain_graph_for_cmp = (
                    self.mode_config.hidden_layer_update_mode
                    == HIDDEN_LAYER_UPDATE_AUTOGRAD
                )
                (
                    pos_autograd_grad,
                    pos_bp_peak_alloc,
                    pos_bp_peak_reserved,
                    pos_bp_time_ms,
                ) = self._autograd_branch_comparison(
                    pos_input_spike_sum_unfold,
                    pos_out_freq,
                    pos_goodness,
                    pos_ln_var,
                    pos_ln_mean,
                    batch_size,
                    True,
                    retain_graph=retain_graph_for_cmp,
                )
            self._apply_manual_update(pos_weight_grad, frozen)
            functional.reset_net(self.layer)

            (
                neg_input_spike_sum_unfold,
                neg_pool_out,
                neg_out_freq,
                neg_goodness,
                neg_ln_mean,
                neg_ln_var,
            ) = self._forward_spike_sequence(neg_encoded)
            (
                neg_weight_grad,
                neg_manual_peak_alloc,
                neg_manual_peak_reserved,
                neg_manual_time_ms,
            ) = self._manual_gradient(
                neg_input_spike_sum_unfold,
                neg_out_freq,
                neg_goodness,
                neg_ln_var,
                neg_ln_mean,
                batch_size,
                False,
            )
            if self.mode_config.profiling.capture_autograd_comparison:
                (
                    neg_autograd_grad,
                    neg_bp_peak_alloc,
                    neg_bp_peak_reserved,
                    neg_bp_time_ms,
                ) = self._autograd_branch_comparison(
                    neg_input_spike_sum_unfold,
                    neg_out_freq,
                    neg_goodness,
                    neg_ln_var,
                    neg_ln_mean,
                    batch_size,
                    False,
                    retain_graph=retain_graph_for_cmp,
                )
            self._apply_manual_update(neg_weight_grad, frozen)
            functional.reset_net(self.layer)

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
                4.0
                * batch_size
                * self.Cout
                * self.Hout
                * self.Wout
                * self.kernel_size
                * self.kernel_size
                * self.Cin
            )

            if self.mode_config.profiling.capture_autograd_comparison:
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
        else:
            (
                pos_input_spike_sum_unfold,
                pos_pool_out,
                pos_out_freq,
                pos_goodness,
                pos_ln_mean,
                pos_ln_var,
            ) = self._forward_spike_sequence(pos_encoded)
            functional.reset_net(self.layer)
            (
                neg_input_spike_sum_unfold,
                neg_pool_out,
                neg_out_freq,
                neg_goodness,
                neg_ln_mean,
                neg_ln_var,
            ) = self._forward_spike_sequence(neg_encoded)
            functional.reset_net(self.layer)
            self._apply_autograd_update(pos_goodness, neg_goodness)

        return (
            pos_pool_out.detach(),
            pos_goodness.detach().mean().cpu().item(),
            pos_cos_sim,
            neg_pool_out.detach(),
            neg_goodness.detach().mean().cpu().item(),
            neg_cos_sim,
        )

    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        return self.train_unsupervised(pos_encoded, neg_encoded, frozen)

    def predict(self, x):
        T, B = x.shape[:2]
        out = torch.zeros(T, B, self.Cout, self.Hp, self.Wp, device=x.device)
        ln_mean = torch.zeros((B,), device=x.device)
        ln_var = torch.zeros((B,), device=x.device)
        for t in range(T):
            spike_out, ln_mean, ln_var = self.forward(x[t], ln_mean, ln_var)
            spike_out = self.layer[2](spike_out)
            out[t] = spike_out
        functional.reset_net(self.layer)
        return out


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


Net = ConvNet
