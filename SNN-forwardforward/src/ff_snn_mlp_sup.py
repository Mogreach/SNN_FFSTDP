from __future__ import annotations

"""
====================================================================
File          : ff_snn_mlp_sup.py
Description   : 定义一个前馈脉冲神经网络（FF-SNN）模型
                该模型由多个层组成，每层使用脉冲神经元进行计算。
                模型的训练和预测方法也在此文件中定义。
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from src.loss import (
    delta_loss_gradient_calculation_mlp,
    ff_supervised_delta_loss,
)
from spikingjelly.activation_based import (
    neuron,
    encoding,
    functional,
    surrogate,
    layer,
)
from src.experiment import ExperimentModeConfig, GradientProfilingSnapshot, StepResult
from src.generate_neg_sample import *


def spike_encoder(images: torch.Tensor, T: int) -> torch.Tensor:
    """
    将图像编码为 T 步脉冲序列。
    
    参数:
        images: torch.Tensor，形状为 [B, C, H, W]，像素值范围为 [0,1]
        T: int，总的时间步数
        
    返回:
        spike_train: torch.Tensor，形状为 [T, B, C, H, W]，脉冲序列（0 或 1）
    """
    B, C, H, W = images.shape
    spike_train = torch.zeros((T, B, C, H, W), device=images.device)
    # v_mem = torch.zeros((B, C, H, W), device=images.device)  # 初始化膜电位为0
    # for t in range(T):
    #     v_mem += images  # 每步累加像素值
    #     spike = (v_mem >= 1.0).to(torch.float)  # 触发放电
    #     spike_train[t] = spike
    #     v_mem = v_mem * (1.0 - spike)  # 膜电位重置：只有放电位置归零
    # Possion编码
    for t in range(T):
        spike_train[t] += encoding.PoissonEncoder()(images)
    return spike_train  # 形状为 [T, B, C, H, W]

class tdLayerNorm(nn.Module):
    def __init__(self, dim, v_threshold,eps=1e-5, alpha=1.0):
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

        # soft combination
        # return x + self.alpha * (ln_x - x)
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
        num_classes=10,
        mode_config: ExperimentModeConfig | None = None,
        device=None,
    ):
        super().__init__()
        self.T = T
        self.layers = []
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
        for d in range(len(dims) - 1):
            if(d==len(dims)-2):
                self.layers += nn.ModuleList(
                    [
                        OutputLayer(
                            in_features=sum(dims[1:d+1]),
                            out_features=dims[d + 1],
                            epoch=epoch,
                            T=T,
                            lr=lr,
                            v_threshold=v_threshold,
                            v_threshold_neg=v_threshold_neg,
                            tau=tau,
                            loss_threshold=loss_threshold,
                            mode_config=self.mode_config,
                        ).to(self.device)
                    ]
                )
            else:
                self.layers += nn.ModuleList(
                    [
                        Layer(
                            in_features=dims[d],
                            out_features=dims[d + 1],
                            epoch=epoch,
                            T=T,
                            lr=lr,
                            v_threshold=v_threshold,
                            v_threshold_neg=v_threshold_neg,
                            tau=tau,
                            loss_threshold=loss_threshold,
                            mode_config=self.mode_config,
                        ).to(self.device)
                    ]
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
            layer_manual_ops = getattr(layer_module, "last_manual_grad_ops_est", None)
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
        self.last_backward_cmp_peak_alloc_bytes = snapshot.backward_cmp_peak_alloc_bytes
        self.last_backward_cmp_peak_reserved_bytes = snapshot.backward_cmp_peak_reserved_bytes
        self.last_backward_cmp_time_ms = snapshot.backward_cmp_time_ms
        return snapshot
        # 通过goodness计算预测结果
    def predict_winner(self, x):
        label = torch.randint(0, self.num_classes, (x.shape[0],), device=x.device)
        h, _ = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
            type="SCFF",
        )
        # 频率编码
        h = spike_encoder(x, self.T)
        h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
        spike_in_of_output_layer = torch.empty((h.shape[0], h.shape[1], 0), device=h.device)
        # spike_in_of_output_layer = torch.cat((spike_in_of_output_layer,h),dim=2)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                spike_out = layer.predict(spike_in_of_output_layer) 
            else:
                h = layer.predict(h)
                spike_in_of_output_layer = torch.cat((spike_in_of_output_layer,h),dim=2)
        spike_out_sum = spike_out.sum(0)  # 计算输出层的总脉冲
        return spike_out_sum.argmax(1)
    def _encode_training_samples(self, x, label):
        x_pos, x_neg = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
        )
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        return x_pos_encoded.flatten(2), x_neg_encoded.flatten(2)

    def train_supervised(self, x, label, frozen) -> StepResult:
        input_pos, input_neg = self._encode_training_samples(x, label)
        return self._train_step(
            input_pos,
            input_neg,
            label,
            frozen,
        )

    def train_ff_stdp(self, x, label, frozen):
        return self.train_supervised(x, label, frozen)

    def _train_step(self, input_pos, input_neg, label, frozen) -> StepResult:
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
                layer_module.train_bp_stdp(pos_spike_in_of_output_layer, label)
                continue

            (
                input_pos,
                pos_goodness,
                pos_cos_sim,
                input_neg,
                neg_goodness,
                neg_cos_sim,
            ) = layer_module.train_supervised(input_pos, input_neg, frozen)
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
                f"layer_{i}": layer.state_dict() for i, layer in enumerate(self.layers)
            },
            "args": args,
        }
        torch.save(check_point, path)

    def load(self, path):
        check_point = torch.load(path)
        # 加载每一层的参数
        for i, layer in enumerate(self.layers):
            key = f"layer_{i}"
            if key in check_point["net"]:
                layer.load_state_dict(check_point["net"][key])
                print(f"[OK] Loaded {key}")
            else:
                print(f"[Skip] {key} not found in checkpoint, skipped.")
        # 打印加载的超参数
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
        # self.opt = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化权重，并添加一个正偏置
                nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.out_features))
                m.weight.data += 0.1  # 添加正偏置，确保权重平均值大于 0
    def cal_goodness(self, freq):
        # goodness = self.T * freq.pow(2)
        goodness = self.T * freq.abs().pow(2) * freq.sign()
        return goodness.mean(dim=1,keepdim=True)

    def forward(self, x, mean, var):
        # 对第1维度（通道维度）计算L2范数，然后进行归一化
        x = self.layer[0](x)   # Flatten
        x = self.layer[1](x)   # Linear
        mean = (1 - 1/self.T) * mean + (1/self.T) * x.mean(dim=1)
        var = (1 - 1/self.T) * var + (1/self.T) * x.var(dim=1, unbiased=False)
        x = ((0.9*self.v_threshold * (x - mean.view(-1,1))) / torch.sqrt(var.view(-1,1) + 1e-5))
        x = self.layer[2](x)   # IFNode  
        # plt.hist(x.detach().flatten().cpu().numpy(), bins=100, density=True)
        return x, mean, var

    def _get_linear_weight(self):
        for module in self.layer.modules():
            if isinstance(module, nn.Linear):
                return module.weight
        raise RuntimeError("Linear layer not found.")

    def _begin_profile(self):
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

    def _manual_delta_gradient(
        self,
        pos_input_spike_sum,
        pos_out_freq,
        pos_goodness,
        pos_ln_var,
        pos_ln_mean,
        neg_input_spike_sum,
        neg_out_freq,
        neg_goodness,
        neg_ln_var,
        neg_ln_mean,
        batch_size,
    ):
        profile_ctx = self._begin_profile()
        weight_grad, delta_loss = delta_loss_gradient_calculation_mlp(
            pos_input_spike_sum,
            pos_out_freq,
            pos_goodness,
            pos_ln_var,
            pos_ln_mean,
            neg_input_spike_sum,
            neg_out_freq,
            neg_goodness,
            neg_ln_var,
            neg_ln_mean,
            self.threshold,
            self.v_threshold,
            batch_size,
        )
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        return (weight_grad.detach(), delta_loss), peak_alloc, peak_reserved, elapsed_ms
    def _apply_manual_update(self, weight_grad, frozen):
        if frozen:
            return
        with torch.no_grad():
            # Keep the original direct weight update rule for the manual branch.
            self._get_linear_weight().add_(self.lr * weight_grad)
    def _apply_autograd_update(self, pos_goodness, neg_goodness):
        profile_ctx = self._begin_profile()
        self.opt.zero_grad()
        delta_loss = ff_supervised_delta_loss(
            pos_goodness,
            neg_goodness,
            self.threshold,
        )
        delta_loss.backward()
        self.opt.step()
        peak_alloc, peak_reserved, _ = self._end_profile(profile_ctx)
        self.last_backward_peak_alloc_bytes = peak_alloc
        self.last_backward_peak_reserved_bytes = peak_reserved
    def _autograd_delta_comparison(self, delta_loss):
        profile_ctx = self._begin_profile()
        self.opt.zero_grad()
        delta_loss.backward()
        grad = self._get_linear_weight().grad.detach().clone()
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        self.opt.zero_grad()
        return grad, peak_alloc, peak_reserved, elapsed_ms

    def train_supervised(self, pos_encoded, neg_encoded, frozen):
        _, batch_size, _ = pos_encoded.shape
        self._reset_runtime_stats()

        needs_manual_grad = (
            self.mode_config.uses_manual_update
        )

        weight_grad = None
        delta_loss = None

        pos_cos_sim = None
        neg_cos_sim = None

        manual_peak_alloc = None
        manual_peak_reserved = None
        manual_time_ms = None

        bp_peak_alloc = None
        bp_peak_reserved = None
        bp_time_ms = None

        # =========================================================
        # Forward pass
        # =========================================================
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

        # =========================================================
        # Manual gradient branch
        # =========================================================
        if needs_manual_grad:

            (
                manual_result,
                manual_peak_alloc,
                manual_peak_reserved,
                manual_time_ms,
            ) = self._manual_delta_gradient(
                pos_input_spike_sum,
                pos_out_freq,
                pos_goodness,
                pos_ln_var,
                pos_ln_mean,
                neg_input_spike_sum,
                neg_out_freq,
                neg_goodness,
                neg_ln_var,
                neg_ln_mean,
                batch_size,
            )

            weight_grad, delta_loss = manual_result

            self.last_manual_grad_peak_alloc_bytes = manual_peak_alloc
            self.last_manual_grad_peak_reserved_bytes = manual_peak_reserved
            self.last_manual_grad_time_ms = float(
                manual_time_ms or 0.0
            )

            self.last_manual_grad_ops_est = float(
                4.0 * batch_size * self.out_features * self.in_features
            )
            # -----------------------------------------------------
            # Optional autograd comparison
            # -----------------------------------------------------
            if self.mode_config.profiling.capture_autograd_comparison:
                # The comparison backward must run before the manual in-place
                # weight update, otherwise the saved graph sees a newer
                # parameter version and backward will fail.

                (
                    autograd_grad,
                    bp_peak_alloc,
                    bp_peak_reserved,
                    bp_time_ms,
                ) = self._autograd_delta_comparison(
                    delta_loss
                )

                self.last_backward_cmp_peak_alloc_bytes = (
                    bp_peak_alloc
                )
                self.last_backward_cmp_peak_reserved_bytes = (
                    bp_peak_reserved
                )
                self.last_backward_cmp_time_ms = float(
                    bp_time_ms or 0.0
                )

                cos_sim = torch.cosine_similarity(
                    autograd_grad.flatten(),
                    -weight_grad.flatten(),
                    dim=0,
                ).detach().cpu().item()

                pos_cos_sim = cos_sim
                neg_cos_sim = cos_sim

            # -----------------------------------------------------
            # Apply manual update
            # -----------------------------------------------------
            self._apply_manual_update(weight_grad, frozen)

        # =========================================================
        # Pure autograd branch
        # =========================================================
        else:
            self._apply_autograd_update(pos_goodness, neg_goodness)
        return (
            pos_output_spike.detach(),
            pos_goodness.detach().mean().cpu().item(),
            pos_cos_sim,
            neg_output_spike.detach(),
            neg_goodness.detach().mean().cpu().item(),
            neg_cos_sim,
        )

    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        return self.train_supervised(pos_encoded, neg_encoded, frozen)

    def predict(self, x):
        N = x.shape[1]
        g = torch.zeros(self.T, N, self.out_features, device=x.device)
        ln_mean = torch.zeros((N), device=x.device)
        ln_var = torch.zeros((N), device=x.device)
        for t in range(self.T):
            spike_out, ln_mean, ln_var = self.forward(x[t], ln_mean, ln_var)
            g[t] += spike_out
        functional.reset_net(self.layer)
        return g
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
        mode_config: ExperimentModeConfig | None = None,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            # neuron.IFNode(
            #     v_reset=None,
            #     v_threshold=v_threshold,
            #     surrogate_function=surrogate.ATan(),
            #     step_mode="s",
            # )
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
    def forward(self, x):
        # 对第1维度（通道维度）计算L2范数，然后进行归一化
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x)
    def train_bp_stdp(self,x_encoded, label):
        self._reset_runtime_stats()
        N = x_encoded.shape[1]
        device = x_encoded.device
        use_cuda_mem_stat = device.type == "cuda"
        output_spike = torch.zeros(self.T, N, self.out_features, device=device)
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
        h = x
        g = torch.zeros(self.T, x.shape[1], self.out_features, device=x.device)
        for t in range(self.T):
            spike_out = self.forward(h[t])
            g[t] += spike_out
        functional.reset_net(self.layer)
        return g
