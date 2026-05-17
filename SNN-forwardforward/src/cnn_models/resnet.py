from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.optim import Adam

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet

from src.cnn_models.common import FFNetDelegatorMixin
from src.experiment import (
    ExperimentModeConfig,
    ExperimentStrategyConfig,
    GradientProfilingSnapshot,
    StepResult,
)
from src.ff_strategies.goodness import (
    GOODNESS_SQUARE,
    compute_goodness,
)
from src.ff_strategies.objectives import (
    compute_hidden_pair_loss,
)
from src.ff_snn_cnn_sup import (
    OutputLayer,
    spike_encoder as supervised_spike_encoder,
)
from src.ff_snn_cnn_unsup import spike_encoder as unsupervised_spike_encoder
from src.generate_neg_sample import generate_pos_n_neg_sample


class OfficialSEWResNet18Backbone(sew_resnet.SEWResNet):
    """
    SpikingJelly official SEW-ResNet18 backbone with small-input adaptation.

    For MNIST / CIFAR-like resolutions we switch to a 3x3 stride-1 stem and
    remove the first max-pool, which is the standard small-image ResNet tweak.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        H: int,
        W: int,
        num_classes: int,
        v_threshold: float,
    ):
        self.input_channels = in_channels
        self.input_height = H
        self.input_width = W
        super().__init__(
            sew_resnet.BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            cnf="ADD",
            spiking_neuron=neuron.IFNode,
            v_reset=None,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            step_mode="s",
        )
        self._adapt_stem_to_project_inputs()

    def _adapt_stem_to_project_inputs(self) -> None:
        is_small_input = min(self.input_height, self.input_width) <= 32
        if is_small_input:
            self.conv1 = layer.Conv2d(
                self.input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = layer.Conv2d(
                self.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def pop_ff_hidden_modules(self) -> list[nn.Module]:
        hidden_modules: list[nn.Module] = []

        stem = nn.Sequential(self.conv1, self.bn1, self.sn1, self.maxpool)
        hidden_modules.append(stem)
        self.conv1 = nn.Identity()
        self.bn1 = nn.Identity()
        self.sn1 = nn.Identity()
        self.maxpool = nn.Identity()

        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(self, stage_name)
            hidden_modules.extend(list(stage.children()))
            setattr(self, stage_name, nn.Identity())

        self.avgpool = nn.Identity()
        self.fc = nn.Identity()
        return hidden_modules


class FFResidualBlockLayer(nn.Module):
    """
    One FF hidden layer backed by an official SEW-ResNet stem or residual block.

    For ResNet blocks we support two hidden-layer update paths:
    1. `autograd`: keep the current Adam-based local optimization behavior.
    2. `manual`: compute a block-local gradient with `torch.autograd.grad`,
       then apply it manually without calling the optimizer step.

    This keeps the project-level `manual` experiment interface available for
    official residual blocks, while being honest that the gradient source here
    is local autodiff rather than the single-convolution closed-form update
    used by the plain CNN implementation.
    """

    def __init__(
        self,
        *,
        layer_module: nn.Module,
        T: int,
        lr: float,
        loss_threshold: float,
        mode_config: ExperimentModeConfig | None = None,
        strategy_config: ExperimentStrategyConfig | None = None,
    ):
        super().__init__()
        self.layer = layer_module
        self.T = T
        self.lr = lr
        self.threshold = loss_threshold
        self.mode_config = mode_config or ExperimentModeConfig()
        self.strategy_config = strategy_config or ExperimentStrategyConfig()
        self.opt = Adam(self.layer.parameters(), lr=lr)
        self._reset_runtime_stats()

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

    def _get_reference_param(self):
        try:
            return next(self.layer.parameters())
        except StopIteration as exc:
            raise RuntimeError("Residual FF layer must contain trainable parameters.") from exc

    def _get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [param for param in self.layer.parameters() if param.requires_grad]

    def _begin_profile(self):
        device = self._get_reference_param().device
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

    def cal_goodness(self, freq):
        return compute_goodness(
            freq,
            T=self.T,
            strategy_name=self.strategy_config.goodness_strategy,
            default_strategy_name=GOODNESS_SQUARE,
        )

    def _forward_spike_sequence(self, encoded):
        outputs = []
        for t in range(self.T):
            outputs.append(self.layer(encoded[t]))
        spike_output = torch.stack(outputs, dim=0)
        out_freq = spike_output.mean(0)
        goodness = self.cal_goodness(out_freq)
        return spike_output, out_freq, goodness

    def _pairwise_loss(self, pos_goodness, neg_goodness):
        return compute_hidden_pair_loss(
            pos_goodness,
            neg_goodness,
            threshold=self.threshold,
            strategy_name=self.strategy_config.hidden_loss_strategy,
            mode_config=self.mode_config,
        )

    def _flatten_grads(self, grads: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([grad.reshape(-1) for grad in grads], dim=0)

    def _manual_local_grad(self, loss):
        params = self._get_trainable_params()
        profile_ctx = self._begin_profile()
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=self.mode_config.profiling.capture_autograd_comparison,
            allow_unused=True,
        )
        manual_grads = [
            torch.zeros_like(param) if grad is None else grad.detach()
            for param, grad in zip(params, grads)
        ]
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        return params, manual_grads, peak_alloc, peak_reserved, elapsed_ms

    def _apply_manual_update(self, params, grads, frozen):
        if frozen:
            return
        with torch.no_grad():
            for param, grad in zip(params, grads):
                param.add_(-self.lr * grad)

    def _autograd_local_grad_comparison(self, loss, params):
        profile_ctx = self._begin_profile()
        self.opt.zero_grad()
        loss.backward()
        autograd_grads = [
            torch.zeros_like(param)
            if param.grad is None
            else param.grad.detach().clone()
            for param in params
        ]
        peak_alloc, peak_reserved, elapsed_ms = self._end_profile(profile_ctx)
        self.opt.zero_grad()
        return autograd_grads, peak_alloc, peak_reserved, elapsed_ms

    def _train_pair(self, pos_encoded, neg_encoded, frozen):
        self._reset_runtime_stats()
        pos_out, _, pos_goodness = self._forward_spike_sequence(pos_encoded)
        functional.reset_net(self.layer)
        neg_out, _, neg_goodness = self._forward_spike_sequence(neg_encoded)
        functional.reset_net(self.layer)

        pos_cos_sim = None
        neg_cos_sim = None
        loss = self._pairwise_loss(pos_goodness, neg_goodness)

        if self.mode_config.uses_manual_update:
            (
                params,
                manual_grads,
                manual_peak_alloc,
                manual_peak_reserved,
                manual_time_ms,
            ) = self._manual_local_grad(loss)
            self.last_manual_grad_peak_alloc_bytes = manual_peak_alloc
            self.last_manual_grad_peak_reserved_bytes = manual_peak_reserved
            self.last_manual_grad_time_ms = float(manual_time_ms or 0.0)
            self.last_manual_grad_ops_est = float(
                sum(int(param.numel()) for param in params)
            )

            if self.mode_config.profiling.capture_autograd_comparison:
                (
                    autograd_grads,
                    bp_peak_alloc,
                    bp_peak_reserved,
                    bp_time_ms,
                ) = self._autograd_local_grad_comparison(loss, params)
                self.last_backward_cmp_peak_alloc_bytes = bp_peak_alloc
                self.last_backward_cmp_peak_reserved_bytes = bp_peak_reserved
                self.last_backward_cmp_time_ms = float(bp_time_ms or 0.0)

                flat_manual = self._flatten_grads(manual_grads)
                flat_autograd = self._flatten_grads(autograd_grads)
                cos_sim = torch.cosine_similarity(
                    flat_autograd,
                    flat_manual,
                    dim=0,
                ).detach().cpu().item()
                pos_cos_sim = cos_sim
                neg_cos_sim = cos_sim

            self._apply_manual_update(params, manual_grads, frozen)

        elif not frozen:
            profile_ctx = self._begin_profile()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            peak_alloc, peak_reserved, _ = self._end_profile(profile_ctx)
            self.last_backward_peak_alloc_bytes = peak_alloc
            self.last_backward_peak_reserved_bytes = peak_reserved

        return (
            pos_out.detach(),
            pos_goodness.detach().mean().cpu().item(),
            pos_cos_sim,
            neg_out.detach(),
            neg_goodness.detach().mean().cpu().item(),
            neg_cos_sim,
        )

    def train_supervised(self, pos_encoded, neg_encoded, frozen):
        return self._train_pair(pos_encoded, neg_encoded, frozen)

    def train_unsupervised(self, pos_encoded, neg_encoded, frozen):
        return self._train_pair(pos_encoded, neg_encoded, frozen)

    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        return self._train_pair(pos_encoded, neg_encoded, frozen)

    def predict(self, x):
        outputs = []
        for t in range(self.T):
            outputs.append(self.layer(x[t]))
        spike_output = torch.stack(outputs, dim=0)
        functional.reset_net(self.layer)
        return spike_output


class OfficialResNetFFCore(nn.Module):
    def __init__(
        self,
        *,
        hidden_layers: list[FFResidualBlockLayer],
        spike_encoder_fn,
        sample_type: str,
        tau,
        epoch,
        T,
        lr,
        v_threshold,
        v_threshold_neg,
        loss_threshold,
        num_classes,
        in_channels,
        H,
        W,
        mode_config: ExperimentModeConfig | None = None,
        strategy_config: ExperimentStrategyConfig | None = None,
        device=None,
    ):
        super().__init__()
        self.T = T
        self.loss_threshold = loss_threshold
        self.num_classes = num_classes
        self.mode_config = mode_config or ExperimentModeConfig()
        self.strategy_config = strategy_config or ExperimentStrategyConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.spike_encoder_fn = spike_encoder_fn
        self.sample_type = sample_type

        self.last_backward_peak_alloc_bytes = None
        self.last_backward_peak_reserved_bytes = None
        self.last_manual_grad_peak_alloc_bytes = None
        self.last_manual_grad_peak_reserved_bytes = None
        self.last_manual_grad_time_ms = None
        self.last_manual_grad_ops_est = None
        self.last_backward_cmp_peak_alloc_bytes = None
        self.last_backward_cmp_peak_reserved_bytes = None
        self.last_backward_cmp_time_ms = None

        self.layers = nn.ModuleList(layer_module.to(self.device) for layer_module in hidden_layers)
        input_feature_of_linear = self._infer_output_feature_count(
            in_channels=in_channels,
            H=H,
            W=W,
        )
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
                mode_config=self.mode_config,
            ).to(self.device)
        )

    def _infer_output_feature_count(self, *, in_channels: int, H: int, W: int) -> int:
        dummy = torch.zeros((1, in_channels, H, W), device=self.device)
        feature_count = 0
        with torch.no_grad():
            for hidden_layer in self.layers:
                dummy = hidden_layer.layer(dummy)
                functional.reset_net(hidden_layer.layer)
                feature_count += int(dummy.flatten(1).shape[1])
        return feature_count

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

    def _encode_training_samples(self, x, label):
        x_pos, x_neg = generate_pos_n_neg_sample(
            x,
            label,
            num_classes=self.num_classes,
            strategy_name=self.strategy_config.neg_sample_strategy or self.sample_type,
            mode_config=self.mode_config,
        )
        return self.spike_encoder_fn(x_pos, self.T), self.spike_encoder_fn(x_neg, self.T)

    def predict_multiple(self, x):
        if self.mode_config.is_supervised:
            return self.predict_winner(x)

        goodness_per_label = []
        for label_idx in range(self.num_classes):
            goodness = []
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
                strategy_name=self.strategy_config.neg_sample_strategy or self.sample_type,
                mode_config=self.mode_config,
            )
            h = self.spike_encoder_fn(h, self.T)
            for hidden_layer in self.layers[:-1]:
                h = hidden_layer.predict(h)
                freq = h.mean(0)
                goodness.append(hidden_layer.cal_goodness(freq).flatten(1).sum(1))

            label_goodness = (
                torch.stack(goodness, dim=1).sum(1)
                if goodness
                else torch.zeros(x.shape[0], device=x.device)
            )
            goodness_per_label.append(label_goodness.unsqueeze(1))

        return torch.cat(goodness_per_label, dim=1).argmax(1)

    def predict_winner(self, x):
        h = self.spike_encoder_fn(x, self.T)
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
                    (spike_in_of_output_layer, h.flatten(2)),
                    dim=2,
                )
        return spike_out.sum(0).argmax(1)

    def train_supervised(self, x, label, frozen) -> StepResult:
        input_pos, input_neg = self._encode_training_samples(x, label)
        return self._train_step(input_pos, input_neg, label, frozen)

    def train_unsupervised(self, x, label, frozen) -> StepResult:
        input_pos, input_neg = self._encode_training_samples(x, label)
        return self._train_step(input_pos, input_neg, label, frozen)

    def train_ff_stdp(self, x, label, frozen):
        if self.mode_config.is_supervised:
            return self.train_supervised(x, label, frozen)
        return self.train_unsupervised(x, label, frozen)

    def _train_step(self, input_pos, input_neg, label, frozen) -> StepResult:
        T, B = input_pos.shape[:2]
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

            hidden_train_fn = (
                layer_module.train_supervised
                if self.mode_config.is_supervised
                else layer_module.train_unsupervised
            )
            (
                input_pos,
                pos_goodness,
                pos_cos_sim,
                input_neg,
                neg_goodness,
                neg_cos_sim,
            ) = hidden_train_fn(input_pos, input_neg, frozen)
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


class OfficialSEWResNet18FFAdapter(OfficialSEWResNet18Backbone, FFNetDelegatorMixin):
    def __init__(
        self,
        *,
        in_channels: int,
        H: int,
        W: int,
        mode_config,
        strategy_config,
        device=None,
        **kwargs,
    ):
        self.mode_config = mode_config
        self.strategy_config = strategy_config
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(
            in_channels=in_channels,
            H=H,
            W=W,
            num_classes=kwargs["num_classes"],
            v_threshold=kwargs["v_threshold"],
        )

        hidden_layers = [
            FFResidualBlockLayer(
                layer_module=module,
                T=kwargs["T"],
                lr=kwargs["lr"],
                loss_threshold=kwargs["loss_threshold"],
                mode_config=mode_config,
                strategy_config=self.strategy_config,
            )
            for module in self.pop_ff_hidden_modules()
        ]

        spike_encoder_fn = (
            unsupervised_spike_encoder if mode_config.is_unsupervised else supervised_spike_encoder
        )
        sample_type = "embed_label_onehot" if mode_config.is_unsupervised else "SCFF"
        self.ff_net = OfficialResNetFFCore(
            hidden_layers=hidden_layers,
            spike_encoder_fn=spike_encoder_fn,
            sample_type=sample_type,
            tau=kwargs["tau"],
            epoch=kwargs["epoch"],
            T=kwargs["T"],
            lr=kwargs["lr"],
            v_threshold=kwargs["v_threshold"],
            v_threshold_neg=kwargs["v_threshold_neg"],
            loss_threshold=kwargs["loss_threshold"],
            num_classes=kwargs["num_classes"],
            in_channels=in_channels,
            H=H,
            W=W,
            mode_config=mode_config,
            strategy_config=self.strategy_config,
            device=self.device,
        )
        self.to(self.device)


class ResNetSupervisedNet(OfficialSEWResNet18FFAdapter):
    pass


class ResNetUnsupervisedNet(OfficialSEWResNet18FFAdapter):
    pass


def build_resnet_model(
    *,
    mode_config,
    strategy_config,
    in_channels: int,
    H: int,
    W: int,
    **kwargs,
):
    net_cls = ResNetUnsupervisedNet if mode_config.is_unsupervised else ResNetSupervisedNet
    return net_cls(
        in_channels=in_channels,
        H=H,
        W=W,
        mode_config=mode_config,
        strategy_config=strategy_config,
        **kwargs,
    )
