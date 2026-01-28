"""
====================================================================
File          : ff_snn_cnn.py
Description   : FF-STDP for CNN
Author        : Morgreach
Version       : 1.0.0
Date          : 2026-01-25
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""


import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from spikingjelly.activation_based import (
    neuron,
    encoding,
    functional,
    surrogate,
    layer,
    monitor,
    learning,
)
from src.loss import gradient_calculation_cnn, delta_loss_gradient_calculation_cnn
from src.generate_neg_sample import *
def pos_derivative(x, theta):
    """
    计算 log(1 + exp(-x + theta)) 关于 x 的导数。

    参数:
        x (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = -1 / (1 + torch.exp(x - theta))
    
    # 返回导数
    return sigmoid
def neg_derivative(y, theta):
    """
    计算 log(1 + exp(y - theta)) 关于 y 的导数。

    参数:
        y (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = 1 / (1 + torch.exp(theta - y))
    
    # 返回导数
    return sigmoid


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

class ConvNet(torch.nn.Module):
    def __init__(
        self,
        conv_cfg,              # 结构列表
        T,
        epoch,
        lr,
        tau,
        v_threshold,
        loss_threshold,
        num_classes=10,
        H=28,
        W=28,
    ):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.num_classes = num_classes
        self.T = T
        self.layers = []
        input_feature_of_linear = 0
        for (in_ch, out_ch, k, s, p) in conv_cfg:
            # Conv2d(k,s,p) : (W + 2*p - k) // s + 1
            H = ((H + 2*p - k) // s + 1)
            W = ((W + 2*p - k) // s + 1)
            # MaxPool(k=2,s=2) : (W + 2*p - k) // s + 1
            Hp = (H - 2) // 2 + 1
            Wp = (W - 2) // 2 + 1
            input_feature_of_linear += out_ch * Hp * Wp
            self.layers += nn.ModuleList(
                [
                    ConvLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        H = H,
                        W = W,
                        Hp = Hp,
                        Wp = Wp,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                        epoch=epoch,
                        T=T,
                        lr=lr,
                        v_threshold=v_threshold,
                        tau=tau,
                        loss_threshold=loss_threshold,
                    ).cuda()
                ]
            )
            H = Hp
            W = Wp
        self.layers += nn.ModuleList(
                [
                    OutputLayer(
                        in_features=input_feature_of_linear,
                        out_features=num_classes,
                        epoch=epoch,
                        T=T,
                        lr=lr,
                        v_threshold=v_threshold,
                        tau=tau,
                        loss_threshold=loss_threshold,
                    ).cuda()
                ]
            )
    def predict_winner(self, x):
        label = torch.randint(0, self.num_classes, (x.shape[0],))
        h, _ = generate_pos_n_neg_sample(x, label, num_classes=self.num_classes, type="SCFF")
        # 频率编码
        h = spike_encoder(x, self.T)
        # h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
        spike_in_of_output_layer = torch.empty((h.shape[0],h.shape[1],0)).cuda()
        # spike_in_of_output_layer = torch.cat((spike_in_of_output_layer,h),dim=2)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                spike_out = layer.predict(spike_in_of_output_layer) 
            else:
                h = layer.predict(h)
                spike_in_of_output_layer = torch.cat((spike_in_of_output_layer, h.flatten(2)),dim=2)
        spike_out_sum = spike_out.sum(0)  # 计算输出层的总脉冲
        return spike_out_sum.argmax(1)
    def train_ff_stdp(self, x, label, frozen):
        x_pos, x_neg = generate_pos_n_neg_sample(x, label, num_classes=self.num_classes, type="SCFF")
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        in_pos = x_pos_encoded
        in_neg = x_neg_encoded
        spike_input_pos = in_pos
        spike_input_neg = in_neg
        goodness_pos, cos_pos, spike_out_pos, goodness_neg, cos_neg, spike_out_neg = self.train_ff_stdp_step(spike_input_pos, spike_input_neg, label, frozen)    
        return goodness_pos, goodness_neg, cos_pos, cos_neg, spike_out_pos, spike_out_neg
    def train_ff_stdp_step(self, input_pos, input_neg, label, frozen):
        T, B, C, H, W = input_pos.shape
        pos_goodness_per_layer = []
        neg_goodness_per_layer = []
        pos_cos_sim_per_layer = []
        neg_cos_sim_per_layer = []
        pos_spike_out_per_layer = []
        neg_spike_out_per_layer = []
        pos_spike_in_of_output_layer = torch.empty((T,B,0)).cuda()
        neg_spike_in_of_output_layer = torch.empty((T,B,0)).cuda()
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                pos_spike_output = layer.train_bp_stdp(pos_spike_in_of_output_layer, label)
                neg_spike_output = pos_spike_output
                pos_spike_out_per_layer.append(pos_spike_output.mean().detach().cpu())
                neg_spike_out_per_layer.append(neg_spike_output.mean().detach().cpu())
                # neg_spike_output = layer.train_bp_stdp(neg_spike_in_of_output_layer, label)
            else:
                input_pos, pos_g , pos_cos_sim, input_neg, neg_g, neg_cos_sim = layer.train_ff_stdp(input_pos, input_neg, frozen)
                pos_goodness_per_layer.append(pos_g.mean().item())
                neg_goodness_per_layer.append(neg_g.mean().item())
                pos_cos_sim_per_layer.append(pos_cos_sim)
                neg_cos_sim_per_layer.append(neg_cos_sim)
                pos_spike_out_per_layer.append(input_pos.mean().detach().cpu())
                neg_spike_out_per_layer.append(input_neg.mean().detach().cpu())
                pos_spike_in_of_output_layer = torch.cat((pos_spike_in_of_output_layer,input_pos.flatten(2)),dim=2)
                neg_spike_in_of_output_layer = torch.cat((neg_spike_in_of_output_layer,input_neg.flatten(2)),dim=2)
        return pos_goodness_per_layer, pos_cos_sim_per_layer , pos_spike_out_per_layer, neg_goodness_per_layer, neg_cos_sim_per_layer, neg_spike_out_per_layer

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
        epoch,
        T,
        lr,
        v_threshold,
        tau,
        loss_threshold,
    ):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            ),
            layer.MaxPool2d(2, 2),
        )

        self.lr = lr
        self.T = T
        self.threshold = loss_threshold
        self.v_threshold = v_threshold
        self.opt = Adam(self.parameters(), lr=lr)

        self.Cin = in_channels
        self.Cout = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Hout = H
        self.Wout = W
        self.Hp = Hp
        self.Wp = Wp
    def cal_goodness(self, freq):
        # freq: [B, C, H, W]
        g = self.T * freq.pow(2)
        return g
    def forward(self, x, mean, var):
        # x: [B, C, H, W]
        x = self.layer[0](x)   # Conv2d
        mean = (1 - 1/self.T) * mean + (1/self.T) * x.mean(dim=(1,2,3))
        var = (1 - 1/self.T) * var + (1/self.T) * x.var(dim=(1,2,3), unbiased=False)
        x = ((self.v_threshold * (x - mean.view(-1,1,1,1))) / torch.sqrt(var.view(-1,1,1,1) + 1e-5))
        x = self.layer[1](x) # IFNode   
        return x, mean, var
    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        # pos_encoded: [T, B, Cin, Hin, Win]
        T, B, Cin, Hin, Win = pos_encoded.shape
        patch = self.kernel_size*self.kernel_size * Cin
        pos_out = torch.empty(T, B, self.Cout, self.Hout, self.Wout).cuda()
        neg_out = torch.empty(T, B, self.Cout, self.Hout, self.Wout).cuda()
        pos_pool_out = torch.empty(T, B, self.Cout, self.Hp, self.Wp).cuda()
        neg_pool_out = torch.empty(T, B, self.Cout, self.Hp, self.Wp).cuda()
        pos_input_spike_sum = pos_encoded.sum(0)  # [B,Cin,Hin,Win]
        neg_input_spike_sum = neg_encoded.sum(0)  # [B,Cin,Hin,Win]
        #===========================================================
        # Positive sample processing
        #===========================================================
        pos_ln_mean = torch.zeros((B),device='cuda')
        pos_ln_var = torch.zeros((B),device='cuda')
        for t in range(T):
            pos_spike, pos_ln_mean, pos_ln_var = self.forward(pos_encoded[t], pos_ln_mean, pos_ln_var)
            pos_out[t] = pos_spike
            pos_pool_out[t] = self.layer[2](pos_out[t])  # MaxPool2d
        #------------------------------------------------------
        # 突触前偏导矩阵：提取卷积patch [B, Cin*Kh*Kw, Hout*Wout]
        #------------------------------------------------------
        pos_input_spike_sum_unfold = F.unfold(pos_input_spike_sum,kernel_size=(self.kernel_size, self.kernel_size),stride=self.stride,padding=self.padding)
        # pos_input_spike_sum_unfold [B, Cin*Kh*Kw, Hout*Wout] → [Cin*Kh*Kw, B*Hout*Wout]
        pos_input_spike_sum_unfold = pos_input_spike_sum_unfold.permute(1, 0, 2).reshape(patch, -1)
        #------------------------------------------------------
        # 突触后偏导矩阵：
        #------------------------------------------------------
        pos_freq = pos_out.mean(0)  # [B,Cout,Hout,Wout]
        pos_goodness = self.cal_goodness(pos_freq)
        # pos_weight_grad, pos_loss = gradient_calculation_cnn(pos_input_spike_sum_unfold, pos_freq, pos_goodness, pos_ln_var, pos_ln_mean,
        #                                                     self.threshold, self.v_threshold, B, self.Cout, True)
        # pos_loss.backward()
        # with torch.no_grad():
        #     for m in self.layer.modules():
        #         if isinstance(m, nn.Conv2d):
        #             w_grad = m.weight.grad
        #             pos_cos_sim = torch.cosine_similarity(w_grad.flatten(),-pos_weight_grad.flatten(),dim=0)
        # self.opt.zero_grad()
        functional.reset_net(self.layer)

        #===========================================================
        # Negative sample processing
        #===========================================================
        neg_ln_mean = torch.zeros((B),device='cuda')
        neg_ln_var = torch.zeros((B),device='cuda')
        for t in range(T):
            neg_spike, neg_ln_mean, neg_ln_var = self.forward(neg_encoded[t], neg_ln_mean, neg_ln_var)
            neg_out[t] = neg_spike
            neg_pool_out[t] = self.layer[2](neg_out[t])  # MaxPool2d
        #------------------------------------------------------
        # 突触前偏导矩阵：提取卷积patch [B, Cin*Kh*Kw, Hout*Wout]
        #------------------------------------------------------
        neg_input_spike_sum_unfold = F.unfold(neg_input_spike_sum,kernel_size=(self.kernel_size, self.kernel_size),stride=self.stride,padding=self.padding)
        # neg_input_spike_sum_unfold [B, Cin*Kh*Kw, Hout*Wout] → [Cin*Kh*Kw, B*Hout*Wout]
        neg_input_spike_sum_unfold = neg_input_spike_sum_unfold.permute(1, 0, 2).reshape(patch, -1)
        #------------------------------------------------------
        # 突触后偏导矩阵：
        #------------------------------------------------------
        neg_freq = neg_out.mean(0)  # [B,Cout,Hout,Wout]
        neg_goodness = self.cal_goodness(neg_freq)
        # neg_weight_grad, neg_loss = gradient_calculation_cnn(neg_input_spike_sum_unfold, neg_freq, neg_goodness, neg_ln_var, neg_ln_mean,
        #                                                     self.threshold, self.v_threshold, B, self.Cout, False)
        # neg_loss.backward()
        # with torch.no_grad():
        #     for m in self.layer.modules():
        #         if isinstance(m, nn.Conv2d):
        #             w_grad = m.weight.grad
        #             neg_cos_sim = torch.cosine_similarity(w_grad.flatten(),-neg_weight_grad.flatten(),dim=0)
        # self.opt.zero_grad()
        # weight_grad = pos_weight_grad + neg_weight_grad
        functional.reset_net(self.layer)


        # Delta loss processing
        weight_grad, delta_loss = delta_loss_gradient_calculation_cnn(pos_input_spike_sum_unfold, pos_freq, pos_goodness, pos_ln_var, pos_ln_mean,
                                                                      neg_input_spike_sum_unfold, neg_freq, neg_goodness, neg_ln_var, neg_ln_mean,
                                                                      self.threshold, self.v_threshold, B, self.Cout)
        delta_loss.backward()
        with torch.no_grad():
            for m in self.layer.modules():
                if isinstance(m, nn.Conv2d):
                    w_grad = m.weight.grad
                    pos_cos_sim = torch.cosine_similarity(w_grad.flatten(), -weight_grad.flatten(), dim=0)
                    neg_cos_sim = torch.cosine_similarity(w_grad.flatten(), -weight_grad.flatten(), dim=0)
        self.opt.zero_grad()

        # Update weights
        if frozen:
            pass
        else:
            with torch.no_grad():
                for m in self.layer.modules():
                    if isinstance(m, nn.Conv2d):         
                        weight_grad = weight_grad.view(m.weight.shape)
                        m.weight.data += self.lr * weight_grad            
        return pos_pool_out.detach(), pos_goodness.detach().mean(1).cpu(),pos_cos_sim.detach().cpu().item(), neg_pool_out.detach(), neg_goodness.detach().mean(1).cpu(),neg_cos_sim.detach().cpu().item()

    def predict(self, x):
        T, B = x.shape[:2]
        out = torch.zeros(T, B, self.Cout, self.Hp, self.Wp).cuda()
        ln_mean = torch.zeros((B)).cuda()
        ln_var = torch.zeros((B)).cuda()
        for t in range(T):
            spike_out, ln_mean, ln_var = self.forward(x[t], ln_mean, ln_var)
            spike_out = self.layer[2](spike_out)  # MaxPool2d
            out[t] = spike_out
        functional.reset_net(self.layer)
        return out

class OutputLayer(nn.Module):
    def __init__(
        self, in_features, out_features, epoch, T, lr, v_threshold, tau, loss_threshold
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
        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)
    def forward(self, x):
        # 对第1维度（通道维度）计算L2范数，然后进行归一化
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x)
    def train_bp_stdp(self,x_encoded, label):
        N = x_encoded.shape[1]
        output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        for t in range(self.T):
            output_spike[t] += self.forward(x_encoded[t])
        spike_freq = output_spike.mean(0)
        self.opt.zero_grad()
        loss = F.cross_entropy(spike_freq.view(-1, self.out_features), label.view(-1))
        loss.backward()
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
        g = torch.zeros(self.T, x.shape[1], self.out_features).cuda()
        for t in range(self.T):
            spike_out = self.forward(h[t])
            g[t] += spike_out
        functional.reset_net(self.layer)
        return g