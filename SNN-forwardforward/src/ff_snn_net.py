"""
====================================================================
File          : ff_snn_net.py
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
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD
from src.loss import Custom_Loss
from spikingjelly.activation_based import (
    neuron,
    encoding,
    functional,
    surrogate,
    layer,
    monitor,
    learning,
)
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
    def __init__(self, dims, tau, epoch, T, lr, v_threshold, v_threshold_neg, opt, loss_threshold):
        super().__init__()
        self.T = T
        self.layers = []
        self.loss_threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
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
                        ).cuda()
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
                        ).cuda()
                    ]
                )
    # 通过goodness计算预测结果
    def predict_multiple(self, x):
        goodness_per_label = []
        # goodness_per_label = 0  # 选择输出频率最大
        for label in range(10):
            goodness = []
            label = torch.full((x.shape[0],), label)
            h, _ = generate_pos_n_neg_sample(x, label, num_classes=10, type="embed_label_onehot")
            h = spike_encoder(h, self.T)
            h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
            spike_in_of_label = h[:,:,0:10]
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    break
                h = layer.predict(h)
                freq = h.mean(0)  # 计算每层的平均频率
                goodness = goodness + [layer.cal_goodness(freq).sum(1)] # 对每个样本的单层goodness求和
                # h = torch.cat((h, spike_in_of_label),dim=2)
            goodness_per_label += [sum(goodness).unsqueeze(1)] # 对所有层求和优度值
        goodness_of_all_label = torch.cat(goodness_per_label, 1)# 拼接所有标签编码对应优度值
        return goodness_of_all_label.argmax(1)
        # 通过goodness计算预测结果
    def predict_winner(self, x):
        label = torch.randint(0, 10, (x.shape[0],))
        h, _ = generate_pos_n_neg_sample(x, label, num_classes=10, type="SCFF")
        # 频率编码
        h = spike_encoder(x, self.T)
        h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
        spike_in_of_output_layer = torch.empty((h.shape[0],h.shape[1],0)).cuda()
        # spike_in_of_output_layer = torch.cat((spike_in_of_output_layer,h),dim=2)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                spike_out = layer.predict(spike_in_of_output_layer) 
            else:
                h = layer.predict(h)
                spike_in_of_output_layer = torch.cat((spike_in_of_output_layer,h),dim=2)
        spike_out_sum = spike_out.sum(0)  # 计算输出层的总脉冲
        return spike_out_sum.argmax(1)
    def train(self, x_pos, x_neg, y, layer_idx):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                train_mode = True
                h_pos, h_neg, loss = layer.train(h_pos, h_neg, y, train_mode)
                break
            else:
                train_mode = False
                h_pos, h_neg, loss = layer.train(h_pos, h_neg, y, train_mode)
        return loss
    def train_ff_stdp(self, x, label, frozen):
        x_pos, x_neg = generate_pos_n_neg_sample(x, label, num_classes=10)
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        in_pos = x_pos_encoded.flatten(2)
        in_neg = x_neg_encoded.flatten(2)
        spike_input_pos = in_pos
        spike_input_neg = in_neg
        goodness_pos, cos_pos, spike_out_pos, goodness_neg, cos_neg, spike_out_neg = self.train_ff_stdp_step(spike_input_pos, spike_input_neg, label, frozen)    
        return goodness_pos, goodness_neg, cos_pos, cos_neg, spike_out_pos, spike_out_neg
    def train_ff_stdp_step(self, input_pos, input_neg, label, frozen):
        T, B, _ = input_pos.shape
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
                pos_spike_in_of_output_layer = torch.cat((pos_spike_in_of_output_layer,input_pos),dim=2)
                neg_spike_in_of_output_layer = torch.cat((neg_spike_in_of_output_layer,input_neg),dim=2)
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


class Layer(nn.Module):
    def __init__(
        self, in_features, out_features, epoch, T, lr, v_threshold, v_threshold_neg, tau, loss_threshold
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
    def train_ff_stdp(self, pos_encoded, neg_encoded, frozen):
        _, N, __ = pos_encoded.shape
        # Positive sample processing
        pos_input_spike_sum = pos_encoded.sum(0)
        pos_output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        pos_ln_mean = torch.zeros((N)).cuda()
        pos_ln_var = torch.zeros((N)).cuda()
        for t in range(self.T):
            pos_spike, pos_ln_mean, pos_ln_var = self.forward(pos_encoded[t], pos_ln_mean, pos_ln_var)
            pos_output_spike[t] += pos_spike
        pos_out_freq = pos_output_spike.mean(0)
        self.opt.zero_grad()
        pos_goodness = self.cal_goodness(pos_out_freq)
        # pos_L_to_s_grad = 2 * pos_out_freq*pos_derivative(pos_goodness,self.threshold) * (self.v_threshold/ torch.sqrt(pos_ln_var.view(N,1) + 1e-5)) # * pos_ln_mean.view(N,1)
        # pos_L_to_s_grad = 2 * pos_out_freq*pos_derivative(pos_goodness,self.threshold) * (self.v_threshold/ torch.sqrt(pos_ln_var.view(N,1) + 1e-5))
        # pos_L_to_s_grad = pos_L_to_s_grad.transpose(0,1)
        # pos_loss = torch.log(1 + torch.exp(-pos_goodness + self.threshold)).mean()
        # pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N
        # pos_loss.backward()
        # with torch.no_grad():
        #     for m in self.layer.modules():
        #         if isinstance(m, nn.Linear):
        #             w_grad = m.weight.grad
        #             pos_cos_sim = torch.cosine_similarity(w_grad.flatten(),-pos_weight_grad.flatten(),dim=0)
        functional.reset_net(self.layer)
        # Negative sample processing
        neg_input_spike_sum = neg_encoded.sum(0)
        neg_output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        neg_ln_mean = torch.zeros((N)).cuda()
        neg_ln_var = torch.zeros((N)).cuda()
        for t in range(self.T):
            neg_spike, neg_ln_mean, neg_ln_var = self.forward(neg_encoded[t], neg_ln_mean, neg_ln_var)
            neg_output_spike[t] += neg_spike
        neg_out_freq = neg_output_spike.mean(0)
        self.opt.zero_grad()
        neg_goodness = self.cal_goodness(neg_out_freq)
        # neg_L_to_s_grad = 2*neg_out_freq*neg_derivative(neg_goodness,self.threshold) * (self.v_threshold / torch.sqrt(neg_ln_var.view(N,1) + 1e-5)) # * neg_ln_mean.view(N,1)
        # neg_L_to_s_grad = neg_L_to_s_grad.transpose(0,1)
        # neg_loss = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        # neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N
        # neg_loss.backward()
        # with torch.no_grad():
        #     for m in self.layer.modules():
        #         if isinstance(m, nn.Linear):
        #             w_grad = m.weight.grad
        #             neg_cos_sim = torch.cosine_similarity(w_grad.flatten(),-1*neg_weight_grad.flatten(),dim=0)
        functional.reset_net(self.layer)

        delta = 8*(pos_goodness - neg_goodness)
        self.opt.zero_grad()
        pos_L_to_s_grad = 8 * pos_derivative(delta,0) * 2 * pos_out_freq  * (self.v_threshold / torch.sqrt(pos_ln_var.view(N,1) + 1e-5))
        pos_L_to_s_grad = pos_L_to_s_grad.transpose(0,1)
        pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N
       
        neg_L_to_s_grad = -8 * pos_derivative(delta,0) * 2 * neg_out_freq * (self.v_threshold / torch.sqrt(neg_ln_var.view(N,1) + 1e-5))
        neg_L_to_s_grad = neg_L_to_s_grad.transpose(0,1)
        neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N
        delta_loss = torch.log(1 + torch.exp(-8*delta)).mean()
        delta_loss.backward()
        with torch.no_grad():
            for m in self.layer.modules():
                if isinstance(m, nn.Linear):
                    w_grad = m.weight.grad
                    pos_cos_sim = torch.cosine_similarity(w_grad.flatten(),(-pos_weight_grad-neg_weight_grad).flatten(),dim=0)
                    neg_cos_sim = torch.cosine_similarity(w_grad.flatten(),(-pos_weight_grad-neg_weight_grad).flatten(),dim=0)
        # Update weights
        if frozen:
            pass
        else:
            with torch.no_grad():
                for m in self.layer.modules():
                    if isinstance(m, nn.Linear):         
                        m.weight += self.lr * (pos_weight_grad + neg_weight_grad) 
        return pos_output_spike.detach(), pos_goodness.detach().mean(1).cpu(),pos_cos_sim.detach().cpu().item(), neg_output_spike.detach(), neg_goodness.detach().mean(1).cpu(),neg_cos_sim.detach().cpu().item()
    def predict(self, x):
        N = x.shape[1]
        g = torch.zeros(self.T, N, self.out_features).cuda()
        ln_mean = torch.zeros((N)).cuda()
        ln_var = torch.zeros((N)).cuda()
        for t in range(self.T):
            spike_out, ln_mean, ln_var = self.forward(x[t], ln_mean, ln_var)
            g[t] += spike_out
        functional.reset_net(self.layer)
        return g
class OutputLayer(nn.Module):
    def __init__(
        self, in_features, out_features, epoch, T, lr, v_threshold, v_threshold_neg, tau, loss_threshold
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