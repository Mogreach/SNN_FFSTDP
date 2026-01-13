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

from visualization_debug import vis_weight_feature
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
    v_mem = torch.zeros((B, C, H, W), device=images.device)  # 初始化膜电位为0
    for t in range(T):
        v_mem += images  # 每步累加像素值
        spike = (v_mem >= 1.0).to(torch.float)  # 触发放电
        spike_train[t] = spike
        v_mem = v_mem * (1.0 - spike)  # 膜电位重置：只有放电位置归零
    # Possion编码
    # spike_train = torch.zeros(T, images.shape[0], images.flatten(1).shape[1]).cuda()
    # for t in range(T):
    #     spike_train[t] += encoding.PoissonEncoder(images).flatten(1)
    return spike_train  # 形状为 [T, B, C, H, W]

class Net(torch.nn.Module):
    def __init__(self, dims, tau, epoch, T, lr, v_threshold_pos, v_threshold_neg, opt, loss_threshold):
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
                            v_threshold_pos=v_threshold_pos,
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
                            v_threshold_pos=v_threshold_pos,
                            v_threshold_neg=v_threshold_neg,
                            tau=tau,
                            loss_threshold=loss_threshold,
                        ).cuda()
                    ]
                )
    # 通过goodness计算预测结果
    def predict(self, x):
        goodness_per_label = []
        # goodness_per_label = 0  # 选择输出频率最大
        for label in range(10):
            goodness = []
            label = torch.full((x.shape[0],), label)
            h, _ = generate_pos_n_neg_sample(x, label, num_classes=10)
            h = spike_encoder(h, self.T)
            h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
            spike_in_of_label = h[:,:,0:10]
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    break
                h = layer.predict(h)
                freq = h.mean(0)  # 计算每层的平均频率
                goodness = goodness + [layer.cal_goodness(freq).sum(1)] # 对每个样本的单层goodness求和
                h = torch.cat((h, spike_in_of_label),dim=2)
            goodness_per_label += [sum(goodness).unsqueeze(1)] # 对所有层求和优度值
        goodness_of_all_label = torch.cat(goodness_per_label, 1)# 拼接所有标签编码对应优度值
        return goodness_of_all_label.argmax(1)
        # 通过goodness计算预测结果
    def predict_winner(self, x):
        label = torch.randint(0, 10, (x.shape[0],))
        h, _ = generate_pos_n_neg_sample(x, label, num_classes=10)
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
    def predict_analyze(self, x):
        goodness_per_label = []
        goodness_label_layer_goodness = torch.zeros(10,len(self.layers),x.shape[0]).cuda()
        freq_label_layer_freq = torch.zeros(10,len(self.layers),x.shape[0],1000).cuda()
        for label in range(10):
            goodness = []
            label = torch.full((x.shape[0],), label)
            h, _ = generate_pos_n_neg_sample(x, label, num_classes=10)
            # 频率编码
            h = spike_encoder(h, self.T)
            h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
            spike_in_of_label = h[:,:,0:10]
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    break
                h = layer.predict(h)
                freq = h.mean(0)  # 计算每层的平均频率
                goodness = goodness + [layer.cal_goodness(freq).sum(1)] # 对每个样本的单层goodness求和
                goodness_label_layer_goodness[label,i,:] = layer.cal_goodness(freq).sum(1)
                freq_label_layer_freq[label,i,:,0:freq.shape[1]] = freq
                h = torch.cat((h, spike_in_of_label),dim=2)
            goodness_per_label += [sum(goodness).unsqueeze(1)] # 对所有层求和优度值
        goodness_of_all_label = torch.cat(goodness_per_label, 1)# 拼接所有标签编码对应优度值
        return goodness_of_all_label.argmax(1),goodness_label_layer_goodness.cpu(),freq_label_layer_freq
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
    def train_ff_stdp(self, x, label):
        x_pos, x_neg = generate_pos_n_neg_sample(x, label, num_classes=10)
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        in_pos = x_pos_encoded.flatten(2)
        in_neg = x_neg_encoded.flatten(2)
        spike_input_pos = in_pos
        spike_input_neg = in_neg
        goodness_pos, cos_pos, spike_out_pos, goodness_neg, cos_neg, spike_out_neg = self.train_ff_stdp_step(spike_input_pos, spike_input_neg, label)      
        return goodness_pos, goodness_neg, cos_pos, cos_neg
    def train_ff_stdp_step(self, input_pos, input_neg, label):
        T, B, _ = input_pos.shape
        pos_goodness_per_layer = []
        neg_goodness_per_layer = []
        pos_cos_sim_per_layer = []
        neg_cos_sim_per_layer = []
        pos_spike_in_of_output_layer = torch.empty((T,B,0)).cuda()
        neg_spike_in_of_output_layer = torch.empty((T,B,0)).cuda()
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                pos_spike_output = layer.train_bp_stdp(pos_spike_in_of_output_layer, label)
                neg_spike_output = 0
                # neg_spike_output = layer.train_bp_stdp(neg_spike_in_of_output_layer, label)
            else:
                input_pos, pos_g , pos_cos_sim, input_neg, neg_g, neg_cos_sim = layer.train_ff_stdp(input_pos, input_neg)
                pos_goodness_per_layer.append(pos_g.mean().item())
                neg_goodness_per_layer.append(neg_g.mean().item())
                pos_cos_sim_per_layer.append(pos_cos_sim)
                neg_cos_sim_per_layer.append(neg_cos_sim)
                pos_spike_in_of_output_layer = torch.cat((pos_spike_in_of_output_layer,input_pos),dim=2)
                neg_spike_in_of_output_layer = torch.cat((neg_spike_in_of_output_layer,input_neg),dim=2)
        return pos_goodness_per_layer, pos_cos_sim_per_layer , pos_spike_output, neg_goodness_per_layer, neg_cos_sim_per_layer, neg_spike_output

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
            layer.load_state_dict(check_point["net"][f"layer_{i}"])
        # 打印加载的超参数
        print(check_point["args"])


class Layer(nn.Module):
    def __init__(
        self, in_features, out_features, epoch, T, lr, v_threshold_pos, v_threshold_neg, tau, loss_threshold
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold_pos,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            ),
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
        # self.opt = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化权重，并添加一个正偏置
                nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.out_features))
                m.weight.data += 0.1  # 添加正偏置，确保权重平均值大于 0


    def visualize_spike_in_timestep(self, layer_forward_out):
        self.spike_vis = torch.cat(
            (self.spike_vis, layer_forward_out[0].cpu().flatten().unsqueeze(1)), dim=1
        )
        if self.visible and self.spike_vis.shape[1] == self.T:
            plt.imshow(
                self.spike_vis.detach().numpy(), cmap="viridis", aspect="auto"
            )  # 使用 'viridis' 颜色映射，自动调整纵横比
            plt.colorbar(label="Spike Intensity")  # 添加颜色条并标注
            plt.title("Spike Visualization")  # 图像标题
            plt.xlabel("Time Steps")  # x 轴标签
            plt.ylabel("Neuron Index")  # y 轴标签
            plt.tight_layout()  # 自动调整子图参数
            plt.show()
        if self.spike_vis.shape[1] == self.T:
            self.spike_vis = torch.zeros(self.out_features).unsqueeze(1)

    def cal_goodness(self, freq):
        # goodness = self.T * freq.pow(2)
        goodness = self.T * freq.abs().pow(2) * freq.sign()
        return goodness

    def forward(self, x):
        # 对第1维度（通道维度）计算L2范数，然后进行归一化
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x)

    def train(self, x_pos, x_neg, y, train_mode):
        g_pos = torch.zeros(self.T, x_pos.shape[0], self.out_features).cuda()
        g_neg = torch.zeros(self.T, x_pos.shape[0], self.out_features).cuda()
        in_pos = torch.zeros(self.T, x_pos.shape[0], self.in_features).cuda()
        in_neg = torch.zeros(self.T, x_pos.shape[0], self.in_features).cuda()
        for t in range(self.T):
            x_pos_encoded = self.encoder(x_pos)
            x_neg_encoded = self.encoder(x_neg)
            g_pos[t] += self.forward(x_pos_encoded)
            in_pos[t] += x_pos_encoded.flatten(1)
            g_neg[t] += self.forward(x_neg_encoded)
            in_neg[t] += x_neg_encoded.flatten(1)
        g_pos_freq = g_pos.mean(0)
        g_neg_freq = g_neg.mean(0)
        if train_mode:
            self.opt.zero_grad()
            pos_goodness = self.cal_goodness(g_pos_freq)
            neg_goodness = self.cal_goodness(g_neg_freq)
            loss = torch.log(
                1
                + torch.exp(
                    torch.cat(
                        [-pos_goodness + self.threshold, neg_goodness - self.threshold]
                    )
                )
            ).mean()
            # loss = torch.log(1 + torch.exp(-pos_goodness + neg_goodness + self.threshold)).mean()
            loss.backward()
            # grad_w_autograd = self.layer[1].weight.grad.clone()
            # loss, grad = Custom_Loss.Frequency_FF_Loss(g_pos,g_neg,in_pos,in_neg,self.in_features, self.out_features, self.T, self.threshold, x_pos, x_neg, g_pos_freq, g_neg_freq)
            # for param in self.layer.parameters():
            #     if param.requires_grad:
            #         # 使用优化器更新权重
            #         param.grad = grad
            # grad_w_manual = grad.clone()
            # cos_sim = torch.nn.functional.cosine_similarity(grad_w_manual.view(-1), grad_w_autograd.view(-1), dim=0)
            # print(cos_sim)
            # print("手写梯度均值:", grad_w_manual.mean().item(), "标准差:", grad_w_manual.std().item())
            # print("自动梯度均值:", grad_w_autograd.mean().item(), "标准差:", grad_w_autograd.std().item())
            self.opt.step()
            functional.reset_net(self.layer)
            return g_pos_freq.detach(), g_neg_freq.detach(), loss.cpu().item()
        else:
            functional.reset_net(self.layer)
            return g_pos_freq.detach(), g_neg_freq.detach(), 0
    def train_ff_stdp(self, pos_encoded, neg_encoded):
        N = pos_encoded.shape[1]
        # Positive sample processing
        pos_input_spike_sum = pos_encoded.sum(0)
        pos_output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        for t in range(self.T):
            pos_output_spike[t] += self.forward(pos_encoded[t])
        pos_out_freq = pos_output_spike.mean(0).transpose(0,1)
        self.opt.zero_grad()
        pos_goodness = self.cal_goodness(pos_out_freq)
        pos_L_to_s_grad = 2*pos_out_freq*pos_derivative(pos_goodness,self.threshold)
        pos_loss = torch.log(1 + torch.exp(-pos_goodness + self.threshold)).mean()
        pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N
        pos_loss.backward()
        with torch.no_grad():   
            for param in self.layer.parameters():
                pos_cos_sim = torch.cosine_similarity(param.grad.flatten(),-1*pos_weight_grad.flatten(),dim=0)
        functional.reset_net(self.layer)
        # Negative sample processing
        neg_input_spike_sum = neg_encoded.sum(0)
        neg_output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        for t in range(self.T):
            neg_output_spike[t] += self.forward(neg_encoded[t])
        neg_out_freq = neg_output_spike.mean(0).transpose(0,1)
        self.opt.zero_grad()
        neg_goodness = self.cal_goodness(neg_out_freq)
        neg_L_to_s_grad = 2*neg_out_freq*neg_derivative(neg_goodness,self.threshold)
        neg_loss = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N
        neg_loss.backward()
        with torch.no_grad():   
            for param in self.layer.parameters():
                neg_cos_sim = torch.cosine_similarity(param.grad.flatten(),-1*neg_weight_grad.flatten(),dim=0)
        functional.reset_net(self.layer)
        # Update weights
        with torch.no_grad():   
            for param in self.layer.parameters():
                # weight_grad = -1 * torch.mean(L_to_s_grad,dim=1,keepdim=True) @ torch.mean(input_spike_sum,dim=0,keepdim=True)
                # 使用优化器更新权重           
                param += self.lr * (pos_weight_grad + neg_weight_grad) 
                # param.grad = weight_grad
                # plt.imshow(np.array(param[511].cpu().reshape(28,28)))
                # 可视化梯度分布
                # plt.figure(figsize=(8, 6))
                # plt.hist(param.grad.cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
                # plt.title("Gradient Distribution")
                # plt.xlabel("Gradient Value")
                # plt.ylabel("Frequency")
                # plt.grid(True)
                # plt.show()
                # param.clamp_(min=-12.0, max=12.0)  # 限制权重在[-12, 12]范围内
        
        return pos_output_spike.detach(), pos_goodness.detach().mean(1).cpu(),pos_cos_sim.detach().cpu().item(), neg_output_spike.detach(), neg_goodness.detach().mean(1).cpu(),neg_cos_sim.detach().cpu().item()
    def predict(self, x):
        h = x
        g = torch.zeros(self.T, x.shape[1], self.out_features).cuda()
        for t in range(self.T):
            spike_out = self.forward(h[t])
            g[t] += spike_out
        functional.reset_net(self.layer)
        return g
class OutputLayer(nn.Module):
    def __init__(
        self, in_features, out_features, epoch, T, lr, v_threshold_pos, v_threshold_neg, tau, loss_threshold
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold_pos,
                surrogate_function=surrogate.ATan(),
                step_mode="s",
            )
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