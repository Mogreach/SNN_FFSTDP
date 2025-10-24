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


def overlay_y_on_x(x, y, classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, :, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        label = y[i].item()  # y[i]是该样本的标签
        # 确保标签在0到9之间（根据设置的 classes）
        # 将第一通道前10个像素位置中对应标签的像素赋值为最大值
        x_[i, :, 0, label] = (
            x_.max()
        )  # 将每个样本前10个像素中，对应标签类别序号赋为当前矩阵最大值
    return x_

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

    return spike_train  # 形状为 [T, B, C, H, W]

class IFNode_Non_T(neuron.IFNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x


class Net(torch.nn.Module):
    def __init__(self, dims, tau, epoch, T, lr, v_threshold, opt, loss_threshold):
        super().__init__()
        self.T = T
        self.layers = []
        self.loss_threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        for d in range(len(dims) - 1):
            self.layers += nn.ModuleList(
                [
                    Layer(
                        in_features=dims[d],
                        out_features=dims[d + 1],
                        epoch=epoch,
                        T=T,
                        lr=lr,
                        v_threshold=v_threshold,
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
            h = overlay_y_on_x(x, label)
            # Possion编码
            # h_encoded = torch.zeros(self.T, h.shape[0], h.flatten(1).shape[1]).cuda()
            # for t in range(self.T):
            #     h_encoded[t] += self.encoder(h).flatten(1)
            # h = h_encoded
            # 频率编码
            h = spike_encoder(h, self.T)
            h = h.flatten(2)  # 将输入展平为 [T, B, C*H*W] 的形状
            for i, layer in enumerate(self.layers):
                h = layer.predict(h)
                freq = h.mean(0)  # 计算每层的平均频率
                goodness = goodness + [layer.cal_goodness(freq).sum(1)] # 对每个样本的单层goodness求和
            goodness_per_label += [sum(goodness).unsqueeze(1)] # 对所有层求和优度值
        goodness_of_all_label = torch.cat(goodness_per_label, 1)# 拼接所有标签编码对应优度值
        return goodness_of_all_label.argmax(1)

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
    def train_ff_stdp(self, x_pos, x_neg):
        x_pos_encoded = spike_encoder(x_pos, self.T)
        x_neg_encoded = spike_encoder(x_neg, self.T)
        in_pos = x_pos_encoded.flatten(2)
        in_neg = x_neg_encoded.flatten(2)

        # in_pos = torch.zeros(self.T, x_pos.shape[0], x_pos.flatten(1).shape[1]).cuda()
        # in_neg = torch.zeros(self.T, x_pos.shape[0], x_neg.flatten(1).shape[1]).cuda()
        # for t in range(self.T):
        #     x_pos_encoded = self.encoder(x_pos)
        #     x_neg_encoded = self.encoder(x_neg)
        #     in_pos[t] += x_pos_encoded.flatten(1)
        #     in_neg[t] += x_neg_encoded.flatten(1)

        spike_input_pos = in_pos
        spike_input_neg = in_neg

        goodness_pos, cos_pos = self.train_ff_stdp_step(spike_input_pos, True)
        goodness_neg, cos_neg = self.train_ff_stdp_step(spike_input_neg, False)
        
        return goodness_pos, goodness_neg, cos_pos, cos_neg
    def train_ff_stdp_step(self, input, is_pos):
        spike_input  = input
        goodness_per_layer = []
        cos_sim_per_layer = []
        for i, layer in enumerate(self.layers):
            spike_input, g ,cos_sim = layer.train_ff_stdp(spike_input, is_pos)
            goodness_per_layer.append(g.mean().item())
            cos_sim_per_layer.append(cos_sim)
        return goodness_per_layer, cos_sim_per_layer

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
        self, in_features, out_features, epoch, T, lr, v_threshold, tau, loss_threshold
    ):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            # neuron.LIFNode(tau=tau, v_threshold= v_threshold, surrogate_function=surrogate.ATan())
            # IFNode_Non_T(v_reset= None, v_threshold= v_threshold, surrogate_function=surrogate.ATan(), step_mode='s')
            neuron.IFNode(
                v_reset=None,
                v_threshold=v_threshold,
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
        goodness = self.T * freq.pow(2)
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
    def train_ff_stdp(self,x_encoded,is_pos):
        N = x_encoded.shape[1]
        input_spike_sum = x_encoded.sum(0)
        output_spike = torch.zeros(self.T, N, self.out_features).cuda()
        for t in range(self.T):
            output_spike[t] += self.forward(x_encoded[t])
        out_freq = output_spike.mean(0).transpose(0,1)
        self.opt.zero_grad()
        goodness = self.cal_goodness(out_freq)

        if is_pos:
            L_to_s_grad = 2*out_freq*pos_derivative(goodness,self.threshold)
            loss = torch.log(1 + torch.exp(-goodness + self.threshold)).mean()
        else:
            L_to_s_grad = 2*out_freq*neg_derivative(goodness,self.threshold)
            loss = torch.log(1 + torch.exp(goodness - self.threshold)).mean()
        weight_grad = -1 * L_to_s_grad @ input_spike_sum / N
        # weight_grad = -1 * torch.mean(L_to_s_grad,dim=1,keepdim=True) @ torch.mean(input_spike_sum,dim=0,keepdim=True)
        loss.backward()
        with torch.no_grad():
            for param in self.layer.parameters():
                    # 使用优化器更新权重           
                    param += self.lr * weight_grad
                    cos_sim = torch.cosine_similarity(param.grad.flatten(),-1*weight_grad.flatten(),dim=0)
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
        # self.opt.step()
        functional.reset_net(self.layer)
        return output_spike.detach(), goodness.detach().mean(1).cpu(),cos_sim.detach().cpu().item()
    def predict(self, x):
        h = x
        g = torch.zeros(self.T, x.shape[1], self.out_features).cuda()
        # self.spike_input_rate = 0
        # h_encoded = spike_encoder(h, self.T)
        for t in range(self.T):
            # h_encoded = self.encoder(h)
            # self.spike_input_rate += h_encoded.mean().detach().cpu() / self.T
            spike_out = self.forward(h[t])
            g[t] += spike_out
            # 用于观察输出层脉冲发放情况
            # if (self.out_features==10):
            # if(g[0].sum() > 0):
            # print(1)
            # self.visualize_spike_in_timestep(spike_out)
        functional.reset_net(self.layer)
        return g
