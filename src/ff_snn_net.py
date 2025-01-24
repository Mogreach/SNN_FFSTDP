import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from src.loss import Frequency_FF_Loss
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
class CustomFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # 保存输入以供反向传播使用
        return input.clamp(min=0)  # 实现 ReLU 操作
 
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors  # 获取保存的输入张量
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # ReLU 的导数是 0 或 1
def overlay_y_on_x(x, y,classes=10):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()  # 创建一个 x 的副本，避免修改原始数据
    batch_size = x.shape[0]  # 获取批量大小
    x_[:, 0, 0, :classes] *= 0.0  # 将N*C*H*W格式向量的每个样本的前10个像素值赋0
    # 遍历每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        label = y[i].item()  # y[i]是该样本的标签
        # 确保标签在0到9之间（根据设置的 classes）
        if 0 <= label < classes:
            # 将第一通道前10个像素位置中对应标签的像素赋值为最大值
            x_[i, 0, label, 0] = x_.max()  # 将每个样本前10个像素中，对应标签类别序号赋为当前矩阵最大值
    return x_
class IFNode_Non_T(neuron.IFNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

class Net(torch.nn.Module):
    def __init__(self, dims, tau, epoch, T, lr, v_threshold, opt, loss_threshold):
        super().__init__()
        self.T = T
        self.layers = []
        self.loss_threshold = loss_threshold
        for d in range(len(dims) - 1):
            self.layers += [Layer(in_features = dims[d], out_features = dims[d + 1], epoch=epoch, T=T, lr=lr,
                                  v_threshold = v_threshold, tau = tau, loss_threshold = loss_threshold).cuda()]
    # 通过goodness计算预测结果
    def predict(self, x):
        goodness_per_label = []   
        # goodness_per_label = 0  # 选择输出频率最大
        for label in range(10):
            goodness = []
            label = torch.full((x.shape[0],),label)
            h = overlay_y_on_x(x, label)
            for i, layer in enumerate(self.layers):
                h = layer.predict(h)
                goodness = goodness + [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    
    # 根据输出层，选择最大输出频率的作为预测结果
    # def predict(self, x):
    #     goodness_per_label = 0  # 选择输出频率最大
    #     for label in range(10):
    #         label = torch.full((x.shape[0],),label)
    #         h = overlay_y_on_x(x, label)
    #         for i, layer in enumerate(self.layers):
    #             h = layer.predict(h)
    #         goodness_per_label += h
    #     return goodness_per_label.argmax(1)

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


class Layer(nn.Module):
    def __init__(self, in_features, out_features, epoch, T, lr, v_threshold, tau, loss_threshold):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            # neuron.LIFNode(tau=tau, v_threshold= v_threshold, surrogate_function=surrogate.ATan())
            # IFNode_Non_T(v_reset= None, v_threshold= v_threshold, surrogate_function=surrogate.ATan(), step_mode='s')
            neuron.IFNode(v_reset= None, v_threshold= v_threshold, surrogate_function=surrogate.ATan(), step_mode='s')
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_epochs = epoch
        self.T = T
        self.threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.opt = Adam(self.parameters(), lr=lr)
        self.visible = False
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)
        self.loss = Frequency_FF_Loss
    def visualize_spike_in_timestep(self,layer_forward_out):
        self.spike_vis = torch.cat((self.spike_vis,layer_forward_out[0].cpu().flatten().unsqueeze(1)),dim=1)
        if (self.visible and self.spike_vis.shape[1] == self.T):
            plt.imshow(self.spike_vis.detach().numpy(), cmap='viridis', aspect='auto')  # 使用 'viridis' 颜色映射，自动调整纵横比
            plt.colorbar(label='Spike Intensity')  # 添加颜色条并标注
            plt.title('Spike Visualization')  # 图像标题
            plt.xlabel('Time Steps')  # x 轴标签
            plt.ylabel('Neuron Index')  # y 轴标签
            plt.tight_layout()  # 自动调整子图参数
            plt.show()
        if (self.spike_vis.shape[1] == self.T):
            self.spike_vis = torch.zeros(self.out_features).unsqueeze(1)

    def forward(self, x):
        # 对第1维度（通道维度）计算L2范数，然后进行归一化
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x_direction)


    def train(self, x_pos, x_neg, y, train_mode):
        g_pos, g_neg = torch.zeros(x_pos.shape[0],self.out_features).cuda(),torch.zeros(x_pos.shape[0],self.out_features).cuda() 
        for t in  range(self.T):
            x_pos_encoded = self.encoder(x_pos)
            x_neg_encoded = self.encoder(x_neg)
            g_pos += self.forward(x_pos_encoded)
            g_neg += self.forward(x_neg_encoded)
        functional.reset_net(self.layer)
        g_pos_freq = g_pos / self.T
        g_neg_freq = g_neg / self.T
        if (train_mode):
            self.opt.zero_grad()
            loss, grad = self.loss(self.in_features, self.out_features, self.T, self.threshold, x_pos, x_neg, g_pos_freq, g_neg_freq)
            for param in self.layer.parameters():
                if param.requires_grad:
                    # 使用优化器更新权重
                    param.grad = grad
            # this backward just compute the derivative and hence is not considered backpropagation.
            # loss.backward()
            self.opt.step()
            return g_pos_freq.detach(), g_neg_freq.detach(), loss
        else:
            return g_pos_freq.detach(), g_neg_freq.detach(), 0
    def predict(self, x):
        h = x
        g = 0
        for t in  range(self.T):
            h_encoded = self.encoder(h)
            spike_out = self.forward(h_encoded)
            g += spike_out
            # 用于观察输出层脉冲发放情况
            # if (self.out_features==10):
                # if(g[0].sum() > 0):
                    # print(1)
                # self.visualize_spike_in_timestep(spike_out)
        functional.reset_net(self.layer)

        return g / self.T
