
"""
====================================================================
File          : ff-snn_opt.py
Description   : SNN-FF训练超参数优化代码
Author        : Morgreach
Version       : 1.0.0
Date          : 2025-04-18
contact       : 1245598043@qq.com
License       : MIT
====================================================================
"""

import torch
import matplotlib.pyplot as plt
import torch
import argparse
import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from src.loss import Custom_Loss
import os
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
Custom_Loss=Custom_Loss()
Frequency_FF_Loss=Custom_Loss.Frequency_FF_Loss
def get_y_neg(y,device):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(10))
        # print("allowed_indices:", allowed_indices)
        # print("y_samp:", y_samp.item())
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)

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
def manual_gradient_computation(x, w, g_pos_freq, g_neg_freq, loss, threshold, T):
    # 计算正负样本的 goodness 梯度
    pos_goodness = T * g_pos_freq.pow(2).mean(1)
    neg_goodness = T * g_neg_freq.pow(2).mean(1)
    
    dL_dg_pos = -torch.sigmoid(-pos_goodness + threshold) * (2 * T * g_pos_freq / g_pos_freq.shape[1])
    dL_dg_neg = torch.sigmoid(neg_goodness - threshold) * (2 * T * g_neg_freq / g_neg_freq.shape[1])
    
    # 计算 dL/dW
    grad_w_manual = torch.zeros_like(w)
    for t in range(T):
        grad_w_manual += torch.matmul(dL_dg_pos[t].T, x[t]) + torch.matmul(dL_dg_neg[t].T, x[t])

    return grad_w_manual

class Layer(nn.Module):
    def __init__(self, in_features, out_features, T, lr, v_threshold, tau, loss_threshold):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            neuron.IFNode(v_reset=None, v_threshold=v_threshold, surrogate_function=surrogate.ATan(), step_mode='s')
        )
        self.T = T
        self.threshold = loss_threshold
        self.encoder = encoding.PoissonEncoder()
        self.opt = Adam(self.parameters(), lr=lr)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x_norm = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x_norm)
    
    def train_step(self, x_pos, x_neg):
        g_pos = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        g_neg = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        v_pos = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        v_neg = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda() 
        for t in range(self.T):
            x_pos_encoded = self.encoder(x_pos)
            x_neg_encoded = self.encoder(x_neg)
            g_pos[t] = self.forward(x_pos_encoded)
            v_pos[t] += self.layer[2].v
            g_neg[t] = self.forward(x_neg_encoded)
            v_neg[t] += self.layer[2].v
        g_pos_freq = g_pos.mean(0)
        g_neg_freq = g_neg.mean(0)
        
        self.opt.zero_grad()
        pos_goodness =  self.T*g_pos_freq.pow(2).mean(1)
        neg_goodness =  self.T*g_neg_freq.pow(2).mean(1)
        loss = torch.log(1 + torch.exp(torch.cat([-pos_goodness + self.threshold, neg_goodness - self.threshold]))).mean()
        loss.backward()
        grad_w_autograd = self.layer[1].weight.grad.clone()
        # 计算手动梯度
        loss_manual, grad_w_manual = Frequency_FF_Loss(g_pos,g_neg,v_pos,v_neg,self.in_features, self.out_features, self.T, self.threshold, x_pos, x_neg, g_pos_freq, g_neg_freq)
        # grad_w_manual = manual_gradient_computation(x_pos, self.layer[1].weight, g_pos_freq, g_neg_freq, loss, self.threshold, self.T)
        for param in self.layer.parameters():
                param.grad = grad_w_manual

        self.opt.step()
        functional.reset_net(self.layer)
        
        return loss.item(), grad_w_autograd, grad_w_manual, pos_goodness, neg_goodness
    def predict(self, x):
        goodness_per_label = []   
        # goodness_per_label = 0  # 选择输出频率最大
        for label in range(10):
            goodness = []
            label = torch.full((x.shape[0],),label)
            h = overlay_y_on_x(x, label)
            g = 0
            for t in  range(self.T):
                h_encoded = self.encoder(h)
                spike_out = self.forward(h_encoded)
                g += spike_out
            g = g / self.T
            functional.reset_net(self.layer)
            goodness = self.T*[g.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)



def eval(lr):
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-dims', default=[784,500], help='dimension of the network')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=1000, type=int, help='batch size')
    parser.add_argument('-epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./data',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')

    parser.add_argument('-lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-v_threshold', default=1.2, type=float, help='V_threshold of LIF neuron')
    parser.add_argument('-loss_threshold', default=1.2, type=float, help='threshold of loss function')
    parser.add_argument('-save-model', action='store_true', help='save the model or not')

    args = parser.parse_args()
    args.lr = lr
    ###########################################################################################
    ####################################前向学习的代码结构######################################
    # 初始化数据加载器
    # 加载训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    # 划分训练集和验证集
    train_size = int(0.95 * len(train_dataset))  # 80% 用于训练
    val_size = len(train_dataset) - train_size  # 20% 用于验证
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    # 初始化超参数
    in_features, out_features, T ,N = 784, 500, 20, 1000
    lr, v_threshold, tau, loss_threshold = args.lr, args.v_threshold, 2.0, args.loss_threshold 
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Layer(in_features, out_features, T, lr, v_threshold, tau, loss_threshold).to(device)
    for i in range(args.epochs):
        torch.cuda.empty_cache()
        batch_samples = 0
        
        for x, y in train_data_loader:
            batch_samples += 1
            x, y = x.to(device), y.to(device)
            label_onehot = F.one_hot(y, 10).float()
            
            x_pos = overlay_y_on_x(x, y)
            y_neg = get_y_neg(y, device)
            x_neg = overlay_y_on_x(x, y_neg)
            
            loss, grad_w_autograd, grad_w_manual, pos, neg = model.train_step(x_pos, x_neg)
            # 计算梯度误差
            error_w = torch.norm(grad_w_manual - grad_w_autograd).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                grad_w_manual.view(-1), grad_w_autograd.view(-1), dim=0
            ).item()
    test_acc = 0
    test_samples = 0
    test_count = 0
    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            test_samples += y_te.numel()
            test_count += 1
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_acc += model.predict(x_te).eq(y_te).cpu().float().mean().item()
    print("test Acc:", 100 * test_acc / test_count, "%")
    return -(test_acc / test_count)

def main():
        # 定义超参数的搜索空间
    space  = [
    Real(1e-6, 1e-1, name='lr'),          # 学习率
    ]
    @use_named_args(space)
    def evaluate_model_pack(**params):
        # configure the model with specific hyperparameters
        # 打印所有参数及其对应的值
        print("Hyperparameters:", params)
        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")
        minimize = eval(**params)
        return minimize

    start_time = time.time()
    # 使用gp_minimize进行优化
    result = gp_minimize(
        func=evaluate_model_pack,   # 目标函数
        dimensions=space,      # 超参数空间
        n_calls=50,            # 迭代次数
        random_state=100        # 随机种子
    )
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Optimizing completed. Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    # 输出最优超参数组合
    print("Best parameters: ", result.x)
    print("Best validation accuracy: ", -result.fun)
if __name__ == '__main__':
    main()