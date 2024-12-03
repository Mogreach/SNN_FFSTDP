### SNN-FF网络定义代码

```Python
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
import torch.nn.functional as F

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
class Net(torch.nn.Module):
    def __init__(self, dims, tau, epoch, T, lr, v_threshold,opt):
        super().__init__()
        self.T = T
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(in_features = dims[d], out_features = dims[d + 1], epoch=epoch, T=T, lr=lr,
                                  v_threshold = v_threshold, tau = tau).cuda()]
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

    def train(self, x_pos, x_neg, y):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg, y)

class Layer(nn.Module):
    def __init__(self, in_features, out_features, epoch, T, lr, v_threshold, tau):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(in_features, out_features, bias=False),
            neuron.LIFNode(tau=tau, v_threshold= v_threshold, surrogate_function=surrogate.ATan()),
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_epochs = epoch
        self.T = T
        self.threshold = 0.35
        self.encoder = encoding.PoissonEncoder()
        self.opt = Adam(self.parameters(), lr=lr)
        self.visible = True
        self.spike_vis = torch.zeros(out_features).unsqueeze(1)

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

    def train(self, x_pos, x_neg, y):
        for i in tqdm(range(self.num_epochs)):
            g_pos, g_neg = torch.zeros(x_pos.shape[0],self.out_features).cuda(),torch.zeros(x_pos.shape[0],self.out_features).cuda() 
            for t in  range(self.T):
                x_pos_encoded = self.encoder(x_pos)
                x_neg_encoded = self.encoder(x_neg)
                g_pos += self.forward(x_pos_encoded)
                g_neg += self.forward(x_neg_encoded)

            # 计算单位时间内平均频率的均方和
            g_pos_loss = (g_pos / self.T).pow(2).mean(1)
            g_neg_loss = (g_neg / self.T).pow(2).mean(1)
            # 计算单位时间内平均频率的平均值
            # g_pos_loss = (g_pos / self.T).mean(1)
            # g_neg_loss = (g_neg / self.T).mean(1)

            # The following loss pushes pos (neg) samples to values larger (smaller) than the self.threshold.
            if (self.out_features == 10):
                # loss = F.mse_loss(g_pos/self.T, y)
                loss = torch.log1p(1 + torch.exp(torch.cat([
                -g_pos_loss + self.threshold,
                g_neg_loss - self.threshold]))).mean()
            else:
                loss = torch.log1p(1 + torch.exp(torch.cat([
                -g_pos_loss + self.threshold,
                g_neg_loss - self.threshold]))).mean()

            self.opt.zero_grad()
            # this backward just compute the derivative and hence is not considered backpropagation.
            loss.backward()
            self.opt.step()
            if i % 100 == 0:
                print("Loss: ", loss.item())
            functional.reset_net(self.layer)
        return (g_pos / self.T).detach(), (g_neg / self.T).detach()
    def predict(self, x):
        h = x
        g = 0
        for t in  range(self.T):
            h_encoded = self.encoder(h)
            spike_out = self.forward(h_encoded)
            g += spike_out
            # 用于观察输出层脉冲发放情况
            if (self.out_features==10):
                # if(g[0].sum() > 0):
                    # print(1)
                self.visualize_spike_in_timestep(spike_out)
        functional.reset_net(self.layer)

        return g / self.T
```

**训练代码思路：**

![image-20241122170342700](assets/image-20241122170342700.png)

### SNN-FF训练主代码

```Python
import matplotlib.pyplot as plt
import torch
import os
import time
import argparse
import sys
import datetime
import torch
import torch.utils.data as data
import torchvision
import numpy as np
# from Dataset.MNIST_encoder import EncodedMNIST
from src.ff_snn_net import Net

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

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().sum(dim=0) #.reshape(1, 28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./data',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=1.5, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-save-model', action='store_true', help='save the model or not')

    args = parser.parse_args()
###########################################################################################
####################################前向学习的代码结构######################################
    # 初始化数据加载器
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

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    device = torch.device("cuda")
    # net = Net(dims=[784, 500, 500],tau=args.tau)
    net = Net(dims=[784, 10],tau=args.tau, epoch=args.epochs, T=args.T)
    x, y = next(iter(train_data_loader))
    x, y = x.to(device), y.to(device)
    #先导入MNIST图像的数据集，生成正负样本后再编码成脉冲序列数据集
    x_pos = overlay_y_on_x(x, y)
    y_neg = get_y_neg(y,device)
    x_neg = overlay_y_on_x(x, y_neg)
    # x_pos_encoded = encoder(x_pos)
    # x_neg_encoded = encoder(x_neg)
    net.train(x_pos, x_neg)
    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())
    x_te, y_te = next(iter(test_data_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    if args.save_model:
        torch.save(net.state_dict(), "mnist_ff.pt")
    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
###########################################################################################

if __name__ == "__main__":
    main()
```

### 训练日志

#### 损失函数定义：

1. $$\text{loss} = \frac{1}{N} \sum_{i=1}^{N} \log\left( \exp\left( \theta - g_{\text{pos}}^{(i)} \right) \right) + \log\left(\exp\left( g_{\text{neg}}^{(i)} - \theta \right) \right)$$
2. $$\text{loss}_l =  \begin{cases}\frac{1}{N} \sum_{i=1}^{N} \log\left( \exp\left( \theta - g_{\text{pos}}^{(i)} \right) \right) + \log\left(\exp\left( g_{\text{neg}}^{(i)} - \theta \right) \right) , \text{if }l < L \\ MSE , \text{else} \end{cases}$$

#### Goodness定义：

1. 均方和

```Python
            g_pos_loss = (g_pos / self.T).pow(2).mean(1)
            g_neg_loss = (g_neg / self.T).pow(2).mean(1)
```

$$g_{pos\_loss}=∑_{t=1}^Tf(x_{pos}^{(t)})^2$$

1. 平均值

```Python
            g_pos_loss = (g_pos / self.T).mean(1)
            g_neg_loss = (g_neg / self.T).mean(1)
```

$$g_{pos\_loss}=∑_{t=1}^Tf(x_{pos}^{(t)})$$

每层的Goodness我是用各个神经元输出频率的平均值计算的，即在T个时间步内，记录每个神经元输出脉冲次数，再除T得到该层所有神经元的输出频率，再取平均值或均方和。

#### 预测方法

1. 通过Goodness计算

```Python
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
```

1. 通过输出层最大输出频率计算

```Python
    # 根据输出层，选择最大输出频率的作为预测结果
    def predict(self, x):
        goodness_per_label = 0  # 选择输出频率最大
        for label in range(10):
            label = torch.full((x.shape[0],),label)
            h = overlay_y_on_x(x, label)
            for i, layer in enumerate(self.layers):
                h = layer.predict(h)
            goodness_per_label += h
        return goodness_per_label.argmax(1)
```

#### 训练记录

| 组别 | 网络结构     | Batchsize | 学习率 | 时间步长 | 脉冲发放阈值 | $$\theta$$ | $$\tau$$ | 代理梯度 | Loss_func | goodness | 预测方法             | Train_acc | Test_acc | 备注                                         |
| ---- | ------------ | --------- | ------ | -------- | ------------ | ---------- | -------- | -------- | --------- | -------- | -------------------- | --------- | -------- | -------------------------------------------- |
| 1    | [784,10]     | 64        | 0.001  | 100      | 0.5          | 0.5        | 2        | Sigmoid  | 1         | 2        | 通过最大输出频率计算 | 11%       | 6.25%    | Loss计算单位时间内平均频率和                 |
| 2    | [784,10]     | 64        | 0.001  | 100      | 0.5          | 0.5        | 2        | Sigmoid  | 1         | 1        | 通过最大输出频率计算 | 17.2%     | 9.94%    | Loss计算单位时间内平均频率均方和             |
| 3    | [784,10]     | 64        | 0.001  | 100      | 0.5          | 0.5        | 2        | Sigmoid  | 1         | 1        | 通过最大输出频率计算 | 14.1%     | 9.94%    | 均方和，Loss修改为log（1+exp）               |
| 4    | [784,10]     | 64        | 0.001  | 100      | 0.45         | 0.5        | 2        | Atan     | 1         | 2        | 通过最大输出频率计算 | 11%       | 5%       | Atan梯度替代，Loss计算单位时间平均频率       |
| 5    | [784,10]     | 64        | 0.001  | 100      | 0.45         | 0.5        | 2        | Atan     | 1         | 1        | 通过最大输出频率计算 | 12.5%     | 12.5%    | Atan梯度替代，Loss计算单位时间平均频率均方和 |
| 6    | [784,500,10] | 64        | 0.001  | 100      | 0.5          | 0.5        | 2        | Atan     | 2         | 1        | 通过最大输出频率计算 | 89.1%     | 49%      | 输出层采用MSE损失函数                        |
| 7    | [784,500,10] | 64        | 0.001  | 100      | 0.45         | 0.35       | 2        | Atan     | 2         | 1        | 通过最大输出频率计算 | 98.5%     | 61%      | 输出层采用MSE损失函数                        |
| 8    | [784,500,10] | 64        | 0.001  | 100      | 0.45         | 0.35       | 2        | Atan     | 1         | 1        | 通过Goodness计算     | 78.79%    | 56.4%    |                                              |

### 问题：关于输出层Loss改进问题

一开始我以为预测输出结果是按照输出层的最大输出频率取计算置信度的，所以一直训练不出来（见训练组别1-7），所以我在调试中去观察了每次预测时，输出层各神经元的脉冲发放情况，发现在一个时间步内有多个神经元同时激发脉冲的现象，理应只有一个神经元激发比较正常，所以提出在输出层用MSE去作为损失函数。我的想法是，这样的话就是前面隐藏层是去学习分类正负样本，而最后一层去根据正负样本隐藏层变量，学习正确标签的映射关系（比如输入1个数字9，负样本是标记为”0“或其他错误标签，正样本通过最后一层能学习到预测标签”9“，而负样本通过最后一层也学习到预测标签”9“），这样在推理的时候是否就能省略历遍10次的操作，只用历遍2次，比较一下就可以得到预测结果？

这个还只是我的想法，具体代码还没有实现，训练组6、7采用的预测方法是取最大输出频率去得到的，精度提升可能只是反向传播中一层全连接层的训练效果。