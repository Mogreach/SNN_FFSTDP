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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from tqdm import tqdm
from src.ff_snn_net import Net
from spikingjelly.activation_based import encoding, functional
import torch.nn.functional as F
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
def eval_model(args, T, batch_size, epochs, lr, v_threshold, loss_threshold):
    # 加载训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
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
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    val_data_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    device = torch.device("cuda")
    
    net = Net(dims=args.dims,tau=args.tau, epoch=epochs, T=T, lr=lr,
              v_threshold=v_threshold, opt=args.opt, loss_threshold=loss_threshold)
    
    for layer_idx in range(len(net.layers)):
        print('training layer', layer_idx+1, '...')
        for i in tqdm(range(epochs)):
            torch.cuda.empty_cache()
            batch_samples = 0
            loss = 0
            for x, y in train_data_loader:
                batch_samples += 1
                x, y = x.to(device), y.to(device)
                label_onehot = F.one_hot(y, 10).float()
            #先导入MNIST图像的数据集，生成正负样本后再编码成脉冲序列数据集
                x_pos = overlay_y_on_x(x, y)
                y_neg = get_y_neg(y,device)
                x_neg = overlay_y_on_x(x, y_neg)
                loss += net.train(x_pos, x_neg, label_onehot, layer_idx)
    val_samples = 0
    val_acc = 0
    for x_val, y_val in val_data_loader:
        val_samples += 1
        x_val, y_val = x_val.to(device), y_val.to(device)
        val_acc += net.predict(x_val).eq(y_val).cpu().float().mean().item()
    minimize =-(val_acc / val_samples)
    return minimize


def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-dims', default=[784,500,500], help='dimension of the network')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=800, type=int, help='batch size')
    parser.add_argument('-epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./data',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')

    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-v_threshold', default=1.2, type=float, help='V_threshold of LIF neuron')
    parser.add_argument('-loss_threshold', default=0.3, type=float, help='threshold of loss function')
    parser.add_argument('-save-model', action='store_true', help='save the model or not')

    args = parser.parse_args()
###########################################################################################
####################################超参数寻优代码######################################

    # 定义超参数的搜索空间
    space  = [
    Integer(1, 100, name='T'),
    Integer(128, 800, name='batch_size'), # 批大小
    Integer(1, 50, name='epochs'),       # 训练轮数
    Real(1e-6, 1e-1, name='lr'),          # 学习率
    Real(0.1, 1.8, name='v_threshold'),    # 脉冲电压阈值
    Real(0.0, 1, name='loss_threshold')   # 损失函数阈值
    ]
    start_time = time.time()
    # 使用gp_minimize进行优化
    result = gp_minimize(
        func=eval_model,   # 目标函数
        dimensions=space,      # 超参数空间
        n_calls=50,            # 迭代次数
        random_state=42        # 随机种子
    )
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Optimizing completed. Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    # 输出最优超参数组合
    print("Best parameters: ", result.x)
    print("Best validation accuracy: ", -result.fun)
    
if __name__ == "__main__":
    main()

