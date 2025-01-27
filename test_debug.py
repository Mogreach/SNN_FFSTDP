import matplotlib.pyplot as plt
import torch
import os
import argparse
import sys
import torch
import torch.utils.data as data
import torchvision
from src.ff_snn_net import Net
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

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().sum(dim=0) #.reshape(1, 28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def plot_loss(loss_of_layer_list, save_path):
    # 获取层数和每层的损失数据
    num_layers = len(loss_of_layer_list)
    
    # 创建一个图形
    plt.figure(figsize=(10, 6))  # 设置图像大小
    
    # 绘制每一层的损失随 epoch 变化的曲线
    for layer_idx in range(num_layers):
        plt.plot(loss_of_layer_list[layer_idx], 'o-', label=f'Layer {layer_idx + 1}')
    
    # 设置图形的标签和标题
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')    # y轴标签
    plt.title('Loss vs Epoch for Each Layer')  # 图形标题
    
    # 显示图例和网格
    plt.legend()  # 显示图例，标识不同的层
    plt.grid(True)  # 显示网格
    
    # 保存图像到文件
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-dims', default=[784,500,500], help='dimension of the network')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=100, type=int, help='batch size')
    parser.add_argument('-epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./data',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')

    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-v_threshold', default=1.2, type=float, help='V_threshold of LIF neuron')
    parser.add_argument('-loss_threshold', default=1.2, type=float, help='threshold of loss function')
    parser.add_argument('-save-model', action='store_true', help='save the model or not')

    args = parser.parse_args()
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
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    device = torch.device("cuda")
    net = Net(dims=args.dims,tau=args.tau, epoch=args.epochs, T=args.T, lr=args.lr,
              v_threshold=args.v_threshold, opt=args.opt, loss_threshold=args.loss_threshold)
    net.load("logs/T100_b100_adam_lr0.0001/checkpoint_max_model.pth")
    test_acc = 0
    test_samples = 0
    test_count = 0
    with torch.no_grad():
        for x_te, y_te in test_data_loader:
            test_samples += y_te.numel()
            test_count += 1
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_acc += net.predict(x_te).eq(y_te).cpu().float().mean().item()
            torch.cuda.empty_cache()
            # if(x_te.shape[0] != args.b or test_samples >= args.b):
            #     break
        print("test Acc:", 100 * test_acc / test_count, "%")
    
if __name__ == "__main__":
    main()

