import torch
import matplotlib.pyplot as plt
import torch
import argparse
import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from src.loss import Custom_Loss
import numpy as np
Custom_Loss=Custom_Loss()
Frequency_FF_Loss=Custom_Loss.Frequency_FF_Loss
FF_Loss_step = Custom_Loss.FF_Loss_step
def vis_w(grad_w_manual, grad_w_autograd):
    # 确保梯度在 CPU 上，并转换为 NumPy
    grad_w_manual_np = grad_w_manual.cpu().numpy()
    grad_w_autograd_np = grad_w_autograd.cpu().numpy()
    grad_diff = grad_w_manual_np - grad_w_autograd_np  # 计算差异

    # 画布设置
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 手写梯度热力图
    im1 = axes[0].imshow(grad_w_manual_np, cmap='viridis', aspect='auto')
    axes[0].set_title("Manual Gradient Heatmap")
    axes[0].set_xlabel("Input Features (784)")
    axes[0].set_ylabel("Output Neurons (500)")
    fig.colorbar(im1, ax=axes[0])

    # 自动梯度热力图
    im2 = axes[1].imshow(grad_w_autograd_np, cmap='viridis', aspect='auto')
    axes[1].set_title("Autograd Gradient Heatmap")
    axes[1].set_xlabel("Input Features (784)")
    axes[1].set_ylabel("Output Neurons (500)")
    fig.colorbar(im2, ax=axes[1])

    # 差异热力图
    im3 = axes[2].imshow(grad_diff, cmap='bwr', aspect='auto', vmin=-np.max(np.abs(grad_diff)), vmax=np.max(np.abs(grad_diff)))
    axes[2].set_title("Gradient Difference Heatmap (Manual - Autograd)")
    axes[2].set_xlabel("Input Features (784)")
    axes[2].set_ylabel("Output Neurons (500)")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # 差异直方图
    plt.figure(figsize=(8, 6))
    plt.hist(grad_diff.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2)  # 0 位置的红色虚线
    plt.title("Gradient Difference Distribution")
    plt.xlabel("Gradient Difference (Manual - Autograd)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
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
        # self.opt = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x_norm = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.layer(x_norm)
    def cal_goodness(self,freq):
        goodness = self.T* (freq.pow(1).mean(1))
        return goodness 
    def train_step(self, x_pos, x_neg):
        x_in_pos = torch.zeros(self.T,x_pos.shape[0],self.in_features).cuda()
        x_in_neg = torch.zeros(self.T,x_pos.shape[0],self.in_features).cuda()
        g_pos = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        g_neg = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        v_pos = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda()
        v_neg = torch.zeros(self.T,x_pos.shape[0],self.out_features).cuda() 
        for t in range(self.T):
            x_pos_encoded = self.encoder(x_pos)
            x_neg_encoded = self.encoder(x_neg)
            x_in_pos[t] = x_pos_encoded.view(x_pos.shape[0],self.in_features)
            x_in_neg[t] = x_pos_encoded.view(x_pos.shape[0],self.in_features)
            g_pos[t] = self.forward(x_pos_encoded)
            v_pos[t] = self.layer[2].v
            g_neg[t] = self.forward(x_neg_encoded)
            v_neg[t] = self.layer[2].v
            # if t>5:
            #     self.opt.zero_grad()
            #     loss_manual, grad_w_manual = FF_Loss_step(x_in_pos[t],x_in_neg[t],g_pos[t],g_neg[t],v_pos[t],v_neg[t],self.in_features,self.out_features,self.T,self.threshold)
            #     for param in self.layer.parameters():
            #             param.grad = grad_w_manual
            #     self.opt.step()
        self.opt.zero_grad()
        g_pos_freq = g_pos.mean(0)
        g_neg_freq = g_neg.mean(0)
        pos_goodness =  self.cal_goodness(g_pos_freq)
        neg_goodness =  self.cal_goodness(g_neg_freq)
        # loss = torch.log(1 + torch.exp(torch.cat([-pos_goodness + self.threshold, neg_goodness - self.threshold]))).mean()
        loss = torch.log(1 + torch.exp(4*(-pos_goodness + neg_goodness))).mean()
        loss.backward()
        grad_w_autograd = self.layer[1].weight.grad.clone()
        # 计算手动梯度
        loss_manual, grad_w_manual = Frequency_FF_Loss(g_pos,g_neg,v_pos,v_neg,self.in_features, self.out_features, self.T, self.threshold, x_pos, x_neg, g_pos_freq, g_neg_freq)
        # # grad_w_manual = manual_gradient_computation(x_pos, self.layer[1].weight, g_pos_freq, g_neg_freq, loss, self.threshold, self.T)
        # for param in self.layer.parameters():
        #         param.grad = grad_w_manual
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
            goodness = [self.cal_goodness(g)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)



def main():
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-dims', default=[784,500], help='dimension of the network')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=1000, type=int, help='batch size')
    parser.add_argument('-epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='./data',type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')

    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
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
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    # 初始化超参数
    in_features, out_features, T ,N = 784, 500, 16, 1000
    lr, v_threshold, tau, loss_threshold = args.lr, args.v_threshold, 2.0, args.loss_threshold 
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Layer(in_features, out_features, T, lr, v_threshold, tau, loss_threshold).to(device)
    # 记录变量
    losses = []
    gradient_errors = []
    manual_mean = []
    manual_std = []
    autograd_mean = []
    autograd_std = []
    cosine_similarities = []
    pos_goodness = []
    neg_goodness = []

    save_path = "./training_metrics.png"  # 设置保存路径

    for i in range(args.epochs):
        batch_samples = 0
        
        for x, y in train_data_loader:
            batch_samples += 1
            x, y = x.to(device), y.to(device)
            label_onehot = F.one_hot(y, 10).float()
            
            x_pos = overlay_y_on_x(x, y)
            y_neg = get_y_neg(y, device)
            x_neg = overlay_y_on_x(x, y_neg)
            
            loss, grad_w_autograd, grad_w_manual, pos, neg = model.train_step(x_pos, x_neg)
            pos_goodness.append(pos.cpu().mean().item())
            neg_goodness.append(neg.cpu().mean().item())
            # 计算梯度误差
            error_w = torch.norm(grad_w_manual - grad_w_autograd).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                grad_w_manual.view(-1), grad_w_autograd.view(-1), dim=0
            ).item()

            # 记录数据
            losses.append(loss)
            gradient_errors.append(error_w)
            manual_mean.append(grad_w_manual.mean().item())
            manual_std.append(grad_w_manual.std().item())
            autograd_mean.append(grad_w_autograd.mean().item())
            autograd_std.append(grad_w_autograd.std().item())
            cosine_similarities.append(cos_sim)

            print(f"Epoch [{i+1}/{args.epochs}] Batch {batch_samples}")
            print(f"Loss: {loss:.6f}")
            print(f"梯度误差: {error_w:.6f}")
            print(f"手写梯度均值: {grad_w_manual.mean().item()}, 标准差: {grad_w_manual.std().item()}")
            print(f"自动梯度均值: {grad_w_autograd.mean().item()}, 标准差: {grad_w_autograd.std().item()}")
            print(f"梯度余弦相似度: {cos_sim:.6f}")
            # if cos_sim<0.5:
            #     break
    test_acc = 0
    test_samples = 0
    test_count = 0
    with torch.no_grad():
        for x_te, y_te in val_data_loader:
            test_samples += y_te.numel()
            test_count += 1
            x_te, y_te = x_te.to(device), y_te.to(device)
            test_acc += model.predict(x_te).eq(y_te).cpu().float().mean().item()
    print("test Acc:", 100 * test_acc / test_count, "%")
    # 绘制曲线
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))

    # Loss 曲线
    axes[0, 0].plot(losses, label="Loss", color="blue")
    # axes[0, 0].set_title("Loss Over Time")
    axes[0, 0].set_xlabel("Batch Iterations")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # 梯度误差曲线
    axes[0, 1].plot(gradient_errors, label="Gradient Error", color="red")
    # axes[0, 1].set_title("Gradient Error Over Time")
    axes[0, 1].set_xlabel("Batch Iterations")
    axes[0, 1].set_ylabel("Error")
    axes[0, 1].legend()

    # 手写梯度均值
    axes[1, 0].plot(manual_mean, label="Manual Gradient Mean", color="green")
    # axes[1, 0].set_title("Manual Gradient Mean Over Time")
    axes[1, 0].set_xlabel("Batch Iterations")
    axes[1, 0].set_ylabel("Mean Value")
    axes[1, 0].legend()

    # 自动梯度均值
    axes[1, 1].plot(autograd_mean, label="Autograd Gradient Mean", color="purple")
    # axes[1, 1].set_title("Autograd Gradient Mean Over Time")
    axes[1, 1].set_xlabel("Batch Iterations")
    axes[1, 1].set_ylabel("Mean Value")
    axes[1, 1].legend()

    # 标准差对比
    axes[2, 0].plot(manual_std, label="Manual Gradient Std", color="cyan")
    axes[2, 0].plot(autograd_std, label="Autograd Gradient Std", color="orange")
    # axes[2, 0].set_title("Gradient Standard Deviation")
    axes[2, 0].set_xlabel("Batch Iterations")
    axes[2, 0].set_ylabel("Standard Deviation")
    axes[2, 0].legend()

    # 余弦相似度
    axes[2, 1].plot(cosine_similarities, label="Cosine Similarity", color="brown")
    # axes[2, 1].set_title("Cosine Similarity Over Time")
    axes[2, 1].set_xlabel("Batch Iterations")
    axes[2, 1].set_ylabel("Similarity")
    axes[2, 1].legend()

    # Pos_goodness
    axes[3, 0].plot(pos_goodness, label="Pos_goodness", color="brown")
    # axes[3, 0].set_title("Pos_goodness Over Time")
    axes[3, 0].set_xlabel("Batch Iterations")
    axes[3, 0].set_ylabel("Pos_goodness")
    axes[3, 0].legend()
    # Neg_goodness
    axes[3, 1].plot(neg_goodness, label="Neg_goodness", color="brown")
    # axes[3, 1].set_title("Neg_goodness Over Time")
    axes[3, 1].set_xlabel("Batch Iterations")
    axes[3, 1].set_ylabel("Neg_goodness")
    axes[3, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 保存图片
    # plt.show()

    print(f"训练指标图已保存至 {save_path}")
    return -(test_acc / test_count)
if __name__ == '__main__':
    main()