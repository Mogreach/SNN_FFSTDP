import torch
import torch.nn as nn
import numpy as np
import torch
from spikingjelly.activation_based import neuron
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
def plot_grad(grad):
    # 将张量转换为 NumPy 数组以进行可视化
    grad_np = grad.cpu().numpy()  # 如果在 GPU 上，先移动到 CPU
    # 绘制热力图
    plt.figure(figsize=(12, 6))  # 设置图像大小
    plt.imshow(grad_np, cmap='viridis', aspect='auto')  # 使用 'viridis' 颜色映射，自动调整纵横比
    plt.colorbar(label='Gradient Magnitude')  # 添加颜色条并标注
    plt.title('Gradient Heatmap')  # 图像标题
    plt.xlabel('Input Dimension (784)')  # x 轴标签
    plt.ylabel('Batch Dimension (500)')  # y 轴标签
    plt.tight_layout()  # 自动调整子图参数
    plt.show()
def compute_f_in_to_grad_function():
    # 设置参数
    T = 128  # 总时间步
    x = torch.arange(-0.2, 1.2, 0.04)  # 输入值范围
    f_in_values = np.linspace(0, 1, 100)  # f_in 的取值范围
    if_node = neuron.IFNode(v_reset=None)  # 定义脉冲神经元

    # 创建网格
    X, F_IN = np.meshgrid(x, f_in_values)
    Z = np.zeros_like(X)  # 存储 firing_rate 的值

    # 计算 firing_rate
    for i, f_in in enumerate(f_in_values):
        s_list = []
        active_steps = int(f_in * T)  # 有效时间步数
        for t in range(T):
            if t < active_steps:  # 有效时间步，使用输入 x
                s_list.append(if_node(torch.tensor(X[i, :], dtype=torch.float32)).unsqueeze(0))
            else:  # 非有效时间步，输入为 0
                s_list.append(if_node(torch.zeros_like(torch.tensor(X[i, :]))).unsqueeze(0))
        
        # 计算发放率
        out_spikes = torch.cat(s_list).numpy()
        Z[i, :] = np.mean(out_spikes, axis=0)  # 平均发放率

    # 计算 firing_rate 关于 x 的导数
    Z_grad = np.gradient(Z, x.numpy(), axis=1).mean(axis=1)

    # 构造插值函数
    grad_interpolator = interp1d(f_in_values, Z_grad, kind='cubic', fill_value="extrapolate")

    return grad_interpolator
def Frequency_FF_Loss(in_features, out_features, T, threshold, in_pos_freq, in_neg_freq, out_pos_freq, out_neg_freq):
# 假设 param 是一个权重矩阵 w (形状: [out_features, in_features])
    pos_goodness = out_pos_freq.pow(2).mean(1)
    neg_goodness = out_neg_freq.pow(2).mean(1)
    loss = torch.log(1 + torch.exp(torch.cat([
        -pos_goodness + threshold,
        neg_goodness - threshold]))).mean()
# 公式为 ∂L/∂f，L 是损失，f是频率
    fr_grad_pos = torch.autograd.grad(
        outputs=loss, 
        inputs=out_pos_freq, 
        retain_graph=True,
        create_graph=False
    )[0].sum(0)
    fr_grad_neg = torch.autograd.grad(
        outputs=loss, 
        inputs=out_neg_freq, 
        retain_graph=True,
        create_graph=False
    )[0].sum(0)
    approximate_grad = compute_f_in_to_grad_function()
# 公式为 ∂f/∂x，f是损失，x是输入总电流
    f_in_grad_pos = torch.tensor(approximate_grad(in_pos_freq.detach().cpu().numpy()),device=in_pos_freq.device).to(dtype=torch.float32)
    f_in_grad_neg = torch.tensor(approximate_grad(in_neg_freq.detach().cpu().numpy()),device=in_neg_freq.device).to(dtype=torch.float32)
    pos_grad = (fr_grad_pos.view(out_features,1)) @ (((f_in_grad_pos*in_pos_freq).mean(0)).view(1,in_features))
    neg_grad = (fr_grad_neg.view(out_features,1)) @ (((f_in_grad_neg*in_neg_freq).mean(0)).view(1,in_features))
    grad = 20*( pos_grad + neg_grad)
    return loss.item(), grad


